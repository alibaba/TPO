import math
import pandas as pd
import subprocess
import transformers
from transformers.utils import logging
from transformers.integrations import WandbCallback
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from functools import partial
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import torch.nn as nn

logger = logging.get_logger(__name__)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logger.info(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


class ProfilerCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()
        if state.global_step > 20:
            print(self.prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))


class LastStepCallback(TrainerCallback):
    def __init__(self):
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save = True
        return control


class MemoryUsageCallback(TrainerCallback):
    def __init__(self):
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if torch.cuda.is_available():
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            gpu_memory = [f"{float(x) / (1024.0 * 1024.0)}:.2f" for x in result.strip().split('\n')]  # 调整单位为GB
            if state.is_local_process_zero:
                print(f'=======> Current GPU Memory Usage: {dict(zip(range(len(gpu_memory)), gpu_memory))}')


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each logging step during training.
    It allows to visualize the model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset for generating predictions.
        num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=16, freq=200):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)

        def decode_predictions(tokenizer, predictions):
            labels = tokenizer.batch_decode(predictions.label_ids)
            prediction_text = tokenizer.batch_decode(predictions.predictions.argmax(axis=-1))
            return {"labels": labels, "predictions": prediction_text}

        # control the frequency of logging by logging the predictions every `freq` epochs
        if state.global_step % self.freq == 0:
            # generate predictions
            predictions = self.trainer.predict(self.sample_dataset)
            # decode predictions and labels
            predictions = decode_predictions(self.tokenizer, predictions)
            # add predictions to a wandb.Table
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # log the table to wandb
            self._wandb.log({"sample_predictions": records_table})


def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_ratio: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(min_lr_ratio, 0.5 * (1.0 + min_lr_ratio + (1 - min_lr_ratio) * math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5,
        min_lr_ratio: float = 0, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr_ratio (`float`, *optional*, defaults to 0.0):
            A scalar. A lower bound ratio for the learning rate of all param groups. Default: 0.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""
    import torch.nn as nn
    from llava.model.multimodal_encoder.evaclip.modeling_evaclip import Attention as EvaAttention
    from llava.model.multimodal_encoder.clipa.modeling_clipa import Attention as ClipaAttention

    def _convert_weights(l):

        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, EvaAttention, ClipaAttention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, nn.Parameter):
            l.data = l.data.to(dtype)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name) and isinstance(l, nn.Parameter):
                attr = getattr(l, name, None)
                if attr is not None:
                    attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)
