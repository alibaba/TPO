# TPO
论文链接： https://arxiv.org/pdf/2412.14487

## 1. 环境配置

本项目依赖的Python库已在 `requirements.txt` 文件中列出。请使用以下命令安装所有必需的依赖项。

```bash
pip install -r requirements.txt
```

数据路径在/tpo/offline_dpo_rlhfv_5K_reform.json 还需要去下载RLHFV图片 https://github.com/RLHF-V/RLHF-V
## 2. 启动训练
项目训练的启动脚本是 /tpo/scripts/token_tpo.sh。该脚本封装了启动训练所需的全部参数和命令。

直接在终端中运行以下命令即可开始训练过程：
```
bash
bash /tpo/scripts/token_tpo.sh
```
## 3. 主程序入口说明
训练的主程序入口文件是 /tpo/llava/train/scaling_train.py

该脚本负责初始化模型、数据加载器以及训练器。一个关键的预处理步骤也在此文件中实现：为输入的图像数据加入噪声。这是我们模型训练中的一个重要环节。

## 4. 核心训练循环说明
TPO训练的核心循环逻辑位于 /tpo/trl/trainer/tpo_trainer.py 文件中。
加入了token_level函数：

    正样本 _get_token_batch_logps_reciprocal()
    
    负样本 _get_token_batch_logps()
## 5. 开源项目致谢
本项目的代码实现主要基于以下优秀的开源项目，我们在此对原作者表示衷心的感谢：

LLaVA (Large Language and Vision Assistant): https://github.com/haotian-liu/LLaVA
TRL (Transformer Reinforcement Learning): https://github.com/huggingface/trl
## 6. 引用
如果您在您的研究中使用了我们的项目，请考虑引用我们的论文：
```
bibtex
@article{gu2024token,
  title={Token preference optimization with self-calibrated visual-anchored rewards for hallucination mitigation},
  author={Gu, Jihao and Wang, Yingyao and Cao, Meng and Bu, Pi and Song, Jun and He, Yancheng and Li, Shilong and Zheng, Bo},
  journal={arXiv preprint arXiv:2412.14487},
  year={2024}
}
```