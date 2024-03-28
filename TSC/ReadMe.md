# 时间跳跃连接攻击方法


## 环境

```
conda env create -f environment.yml
```
## GPU infos
```
NVIDIA GeForce RTX 4080
NVIDIA-SMI 545.37.02       Driver Version: 546.65       
CUDA Version: 12.3
```

# 数据集
被用于生成对抗样本的Kinetics-400以及UCF-101数据集从[这里](https://drive.google.com/drive/folders/1UFU2cCrm8RHk1L6PJy-rdiSzTU6-nal0?usp=drive_link)下载。 
将utils.py文件中的**UCF_DATA_ROOT**以及**Kinetic_DATA_ROOT**修改为你的数据集路径

# Models
以ResNet-50/101为主干网络的Non-local, SlowFast, TPN被作为白盒、黑盒模型。
## 权值文件
在[这里](https://drive.google.com/drive/folders/1yfDOYPYK_JbKKBX0dxpZ3RZqBts53NuZ?usp=sharing)下载

# Attack
在utils.py中的OPT_PATH设置输出路径
## 生成对抗样本.
```
python attack_kinetics.py/attack_ucf101.py
```
## 查看ASR(%)
```
python reference_kinetics.py/reference_ucf101.py
```