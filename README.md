# ReChorus-ADT-Reproduction
Course Project: Reproduction of ADT model using ReChorus
# ADT: Denoising Implicit Feedback for Recommendation (ReChorus复现)

这是 机器学习 期末大作业代码仓库。
我们基于 **ReChorus** 框架，复现了 WSDM 2021 论文 **"Denoising Implicit Feedback for Recommendation" (ADT)**。

##  小组信息
* **成员**：孟昊  黄键乐
* **学号**：23330096  

##  运行环境
* Python 3.10
* PyTorch (CPU Version)
* NumPy, Pandas, Tqdm

##  快速开始

### 1. 数据准备
我们在项目中提供了自动下载和预处理数据的脚本。请在 `src` 目录下运行：

```bash
# 准备 MovieLens-1M 数据集
python get_data.py

# 准备 Amazon Grocery 数据集
python get_amazon_data.py
```
# 2. 运行模型
复现实验包含 ADT (本文模型) 以及 BPRMF、NeuMF 两个基线模型。
在 MovieLens-1M 数据集上运行：
```bash
#运行 ADT (Ours)
thon main.py --model_name ADT --dataset ml-1m --denoise_drop_rate 0.1 --path ../data/ --test_all 1 --epoch 10

# 运行 BPRMF (Baseline)
python main.py --model_name BPRMF --dataset ml-1m --path ../data/ --test_all 1 --epoch 10

# 运行 NeuMF (Baseline)
python main.py --model_name NeuMF --dataset ml-1m --path ../data/ --test_all 1 --epoch 10

#
```
在 Amazon-Grocery 数据集上运行：
```bash
# 运行 ADT
python main.py --model_name ADT --dataset Grocery_and_Gourmet_Food --denoise_drop_rate 0.1 --path ../data/ --test_all 1 --epoch 10

# 运行 BPRMF
python main.py --model_name BPRMF --dataset Grocery_and_Gourmet_Food --denoise_drop_rate 0.1 --path ../data/ --test_all 1 --epoch 10

# 运行 NeuMF
python main.py --model_name NeuMF --dataset Grocery_and_Gourmet_Food --denoise_drop_rate 0.1 --path ../data/ --test_all 1 --epoch 10
```
# 3. 运行说明:
为快速得到结果,运行代码均进行10轮训练(--epoch 10),建议再指令后加上--batch_size 2048 以提高运行速度。

# 4. 文件说明:

本项目基于 ReChorus 框架进行 ADT 模型复现，主要包含以下核心文件与修改：

### 1. 核心模型代码
* **`src/models/general/ADT.py`** **【核心工作】** ADT (Adaptive Denoising Training) 模型的完整实现代码。
  * 实现了论文中的自适应去噪损失函数 (Adaptive Denoising Loss)。
  * 包含 `Truncated Cross-Entropy` 和 `Drop Rate` 的动态计算逻辑。

### 2. 框架修复与优化
* **`src/helpers/BaseRunner.py`**  针对 ADT 模型的 Point-wise 训练方式进行了适配修改：若运行出现错误可替换为改文件。
 
### 3. 数据处理脚本
* **`src/get_data.py`** 用于自动下载并预处理 **MovieLens-1M** 数据集（转换格式、划分训练/测试集）。
* **`src/get_amazon_data.py`** 用于自动下载并预处理 **Amazon Grocery & Gourmet Food** 数据集，以满足多数据集对比实验的要求。

### 4. 项目入口
* **`src/main.py`** 项目的启动入口文件，用于解析命令行参数并加载对应的 Runner 进行训练与测试。
* **`requirements.txt`** 项目运行所需的 Python 依赖库列表（PyTorch, Pandas, NumPy, Tqdm 等）。
