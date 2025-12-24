# -*- coding: UTF-8 -*-
import os
import io
import requests
import zipfile
import pandas as pd

# 1. 自动定位到项目根目录下的 data/ml-1m
# 无论你在哪运行，这行代码都能找到正确位置
current_file_path = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_dir)
data_dir = os.path.join(project_root, 'data', 'ml-1m')

print(f"目标文件夹: {data_dir}")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 2. 下载官方数据 (MovieLens-1M)
url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
print(f"正在下载数据: {url} ...")

try:
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # 解压到 data 目录下 (zip包里自带 ml-1m 文件夹结构)
    extract_root = os.path.join(project_root, 'data')
    z.extractall(extract_root)
    print("下载并解压成功！")
except Exception as e:
    print(f"下载失败: {e}")
    print("请检查网络，或手动下载解压到 data 目录。")
    exit()

# 3. 格式转换与切分
print("正在处理数据格式 (转换为 train/dev/test.csv)...")
raw_file = os.path.join(data_dir, 'ratings.dat')

if not os.path.exists(raw_file):
    print(f"❌ 错误: 未找到 {raw_file}")
    exit()

# 读取原始 .dat 文件
df = pd.read_csv(raw_file, sep='::', header=None, names=['user_id', 'item_id', 'rating', 'time'], engine='python', encoding='ISO-8859-1')

# 随机打乱
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df)

# 按 8:1:1 切分
train_df = df.iloc[:int(0.8*n)]
dev_df = df.iloc[int(0.8*n):int(0.9*n)]
test_df = df.iloc[int(0.9*n):]

# 保存为 ReChorus 需要的 csv 格式 (tab分隔)
train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False, sep='\t')
dev_df.to_csv(os.path.join(data_dir, 'dev.csv'), index=False, sep='\t')
test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False, sep='\t')

print("-" * 30)
print("✅ 修复完成！")
print(f"数据已生成在: {data_dir}")
print("包含文件: train.csv, dev.csv, test.csv")
print("现在你可以直接运行 main.py 了！")
