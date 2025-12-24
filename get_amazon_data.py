# -*- coding: UTF-8 -*-
import os
import gzip
import json
import pandas as pd
import numpy as np
import urllib.request

# 1. 设置路径
dataset_name = 'Grocery_and_Gourmet_Food'
# 定位到 data 目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_dir = os.path.join(project_root, 'data', dataset_name)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"创建目录: {data_dir}")

# 2. 下载数据 (使用 5-core 版本，数据量适中)
# 链接来自 UCSD Amazon 数据集
file_name = 'reviews_Grocery_and_Gourmet_Food_5.json.gz'
url = f'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{file_name}'
save_path = os.path.join(data_dir, file_name)

if not os.path.exists(save_path):
    print(f"正在下载 {dataset_name} (约 30MB)...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print("下载完成。")
    except Exception as e:
        print(f"下载失败: {e}")
        exit()
else:
    print("文件已存在，跳过下载。")

# 3. 读取并处理 (JSON -> CSV)
print("正在读取并转换格式...")
data = []
with gzip.open(save_path, 'r') as f:
    for line in f:
        line_data = json.loads(line)
        # 提取 ReChorus 需要的字段: user_id, item_id, rating, timestamp
        data.append([
            line_data['reviewerID'], 
            line_data['asin'], 
            line_data['overall'], 
            line_data.get('unixReviewTime', 0)
        ])

df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'time'])

# 4. 重新编码 ID (字符串 -> 数字)
print("正在重新编码 ID...")
user_id_map = {id: i for i, id in enumerate(df['user_id'].unique())}
item_id_map = {id: i for i, id in enumerate(df['item_id'].unique())}
df['user_id'] = df['user_id'].map(user_id_map)
df['item_id'] = df['item_id'].map(item_id_map)

# 5. 切分数据集 (8:1:1)
print("正在切分 Train/Dev/Test...")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df)
train = df.iloc[:int(0.8*n)]
dev = df.iloc[int(0.8*n):int(0.9*n)]
test = df.iloc[int(0.9*n):]

# 6. 保存
train.to_csv(os.path.join(data_dir, 'train.csv'), index=False, sep='\t')
dev.to_csv(os.path.join(data_dir, 'dev.csv'), index=False, sep='\t')
test.to_csv(os.path.join(data_dir, 'test.csv'), index=False, sep='\t')

print("-" * 30)
print(f"✅ {dataset_name} 数据集准备完毕！")
print("包含文件: train.csv, dev.csv, test.csv")