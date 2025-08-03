#!/usr/bin/env python3
"""
简化的数据加载器
只处理基本的文本分类任务
"""

import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class SimpleHateSpeechDataset(Dataset):
    """简单的仇恨言论数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # 处理NaN标签
        label_value = self.labels[idx]
        if pd.isna(label_value):
            label = 0  # 默认为非仇恨言论
        else:
            label = int(float(label_value))  # 先转float再转int，处理可能的字符串数字

        # 编码文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_simple_data(data_dir='dataset', model_name='bert-base-uncased',
                    batch_size=16, max_length=128):
    """加载简化的数据"""

    print("加载数据...")

    # 加载数据文件
    train_df = pd.read_csv(os.path.join(data_dir, 'hateval2019_en_train.csv'))
    dev_df = pd.read_csv(os.path.join(data_dir, 'hateval2019_en_dev.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'hateval2019_en_test.csv'))

    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(dev_df)}")
    print(f"测试集大小: {len(test_df)}")

    # 数据清理和过滤
    def clean_and_filter_data(df):
        df = df.copy()

        # 文本清理
        df['text'] = df['text'].str.replace('&amp;', '&')
        df['text'] = df['text'].str.replace('&lt;', '<')
        df['text'] = df['text'].str.replace('&gt;', '>')

        # 过滤掉空文本
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.strip() != '']

        # 处理标签
        # 如果HS列有NaN，填充为0
        if 'HS' in df.columns:
            df['HS'] = df['HS'].fillna(0)
            # 确保标签是0或1
            df['HS'] = df['HS'].astype(float).astype(int)
            # 过滤掉无效标签
            df = df[df['HS'].isin([0, 1])]

        return df

    print("清理数据...")
    train_df = clean_and_filter_data(train_df)
    dev_df = clean_and_filter_data(dev_df)
    test_df = clean_and_filter_data(test_df)

    print(f"清理后训练集大小: {len(train_df)}")
    print(f"清理后验证集大小: {len(dev_df)}")
    print(f"清理后测试集大小: {len(test_df)}")

    # 标签分布
    print(f"训练集标签分布: {train_df['HS'].value_counts().to_dict()}")

    # 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 创建数据集
    train_dataset = SimpleHateSpeechDataset(
        train_df['text'].values,
        train_df['HS'].values,
        tokenizer,
        max_length
    )

    dev_dataset = SimpleHateSpeechDataset(
        dev_df['text'].values,
        dev_df['HS'].values,
        tokenizer,
        max_length
    )

    test_dataset = SimpleHateSpeechDataset(
        test_df['text'].values,
        test_df['HS'].values,
        tokenizer,
        max_length
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # macOS兼容性
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, dev_loader, test_loader, tokenizer

