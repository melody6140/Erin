#!/usr/bin/env python3
"""
ASRM改进实验
对比基线BERT和改进ASRM的效果
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class ASRMTrainer:
    """ASRM实验训练器"""

    def __init__(self, model, train_loader, dev_loader, test_loader,
                 learning_rate=2e-5, num_epochs=5, device='cpu',
                 model_name='asrm_model', output_dir='asrm_outputs'):

        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.device = torch.device(device)
        self.model_name = model_name
        self.output_dir = Path(output_dir)

        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)

        # 移动模型到设备
        self.model.to(self.device)

        # 优化器和调度器
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 训练历史
        self.train_history = {
            'epochs': [],
            'train_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'val_f1_scores': []
        }

        # 最佳模型
        self.best_val_f1 = 0.0
        self.best_model_path = self.output_dir / f'best_{model_name}.pt'

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()

        total_loss = 0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(self.train_loader, desc=f"[{self.model_name}] Epoch {epoch+1}/{self.num_epochs}")

        for batch in progress_bar:
            # 移动数据到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向传播
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 计算平均损失和准确率
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)

        return {'avg_loss': avg_loss, 'accuracy': accuracy}

    def evaluate(self, data_loader, dataset_name="验证"):
        """评估模型"""
        self.model.eval()

        all_predictions = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"[{self.model_name}] 评估{dataset_name}集"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )

        # 详细分类报告
        if dataset_name == "测试":
            print(f"\n📊 [{self.model_name}] 详细分类报告:")
            print(classification_report(all_labels, all_predictions,
                                      target_names=['正常', '仇恨'], digits=4))

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(self):
        """完整训练流程"""
        print(f"\n🚀 开始训练 {self.model_name} 模型...")

        for epoch in range(self.num_epochs):
            print(f"\n{'='*50}")
            print(f"[{self.model_name}] Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*50}")

            # 训练
            train_metrics = self.train_epoch(epoch)

            # 验证
            val_metrics = self.evaluate(self.dev_loader, "验证")

            # 记录历史
            self.train_history['epochs'].append(epoch + 1)
            self.train_history['train_losses'].append(train_metrics['avg_loss'])
            self.train_history['train_accuracies'].append(train_metrics['accuracy'])
            self.train_history['val_accuracies'].append(val_metrics['accuracy'])
            self.train_history['val_f1_scores'].append(val_metrics['f1'])

            # 打印指标
            print(f"\n📊 [{self.model_name}] 训练指标:")
            print(f"  - 损失: {train_metrics['avg_loss']:.4f}")
            print(f"  - 准确率: {train_metrics['accuracy']:.4f}")

            print(f"\n📈 [{self.model_name}] 验证指标:")
            print(f"  - 准确率: {val_metrics['accuracy']:.4f}")
            print(f"  - F1分数: {val_metrics['f1']:.4f}")
            print(f"  - 精确率: {val_metrics['precision']:.4f}")
            print(f"  - 召回率: {val_metrics['recall']:.4f}")

            # 保存最佳模型
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.save_model()
                print(f"💾 保存最佳模型，验证F1: {val_metrics['f1']:.4f}")

        # 测试最佳模型
        print(f"\n🧪 [{self.model_name}] 在测试集上评估最佳模型...")
        self.load_model()
        test_metrics = self.evaluate(self.test_loader, "测试")

        print(f"\n🎯 [{self.model_name}] 最终测试结果:")
        print(f"  - 准确率: {test_metrics['accuracy']:.4f}")
        print(f"  - F1分数: {test_metrics['f1']:.4f}")
        print(f"  - 精确率: {test_metrics['precision']:.4f}")
        print(f"  - 召回率: {test_metrics['recall']:.4f}")

        # 保存结果
        results = {
            'model_name': self.model_name,
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'best_val_f1': self.best_val_f1,
            'train_history': self.train_history
        }

        results_path = self.output_dir / f'{self.model_name}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return test_metrics

    def save_model(self):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_f1': self.best_val_f1,
            'train_history': self.train_history
        }, self.best_model_path)

    def load_model(self):
        """加载模型"""
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def run_asrm_experiments(args):
    """运行ASRM改进实验"""

    print("="*80)
    print("🔬 ASRM改进实验")
    print("="*80)
    print(f"📊 实验配置:")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 学习率: {args.learning_rate}")
    print(f"  - 训练轮数: {args.num_epochs}")
    print(f"  - 最大长度: {args.max_length}")
    print(f"  - 设备: {args.device}")
    print(f"  - ASRM类型: {args.asrm_type}")
    print("="*80)

    # 加载数据
    print("\n📊 加载数据...")
    from simple_data_loader import load_simple_data
    train_loader, dev_loader, test_loader, tokenizer = load_simple_data(
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # 实验结果存储
    all_results = {}

    # 1. 基线实验
    print("\n" + "="*60)
    print("🏁 实验1: BERT基线")
    print("="*60)

    from baseline_model import BERTBaseline
    baseline_model = BERTBaseline()

    baseline_trainer = ASRMTrainer(
        model=baseline_model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        model_name='bert_baseline',
        output_dir=args.output_dir
    )

    baseline_results = baseline_trainer.train()
    all_results['baseline'] = baseline_results

    # 2. ASRM实验
    print("\n" + "="*60)
    print(f"🚀 实验2: BERT + {args.asrm_type.upper()} ASRM")
    print("="*60)

    from baseline_model import BERTWithASRM
    asrm_model = BERTWithASRM(asrm_type=args.asrm_type)

    asrm_trainer = ASRMTrainer(
        model=asrm_model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        model_name=f'bert_{args.asrm_type}_asrm',
        output_dir=args.output_dir
    )

    asrm_results = asrm_trainer.train()
    all_results['asrm'] = asrm_results

    # 3. 结果对比
    print("\n" + "="*80)
    print("📈 实验结果对比")
    print("="*80)

    print(f"🏁 BERT基线:")
    print(f"  - 准确率: {baseline_results['accuracy']:.4f}")
    print(f"  - F1分数: {baseline_results['f1']:.4f}")
    print(f"  - 精确率: {baseline_results['precision']:.4f}")
    print(f"  - 召回率: {baseline_results['recall']:.4f}")

    print(f"\n🚀 BERT + {args.asrm_type.upper()} ASRM:")
    print(f"  - 准确率: {asrm_results['accuracy']:.4f}")
    print(f"  - F1分数: {asrm_results['f1']:.4f}")
    print(f"  - 精确率: {asrm_results['precision']:.4f}")
    print(f"  - 召回率: {asrm_results['recall']:.4f}")

    # 计算改进幅度
    accuracy_improvement = (asrm_results['accuracy'] - baseline_results['accuracy']) / baseline_results['accuracy'] * 100
    f1_improvement = (asrm_results['f1'] - baseline_results['f1']) / baseline_results['f1'] * 100

    print(f"\n📊 改进幅度:")
    print(f"  - 准确率改进: {accuracy_improvement:+.2f}%")
    print(f"  - F1分数改进: {f1_improvement:+.2f}%")

    if f1_improvement > 5:
        print("🎉 显著改进！ASRM模块有效！")
    elif f1_improvement > 0:
        print("✅ 有所改进，ASRM模块有一定效果")
    else:
        print("❌ 性能下降，需要进一步调优")

    # 保存对比结果
    comparison_results = {
        'baseline': baseline_results,
        'asrm': asrm_results,
        'improvements': {
            'accuracy': accuracy_improvement,
            'f1': f1_improvement
        },
        'config': vars(args)
    }

    comparison_path = Path(args.output_dir) / 'asrm_comparison_results.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)

    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="ASRM改进实验")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--output_dir", type=str, default="asrm_outputs", help="输出目录")
    parser.add_argument("--asrm_type", type=str, default="improved",
                       choices=['improved', 'adaptive', 'multiscale'], help="ASRM类型")
    parser.add_argument("--quick_test", action="store_true", help="快速测试模式")

    args = parser.parse_args()

    # 快速测试模式
    if args.quick_test:
        args.batch_size = 8
        args.num_epochs = 3
        args.max_length = 64
        args.output_dir = "asrm_test_outputs"
        print("🚀 快速测试模式")

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 运行实验
    results = run_asrm_experiments(args)


if __name__ == "__main__":
    main()

