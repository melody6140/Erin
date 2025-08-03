#!/usr/bin/env python3
"""
BERT Baseline 主程序
最简单的仇恨言论检测实验
"""

import torch
import argparse
from baseline_model import BERTBaseline
from baseline_trainer import BaselineTrainer
from simple_data_loader import load_simple_data

def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description="BERT Baseline 仇恨言论检测")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--device", type=str, default="cpu", help="设备")

    args = parser.parse_args()

    print("="*50)
    print("BERT Baseline 仇恨言论检测")
    print("="*50)
    print(f"设备: {args.device}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"最大长度: {args.max_length}")
    print("="*50)

    try:
        # 加载数据
        train_loader, dev_loader, test_loader, tokenizer = load_simple_data(
            batch_size=args.batch_size,
            max_length=args.max_length
        )

        # 创建模型
        print("\n创建BERT模型...")
        model = BERTBaseline()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {param_count:,}")

        # 创建训练器
        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            device=args.device
        )

        # 训练
        test_metrics = trainer.train()

        print(f"\n🎉 训练完成！")
        print(f"📊 最终测试结果:")
        print(f"  - 准确率: {test_metrics['accuracy']:.4f}")
        print(f"  - 精确率: {test_metrics['precision']:.4f}")
        print(f"  - 召回率: {test_metrics['recall']:.4f}")
        print(f"  - F1分数: {test_metrics['f1']:.4f}")

        # 预测示例
        print(f"\n🔮 预测示例:")
        sample_texts = [
            "I love everyone regardless of their background.",
            "These people are ruining our country.",
            "Diversity makes us stronger.",
            "We need to get rid of these immigrants."
        ]

        predictions = trainer.predict(sample_texts, tokenizer, args.max_length)

        for i, pred in enumerate(predictions, 1):
            print(f"\n示例 {i}:")
            print(f"  文本: {pred['text']}")
            print(f"  预测: {pred['predicted_class']}")
            print(f"  置信度: {pred['confidence']:.4f}")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

