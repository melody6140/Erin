#!/usr/bin/env python3
"""
简单的BERT基线模型
用于与改进的ASRM模型进行对比
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class BERTBaseline(nn.Module):
    """简单的BERT分类器基线"""

    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 2,
                 dropout_rate: float = 0.1):
        super(BERTBaseline, self).__init__()

        # BERT编码器
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """前向传播"""
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 使用[CLS]token的表示
        pooled_output = outputs.pooler_output

        # Dropout和分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


class BERTWithASRM(nn.Module):
    """集成改进ASRM的BERT模型"""

    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 2,
                 dropout_rate: float = 0.1, asrm_type: str = 'improved'):
        super(BERTWithASRM, self).__init__()

        # BERT编码器
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # 导入改进的ASRM模块
        from improved_asrm import ImprovedASRM, AdaptiveASRM, MultiScaleASRM

        # 选择ASRM类型
        if asrm_type == 'improved':
            self.asrm = ImprovedASRM(self.hidden_size)
        elif asrm_type == 'adaptive':
            self.asrm = AdaptiveASRM(self.hidden_size)
        elif asrm_type == 'multiscale':
            self.asrm = MultiScaleASRM(self.hidden_size)
        else:
            raise ValueError(f"Unknown ASRM type: {asrm_type}")

        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """前向传播"""
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 获取最后一层隐藏状态
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 应用改进的ASRM
        enhanced_states = self.asrm(hidden_states, apply_augmentation=True)

        # 获取[CLS]表示
        cls_representation = enhanced_states[:, 0, :]  # [batch_size, hidden_size]

        # 分类
        pooled_output = self.dropout(cls_representation)
        logits = self.classifier(pooled_output)

        return logits


def test_models():
    """测试模型"""
    print("🧪 测试基线模型...")

    # 创建测试数据
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # 测试基线模型
    print("\n1️⃣ 测试BERT基线...")
    baseline_model = BERTBaseline()
    baseline_output = baseline_model(input_ids, attention_mask)
    print(f"✅ 基线输出形状: {baseline_output.shape}")

    # 测试ASRM模型
    print("\n2️⃣ 测试BERT+改进ASRM...")
    asrm_model = BERTWithASRM(asrm_type='improved')
    asrm_output = asrm_model(input_ids, attention_mask)
    print(f"✅ ASRM输出形状: {asrm_output.shape}")

    print("\n🎉 模型测试通过！")


if __name__ == "__main__":
    test_models()

