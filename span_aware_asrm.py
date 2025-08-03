#!/usr/bin/env python3
"""
Span-aware ASRM模块
基于深度分析，重新设计ASRM以捕获span-level语义信息
解决原始ASRM的根本问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class SpanAwareASRM(nn.Module):
    """
    Span感知的ASRM模块

    核心改进：
    1. 从token-level到span-level建模
    2. 保持位置和结构信息
    3. 增强特征判别性
    4. 适配对比学习需求
    """

    def __init__(self, hidden_size: int = 768, window_sizes: List[int] = [1, 3, 5],
                 reduction_ratio: int = 4):
        super(SpanAwareASRM, self).__init__()

        self.hidden_size = hidden_size
        self.window_sizes = window_sizes
        self.num_windows = len(window_sizes)

        # 多窗口卷积层 - 捕获不同长度的span
        self.span_convs = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size=ws,
                     padding=ws//2, groups=hidden_size//4)  # 使用分组卷积减少参数
            for ws in window_sizes
        ])

        # Span融合层
        self.span_fusion = nn.Sequential(
            nn.Linear(hidden_size * self.num_windows, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # 位置感知的注意力机制
        self.position_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 语义重要性预测器
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // reduction_ratio),
            nn.GELU(),
            nn.Linear(hidden_size // reduction_ratio, 1),
            nn.Sigmoid()
        )

        # 对比学习友好的特征增强器
        self.contrastive_enhancer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                apply_augmentation: bool = True) -> torch.Tensor:
        """
        前向传播

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size] - 用于对比学习增强
            apply_augmentation: 是否应用增强
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 1. Span-level特征提取
        span_features = self._extract_span_features(hidden_states)

        # 2. 位置感知的注意力
        position_enhanced_features = self._apply_position_attention(
            span_features, attention_mask
        )

        # 3. 语义重要性计算
        importance_scores = self.importance_predictor(position_enhanced_features)

        # 4. 特征校准
        calibrated_features = position_enhanced_features * importance_scores

        # 5. 对比学习友好的增强
        if apply_augmentation and self.training:
            enhanced_features = self._apply_contrastive_enhancement(
                calibrated_features, labels
            )
        else:
            enhanced_features = calibrated_features

        # 6. 残差连接
        output = (
            torch.sigmoid(self.residual_weight) * enhanced_features +
            (1 - torch.sigmoid(self.residual_weight)) * hidden_states
        )

        return output

    def _extract_span_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """提取多尺度span特征"""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 转换为卷积格式 [batch_size, hidden_size, seq_len]
        conv_input = hidden_states.transpose(1, 2)

        span_outputs = []
        for conv in self.span_convs:
            # 应用卷积获取span特征
            span_output = conv(conv_input)  # [batch_size, hidden_size, seq_len]
            span_output = F.gelu(span_output)
            span_outputs.append(span_output.transpose(1, 2))  # 转回 [batch_size, seq_len, hidden_size]

        # 拼接多尺度特征
        concatenated = torch.cat(span_outputs, dim=-1)  # [batch_size, seq_len, hidden_size * num_windows]

        # 融合多尺度特征
        fused_features = self.span_fusion(concatenated)  # [batch_size, seq_len, hidden_size]

        return fused_features

    def _apply_position_attention(self, features: torch.Tensor,
                                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """应用位置感知的注意力机制"""

        # 创建注意力掩码
        if attention_mask is not None:
            # 转换为注意力掩码格式
            attn_mask = attention_mask.unsqueeze(1).expand(-1, features.size(1), -1)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
        else:
            attn_mask = None

        # 自注意力机制
        attended_features, _ = self.position_attention(
            features, features, features,
            attn_mask=attn_mask
        )

        return attended_features

    def _apply_contrastive_enhancement(self, features: torch.Tensor,
                                     labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """应用对比学习友好的特征增强"""

        # 基础特征增强
        enhanced = self.contrastive_enhancer(features)

        if labels is not None and self.training:
            # 类别感知的特征增强
            enhanced = self._enhance_inter_class_difference(enhanced, labels)

        return enhanced

    def _enhance_inter_class_difference(self, features: torch.Tensor,
                                      labels: torch.Tensor) -> torch.Tensor:
        """增强类间差异"""
        batch_size = features.size(0)

        # 计算类别中心
        unique_labels = torch.unique(labels)
        class_centers = []

        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                class_center = features[mask].mean(dim=0, keepdim=True)
                class_centers.append(class_center)

        if len(class_centers) > 1:
            # 计算类间距离
            center_diff = class_centers[0] - class_centers[1]

            # 增强类间差异
            for i, label in enumerate(labels):
                if label == unique_labels[0]:
                    features[i] = features[i] + 0.1 * center_diff.squeeze(0)
                else:
                    features[i] = features[i] - 0.1 * center_diff.squeeze(0)

        return features


class ContrastiveFriendlyASRM(nn.Module):
    """
    对比学习友好的ASRM模块
    专门为对比学习任务设计
    """

    def __init__(self, hidden_size: int = 768, temperature: float = 0.07):
        super(ContrastiveFriendlyASRM, self).__init__()

        self.hidden_size = hidden_size
        self.temperature = temperature

        # Span-aware基础模块
        self.span_asrm = SpanAwareASRM(hidden_size)

        # 对比学习特定的投影头
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2)
        )

        # 特征判别器
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 2)  # 二分类
        )

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_contrastive_features: bool = False) -> dict:
        """
        前向传播，返回多种特征用于对比学习
        """

        # 基础span-aware增强
        enhanced_features = self.span_asrm(
            hidden_states, attention_mask, labels, apply_augmentation=True
        )

        # CLS表示
        cls_features = enhanced_features[:, 0, :]  # [batch_size, hidden_size]

        # 对比学习投影
        contrastive_features = self.projection_head(cls_features)
        contrastive_features = F.normalize(contrastive_features, p=2, dim=1)

        # 分类特征
        classification_logits = self.discriminator(cls_features)

        results = {
            'enhanced_features': enhanced_features,
            'cls_features': cls_features,
            'classification_logits': classification_logits
        }

        if return_contrastive_features:
            results['contrastive_features'] = contrastive_features

        return results


class HybridDCLASRM(nn.Module):
    """
    混合DCL-ASRM模块
    结合双重对比学习和Span-aware ASRM的优势
    """

    def __init__(self, hidden_size: int = 768, num_classes: int = 2):
        super(HybridDCLASRM, self).__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # Span-aware ASRM核心
        self.span_asrm = SpanAwareASRM(hidden_size)

        # 自监督对比学习分支
        self.self_supervised_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 128)
        )

        # 监督对比学习分支
        self.supervised_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 128)
        )

        # 主分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # 焦点损失权重
        self.focal_alpha = nn.Parameter(torch.tensor(1.0))
        self.focal_gamma = nn.Parameter(torch.tensor(2.0))

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        """
        前向传播，实现混合DCL-ASRM
        """

        # Span-aware特征增强
        enhanced_features = self.span_asrm(
            hidden_states, attention_mask, labels, apply_augmentation=True
        )

        # CLS表示
        cls_features = enhanced_features[:, 0, :]

        # 分类输出
        classification_logits = self.classifier(cls_features)

        # 对比学习特征
        self_supervised_features = F.normalize(
            self.self_supervised_head(cls_features), p=2, dim=1
        )
        supervised_features = F.normalize(
            self.supervised_head(cls_features), p=2, dim=1
        )

        return {
            'classification_logits': classification_logits,
            'self_supervised_features': self_supervised_features,
            'supervised_features': supervised_features,
            'enhanced_features': enhanced_features,
            'cls_features': cls_features
        }


def test_span_aware_asrm():
    """测试Span-aware ASRM模块"""
    print("🧪 测试Span-aware ASRM模块...")

    # 创建测试数据
    batch_size, seq_len, hidden_size = 4, 128, 768
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.tensor([0, 1, 0, 1])

    # 测试基础Span-aware ASRM
    print("\n1️⃣ 测试基础Span-aware ASRM...")
    span_asrm = SpanAwareASRM(hidden_size=hidden_size)
    output1 = span_asrm(hidden_states, attention_mask, labels)
    print(f"✅ 输入: {hidden_states.shape} -> 输出: {output1.shape}")

    # 测试对比学习友好的ASRM
    print("\n2️⃣ 测试对比学习友好的ASRM...")
    contrastive_asrm = ContrastiveFriendlyASRM(hidden_size=hidden_size)
    output2 = contrastive_asrm(hidden_states, attention_mask, labels, return_contrastive_features=True)
    print(f"✅ 分类logits: {output2['classification_logits'].shape}")
    print(f"✅ 对比特征: {output2['contrastive_features'].shape}")

    # 测试混合DCL-ASRM
    print("\n3️⃣ 测试混合DCL-ASRM...")
    hybrid_asrm = HybridDCLASRM(hidden_size=hidden_size)
    output3 = hybrid_asrm(hidden_states, attention_mask, labels)
    print(f"✅ 分类logits: {output3['classification_logits'].shape}")
    print(f"✅ 自监督特征: {output3['self_supervised_features'].shape}")
    print(f"✅ 监督特征: {output3['supervised_features'].shape}")

    print("\n🎉 所有Span-aware ASRM模块测试通过！")


if __name__ == "__main__":
    test_span_aware_asrm()

