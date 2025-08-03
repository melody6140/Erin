#!/usr/bin/env python3
"""
增强的SRAF模型
集成SpanAware ASRM，解决原始ASRM的语义层次不匹配问题
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# 导入SpanAware ASRM
from span_aware_asrm import SpanAwareASRM, ContrastiveFriendlyASRM, HybridDCLASRM


class AttentionEntropyRegularizer(nn.Module):
    """注意力熵正则化器"""

    def __init__(self):
        super(AttentionEntropyRegularizer, self).__init__()

    def compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        计算注意力熵
        Args:
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        Returns:
            注意力熵 [scalar]
        """
        eps = 1e-8
        attention_weights_safe = attention_weights.clone() + eps

        # 计算熵: H(α) = -Σ(α_i * log(α_i))
        entropy = -torch.sum(attention_weights_safe * torch.log(attention_weights_safe), dim=-1)

        # 对所有头和位置求平均
        mean_entropy = torch.mean(entropy)

        return mean_entropy


class CognitiveDistortionActivationOperator(nn.Module):
    """认知扭曲激活算子 (CDAO)"""

    def __init__(self, hidden_size: int, num_distortion_types: int = 5):
        super(CognitiveDistortionActivationOperator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_distortion_types)
        )

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_output: [batch_size, hidden_size]
        Returns:
            认知扭曲预测 [batch_size, num_distortion_types]
        """
        return self.classifier(pooled_output)


class ContrastiveLearningHead(nn.Module):
    """对比学习头"""

    def __init__(self, hidden_size: int, projection_dim: int = 128):
        super(ContrastiveLearningHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, hidden_size]
        Returns:
            投影特征 [batch_size, projection_dim]
        """
        return F.normalize(self.projection(x), p=2, dim=1)


class EnhancedSRAFModel(nn.Module):
    """
    增强的SRAF模型
    集成SpanAware ASRM，解决原始实现的问题
    """

    def __init__(self, config, asrm_type: str = 'span_aware'):
        super(EnhancedSRAFModel, self).__init__()
        self.config = config

        # 加载预训练BERT模型
        self.bert_config = AutoConfig.from_pretrained(config.model_name)
        self.bert_config.output_attentions = True
        self.bert = AutoModel.from_pretrained(config.model_name, config=self.bert_config)

        # 选择ASRM类型
        if asrm_type == 'span_aware':
            self.asrm = SpanAwareASRM(
                hidden_size=config.hidden_size,
                window_sizes=[1, 3, 5],
                reduction_ratio=4
            )
        elif asrm_type == 'contrastive_friendly':
            self.asrm = ContrastiveFriendlyASRM(
                hidden_size=config.hidden_size,
                temperature=config.contrastive_temperature
            )
        elif asrm_type == 'hybrid_dcl':
            self.asrm = HybridDCLASRM(
                hidden_size=config.hidden_size,
                num_classes=config.num_labels
            )
        else:
            raise ValueError(f"Unknown ASRM type: {asrm_type}")

        self.asrm_type = asrm_type

        # 注意力熵正则化器
        self.entropy_regularizer = AttentionEntropyRegularizer()

        # 认知扭曲激活算子
        self.cdao = CognitiveDistortionActivationOperator(
            hidden_size=config.hidden_size,
            num_distortion_types=config.num_cognitive_distortions
        )

        # 对比学习头
        self.contrastive_head = ContrastiveLearningHead(config.hidden_size)

        # 主分类器
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_labels)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        cognitive_distortion_labels: Optional[torch.Tensor] = None,
        return_contrastive_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        attentions = outputs.attentions

        # 应用增强的ASRM
        if self.asrm_type == 'contrastive_friendly':
            asrm_outputs = self.asrm(
                sequence_output,
                attention_mask,
                labels,
                return_contrastive_features=True
            )
            enhanced_pooled_output = asrm_outputs['cls_features']
            logits = asrm_outputs['classification_logits']
            contrastive_features = asrm_outputs.get('contrastive_features', None)
        elif self.asrm_type == 'hybrid_dcl':
            asrm_outputs = self.asrm(sequence_output, attention_mask, labels)
            enhanced_pooled_output = asrm_outputs['cls_features']
            logits = asrm_outputs['classification_logits']
            contrastive_features = asrm_outputs['supervised_features']
        else:  # span_aware
            enhanced_sequence_output = self.asrm(
                sequence_output, attention_mask, labels, apply_augmentation=True
            )
            enhanced_pooled_output = torch.mean(enhanced_sequence_output, dim=1)
            logits = self.classifier(enhanced_pooled_output)
            contrastive_features = self.contrastive_head(enhanced_pooled_output)

        # 认知扭曲预测
        cdao_logits = self.cdao(enhanced_pooled_output)

        # 计算注意力熵
        attention_entropy = 0
        if attentions:
            for attention in attentions:
                attention_entropy += self.entropy_regularizer.compute_attention_entropy(attention)
            attention_entropy /= len(attentions)

        results = {
            'logits': logits,
            'cdao_logits': cdao_logits,
            'contrastive_features': contrastive_features,
            'attention_entropy': attention_entropy,
            'enhanced_features': enhanced_pooled_output
        }

        # 计算损失
        if labels is not None:
            loss_dict = self.compute_loss(
                logits=logits,
                labels=labels,
                cdao_logits=cdao_logits,
                cognitive_distortion_labels=cognitive_distortion_labels,
                attention_entropy=attention_entropy,
                contrastive_features=contrastive_features
            )
            results.update(loss_dict)

        return results

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        cdao_logits: torch.Tensor,
        cognitive_distortion_labels: Optional[torch.Tensor],
        attention_entropy: torch.Tensor,
        contrastive_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算总损失"""

        # 主任务损失
        main_loss = F.cross_entropy(logits, labels)

        # 注意力熵正则化损失
        entropy_loss = -attention_entropy

        # 认知扭曲损失
        cdao_loss = torch.tensor(0.0, device=logits.device)
        if cognitive_distortion_labels is not None:
            cdao_loss = F.cross_entropy(cdao_logits, cognitive_distortion_labels)

        # 对比学习损失
        contrastive_loss = self.compute_contrastive_loss(contrastive_features, labels)

        # 总损失
        total_loss = (
            main_loss +
            self.config.entropy_weight * entropy_loss +
            self.config.cdao_weight * cdao_loss +
            self.config.contrastive_weight * contrastive_loss
        )

        return {
            'loss': total_loss,
            'main_loss': main_loss,
            'entropy_loss': entropy_loss,
            'cdao_loss': cdao_loss,
            'contrastive_loss': contrastive_loss
        }

    def compute_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算监督对比学习损失"""
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.config.contrastive_temperature

        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 移除对角线
        diagonal_mask = torch.eye(batch_size, device=device)
        mask = mask * (1 - diagonal_mask)

        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        exp_sim = exp_sim * (1 - diagonal_mask)

        pos_sim = exp_sim * mask
        neg_sim = exp_sim * (1 - mask)

        pos_sum = pos_sim.sum(dim=1)
        neg_sum = neg_sim.sum(dim=1)

        pos_sum = torch.clamp(pos_sum, min=1e-8)
        total_sum = pos_sum + neg_sum

        loss = -torch.log(pos_sum / total_sum)

        mask_pos = (mask.sum(dim=1) > 0).float()
        loss = loss * mask_pos

        return loss.sum() / torch.clamp(mask_pos.sum(), min=1.0)


def test_enhanced_sraf():
    """测试增强的SRAF模型"""
    print("🧪 测试增强的SRAF模型...")

    # 创建配置
    class TestConfig:
        model_name = "bert-base-uncased"
        hidden_size = 768
        num_labels = 2
        dropout = 0.1
        num_cognitive_distortions = 5
        entropy_weight = 0.1
        cdao_weight = 0.05
        contrastive_weight = 0.1
        contrastive_temperature = 0.07

    config = TestConfig()

    # 创建测试数据
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.tensor([0, 1, 0, 1])

    # 测试不同类型的ASRM
    asrm_types = ['span_aware', 'contrastive_friendly', 'hybrid_dcl']

    for asrm_type in asrm_types:
        print(f"\n🔬 测试 {asrm_type} ASRM...")

        try:
            model = EnhancedSRAFModel(config, asrm_type=asrm_type)
            outputs = model(input_ids, attention_mask, labels)

            print(f"✅ {asrm_type} 测试通过:")
            print(f"  - 分类logits: {outputs['logits'].shape}")
            print(f"  - CDAO logits: {outputs['cdao_logits'].shape}")
            print(f"  - 对比特征: {outputs['contrastive_features'].shape}")
            print(f"  - 总损失: {outputs['loss'].item():.4f}")

        except Exception as e:
            print(f"❌ {asrm_type} 测试失败: {e}")

    print("\n🎉 增强SRAF模型测试完成！")


if __name__ == "__main__":
    test_enhanced_sraf()

