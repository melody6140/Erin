#!/usr/bin/env python3
"""
改进的ASRM模块 (Attention-guided Semantic Recalibration Module)
解决原始实现中的过度正则化和语义破坏问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedASRM(nn.Module):
    """
    改进的注意力引导语义校准模块

    主要改进：
    1. 添加残差连接，保持原始语义信息
    2. 移除过度正则化的dropout
    3. 使用更细粒度的语义捕获
    4. 添加可学习的融合权重
    """

    def __init__(self, hidden_size: int = 768, reduction_ratio: int = 8,
                 use_max_pooling: bool = False, residual_weight: float = 0.5):
        super(ImprovedASRM, self).__init__()

        self.hidden_size = hidden_size
        self.reduction_ratio = reduction_ratio
        self.use_max_pooling = use_max_pooling

        # 使用更小的压缩比例，保留更多信息
        reduced_size = max(1, hidden_size // reduction_ratio)

        # 改进的激励网络
        self.excitation = nn.Sequential(
            nn.Linear(hidden_size, reduced_size),
            nn.SiLU(),  # 使用SiLU激活函数，比ReLU更平滑
            nn.Linear(reduced_size, hidden_size),
            nn.Sigmoid()
        )

        # 可学习的残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(residual_weight))

        # 层归一化，稳定训练
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 移除dropout，减少过度正则化

    def forward(self, hidden_states: torch.Tensor,
                apply_augmentation: bool = True) -> torch.Tensor:
        """
        前向传播

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            apply_augmentation: 是否应用增强

        Returns:
            enhanced_states: 增强后的隐藏状态
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 1. 全局信息提取（改进版）
        if self.use_max_pooling:
            # 使用最大池化捕获显著特征
            global_info = torch.max(hidden_states, dim=1)[0]
        else:
            # 使用平均池化捕获整体语义
            global_info = torch.mean(hidden_states, dim=1)

        # 2. 生成注意力权重
        attention_weights = self.excitation(global_info)  # [batch_size, hidden_size]
        attention_weights = attention_weights.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # 3. 校准操作
        calibrated_states = attention_weights * hidden_states

        # 4. 残差连接（关键改进）
        if apply_augmentation:
            # 使用可学习的权重融合原始和校准后的特征
            enhanced_states = (
                torch.sigmoid(self.residual_weight) * calibrated_states +
                (1 - torch.sigmoid(self.residual_weight)) * hidden_states
            )
        else:
            enhanced_states = calibrated_states

        # 5. 层归一化
        enhanced_states = self.layer_norm(enhanced_states)

        return enhanced_states


class AdaptiveASRM(nn.Module):
    """
    自适应ASRM模块
    根据输入动态调整校准强度
    """

    def __init__(self, hidden_size: int = 768, reduction_ratio: int = 8):
        super(AdaptiveASRM, self).__init__()

        self.hidden_size = hidden_size

        # 校准强度预测器
        self.intensity_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

        # 基础ASRM模块
        self.base_asrm = ImprovedASRM(hidden_size, reduction_ratio)

    def forward(self, hidden_states: torch.Tensor,
                apply_augmentation: bool = True) -> torch.Tensor:
        """
        自适应前向传播
        """
        # 预测校准强度
        global_context = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]
        intensity = self.intensity_predictor(global_context)  # [batch_size, 1]

        # 应用基础ASRM
        enhanced_states = self.base_asrm(hidden_states, apply_augmentation)

        # 根据预测强度调整输出
        intensity = intensity.unsqueeze(1)  # [batch_size, 1, 1]
        adaptive_output = (
            intensity * enhanced_states +
            (1 - intensity) * hidden_states
        )

        return adaptive_output


class MultiScaleASRM(nn.Module):
    """
    多尺度ASRM模块
    在不同尺度上进行语义校准
    """

    def __init__(self, hidden_size: int = 768, num_scales: int = 3):
        super(MultiScaleASRM, self).__init__()

        self.hidden_size = hidden_size
        self.num_scales = num_scales

        # 多尺度ASRM模块
        self.scale_modules = nn.ModuleList([
            ImprovedASRM(hidden_size, reduction_ratio=4 * (i + 1))
            for i in range(num_scales)
        ])

        # 尺度融合权重
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

        # 最终融合层
        self.fusion = nn.Linear(hidden_size * num_scales, hidden_size)

    def forward(self, hidden_states: torch.Tensor,
                apply_augmentation: bool = True) -> torch.Tensor:
        """
        多尺度前向传播
        """
        scale_outputs = []

        # 在不同尺度上应用ASRM
        for i, scale_module in enumerate(self.scale_modules):
            scale_output = scale_module(hidden_states, apply_augmentation)
            scale_outputs.append(scale_output)

        # 加权融合
        weighted_outputs = []
        for i, output in enumerate(scale_outputs):
            weight = F.softmax(self.scale_weights, dim=0)[i]
            weighted_outputs.append(weight * output)

        # 拼接并融合
        concatenated = torch.cat(scale_outputs, dim=-1)
        fused_output = self.fusion(concatenated)

        return fused_output


def test_improved_asrm():
    """测试改进的ASRM模块"""
    print("🧪 测试改进的ASRM模块...")

    # 创建测试数据
    batch_size, seq_len, hidden_size = 2, 10, 768
    test_input = torch.randn(batch_size, seq_len, hidden_size)

    # 测试基础改进ASRM
    print("\n1️⃣ 测试基础改进ASRM...")
    improved_asrm = ImprovedASRM(hidden_size=hidden_size)
    output1 = improved_asrm(test_input, apply_augmentation=True)
    print(f"✅ 输入形状: {test_input.shape} -> 输出形状: {output1.shape}")

    # 测试自适应ASRM
    print("\n2️⃣ 测试自适应ASRM...")
    adaptive_asrm = AdaptiveASRM(hidden_size=hidden_size)
    output2 = adaptive_asrm(test_input, apply_augmentation=True)
    print(f"✅ 输入形状: {test_input.shape} -> 输出形状: {output2.shape}")

    # 测试多尺度ASRM
    print("\n3️⃣ 测试多尺度ASRM...")
    multiscale_asrm = MultiScaleASRM(hidden_size=hidden_size)
    output3 = multiscale_asrm(test_input, apply_augmentation=True)
    print(f"✅ 输入形状: {test_input.shape} -> 输出形状: {output3.shape}")

    print("\n🎉 所有ASRM模块测试通过！")


if __name__ == "__main__":
    test_improved_asrm()

