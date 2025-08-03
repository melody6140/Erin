# SRAF: 语义感知增强的注意力熵平衡框架

基于论文实现的仇恨言论检测系统，采用BERT作为基础模型，集成了注意力引导的语义校准模块(ASRM)和分层注意力引导框架(HAGF)。

## 🚀 项目特点

- **ASRM模块**: 注意力引导的语义校准，通过"压缩-激励-校准"机制提升对比学习效果
- **HAGF框架**: 分层注意力引导，包含注意力熵正则化和认知扭曲激活算子
- **多任务学习**: 同时进行仇恨言论检测和认知扭曲识别
- **对比学习**: 增强语义表示学习能力
- **完整流程**: 从数据预处理到模型训练、评估和可视化分析

## 📁 项目结构

```
.
├── config.py              # 配置文件
├── data_loader.py          # 数据加载和预处理
├── model.py               # SRAF模型实现
├── trainer.py             # 训练器
├── main.py                # 主程序入口
├── utils.py               # 可视化和分析工具
├── requirements.txt       # 依赖包
├── README.md             # 项目说明
├── dataset/              # 数据集目录
│   ├── hateval2019_en_train.csv
│   ├── hateval2019_en_dev.csv
│   └── hateval2019_en_test.csv
└── outputs/              # 输出目录 (训练后生成)
    ├── best_model.pt
    ├── training_curves.png
    ├── final_results.json
    └── analysis/
```

## 🛠️ 环境要求

- Python 3.9+
- PyTorch 2.0+
- transformers 4.30+
- Apple Silicon MPS 支持

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ 快速开始

### 方法一：完整版本（包含可视化）

#### 1. 训练模型

```bash
# 使用默认参数训练
python main.py --mode train

# 自定义参数训练
python main.py --mode train \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --num_epochs 10 \
    --entropy_weight 0.15 \
    --cdao_weight 0.08
```

#### 2. 测试模型

```bash
# 使用最佳模型测试
python main.py --mode test

# 指定模型路径测试
python main.py --mode test --model_path outputs/best_model.pt
```

#### 3. 预测新文本

```bash
# 对示例文本进行预测
python main.py --mode predict

# 交互式预测
python main.py interactive
```

### 方法二：简化版本（推荐，无可视化问题）

#### 1. 训练模型

```bash
# 快速训练（推荐）
python simple_main.py --num_epochs 3 --batch_size 4

# 自定义参数训练
python simple_main.py \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --num_epochs 5 \
    --entropy_weight 0.15 \
    --cdao_weight 0.08
```

#### 2. 快速验证

```bash
# 运行基础功能测试
python test_model.py
```

## 🧠 模型架构

### SRAF框架核心组件

#### 1. ASRM (注意力引导的语义校准模块)

```python
# 压缩操作: z = 1/L * Σ(x_i)
z = torch.mean(x, dim=1)

# 激励操作: s = σ(W2 * ReLU(W1 * z))
s = self.excitation(z)

# 校准输出: X_tilde = s ⊙ X
enhanced_x = x * s
```

#### 2. HAGF (分层注意力引导框架)

- **注意力熵正则化**: `L_entropy = -1/N * Σ H(α_i)`
- **认知扭曲激活算子**: 识别文本中的逻辑谬误模式
- **统一训练目标**: `L_total = L_task + λ1*L_entropy + λ2*L_CDAO + λ3*L_contrastive`

## 📊 实验配置

### 默认超参数

| 参数 | 值 | 说明 |
|------|----|----|
| `batch_size` | 16 | 批次大小 |
| `learning_rate` | 2e-5 | 学习率 |
| `num_epochs` | 5 | 训练轮数 |
| `entropy_weight` | 0.1 | 注意力熵权重 λ1 |
| `cdao_weight` | 0.05 | 认知扭曲权重 λ2 |
| `contrastive_weight` | 0.1 | 对比学习权重 |

### 损失函数组件

1. **主任务损失**: 仇恨言论二分类交叉熵
2. **注意力熵损失**: 正则化注意力分布
3. **认知扭曲损失**: 辅助任务多分类损失
4. **对比学习损失**: 监督对比学习损失

## 📈 结果分析

训练完成后，系统会自动生成：

1. **训练曲线图**: 显示各损失组件的变化趋势
2. **性能指标**: 准确率、精确率、召回率、F1分数
3. **分析报告**: 包含特征可视化和模型分析

### 示例输出

```
测试集详细报告:
              precision    recall  f1-score   support

      非仇恨       0.85      0.89      0.87      1500
        仇恨       0.83      0.78      0.80      1200

    accuracy                           0.84      2700
   macro avg       0.84      0.84      0.84      2700
weighted avg       0.84      0.84      0.84      2700
```

## 🔍 模型分析工具

### 注意力可视化

```python
from utils import AttentionVisualizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
visualizer = AttentionVisualizer(tokenizer)

# 可视化注意力权重
visualizer.visualize_attention(
    model=model,
    text="这是一个示例文本",
    layer_idx=-1,
    head_idx=0,
    save_path="attention_heatmap.png"
)
```

### 特征空间可视化

```python
from utils import ModelAnalyzer

analyzer = ModelAnalyzer(model, tokenizer)

# 生成完整分析报告
analyzer.generate_analysis_report(
    train_loader=train_loader,
    test_loader=test_loader,
    save_dir="outputs/analysis"
)
```

## 🎯 关键创新点

### 1. 语义感知的数据增强
- 通过ASRM模块生成保留核心语义的增强样本
- 避免传统随机增强破坏语义一致性的问题

### 2. 注意力熵约束
- 防止模型过度聚焦于偏见特征
- 鼓励模型关注更广泛的上下文信息

### 3. 认知扭曲识别
- 引入外部知识识别逻辑谬误
- 从表层词汇转向深层语义推理

### 4. 多层级优化策略
- "广度约束"到"深度激活"的分层协同
- 系统性解决注意力偏移问题

## 📚 数据集说明

使用SemEval-2019 Task 5数据集：
- **训练集**: 9,077条样本
- **验证集**: 1,000条样本
- **测试集**: 2,971条样本
- **标签**: 二分类 (0: 非仇恨, 1: 仇恨)

## 🚨 注意事项

1. **设备兼容性**: 代码已针对Apple Silicon MPS进行优化
2. **内存使用**: 建议至少16GB内存用于训练
3. **模型大小**: 完整模型约110M参数
4. **训练时间**: 在M4 Pro芯片上约需1-2小时

## 🤝 贡献指南

1. Fork本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目仅供学术研究使用。

## 🙏 致谢

- HuggingFace Transformers库
- SemEval-2019仇恨言论检测任务
- PyTorch深度学习框架

---

**注**: 本实现基于学术研究论文，旨在复现和验证SRAF框架的有效性。如有任何问题或建议，欢迎提交Issue。

