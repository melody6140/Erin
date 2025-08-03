# 🏗️ 项目结构说明

## 📁 清理后的项目结构

```
pythonProject/
├── 📂 dataset/                    # 数据集目录
│   ├── hateval2019_en_train.csv   # 训练集
│   ├── hateval2019_en_dev.csv     # 验证集
│   └── hateval2019_en_test.csv    # 测试集
│
├── 📂 paper/                      # 论文相关文件
│   └── PECOLA.pdf                 # 论文PDF
│
├── 📂 outputs/                    # 训练输出目录
│   ├── best_model.pt              # 最佳模型权重
│   ├── final_results.json         # 最终结果
│   └── training_curves.png        # 训练曲线图
│
├── 🔧 核心模块文件
├── config.py                      # 配置文件
├── model.py                       # 原始SRAF模型
├── trainer.py                     # 原始训练器
├── data_loader.py                 # 原始数据加载器
├── utils.py                       # 工具函数
│
├── 🚀 改进的ASRM实现
├── improved_asrm.py               # 改进的ASRM模块（新增）
├── baseline_model.py              # 基线和ASRM对比模型（新增）
├── asrm_experiment.py             # ASRM改进实验脚本（新增）
│
├── 📋 主程序文件
├── main.py                        # 原始主程序
├── simple_main.py                 # 简化主程序
├── baseline_main.py               # 基线主程序
├── quick_start.py                 # 快速开始脚本
│
├── 🔬 实验和测试文件
├── test_model.py                  # 模型测试
├── simple_data_loader.py          # 简化数据加载器
├── contrastive_main.py            # 对比学习主程序
├── contrastive_main_stable.py     # 稳定版对比学习
├── single_view_baseline.py        # 单视图基线
├── radical_improvement_experiment.py  # 根本性改进实验
│
├── 📚 文档和配置
├── README.md                      # 项目说明
├── requirements.txt               # 依赖包列表
└── PROJECT_STRUCTURE.md           # 项目结构说明（本文件）
```

## 🎯 核心文件说明

### 🔧 **核心模块**

1. **`improved_asrm.py`** ⭐ **[新增核心文件]**
   - 改进的ASRM模块实现
   - 解决原始ASRM的过度正则化问题
   - 包含三种ASRM变体：基础改进版、自适应版、多尺度版

2. **`baseline_model.py`** ⭐ **[新增核心文件]**
   - BERT基线模型
   - 集成改进ASRM的BERT模型
   - 用于对比实验

3. **`asrm_experiment.py`** ⭐ **[新增核心文件]**
   - ASRM改进实验主脚本
   - 自动对比基线和ASRM模型性能
   - 生成详细的实验报告

### 📋 **主要程序**

4. **`simple_data_loader.py`**
   - 简化的数据加载器
   - 支持HatEval2019数据集
   - 清洁的数据预处理流程

5. **`config.py`**
   - 统一的配置管理
   - 模型、训练、数据参数配置

### 🔬 **实验文件**

6. **`radical_improvement_experiment.py`**
   - 根本性改进实验
   - 包含多种先进技术的集成

7. **`contrastive_main.py`** / **`contrastive_main_stable.py`**
   - 对比学习实验
   - 双视图数据增强

## 🚀 **快速开始指南**

### 1. **测试改进的ASRM模块**
```bash
# 测试ASRM模块功能
python improved_asrm.py

# 测试基线模型
python baseline_model.py
```

### 2. **运行ASRM改进实验**
```bash
# 快速测试（推荐首次运行）
python asrm_experiment.py --quick_test

# 完整实验
python asrm_experiment.py --num_epochs 5 --batch_size 16

# 测试不同ASRM类型
python asrm_experiment.py --asrm_type adaptive --quick_test
python asrm_experiment.py --asrm_type multiscale --quick_test
```

### 3. **运行原始实验**
```bash
# 原始SRAF框架
python simple_main.py

# 对比学习实验
python contrastive_main_stable.py
```

## 📊 **实验建议顺序**

1. **🧪 基础测试**
   ```bash
   python improved_asrm.py          # 测试ASRM模块
   python baseline_model.py         # 测试基线模型
   ```

2. **🔬 ASRM改进实验**
   ```bash
   python asrm_experiment.py --quick_test --asrm_type improved
   ```

3. **📈 完整对比实验**
   ```bash
   python asrm_experiment.py --num_epochs 5 --asrm_type improved
   python asrm_experiment.py --num_epochs 5 --asrm_type adaptive
   python asrm_experiment.py --num_epochs 5 --asrm_type multiscale
   ```

4. **🚀 根本性改进实验**
   ```bash
   python radical_improvement_experiment.py --quick_test
   ```

## 🗂️ **已清理的冗余文件**

以下文件已被清理，以简化项目结构：

- ❌ 重复的基线实验文件
- ❌ 多个版本的测试脚本
- ❌ 临时调试文件
- ❌ 重复的输出目录
- ❌ 过时的实验脚本

## 📝 **下一步计划**

1. **验证改进的ASRM模块**
2. **运行对比实验**
3. **分析实验结果**
4. **根据结果进一步优化**

---

**注意**: 项目已经过系统性清理，保留了核心功能文件，移除了冗余和临时文件。现在可以专注于ASRM模块的改进和实验验证。

