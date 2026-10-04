# MITK_AI_Project

基于 MITK 平台的 AI 医学影像智能分割系统——从 MITK 插件设计、TotalSegmentator 源码分析到 nnUNet 流程复现的完整工程。

## 目录结构

| 目录 | 说明 |
|------|------|
| `AI/Inference/` | 推理入口脚本（`predict.py`、`organ_analysis.py`、`test_totalseg.py`） |
| `AI/TotalSegmentator/` | TotalSegmentator 源码克隆（用于源码流程分析） |
| `AI/nnUNet/` | nnUNet v2 源码克隆（用于训练/推理流程复现） |
| `AI/Utils/` | 辅助工具脚本 |
| `Plugin/` | MITK 插件工程 `org.mitk.gui.qt.ai.segmentation` |
| `Models/` | TotalSegmentator 预训练权重（约 1.5 GB，已被 .gitignore 排除） |
| `Dataset/` | CT 影像数据与 nnUNet 格式转换结果（已被 .gitignore 排除） |
| `Results/` | 分割与量化分析结果（肝脏面积/体积曲线、预测结果等） |
| `Documentation/` | 项目文档 |
| `Python/` | Python 环境相关脚本 |

## 复现要点

- **MITK**：2024.12 Release（未自行 SuperBuild 编译）；Qt 6.8.3 MSVC2022；CMake ≥ 3.29；Visual Studio 2022。
- **AI 环境**：`conda create -n mitk-ai python=3.10`，安装 GPU 版 PyTorch（CUDA 12.1）与 `TotalSegmentator`。
- **模型权重**：首次运行 `TotalSegmentator` 时自动下载，或从官方渠道获取后放入 `Models/`。
- **推理链路**：MITK 显示 CT → 外部调用 AI → TotalSegmentator 分割 → 生成 NIfTI → MITK 重新载入显示与量化。

## 关联仓库

课件（已脱敏）见 [Research-Innovation-Class](https://github.com/wangziwen2794928256/Research-Innovation-Class)。
