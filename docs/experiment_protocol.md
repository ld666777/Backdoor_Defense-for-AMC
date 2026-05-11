# 实验协议建议

## 最小闭环

- Dataset: RML2016.10a
- Model: VT-CNN2-Lite
- Attack: time_add
- Poison rate: 3%
- Target class: QPSK
- Defense: SNR-aware feature detection + dual-domain consistency + filter-and-finetune

## 论文图表

1. ACC vs SNR
2. ASR vs SNR
3. 不同投毒率下 ACC/ASR 表格
4. 消融实验表格
5. 检测 precision/recall/F1 表格
6. 混淆矩阵

## 注意事项

- 验证集和测试集必须保持干净。
- 统计 ASR 时排除原本就是目标类别的测试样本。
- CPU 版实验建议先固定一个模型和一个攻击，跑通后再扩展。
