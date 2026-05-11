# 论文批量实验 CPU 版说明

本说明对应 `configs/paper_experiments_cpu.yaml` 和 `scripts/paper_experiments_cpu.py`。
该批处理版本用于生成 TCCN 初稿中需要的主结果表、投毒比例表、多模型对比表、检测表和消融实验表。

## 1. 先预览任务

建议先运行 dry-run，确认实验数量和目录是否符合预期：

```bash
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode all --dry_run
```

Windows CMD 如中文显示乱码，先执行：

```bat
chcp 65001
```

## 2. 运行全部论文实验

```bash
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode all
```

CPU 上完整矩阵会比较慢。可以分批运行：

```bash
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode sweep
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode model
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode ablation
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode aggregate
```

## 3. 只补某几个阶段

如果模型已经训练好，只想重新检测、修复和评估：

```bash
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode ablation --stages detect,repair,evaluate,plot
```

如果只想汇总已有结果并重新导出表格：

```bash
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode aggregate
```

## 4. 输出目录

每个单次实验结果位于：

```text
results/paper_experiments/runs/<run_id>/
```

总汇总文件位于：

```text
results/paper_experiments/summary/all_runs_summary.csv
results/paper_experiments/summary/all_snr_curves.csv
results/paper_experiments/summary/paper_experiment_report.md
```

LaTeX 表格位于：

```text
results/paper_experiments/latex_tables/table_main_acc_asr.tex
results/paper_experiments/latex_tables/table_poison_rate.tex
results/paper_experiments/latex_tables/table_ablation.tex
results/paper_experiments/latex_tables/table_detection.tex
results/paper_experiments/latex_tables/table_model_comparison.tex
```

## 5. 与论文实验小节的对应关系

- `table_main_acc_asr.tex`：总体防御效果，对应 Overall Defense Performance。
- `table_poison_rate.tex`：不同投毒比例下的鲁棒性，对应 Impact of Poisoning Rate。
- `table_ablation.tex`：消融实验，对应 Ablation Study。
- `table_detection.tex`：后门样本检测性能，对应 Detection Performance。
- `table_model_comparison.tex`：不同 AMR backbone 对比，对应 Generalization Across Backbones。
- `all_snr_curves.csv`：按 SNR 的 ACC/ASR 曲线数据，对应 SNR-wise Performance Analysis。

## 6. CPU 运行建议

如果电脑较慢，建议先修改 `configs/paper_experiments_cpu.yaml`：

```yaml
paper:
  sample_limit: 30000

overrides:
  train:
    epochs: 15
  repair:
    epochs: 5
  defense:
    transform_num: 4
```

确认流程正常后，再恢复为正式配置。

## 7. 断点续跑

默认：

```yaml
paper:
  skip_existing: true
```

脚本会自动跳过已经存在的投毒数据、后门 checkpoint、检测结果、修复日志和评估 summary。
如果你修改了方法或配置，建议删除对应 run 目录和 checkpoint 后重新运行。

## 8. 快速试跑配置

如果你想先在 CPU 上确认批量流程是否完整，可以运行较小的快速配置：

```bash
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu_quick.yaml --mode all --dry_run
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu_quick.yaml --mode all
```

该配置只用于调试流程，不建议把结果直接写入论文。
