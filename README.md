# AMR 后门攻击防御 CPU 版实验代码

本项目用于复现实验：**基于 RML2016.10a 的深度学习自动调制识别（AMR）后门攻击与防御**。代码默认使用 **PyTorch CPU**，适合集显/无独立 GPU 的电脑。

## 1. 目录说明

```text
amr_backdoor_defense_cpu/
├── configs/              # 实验配置
├── data/raw/             # 放置 RML2016.10a_dict.pkl
├── data/processed/       # 预处理后的 train/val/test
├── data/poisoned/        # 投毒后的训练集
├── src/                  # 核心代码
├── scripts/              # 分阶段运行脚本
├── checkpoints/          # 模型权重
└── results/              # 日志、CSV、图、LaTeX 表格
```

## 2. 环境安装

建议 Python 3.9+。

```bash
pip install -r requirements.txt
```

如果是 Windows CMD，中文显示乱码时先执行：

```bat
chcp 65001
```

## 3. 数据集放置

请将数据集文件放到：

```text
data/raw/RML2016.10a_dict.pkl
```

常见的 RML2016.10a pickle 结构为：

```python
key = (modulation_name, snr)
value = ndarray, shape = [1000, 2, 128]
```

## 4. 快速 CPU 闭环

先运行 quick 配置，验证流程是否正常：

```bash
python main.py --config configs/quick_cpu.yaml --stage all
```

也可以分阶段运行：

```bash
python main.py --config configs/quick_cpu.yaml --stage prepare_data
python main.py --config configs/quick_cpu.yaml --stage train_clean
python main.py --config configs/quick_cpu.yaml --stage make_poison
python main.py --config configs/quick_cpu.yaml --stage train_backdoor
python main.py --config configs/quick_cpu.yaml --stage detect
python main.py --config configs/quick_cpu.yaml --stage repair
python main.py --config configs/quick_cpu.yaml --stage evaluate
python main.py --config configs/quick_cpu.yaml --stage plot
```

## 5. 结果文件

运行后主要结果在：

```text
results/csv/              # 论文表格用 CSV
results/figures/          # PDF/PNG 图
results/latex_tables/     # LaTeX 表格
results/reports/          # 实验摘要 Markdown
```

关键指标：

- `clean_acc`：干净测试精度
- `asr`：攻击成功率
- `acc_by_snr`：不同 SNR 下干净精度
- `asr_by_snr`：不同 SNR 下攻击成功率
- `detection_precision/recall/f1`：投毒样本检测效果

## 6. CPU 运行建议

- 首次运行用 `quick_cpu.yaml`。
- `num_workers` 默认设为 0，避免 Windows 多进程问题。
- `batch_size` 可根据内存改小，例如 128。
- 如果检测阶段很慢，可降低 `defense.transform_num` 或设置 `defense.sample_limit`。

## 7. 论文实验建议

最小闭环：

```text
模型：vtcnn2_lite
攻击：time_add
投毒比例：3%
目标类：QPSK
防御：SNR-aware + dual-domain + filter-and-finetune
```

完整实验再扩展攻击类型、投毒比例和目标类别。

## 论文批量实验：paper_experiments_cpu

如果需要生成 TCCN 初稿中的完整表格、消融实验和批量对比结果，请使用：

```bash
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode all --dry_run
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode all
```

也可以分批运行：

```bash
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode sweep
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode model
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode ablation
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode aggregate
```

生成的论文汇总结果位于：

```text
results/paper_experiments/summary/
results/paper_experiments/latex_tables/
```

详细说明见：

```text
docs/paper_experiments_cpu.md
```
