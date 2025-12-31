# Exp3 实验运行命令

本文档包含运行 Exp3 所有实验所需的完整命令。

---

## 📋 前置条件

### 1. 生成任务数据（如果还没有）

```bash
cd /Users/caohuixi/symphony2.0/symphony-data-generator
conda activate symphony-data-gen  # 或使用你的环境
python src/quick_start.py
```

这将生成 `symphony-data-generator/data/exp3/task_pool.jsonl`

---

## 🔬 1. Simulation 实验

### 1.1 Shock A (A_unavailable)

```bash
python3 experiments/exp3/sim/exp3_sim.py \
  --n 1000 \
  --shock A_unavailable \
  --shock-point 500 \
  --seed 42 \
  --outdir experiments/exp3/sim/results/sim_exp3_robustness
```

**结果位置**: `experiments/exp3/sim/results/sim_exp3_robustness/ShockA/[时间戳]/`

### 1.2 Shock B (A_degraded)

```bash
python3 experiments/exp3/sim/exp3_sim.py \
  --n 1000 \
  --shock A_degraded \
  --shock-point 500 \
  --seed 42 \
  --outdir experiments/exp3/sim/results/sim_exp3_robustness
```

**结果位置**: `experiments/exp3/sim/results/sim_exp3_robustness/ShockB/[时间戳]/`

---

## 🎯 2. Real Execution 实验

### 2.1 Shock A (A_unavailable)

```bash
python3 experiments/exp3/real/exp3_real.py \
  --tasks symphony-data-generator/data/exp3/task_pool.jsonl \
  --n 200 \
  --shock A_unavailable \
  --shock-point 100 \
  --seed 42 \
  --outdir experiments/exp3/real/results/real_exp3_robustness
```

**结果位置**: `experiments/exp3/real/results/real_exp3_robustness/ShockA/[时间戳]/`

### 2.2 Shock B (A_degraded)

⚠️ **重要**: 使用 `shock-point=400` 而不是 500，确保有足够的 post-shock 任务来观察恢复。

```bash
python3 experiments/exp3/real/exp3_real.py \
  --tasks symphony-data-generator/data/exp3/task_pool.jsonl \
  --n 500 \
  --shock A_degraded \
  --shock-point 400 \
  --seed 42 \
  --outdir experiments/exp3/real/results/real_exp3_robustness
```

**结果位置**: `experiments/exp3/real/results/real_exp3_robustness/ShockB/[时间戳]/`

---

## 📊 3. 生成对比图 (Sim vs Real)

### 3.1 Shock A 对比图

首先找到最新的结果目录，然后运行：

```bash
# 方式 1: 使用最新结果（推荐）
SIM_SHOCKA=$(ls -td experiments/exp3/sim/results/sim_exp3_robustness/ShockA/*/ | head -1)
REAL_SHOCKA=$(ls -td experiments/exp3/real/results/real_exp3_robustness/ShockA/*/ | head -1)

python3 experiments/exp3/plot/plot_sim_vs_real.py \
  --sim "$SIM_SHOCKA/trajectory_linucb.csv" \
  --real "$REAL_SHOCKA/trajectory_real.json" \
  --shock-point-sim 500 \
  --shock-point-real 100 \
  --shock-type A_unavailable \
  --out experiments/exp3/plot/sim_vs_real_ShockA.png
```

**或者直接指定路径**：

```bash
python3 experiments/exp3/plot/plot_sim_vs_real.py \
  --sim experiments/exp3/sim/results/sim_exp3_robustness/ShockA/2025-12-31_01-03-42/trajectory_linucb.csv \
  --real experiments/exp3/real/results/real_exp3_robustness/ShockA/2025-12-31_14-38-45/trajectory_real.json \
  --shock-point-sim 500 \
  --shock-point-real 100 \
  --shock-type A_unavailable \
  --out experiments/exp3/plot/sim_vs_real_ShockA.png
```

### 3.2 Shock B 对比图

⚠️ **注意**: Sim 和 Real 的 shock_point 不同，需要分别指定：
- Sim: shock_point=500 (n=1000)
- Real: shock_point=400 (n=500)

```bash
# 方式 1: 使用最新结果（推荐）
SIM_SHOCKB=$(ls -td experiments/exp3/sim/results/sim_exp3_robustness/ShockB/*/ | head -1)
REAL_SHOCKB=$(ls -td experiments/exp3/real/results/real_exp3_robustness/ShockB/*/ | head -1)

python3 experiments/exp3/plot/plot_sim_vs_real.py \
  --sim "$SIM_SHOCKB/trajectory_linucb.csv" \
  --real "$REAL_SHOCKB/trajectory_real.json" \
  --shock-point-sim 500 \
  --shock-point-real 400 \
  --shock-type A_degraded \
  --out experiments/exp3/plot/sim_vs_real_ShockB.png
```

**或者直接指定路径**：

```bash
python3 experiments/exp3/plot/plot_sim_vs_real.py \
  --sim experiments/exp3/sim/results/sim_exp3_robustness/ShockB/2025-12-31_14-56-07/trajectory_linucb.csv \
  --real experiments/exp3/real/results/real_exp3_robustness/ShockB/2025-12-31_16-23-47/trajectory_real.json \
  --shock-point-sim 500 \
  --shock-point-real 400 \
  --shock-type A_degraded \
  --out experiments/exp3/plot/sim_vs_real_ShockB.png
```

---

## 🚀 一键运行所有实验

使用提供的脚本一次性运行所有实验：

```bash
chmod +x experiments/exp3/run_all_experiments.sh
bash experiments/exp3/run_all_experiments.sh
```

---

## 📝 参数说明

### Simulation 参数
- `--n`: 任务数（默认: 1000）
- `--shock`: Shock 类型，`A_unavailable` 或 `A_degraded`
- `--shock-point`: Shock 发生位置（默认: n//2）
- `--seed`: 随机种子（默认: 42）
- `--outdir`: 输出目录

### Real Execution 参数
- `--tasks`: **必需** - 任务文件路径（symphony-data-generator 生成）
- `--n`: 任务数（默认: 200）
- `--shock`: Shock 类型，`A_unavailable` 或 `A_degraded`
- `--shock-point`: Shock 发生位置（默认: n//2）
- `--seed`: 随机种子（默认: 42）
- `--outdir`: 输出目录
- `--no-plots`: 不生成图表（可选）
- `--no-excel`: 不生成 Excel 文件（可选）

### 对比图参数
- `--sim`: Sim trajectory CSV 文件路径
- `--real`: Real trajectory JSON 文件路径
- `--shock-point`: Shock point（如果 sim 和 real 相同）
- `--shock-point-sim`: Sim 的 shock point
- `--shock-point-real`: Real 的 shock point
- `--shock-type`: Shock 类型（用于标题）
- `--out`: 输出图片路径

---

## 📂 输出文件结构

```
experiments/exp3/
├── sim/results/sim_exp3_robustness/
│   ├── ShockA/[时间戳]/
│   │   ├── summary.csv
│   │   ├── trajectory_*.csv
│   │   ├── exp3_results.xlsx
│   │   └── plot_*.png
│   └── ShockB/[时间戳]/
│       └── ...
├── real/results/real_exp3_robustness/
│   ├── ShockA/[时间戳]/
│   │   ├── summary.csv
│   │   ├── trajectory_linucb.csv
│   │   ├── trajectory_real.json
│   │   ├── exp3_results.xlsx
│   │   └── plot_*.png
│   └── ShockB/[时间戳]/
│       └── ...
└── plot/
    ├── sim_vs_real_ShockA.png
    └── sim_vs_real_ShockB.png
```

---

## ⚠️ 注意事项

1. **任务文件**: Real 实验需要先运行 `quick_start.py` 生成任务文件
2. **Shock Point**: Real ShockB 使用 `shock-point=400` 而不是 500，确保有足够的 post-shock 任务
3. **任务数**: Real 实验任务数较少（200-500），Sim 使用 1000，这是为了控制 API 成本
4. **结果目录**: 每次运行都会创建新的时间戳目录，使用最新目录进行对比

---

**最后更新**: 2025-12-31

