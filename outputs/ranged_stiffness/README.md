# Ranged-stiffness EE compliance sweep

`eval_ranged_stiffness.py` runs the 14 stiffness commands for the three
range-capable policies plus the oracle-force analytical baseline:

- analytical MoE;
- two-layer ranged high-level v2 policy (`ee_xyz_range_200_600_v2_3kp_force_b_stu_adapt_20260802_204248`);
- end-to-end ranged policy;
- oracle force + 3kp stiff low-level policy (`gentle_3kp_stiff_finetune_limmt_full_force30`).

That is 56 eval processes total and 840 force probes per policy. The oracle
baseline does not use a learned force estimator: it sends the evaluator's
ground-truth force through the analytical `nominal + F / K_xyz` command. Each child
process is monitored until its 60-probe JSON report is complete. The runner
then sends `SIGINT` to stop the lingering Isaac Sim process; if needed it
escalates to `SIGTERM` and `SIGKILL`.

Run a command-only validation first:

```bash
python outputs/ranged_stiffness/eval_ranged_stiffness.py --dry-run
```

Run one representative job:

```bash
python outputs/ranged_stiffness/eval_ranged_stiffness.py \
  --policy analytical_moe --stiffness-index 0
```

Run the full sweep:

```bash
python outputs/ranged_stiffness/eval_ranged_stiffness.py
```

To run only the v2 two-layer policy:

```bash
python outputs/ranged_stiffness/eval_ranged_stiffness.py \
  --policy two_layer_range_v2
```

Raw policy-evaluation JSON reports, logs, and `ranged_eval_manifest.json` are
written under `eval_results/`. This keeps them separate from plotting code,
figures, tables, and documentation.
Complete reports are skipped on later invocations unless `--no-resume` is
provided.

## Benchmark plots and table

The plotting script only reads existing complete JSON reports from
`eval_results/`; it does not launch evaluation:

```bash
python outputs/ranged_stiffness/plot_ranged_stiffness.py
```

It generates five figures under `figures/`, the aggregate CSV
`ranged_stiffness_metrics.csv`, and the Markdown table
`ranged_stiffness_benchmark.md`. The script recognizes all four methods and
only includes methods with complete reports. An end-to-end range sweep can be
supplied by glob, for example:

```bash
python outputs/ranged_stiffness/plot_ranged_stiffness.py \
  --end2end-glob '/path/to/end2end_range_*_k*.json'
```
