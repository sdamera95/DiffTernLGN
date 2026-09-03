# Checkpoints

Hardened circuits and trained models behind the CIFAR figures, plus the WDBC circuits.
CIFAR-10 itself is **not** shipped — the notebooks download and cache it on first run.

## What is here

`cifar10/` — seed 42, both architectures (`ternary` = PST-DTLGN, `binary` = DLGN), with
`_soft.pkl` (continuous network) and `_harden.pkl` (discrete circuit) per run.

| scale | neurons | layers | ternary | binary |
|---|---|---|---|---|
| small | 48,000 | 4 | 4.9 MB | 3.9 MB |
| deeper-small | 60,000 | 5 | 6.1 MB | 4.8 MB |
| medium | 96,000 | 4 | 9.7 MB | 7.7 MB |
| large | 144,000 | 4 | 14.5 MB | 11.5 MB |
| vlarge | 192,000 | 4 | 19.3 MB | 15.4 MB |

`results.jsonl` — soft and circuit accuracy, gap, unknown fraction, timings and gate counts
for **all 84 runs**: both datasets, all 7 scales, 3 seeds. Complete even where checkpoints
are not, so the summary tables and scaling curves cover everything.

`wdbc/` — hardened circuits for the 21 breast-cancer runs.

## What is not here, and why

| scale | neurons | ternary | binary | pair |
|---|---|---|---|---|
| huge | 512,000 | 51.4 MB | 41.0 MB | 92.4 MB |
| vhuge | 1,024,000 | 102.6 MB | 82.0 MB | 184.6 MB |

Shipping those two would take one seed from 98 MB to 375 MB, and all three seeds plus
CIFAR-100 to about 2.2 GB. GitHub blocks single files over 100 MB and degrades on large
repositories. Seeds 43 and 44 and the CIFAR-100 checkpoints are omitted for the same
reason; their metrics are in `results.jsonl`.

## Regenerating what is missing

`../final-notebooks/00_cifar_scaling_battery.ipynb` trains and writes checkpoints here.
Set `SCALES_TO_RUN`, `SEEDS` and `DATASET` at the top, and run it. It skips runs already
recorded in `results.jsonl`, so it is safe to stop and restart, or to add one scale later.
Set `SMOKE = True` first for a two-minute end-to-end check.

`../final-notebooks/02_cifar_scaling_figures.ipynb` reads whatever is here and reports what
it cannot find. Its `SEEDS` switch selects which runs to include: one seed plots that run,
several plot the mean with the spread across seeds.

`huge` and `vhuge` need a large-memory GPU and take hours per run; `train_wall_s` and
`steps_per_s` in `results.jsonl` give per-scale timings measured on an RTX PRO 6000.
