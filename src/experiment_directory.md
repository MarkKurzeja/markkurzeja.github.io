---
title: "🔶 The AMBER Directory"
subtitle: "A concrete directory layout for ambered experiments"
date: 2026-03-12
author: Mark Kurzeja
publish: true
---

<div class="abstract">
A short companion to <a href="reproducible_research.html">The AMBER Workflow</a>. What does an AMBER-compliant experiment repo actually look like on disk? Here is the directory structure I reach for every time.
</div>

## The Layout

```
my-experiment/
├── main.py
├── run.sh
├── pyproject.toml
├── configs/
│   ├── baseline.yaml
│   ├── 20260105a_larger_lr.yaml
│   └── 20260107a_dropout_sweep.yaml
├── reports/
│   ├── README.md
│   ├── 20260105a_baseline_results.ipynb
│   └── 20260108a_dropout_comparison.ipynb
├── runs/
│   ├── 20260105a_baseline/
│   │   ├── config.yaml
│   │   ├── git_state.txt
│   │   ├── metrics.json
│   │   └── model.pt
│   └── 20260107a_dropout_sweep/
│       ├── config.yaml
│       ├── git_state.txt
│       ├── metrics.json
│       └── model.pt
└── src/
    ├── data.py
    ├── model.py
    └── train.py
```

## What Each Piece Does

**`main.py`** is the single entry point. It takes a config path and runs the full pipeline: data loading, training, evaluation, artifact saving. One command, one config, one run.

**`run.sh`** records the exact shell commands<span class="sidenote-number"></span><span class="sidenote">This can be a Makefile or a Justfile. The format does not matter. What matters is that the commands are checked in, not living in your terminal history.</span>. A typical `run.sh` looks like this:

```bash
#!/bin/bash
set -euo pipefail
CONFIG=${1:?Usage: ./run.sh configs/my_config.yaml}
RUN_DIR="runs/$(basename "$CONFIG" .yaml)"
mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/config.yaml"
git log -1 --format="%H" > "$RUN_DIR/git_state.txt"
git diff >> "$RUN_DIR/git_state.txt"
python main.py --config "$CONFIG" --output-dir "$RUN_DIR"
```

**`configs/`** holds one YAML file per experiment. The naming convention is `YYYYMMDD[a-z]_description.yaml`, where the letter suffix distinguishes multiple configs created on the same day<span class="sidenote-number"></span><span class="sidenote">The date prefix makes configs sort chronologically in `ls`. The letter suffix is just a tiebreaker: `a`, `b`, `c` for the first, second, third config of the day.</span>. Each config is complete: it specifies everything needed to reproduce the run.

**`runs/`** holds one folder per completed experiment. Each folder is a frozen snapshot: the config used, the git state at run time, and every artifact the run produced. Once written, nothing in here changes. This is the amber.

**`reports/`** holds standalone analysis notebooks. They follow the same date-prefix naming. Each report reads directly from `runs/` and does not import from `src/`<span class="sidenote-number"></span><span class="sidenote">Standard library and third-party imports are fine. The rule is: do not import from your own project code. If you need a helper, copy it into the report. A little copying is better than a little dependency.</span>.

**`reports/README.md`** is the lab notebook. It is a running log of what was tried, what the results were, and what to try next. Keep it casual, keep it current.

**`src/`** is where the actual code lives: data loading, model definitions, training loops. `main.py` orchestrates, `src/` implements.

## The Naming Convention

The date-prefix convention (`YYYYMMDD[a-z]_name`) threads through configs, runs, and reports. This makes it trivial to trace a result back to its origin:

- See an interesting plot in `reports/20260108a_dropout_comparison.ipynb`?
- It reads from `runs/20260107a_dropout_sweep/`.
- That run used `configs/20260107a_dropout_sweep.yaml`.

One date string, three locations, full traceability.
