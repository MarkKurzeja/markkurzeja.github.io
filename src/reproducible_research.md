---
title: "The Checked-In Experiment"
subtitle: "A reproducible research workflow that fits in your head"
date: 2026-03-10
author: Mark Kurzeja
---

<div class="abstract">
A short how-to guide for running experiments that are always reproducible, always traceable, and always ready to share. The entire workflow reduces to one rule: if it isn't checked in, it doesn't exist.
</div>

If you've done any empirical research &mdash; stats, ML, computational science, bench experiments with code &mdash; you've been burned by a failed reproduction. Someone asks about a result from two months ago. The config was in a Slack message. The code has drifted. The notebook imported a helper that's since been refactored. You spend a day reconstructing what should have taken five minutes to look up.

I've lost count of how many times I've seen this happen in my own statistics work, and in published papers where the authors themselves couldn't reproduce their own numbers. Adopting the habits in this post has saved me hundreds of hours of debugging. The investment is small; the compounding returns are enormous.

This post describes a lightweight workflow that prevents these failures. It's a set of habits, not a tool &mdash; and every habit serves the same principle.

## The One Rule

**If it isn't checked in, it doesn't exist.**

Version control is the single source of truth. Not your memory, not Slack, not a shared drive. Code, config, analysis, results &mdash; if it matters, it lives in the repo.<span class="sidenote">This idea isn't new. Knuth's literate programming, Claerbout's reproducible research, and the broader open-science movement all share this DNA. This workflow is a pragmatic distillation for anyone running experiments.</span>

Everything below is a consequence of this one rule.

## The Workflow

Here's the full loop at a glance:

```
 Config ──→ Runfile ──→ Experiment ──→ Notebook ──→ README
   │           │            │              │           │
   │           │            │              │           │
   ▼           ▼            ▼              ▼           ▼
 checked     checked    HEAD + diff     standalone   experiment
   in          in        captured      with saved     summary
                                        outputs      and plots
                          ╲               │           ╱
                           ╲              │          ╱
                            ▼             ▼         ▼
                         ┌──────────────────────────┐
                         │  Only share what's in     │
                         │  the repo as a report     │
                         └──────────────────────────┘
```

Each step below maps to a habit.

## Make Config Complete

The config file should be sufficient to reproduce an experiment from scratch, with no ambient state. If you need to know anything beyond the config to re-run, the config is incomplete.<span class="sidenote">A useful litmus test: can a colleague reproduce your experiment using only the config and the README, without asking you a single question?</span>

This is the hardest habit to adopt and the most valuable. It forces you to be explicit about every decision: hyperparameters, data paths, preprocessing steps, random seeds, all of it.

Here's what a complete config looks like:

```toml
# Good: everything needed to reproduce from scratch
[experiment]
name = "baseline_v3"
seed = 42

[data]
path = "data/processed/train.parquet"
val_path = "data/processed/val.parquet"
preprocessing = "standardize"  # no ambient assumptions

[model]
type = "ridge"
alpha = 0.1
features = ["age", "income", "region"]

[output]
dir = "results/baseline_v3"
```

And here are three common ways configs fall short:

```python
# Bad #1: Missing the data path — relies on whoever runs it
#   to "just know" which file to use
config = {
    "model": "ridge",
    "alpha": 0.1,
}
# Where's the data? Who knows. Ask Mark, he might remember.
```

```python
# Bad #2: Hardcoded paths that only work on one machine
config = {
    "data": "/home/mark/Desktop/thursday_data_final_v2.csv",
    "model": "ridge",
    "alpha": 0.1,
}
# Good luck running this on any other machine.
```

```python
# Bad #3: Missing the random seed and preprocessing details
config = {
    "data": "data/train.parquet",
    "model": "ridge",
    "alpha": 0.1,
    # No seed — results will differ every run.
    # No mention of preprocessing — was the data standardized?
    #   Log-transformed? Filtered? Hope you remember.
}
```

## Capture the Code State

Prefer to check in the code at the commit that produced the result. When that isn't practical &mdash; you're iterating quickly, the working tree is dirty &mdash; save three things as experiment artifacts:

1. **HEAD** commit hash
2. **git diff** (the uncommitted changes)
3. **Config** used for the run

An experiment's identity is (code state, config). If you have both, you can reproduce. If you're missing either, you're guessing.<span class="marginnote">In practice, I use branch-per-experiment or tag conventions, but the specific Git workflow matters less than the discipline of always recording the code state.</span>

## Use One Codebase for Train and Eval

Separate codebases for training and evaluation are a common source of silent divergence. When eval code drifts from training code, your metrics become unreliable and you won't notice.<span class="sidenote">This is especially insidious when eval involves preprocessing or feature engineering. A mismatch between train-time and eval-time preprocessing is one of the most common and hardest-to-catch bugs in empirical work.</span>

One codebase. The config determines which mode runs, not which code runs.

## Save Commands in Runfiles

Every shell command needed to execute a pipeline lives in a checked-in "runfile" &mdash; a script that captures the exact invocation.<span class="sidenote">You could use Make, Just, a shell script, or a plain text file with copy-pasteable commands. The format doesn't matter. What matters is that the commands are recorded.</span>

Runfiles aren't fancy. Their value is being explicit, version-controlled, and runnable. No one has to reverse-engineer your pipeline from memory.

## Analyze in Standalone Notebooks

Results go into a notebook &mdash; Colab, Jupyter, Quarto, Plaque, whatever you prefer &mdash; saved with its outputs. The critical rule: **the notebook does not import from your custom experiment codebase.** It is self-contained.<span class="sidenote">The specific tool doesn't matter. What matters is the contract: the notebook is a standalone document with saved outputs that you can re-read (and re-run) years later.</span>

To be clear: standard library and third-party imports (`numpy`, `pandas`, `matplotlib`, etc.) are fine. The problem is importing from your own project &mdash; your custom analysis library, your plotting utilities, your data loading helpers.

Why? Because your code will change. If your analysis notebook imports a plotting helper from your experiment codebase, and someone refactors that module next month, your saved analysis becomes un-runnable. Worse, if the function's *behavior* changes silently, re-running the notebook produces different results with no indication that anything changed. Over time, internal code drifts in ways that break backwards compatibility &mdash; renamed arguments, changed defaults, deleted functions. A notebook that depends on your codebase has an expiration date you can't predict.<span class="marginnote">If you find yourself wanting to import code into your analysis notebook, copy the relevant function directly into the notebook. Duplication here is a feature, not a bug &mdash; it freezes the behavior at the time of analysis.</span>

A standalone notebook with frozen outputs is a time capsule. Check the notebooks into the repo. Results, plots, and analysis are now versioned alongside the code and config that produced them.

## Maintain a README Log

Keep a running summary in the README: what was tried, what the results were, key plots, brief interpretation. This is a lab notebook, not a formal report. Its audience is future-you and your collaborators.<span class="sidenote">The discipline of writing a one-paragraph summary after each experiment dramatically improves your own understanding of what you're learning over the course of a project.</span>

## Only Share What's Checked In

The final habit ties it all together: when you share results &mdash; in a meeting, a paper, a design doc &mdash; share only what exists as a checked-in report.

This creates a forcing function. If a result is worth sharing, it's worth checking in properly. If it isn't checked in, it's provisional.<span class="marginnote">This habit also protects you. When someone questions a number six months later, you don't need to reconstruct anything. You point them to the commit.</span>

## That's It

None of these habits are heavy. A complete config, a runfile, a standalone notebook, a README entry, and a commit. The power is in the composition &mdash; together they make reproducibility a byproduct of the workflow rather than an afterthought.

The goal isn't perfection. The goal is that when future-you needs to understand an experiment, everything is in one place, and it still runs.
