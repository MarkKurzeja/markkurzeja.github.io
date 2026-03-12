---
title: "🔶 The AMBER Workflow"
subtitle: "Twelve principles for preserving research in amber"
date: 2026-03-10
author: Mark Kurzeja
---

<div class="abstract">
An opinionated guide for recording and reproducing research results.<span class="sidenote-number"></span><span class="sidenote">Adapted from a workflow I learned from <a href="https://www.linkedin.com/in/john-aslanides/">John Aslanides</a>.</span>
</div>

Good empirical research should be:

- **Durable.** Results still make sense and still run months or years later, without bit rot or dependency decay.<span class="sidenote-number"></span><span class="sidenote">Amber preserves things perfectly: an insect trapped in resin 40 million years ago is still there, every leg, every wing vein, frozen exactly as it was. That is what a good research workflow does to experiments.</span>
- **Statistically reproducible.** Anyone can re-run an experiment and get statistically indistinguishable results.<span class="sidenote-number"></span><span class="sidenote">Put differently, A/A testing is easy to pull off: re-run the same experiment and confirm your pipeline produces stable results.</span>
- **Traceable.** Every result points back to the exact code and config that produced it.
- **Shareable.** A colleague can pick up where you left off with minimal questioning.

In practice, most research workflows fail at least one of these.<span class="sidenote-number"></span><span class="sidenote">The standard pathologies: (1) the config was in a Slack message and nobody can find it, (2) the code has drifted and nobody recorded which commit produced the result, (3) the analysis notebook imported a helper that's since been refactored, (4) training and eval used separate codebases and a preprocessing change didn't propagate, (5) the pipeline was run with shell commands that lived in someone's terminal history and that person is on vacation.</span> You iterate in a development branch, try dozens of things, and eventually land on a result. But the lineage is lost. Tribal knowledge accumulates in people's heads instead of in the repo. 
The workflow below is a set of habits, not a tool, that ended up producing much better research.

## The Setup

This workflow relies on five things:

- **A codebase.** One repo containing all code for training, evaluation, preprocessing, and analysis.
- **A config.** A file (JSON, YAML, TOML) that fully specifies an experiment: hyperparameters, data paths, preprocessing steps, random seeds.
- **A runfile.** A checked-in script (Makefile, Justfile, shell script) that records every command needed to execute the pipeline.
- **A run directory.** A dedicated output folder per experiment containing the config, the code state (<code>HEAD</code> hash and <code>git diff</code>), and every artifact the run produces. Once complete, this folder is read-only: a frozen snapshot of exactly what happened.
- **A report.** A standalone notebook or script (<a href="https://colab.research.google.com/">Colab</a>, <a href="https://jupyter.org/">Jupyter</a>, <a href="https://quarto.org/">Quarto</a>, <a href="https://blog.alexalemi.com/plaque.html">Plaque</a>, static HTML) that reads from a run directory and contains all statistical analysis, plots, and interpretation in one place.

## The Twelve Principles

<div class="proof-block">
<div class="proof-label">The Twelve Principles of Reproducible Research</div>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="repro:1">1.</span>
Check everything into version control.
</summary>
<div class="subproof">
Version control is the single source of truth. Not your memory, not Slack, not a shared drive. Code, config, analysis, results: if it matters, it lives in the repo.<span class="sidenote-number"></span><span class="sidenote">This idea isn't new. Knuth's literate programming is the most famous articulation. This workflow is a pragmatic distillation for anyone running experiments.</span>
</div>
</details>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="repro:2">2.</span>
Use one codebase.
</summary>
<div class="subproof">
Training, evaluation, preprocessing, analysis: all in one repo. The config determines what runs, not which code runs.<span class="sidenote-number"></span><span class="sidenote">This is how you avoid the most insidious class of bugs: silent divergence between training and evaluation code, where a preprocessing change propagates to one but not the other.</span>
</div>
</details>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="repro:3">3.</span>
Make dependencies explicit.
</summary>
<div class="subproof">
Pin your dependencies and use a lockfile.<span class="sidenote-number"></span><span class="sidenote"><code>uv</code> handles this well for Python. The point is that anyone checking out your repo can recreate your environment exactly, not approximately.</span> A colleague should be able to clone, install, and run without asking you a single question.
</div>
</details>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="repro:4">4.</span>
Make configs complete.
</summary>
<div class="subproof">
The config file should be sufficient to reproduce an experiment from scratch, with no ambient state: hyperparameters, data paths, preprocessing steps, random seeds, etc.<span class="sidenote-number"></span><span class="sidenote">A useful litmus test: can a colleague reproduce your experiment using only the config and the README, with minimal questioning?</span> If you need to know anything beyond the config to re-run, the config is incomplete.

The common failures: missing data paths, hardcoded paths that only work on one machine, missing random seeds, and behavior controlled by command-line flags that nobody saved down.
</div>
</details>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="repro:5">5.</span>
Put commands in runfiles.
</summary>
<div class="subproof">
Every shell command needed to execute a pipeline lives in a checked-in script: a Makefile, a Justfile, a shell script.<span class="sidenote-number"></span><span class="sidenote">The format doesn't matter. What matters is that the commands are recorded. No one should have to reverse-engineer your pipeline from terminal history.</span> The config defines <em>what</em> to run, and the runfile defines <em>how</em>.
</div>
</details>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="repro:6">6.</span>
Produce a folder of artifacts per run.
</summary>
<div class="subproof">
Each run lands in a dedicated output directory: the config, the <code>HEAD</code> commit hash, the <code>git diff</code>, and every artifact the run produces (logs, metrics, outputs). Once a run is complete, this folder is read-only. Nothing in it changes. It is a frozen snapshot of exactly what happened, and reports read directly from it.
</div>
</details>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="repro:7">7.</span>
Record the code state with every run.
</summary>
<div class="subproof">
Every run records the <code>HEAD</code> commit hash, the <code>git diff</code> (if the working tree is dirty), and the config used. An experiment's identity is (code state, config). If you have both, you can reproduce. If you're missing either, you're guessing.<span class="sidenote-number"></span><span class="sidenote">Ideally, you run all experiments from a clean <code>HEAD</code> with different configs rather than relying on uncommitted changes. Large <code>git diff</code>s are a smell: if you need to change the code, commit first, then run. The discipline of committing before running keeps your experiments tied to real, recoverable code states.</span>
</div>
</details>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="repro:8">8.</span>
Make reports durable and standalone.
</summary>
<div class="subproof">
Results go into a report: a notebook (<a href="https://colab.research.google.com/">Colab</a>, <a href="https://jupyter.org/">Jupyter</a>, <a href="https://quarto.org/">Quarto</a>, <a href="https://blog.alexalemi.com/plaque.html">Plaque</a>), a static HTML file, a script that renders plots. The format doesn't matter. What matters is that the statistical analysis, plots, and outputs all live in the same place, the report reads from frozen run artifacts, and it does not import from your experiment codebase.<span class="sidenote-number"></span><span class="sidenote">Standard library and third-party imports (<code>numpy</code>, <code>pandas</code>) are fine. The problem is importing from your own project. If you need a helper function, copy it into the report. Duplication here is a feature: it freezes the behavior at the time of analysis. As Rob Pike <a href="https://www.youtube.com/watch?v=PAAkCSZUG1c&t=568s">put it</a>, "a little copying is better than a little dependency."</span> Your code will change. A report that depends on it has an expiration date you can't predict.
</div>
</details>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="repro:9">9.</span>
Make reports easy to re-run.
</summary>
<div class="subproof">
A report should be re-runnable from its own directory at any point in the future and produce exactly the same results. Save the outputs alongside the code that produced them, and check everything into the repo. Your statistical analysis is inherently durable: a standalone report with frozen outputs is a time capsule.
</div>
</details>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="repro:10">10.</span>
Keep a README log.
</summary>
<div class="subproof">
Write a running summary: what was tried, what the results were, key plots, brief interpretation. Screenshot key results and embed them directly, especially if your screenshot tool generates shareable links.<span class="sidenote-number"></span><span class="sidenote">The discipline of writing a one-paragraph summary after each experiment dramatically improves your own understanding of what you're learning over the course of a project.</span> This is a lab notebook, not a formal report. Its audience is future-you and your collaborators.
</div>
</details>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="repro:11">11.</span>
Share results from committed reports.
</summary>
<div class="subproof">
When you share results (in a meeting, a paper, a design doc) share only what exists as a committed report. If it's worth sharing, it's worth committing.<span class="marginnote">This habit also protects you. When someone questions a number six months later, you point them to the commit.</span>
</div>
</details>

<details class="step-details" open>
<summary class="step">
<span class="step-number" id="repro:12">12.</span>
Use each experiment to design the next one.
</summary>
<div class="subproof">
The loop closes when you take what you learned and use it to design the next experiment. Good research is a clean sequence of runs, each building on the last, each fully traceable.
</div>
</details>

</div>

## The Loop

In practice, the principles above produce a tight loop:

<div class="algorithm">
<div class="algorithm-header"><strong>Algorithm A</strong> (AMBER). Given a codebase and a config, produce a traceable result.</div>
<ol class="algorithm-steps">
<li><span class="algorithm-step-label">A1.</span><span class="algorithm-step-body"><span class="step-action">[Configure]</span> Write a complete config. Check it in.</span></li>
<li><span class="algorithm-step-label">A2.</span><span class="algorithm-step-body"><span class="step-action">[Run]</span> Execute the pipeline via the runfile. Save all artifacts to a dedicated output directory. Record the code state alongside the artifacts.</span></li>
<li><span class="algorithm-step-label">A3.</span><span class="algorithm-step-body"><span class="step-action">[Report]</span> Write a standalone report with frozen outputs. Check it in.</span></li>
<li><span class="algorithm-step-label">A4.</span><span class="algorithm-step-body"><span class="step-action">[Learn]</span> Update the README log with results and interpretation. Commit everything.</span></li>
<li><span class="algorithm-step-label">A5.</span><span class="algorithm-step-body"><span class="step-action">[Share]</span> If the result is worth sharing, share only from the repo.</span></li>
<li><span class="algorithm-step-label">A6.</span><span class="algorithm-step-body"><span class="step-action">[Iterate]</span> Design the next config from what you learned. Return to A1.</span></li>
</ol>
</div>

The goal is simple: when future-you revisits an experiment, everything is in one place, and it still runs.
