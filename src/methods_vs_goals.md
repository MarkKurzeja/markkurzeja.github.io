---
title: "🔧 Methods-Forward vs 🎯 Goal-Backward Research"
subtitle: "Prefer goal-backward research"
date: 2026-03-13
author: Mark Kurzeja
---

During my time at Google DeepMind, I noticed that there were two predominant paradigms of research. I started calling them 🔧 **methods-forward** and 🎯 **goal-backward**. The distinction is subtle, but it shapes how teams form, how projects are scoped, and how they approach problems.

## 🔧 Methods-Forward

In methods-forward research, you start with a technique and go looking for problems it can solve. You pick up neural nets, or kernel machines, or Monte Carlo tree search, and you ask: where is this useful? How can I make it better?

This is how a lot of empirical ML research works. You build a codebase, pay the fixed cost of development once, and then sweep it across a suite of benchmarks. The codebase is the expensive part - the evals are comparatively cheap to plug in. So methods-forward research is often a rational response to the fixed cost of producing a codebase: you amortize that investment across as many problems as possible. Because codebases are expensive to build and maintain, they naturally get owned by groups of people.<span class="sidenote-number"></span><span class="sidenote">Think of the lab that builds a large-scale simulation engine, or a custom inference framework. That infrastructure investment defines what problems they can even attempt.</span> Entire research "areas" form around shared infrastructure: the diffusion group, the long context group, the reinforcement learning group. The method becomes the identity.<span class="sidenote-number"></span><span class="sidenote">Even the distinction between pre-training and post-training is somewhat artificial in this framing - it's a methods-forward split, organized around infrastructure and techniques rather than around the problems being solved. Though this may also be a function of the incredible cost and variety of methods required at each stage.</span>

Methods-forward is genuinely useful for exploration - when you don't know what problems exist, wandering with a powerful tool is a reasonable strategy. But it comes with a cost: you tend to see every problem through the lens of your method, and you can end up optimizing the tool rather than solving the problem.

## 🎯 Goal-Backward

In goal-backward research, you start with a problem and throw "the kitchen sink" at it. The eval is fixed. The methods are variables. You don't care how you get there, only whether you get there.

The cleanest version of this is a leaderboard. Think of a Kaggle competition: the objective is defined, the metric is defined, and nobody cares whether you use gradient boosting or a neural net or an ensemble of both.<span class="sidenote-number"></span><span class="sidenote">Kaggle is almost purely goal-backward. The winning solutions are often bizarre Frankenstein ensembles that no methods-forward researcher would ever build, because they work.</span> The same dynamic shows up wherever a community rallies around a fixed eval: ImageNet, GLUE, WMT translation benchmarks.<span class="sidenote-number"></span><span class="sidenote">ImageNet is the canonical example. A fixed benchmark organized an entire field's effort for years, and the progress was rapid precisely because everyone was attacking the same target with different methods.</span>

The main selling point of goal-backward research is loose binding. People and teams aren't tied to a particular method, so pivoting is easy. Because the focus is on the goal rather than the tool, you're quick to ditch approaches that aren't working. The eval outlasts any single method, and that's a feature. It also makes coordination at scale much simpler - you can point a large team at a shared objective without needing everyone to agree on how to get there.

## Why We Default to Methods-Forward

Despite its advantages, most researchers default to methods-forward. Codebases are expensive, so if you've spent six months building a training pipeline, you're going to use it.<span class="sidenote-number"></span><span class="sidenote">Sunk cost reasoning is powerful. It's hard to abandon infrastructure you've invested in, even when a different method would solve the problem faster.</span> Institutions organize around methods because it's easier to share infrastructure than to share goals. And publishing incentives reinforce it: researchers become known as the authors of a particular technique, and it's difficult to publish improvements on an eval if the method keeps shifting underneath you.<span class="sidenote-number"></span><span class="sidenote">A notable counter-example is AlphaFold, where the team was relatively agnostic to architecture throughout the project's lifetime. The goal (protein structure prediction) was fixed, and they were willing to change everything else - the <a href="https://www.nature.com/articles/s41586-021-03819-2">system entered in CASP14 was entirely different</a> from the one entered in CASP13, rebuilt from scratch around a new architecture (Evoformer) when the old one hit a ceiling. Most academic labs can't afford that posture.</span>

## The Case for Goal-Backward

For most applied research, goal-backward is the stronger default. Three reasons.

<span class="newthought">It prevents method lock-in.</span> Methods-forward teams get attached to their method. The fixed cost bias and the difficulty of producing new codebases often makes it hard to walk away from an approach, even when a different one would solve the problem faster. Goal-backward teams swap methods freely. When something better comes along, there's no sunk cost in the old approach because the investment was in the eval and the problem understanding, not in the implementation.<span class="sidenote-number"></span><span class="sidenote">This is the difference between "we need to make transformers work on this" and "we need to solve this, and right now transformers happen to be the best approach." The second framing makes it natural to switch when the landscape changes.</span>

<span class="newthought">It coordinates teams better.</span> Goal-backward gives a team a shared objective function. Everyone is pointed at the same target. Methods-forward teams tend to optimize locally - each subgroup improves their own method - which can make it harder to step back and ask whether a completely different approach would be more effective. Goal-backward research is more composable: different people can try different approaches, and the eval arbitrates.<span class="sidenote-number"></span><span class="sidenote">This is essentially why leaderboards work. They turn coordination into competition on a shared metric, which is much simpler than trying to align a team on which method to invest in.</span>

<span class="newthought">It is amenable to LLMs.</span> Language models are comfortable implementing a wide range of methods. The bottleneck has started to shift: it used to be "can I implement this?" and increasingly it's "do I know what success looks like?" Goal-backward research gives an LLM a clear target. You define the eval, and the LLM can help you try dozens of approaches in the time it used to take to try one.

The clearest example of this is Karpathy's <a href="https://github.com/karpathy/autoresearch">autoresearch</a>, which is purely goal-backward: you specify a metric (validation bits per byte), and an LLM agent runs hundreds of experiments overnight, modifying code, training, evaluating, and iterating. The current implementation forms a strong prior, but nothing in principle stops the agent from rearchitecting the method entirely.<span class="sidenote-number"></span><span class="sidenote">The entire codebase is ~630 lines, small enough to fit in a single context window. The agent doesn't care what architecture changes it makes, only whether the metric improves.</span> DeepMind's <a href="https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/">AlphaEvolve</a> follows the same pattern: define an evaluation function, and let a Gemini-powered agent evolve algorithms to optimize it. Both systems only work because the eval is clearly defined up front. Methods-forward research doesn't leverage LLMs nearly as well, because the value was in the accumulated codebase, and LLMs have made that accumulation cheap.

## Where This Leaves Us

Next time you start a project, define the eval before you write any code. Get the team aligned on what success looks like, and let the methods be disposable. The hard part is no longer implementation - it's knowing what to optimize for.

