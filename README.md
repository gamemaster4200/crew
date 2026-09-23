# CREW

Local experimental application for orchestrating multiple LLM agents.

## Current version

CREW v0.0.3.

Two independently configurable panels solve the same task.

Each panel supports 1 to 4 sequential agents. Each agent has:

```text
role: Solver / Critic / Improver / Integrator
model: GPT-5.6 Luna / GPT-5.6 Sol
```

Default comparison:

```text
Panel A:
Solver Luna -> Critic Luna -> Improver Luna -> Integrator Luna

Panel B:
Solver Luna
```

Panel A and Panel B run concurrently.

## Three-call Sol review

The final answers are randomized into anonymous Candidate A and Candidate B.

Three independent GPT-5.6 Sol API calls are used:

```text
Sol Advocate ----\
                  -> Sol Judge
Sol Adversary ---/
```

Advocate and Adversary run concurrently. The Judge receives the original task,
both anonymous candidate answers, and both pole reports.

The reviewer never receives the panel configurations.

## UI

All large text blocks are collapsible.

Metrics are displayed as:

- seconds;
- ktok;
- USD cost;
- panel cost difference;
- panel cost ratio.

Progress bars reflect real backend stage transitions.

## Cost accounting

Each completed run stores the pricing snapshot used for its cost calculation.

Snapshot date:

```text
2026-09-23
```

Standard short-context text pricing per 1M tokens:

```text
GPT-5.6 Luna: input 0.20 USD, cached input 0.02 USD, output 1.20 USD
GPT-5.6 Sol:  input 4.00 USD, cached input 0.40 USD, output 20.00 USD
```

Long-context pricing is applied per call when input exceeds 272K tokens.

## Run artifacts

Completed experiments are stored as UTF-8 JSON under:

```text
runs/
```

The JSON contains:

- original task;
- both panel configurations;
- every agent artifact;
- model, latency, usage and cost for every call;
- three complete Sol reviewer calls;
- blind mapping;
- scores and verdict;
- pricing snapshot;
- aggregate metrics and cost differences.

`runs/` is ignored by Git.

## Run

```powershell
.\start-crew.ps1
```

Open:

```text
http://127.0.0.1:8000
```

## Security

`.env` is ignored by Git and must never be committed.