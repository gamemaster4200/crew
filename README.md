# CREW

Local experimental application for orchestrating multiple LLM agents.

## Current version

CREW v0.0.2 benchmark UI.

Current comparison:

```text
4x GPT-5.6 Luna Crew
vs
1x GPT-5.6 Luna
```

Both solution branches start concurrently. The four-agent branch remains
sequential internally:

```text
Solver -> Critic -> Improver -> Integrator
```

The browser receives real stage progress from the backend.

All large text sections are collapsible. UI metrics use seconds and ktok.

## Cost accounting

The run JSON stores raw token counts, calculated USD cost, and the pricing
snapshot used for the calculation.

Pricing snapshot date:

```text
2026-09-23
```

Standard short-context text pricing per 1M tokens:

```text
GPT-5.6 Luna: input 0.20 USD, cached input 0.02 USD, output 1.20 USD
GPT-5.6 Sol:  input 4.00 USD, cached input 0.40 USD, output 20.00 USD
```

Long-context pricing is applied per call when input exceeds 272K tokens.

## Run

```powershell
.\start-crew.ps1
```

Open:

```text
http://127.0.0.1:8000
```

## Run artifacts

Each completed test is saved locally as UTF-8 JSON under `runs/`.
`runs/` is ignored by Git.

## Roadmap

- v0.0.0 - project skeleton
- v0.0.1 - first OpenAI API response in browser
- v0.0.2 - fixed Crew, comparison Test Mode, progress and benchmark metrics
- v0.0.3 - configurable Panel A / Panel B and three-call Sol review

## Security

`.env` is ignored by Git and must never be committed.