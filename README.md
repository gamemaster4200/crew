# CREW

Local experimental application for orchestrating multiple LLM agents.

## Current version

CREW v0.0.2 - fixed four-agent Crew with comparison Test Mode and live stage progress.

## Models

Default worker model:

```text
gpt-5.6-luna
```

Default independent evaluator:

```text
gpt-5.6-sol
```

Optional local override in `.env`:

```text
OPENAI_MODEL=gpt-5.6-luna
OPENAI_EVALUATOR_MODEL=gpt-5.6-sol
```

## Run

```powershell
.\start-crew.ps1
```

Open:

```text
http://127.0.0.1:8000
```

## Test Mode

One user task is solved by two independent branches:

```text
                         +-> Solver -> Critic -> Improver -> Integrator --+
User task ---------------+                                                +-> Sol evaluator
                         +-> 1x Luna baseline -----------------------------+
```

The 4x Luna branch and 1x Luna baseline run concurrently.

The browser receives real progress events from the backend:

```text
4x Luna:
Solver -> Critic -> Improver -> Integrator

1x Luna:
Single -> done

Sol:
waiting -> evaluating -> done
```

The progress bars reflect actual stage transitions, not estimated elapsed time.

The Sol evaluator receives anonymous Candidate A and Candidate B in randomized
order. It does not receive the internal Crew trace and is not told which
candidate came from the four-agent workflow.

## Evaluator output

The evaluator produces:

- a concise advocate case for each candidate;
- a concise adversarial case against each candidate;
- 0-10 scores for correctness, completeness, robustness, relevance, and
  actionability;
- A, B, or TIE preference;
- confidence;
- a short verdict.

## Metrics

The UI shows:

- API call count;
- input tokens;
- output tokens;
- reasoning tokens when reported by the API;
- total tokens;
- latency;
- total Test Mode wall time.

## Run artifacts

Each run is saved locally as UTF-8 JSON under:

```text
runs/
```

`runs/` is ignored by Git.

## Deliberate limitations

v0.0.2 still does not have:

- configurable roles;
- configurable per-agent models in the UI;
- parallel agents inside the four-agent Crew;
- tools;
- database storage;
- long-term memory;
- autonomous routing;
- token-by-token response streaming.

## Roadmap

- v0.0.0 - project skeleton
- v0.0.1 - first OpenAI API response in browser
- v0.0.2 - fixed 4-agent Crew + comparison Test Mode + live progress
- v0.0.3 - configurable agents, roles and models

## Security

`.env` is ignored by Git and must never be committed.