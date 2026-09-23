# CREW

Local experimental application for orchestrating multiple LLM agents.

## Current version

CREW v0.0.2 - first real fixed Crew.

## Run

Put the CREW project API key into the local `.env`:

```text
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.6-luna
```

Start:

```powershell
.\start-crew.ps1
```

Open:

```text
http://127.0.0.1:8000
```

## v0.0.2 workflow

```text
User task
   |
Solver
   |
Critic
   |
Improver
   |
Integrator
   |
Final answer
```

All four roles use the same configured model. The default is `gpt-5.6-luna`.

The UI also exposes a `1x Luna baseline` button so the same task can be
compared against the four-agent workflow.

Each run is saved locally as UTF-8 JSON under:

```text
runs/
```

The `runs/` directory is ignored by Git.

## Deliberate limitations

v0.0.2 does not yet have:

- configurable roles;
- configurable per-agent models;
- parallel execution;
- tools;
- database storage;
- long-term memory;
- autonomous routing;
- live per-stage streaming.

Those belong to later experiments.

## Roadmap

- v0.0.0 - project skeleton
- v0.0.1 - first OpenAI API response in browser
- v0.0.2 - fixed 4-agent Crew
- v0.0.3 - configurable agents, roles and models

## Security

`.env` is ignored by Git and must never be committed.