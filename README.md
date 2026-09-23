# CREW

Local experimental application for orchestrating multiple LLM agents.

## Current version

CREW v0.0.1 вЂ” first OpenAI API response in the browser.

## Run

1. Put the CREW project API key into the local `.env`:

```text
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.6-luna
```

2. Start CREW:

```powershell
.\start-crew.ps1
```

3. Open:

```text
http://127.0.0.1:8000
```

## v0.0.1 architecture

```text
Browser
  в†“
FastAPI localhost
  в†“
OpenAI Responses API
  в†“
GPT-5.6 Luna
  в†“
Browser
```

There is deliberately no multi-agent workflow yet.

## Roadmap

- v0.0.0 вЂ” project skeleton
- v0.0.1 вЂ” first OpenAI API response in browser
- v0.0.2 вЂ” fixed 4-agent Crew
- v0.0.3 вЂ” configurable agents, roles and models

## Security

`.env` is ignored by Git and must never be committed.