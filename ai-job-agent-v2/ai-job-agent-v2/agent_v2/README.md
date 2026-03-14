# 🤖 AI Job Search Agent

A Single LLM-Based AI Agent that autonomously performs job filtering, ranking, and resume tailoring using Google Gemini.

## Pipeline
```
Candidate Profile → LLM Reasoning → Filter Tool → Ranking Tool → Resume Tailor Tool → Output
```

## Setup

```bash
pip install -r requirements.txt
```

Set your Gemini API key in `main.py`:
```python
GEMINI_API_KEY = "your_key_here"  # Get free at aistudio.google.com
```

## Run
```bash
python main.py
```

## Output Files
- `output/agent_output.txt` — Full agent output with reasoning trace
- `output/filtered_jobs.json` — Jobs after filtering
- `output/ranked_jobs.json` — Ranked jobs with scores
- `output/tailored_resume.json` — Tailored summary + modified bullets
- `logs/agent.log` — Complete decision log
