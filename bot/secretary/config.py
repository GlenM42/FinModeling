import os

# Cheapest current OpenAI text model (per live pricing lookup: $0.20/$1.25 per 1M
# input/output tokens) — matches the "very simple function calling" cost priority.
# Nano-tier can be less reliable at multi-step tool-calling/argument extraction than
# mini-tier; if quality issues appear (bad dates, wrong tickers), change ONLY this
# line (e.g. to "gpt-5.4-mini") — no other file references a model name.
OPENAI_MODEL = "gpt-5.4-nano"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MAX_TOOL_HOPS = 4        # guards against an infinite tool-call loop
MAX_SESSION_TURNS = 20   # bounds in-memory chat history growth per user
