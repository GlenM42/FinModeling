from dotenv import load_dotenv

# Must run before any test module imports finmodeling.database (engine) or
# bot.secretary.agent (OpenAI client) — both are built from env vars at import time.
load_dotenv()
