FROM python:3.12-slim

# Set up a working directory
WORKDIR /app

# Copy & install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries=5 -r requirements.txt --progress-bar off -v

# Copy the rest of the code
COPY . .

# A non-root user:
RUN adduser --system --group bot \
  && chown -R bot:bot /app

USER bot

# Run the bot
CMD ["python", "./telegram_bot.py"]
