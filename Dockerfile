FROM python:3.12-slim

# Set up a working directory
WORKDIR /app

# Copy & install dependencies first (so you cache layers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# A non-root user:
RUN adduser --system --group bot \
  && chown -R bot:bot /app

USER bot

# 7) Run your bot
CMD ["python", "./telegram_bot.py"]
