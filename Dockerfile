FROM python:3.12-slim

# Set up a working directory
WORKDIR /app

# Install uv into the image
RUN pip install --no-cache-dir uv

# Copy only dependency first
COPY pyproject.toml uv.lock ./

# Create venv and install base and bot deps
RUN uv venv /app/.venv \
 && uv sync --group bot --frozen --no-dev

# Copy the rest of the code
COPY . .

# now drop privileges
RUN useradd -ms /bin/bash bot
USER bot

# Ensure the venv is used
ENV PATH="/app/.venv/bin:$PATH"

# Run the bot
CMD ["python", "./telegram_bot.py"]
