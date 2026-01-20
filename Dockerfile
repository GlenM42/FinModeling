FROM python:3.12-slim

# Set up a working directory
WORKDIR /app

# Install curl, install uv, then remove curl in same layer to minimize size
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh \
    && apt-get remove -y curl \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and group (UID/GID 1000) and give ownership of /app
RUN groupadd -g 1000 cthulhu && \
    useradd -u 1000 -g cthulhu -m -s /bin/bash cthulhu && \
    chown cthulhu:cthulhu /app

# Copy dependency files first (better caching)
# Use --chown to avoid creating duplicate layer
COPY --chown=cthulhu:cthulhu pyproject.toml uv.lock ./

# Create venv and install base and bot deps
RUN uv venv /app/.venv \
 && uv sync --group bot --frozen --no-dev

# Copy application code with proper ownership
COPY --chown=cthulhu:cthulhu . .

# Switch to non-root user for runtime
USER cthulhu

# Ensure the venv is used
ENV PATH="/app/.venv/bin:$PATH"

# Run the bot
CMD ["python", "./telegram_bot.py"]
