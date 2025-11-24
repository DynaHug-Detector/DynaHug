FROM python:3.10.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Installing strace
RUN apt-get update && apt-get install -y strace && rm -rf /var/lib/apt/lists/*

# Installing git 
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Installing tmux
RUN apt-get update && apt-get install -y tmux && rm -rf /var/lib/apt/lists/*

# Installing netcat and zsh for pypi injected payloads
RUN apt-get update && apt-get install -y netcat-traditional && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y zsh && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/app

# Optional: Set environment variables for uv
ENV UV_SYSTEM_PYTHON=1

# Copy your project files (if any)
COPY . /usr/src/app

# Example: Install dependencies using uv
RUN uv add -r requirements.txt

CMD ["/bin/bash"]