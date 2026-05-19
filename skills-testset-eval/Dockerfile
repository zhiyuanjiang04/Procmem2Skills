FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Node.js for Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @anthropic-ai/claude-code

# Python deps
RUN pip install --no-cache-dir \
    faiss-cpu numpy pandas sentence-transformers pytest anthropic \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Create non-root user (matching host UID for volume permissions)
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd -g ${USER_GID} evaluser \
    && useradd -m -u ${USER_UID} -g ${USER_GID} evaluser

# Directory layout matching code expectations:
#   /workspace/procmem2skills/  <- REPO_ROOT
#   /workspace/skillsbench_repo/
WORKDIR /workspace

COPY --chown=evaluser:evaluser testsets/ /workspace/procmem2skills/testsets/
COPY --chown=evaluser:evaluser data/ /workspace/procmem2skills/data/
COPY --chown=evaluser:evaluser skillsbench_repo/ /workspace/skillsbench_repo/
COPY --chown=evaluser:evaluser original-tasks/ /workspace/terminal-bench/original-tasks/

RUN mkdir -p /workspace/results && chown evaluser:evaluser /workspace/results
RUN mkdir -p /tmp/exec_prefill && chown evaluser:evaluser /tmp/exec_prefill

ENV SKILL_CORPUS_PATH=/workspace/procmem2skills/data/processed/skill_corpus.jsonl
ENV SKILL_INDEX_PATH=/workspace/procmem2skills/data/embeddings/index/index.faiss
ENV SKILL_META_PATH=/workspace/procmem2skills/data/embeddings/skill_metadata.jsonl
ENV SKILL_EMBEDDINGS_PATH=/workspace/procmem2skills/data/embeddings/skill_embeddings.npy
ENV PYTHONPATH=/workspace/procmem2skills/testsets

USER evaluser
ENTRYPOINT ["bash"]
