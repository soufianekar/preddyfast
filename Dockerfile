# Hugging Face Spaces (Docker SDK). The Space runs the container as uid 1000,
# so everything the app writes — flask cache, HTTP cache, point extracts, the
# precomputed mirror — lives under that user's home. The disk is ephemeral:
# caches rebuild after a restart, and the precomputed metadata re-syncs from
# the Dataset (utils/hub_sync.py).
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src/ ./src/
WORKDIR $HOME/app/src

ENV REACT_VERSION=18.2.0 \
    URL_BASE_PATHNAME=/ \
    PORT=7860

EXPOSE 7860

# Two workers for the Space's two vCPUs; 90 s stays above the slowest bounded
# read. No --preload: forking after import is unsafe here. The rate limiter's
# storage starts a timer thread at import that a forked worker would not have,
# and on macOS the forked workers die in the Objective-C runtime. Nothing calls
# an API at startup, so importing once per worker costs only a few seconds.
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT} --workers=2 --timeout=90 app:server"]
