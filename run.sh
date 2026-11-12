#!/bin/sh
# Starts Heartwise. The first run sets up a private Python environment in .venv.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt || exit 1
fi
exec .venv/bin/python server.py "$@"
