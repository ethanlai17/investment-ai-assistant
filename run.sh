#!/bin/bash
cd "$(dirname "$0")"
exec python -B main.py "$@"
