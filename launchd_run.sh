#!/bin/bash
cd "$(dirname "$0")"
now=$(date +%s)
scheduled=$(date -j -f "%H:%M" "09:30" +%s)
diff=$(( now - scheduled ))
if [ "$diff" -ge 0 ] && [ "$diff" -le 7200 ]; then
    exec .venv/bin/python -B main.py --run-now
fi
