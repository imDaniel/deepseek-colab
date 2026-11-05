#!/bin/bash
set -u
if python3 /content/vm/github-cache-restore.py; then
  exit 0
fi
printf '%s\n' 'GitHub cache unavailable or incompatible; building original source.'
exec bash /content/vm/build.sh
