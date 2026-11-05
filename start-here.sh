#!/bin/sh
# colibri: streamlined launcher for Qwen3.6 / Qwen3-Coder on Apple Silicon (Metal + Unified Memory).
#
# Usage:
#   ./start-here.sh serve --model-dir /path/to/qwen36-model
#   ./start-here.sh chat  --model-dir /path/to/qwen36-model
#   ./start-here.sh info
#   ./start-here.sh build
#   ./start-here.sh stop

here=$(cd "$(dirname "$0")" && pwd)
launcher="$here/c/coli"
[ -f "$launcher" ] || launcher="$here/coli"

subcmd="${1:-serve}"
if [ $# -gt 0 ]; then
    shift
fi

for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        exec "$candidate" "$launcher" "$subcmd" "$@"
    fi
done

echo "colibri requires Python 3.10 or newer."
echo "On macOS, install with: brew install python"
exit 1
