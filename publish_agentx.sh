#!/usr/bin/env bash
set -uo pipefail
cd /home/liangdi/workspace/agent/adk-rust
ORDER=(agentx-adk-telemetry agentx-adk-rust)
for c in "${ORDER[@]}"; do
  for attempt in 1 2 3 4 5 6; do
    echo "=== [$c] attempt $attempt $(date -u +%H:%M:%S) ==="
    if out=$(cargo publish -p "$c" --registry crates-io --allow-dirty 2>&1); then
      echo "$out" | tail -2
      echo "=== [$c] DONE ==="
      break
    fi
    echo "$out" | tail -1
    if echo "$out" | grep -q "429"; then
      when=$(echo "$out" | grep -oE "after [A-Za-z]{3}, [0-9]{2} [A-Za-z]{3} [0-9]{4} [0-9:]+ GMT" | sed 's/after //')
      target=$(date -u -d "$when" +%s 2>/dev/null || date -u -d "now + 5 min" +%s)
      now=$(date -u +%s); wait=$(( target - now + 30 )); [[ $wait -lt 0 ]] && wait=30
      echo "  [429] waiting ${wait}s until $when"
      sleep "$wait"
    else
      echo "  [err] retrying in 120s"
      sleep 120
    fi
  done
  sleep 360
done
echo "ALL PUBLISHED"
