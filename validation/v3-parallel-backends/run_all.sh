#!/bin/bash
# Iterate all joblib backends, appending rows to results.md.
# Uses tester image pybroker-validator:latest (must be built beforehand).
set -u
cat > results.md << 'EOF'
| Backend         |  Wall time |  Correct  | Notes |
|-----------------|-----------:|:---------:|-------|
EOF

fail=0
for b in loky threading multiprocessing ray dask spark; do
    echo "===== $b =====" >&2
    if ! ./run_one.sh "$b"; then
        echo "| ${b} | - | FAIL | see stderr /tmp/test-${b}.stderr |" >> results.md
        fail=1
    fi
done

# Final teardown: remove shared network, no named volumes were created
podman network rm pybroker-net >/dev/null 2>&1 || true
echo "===== done (fail=$fail) =====" >&2
exit "$fail"
