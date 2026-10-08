#!/usr/bin/env bash
# Waits for the GPU to go idle (the user's Gamma training finishing),
# then runs the shading closure + metrics for the remaining 4 styles.
#
# GPU-idle detection is process-based: a CUDA training's utilization dips
# to 0% during data loading, so utilization polling gets fooled - the
# presence of ANY compute process means occupied.
cd "$(dirname "$0")/.."

while true; do
    # Occupancy = a large python process (the Gamma training holds ~5 GB;
    # desktop apps show [N/A] memory in nvidia-smi here, so process-based
    # queries are useless - check python RAM via PowerShell instead).
    HEAVY_PYTHON=$(powershell -Command         "@(Get-Process python -ErrorAction SilentlyContinue | Where-Object {\$_.WorkingSet64 -gt 2GB}).Count" 2>/dev/null)
    if [ "${HEAVY_PYTHON:-0}" -eq 0 ]; then
        echo "supervisor: no heavy python process - GPU free, starting"
        break
    fi
    echo "supervisor: Gamma training detected (${HEAVY_PYTHON} heavy python), waiting..."
    sleep 180
done

for style in bear wstraight wwavy wcurly; do
    echo "=== shading $style ==="
    timeout 900 ./bin/vkhr.exe --shade yes --shade-dir "dumps/dataset_full/${style}_eval" \
        --shade-source recon "share/scenes/${style}.vkhr" > "shade_${style}.log" 2>&1
    CODE=$?
    FRAMES=$(grep -c "shade: frame" "shade_${style}.log" || true)
    echo "shading $style: exit $CODE, $FRAMES frames shaded (log: shade_${style}.log)"
done

echo "=== metrics ==="
for style in bear wstraight wwavy wcurly ponytail; do
    echo "--- $style ---"
    KMP_DUPLICATE_LIB_OK=TRUE python nn/metrics.py --dump "dumps/dataset_full/${style}_eval" 2>&1 |
        grep -vE "Warning|warn" | tail -9
done

echo "supervisor: all styles complete"
