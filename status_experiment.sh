#!/bin/bash
# status_experiment.sh
# One-line summary of the running grokking ablation experiment

# Default to the most recent combined results folder if none specified
RESULTS_DIR="${RESULTS_DIR:-results_2026-07-03_4condition_125seeds_FINAL}"
LOG_FILE="$RESULTS_DIR/experiment.log"

# If no log file exists yet, try to find recent output
if [ ! -f "$LOG_FILE" ]; then
    # Look for the most recent progress in the results directory
    LATEST=$(find "$RESULTS_DIR" -name "metrics.json" -o -name "history.npz" 2>/dev/null | head -1)
    if [ -z "$LATEST" ]; then
        echo "No experiment output found yet. Still initializing..."
        exit 0
    fi
fi

# Try to read the last few lines from the running terminal if possible
# Fall back to checking the results directory structure

CONDITIONS=(baseline full_steerage_v1 full_steerage full_steerage_v2)
CURRENT_CONDITION=""
CURRENT_SEED=""
LAST_STEP=""

# Check which condition folders have seeds
for cond in "${CONDITIONS[@]}"; do
    if [ -d "$RESULTS_DIR/$cond" ]; then
        SEED_COUNT=$(ls "$RESULTS_DIR/$cond"/seed_* 2>/dev/null | wc -l)
        if [ "$SEED_COUNT" -gt 0 ]; then
            CURRENT_CONDITION="$cond"
            # Get the highest seed number
            HIGHEST_SEED=$(ls "$RESULTS_DIR/$cond"/seed_* 2>/dev/null | sort -V | tail -1 | sed 's/.*seed_//')
            CURRENT_SEED="$HIGHEST_SEED"
            
            # Try to get the last step from metrics if available
            METRICS="$RESULTS_DIR/$cond/seed_$HIGHEST_SEED/metrics.json"
            if [ -f "$METRICS" ]; then
                LAST_STEP=$(python3 -c "
import json
with open('$METRICS') as f:
    data = json.load(f)
print(data.get('steps_to_0.9', 'N/A'))
" 2>/dev/null || echo "N/A")
            fi
        fi
    fi
done

if [ -n "$CURRENT_CONDITION" ]; then
    printf "Condition: %-22s | Seed: %3s | Last completed: %s\n" \
        "$CURRENT_CONDITION" "$CURRENT_SEED" "${LAST_STEP:-in progress}"
else
    echo "Experiment starting... no seeds completed yet."
fi

echo "Results: $RESULTS_DIR"
