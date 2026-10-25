#!/system/bin/sh
set -u

cd /data/local/tmp/layerskip-hetero-resident || exit 1

EXP=/data/local/tmp/llama-exp-layerskip-hetero-first5
PROMPTS=/data/local/tmp/llama-exp-layerskip-humaneval/f16-window46-first20/prompts
MODEL=/data/local/tmp/gguf/layerskip/layerskip-llama2-7B-F16.gguf
CPU_RESULTS=/data/local/tmp/llama-exp-layerskip-humaneval/f16-max512-first5/results

export LD_LIBRARY_PATH="$PWD"
export ADSP_LIBRARY_PATH="$PWD/dsp-current"
export DSP_LIBRARY_PATH="$PWD/dsp-current"
export GGML_HEXAGON_DEVICES=1
unset GGML_HEXAGON_NDEV

mkdir -p "$EXP/results" "$EXP/logs" "$EXP/status" "$EXP/runners"

for ID in 000 001 002 003 004; do
    PROMPT="$PROMPTS/human_eval_${ID}_lf.txt"
    if [ ! -f "$PROMPT" ]; then
        echo "missing prompt: $PROMPT"
        exit 1
    fi

    TAG="humaneval_${ID}_f16_e8_w6_self_htp25_n512"
    if [ -f "$EXP/status/$TAG.exit" ] && [ -s "$EXP/results/$TAG.txt" ] && [ "$(cat "$EXP/status/$TAG.exit")" = 0 ]; then
        echo "$TAG already completed; skipping"
        continue
    fi
    LOG="$EXP/logs/$TAG.log"
        RESULT="$EXP/results/$TAG.txt"

        echo "===== $TAG: cooling 60 seconds ====="
        sleep 60

        ATTEMPT=1
        while :; do
            echo "$TAG attempt=$ATTEMPT"
            ./llama-layerskip \
                -m "$MODEL" \
                -f "$PROMPT" \
                -n 512 -c 2048 \
                -b 512 -ub 128 \
                -t 4 \
                --exit-layer 8 \
                --draft-max 6 \
                --tail-device HTP0 \
                --tail-offload-layers 25 \
                > "$RESULT" 2> "$LOG"
            STATUS=$?
            echo "$STATUS" > "$EXP/status/$TAG.exit"
            echo "$TAG attempt=$ATTEMPT exit=$STATUS"
            if [ "$STATUS" -eq 0 ]; then
                break
            fi
            if [ "$ATTEMPT" -ge 2 ] || ! grep -q 'failed to open session' "$LOG"; then
                tail -n 40 "$LOG"
                exit "$STATUS"
            fi
            ATTEMPT=2
            sleep 120
        done

        grep -E 'layerskip (split-target-only )?generation|n_drafted|n_accept|n_verify|n_bonus|window accept|phase draft build|phase tail verify' "$LOG"
        dumpsys battery | grep -E 'level:|temperature:'

    SELF_RESULT="$EXP/results/humaneval_${ID}_f16_e8_w6_self_htp25_n512.txt"
    CPU_RESULT="$CPU_RESULTS/humaneval_${ID}_f16_target_n512.txt"
    if [ -f "$CPU_RESULT" ]; then
        if cmp -s "$CPU_RESULT" "$SELF_RESULT"; then
            echo "humaneval_${ID} CPU target/hetero self output: identical"
        else
            echo "humaneval_${ID} CPU target/hetero self output: DIFFERENT"
        fi
    fi
done

echo '===== hetero first-five experiment completed ====='
touch "$EXP/runners/complete"
