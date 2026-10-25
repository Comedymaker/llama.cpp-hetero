#!/system/bin/sh

cd /data/local/tmp/layerskip-hetero-resident || exit 1

export LD_LIBRARY_PATH="$PWD"
export ADSP_LIBRARY_PATH="$PWD/dsp-current"
export DSP_LIBRARY_PATH="$PWD/dsp-current"
unset CDSP_LIBRARY_PATH
unset GGML_HEXAGON_NDEV
export GGML_HEXAGON_DEVICES=1

MODEL=/data/local/tmp/gguf/layerskip/layerskip-llama2-7B-F16.gguf
PROMPT=/data/local/tmp/llama-exp-layerskip-cnndm-lm/prompts/cnn_dm_lm_03_prefix100.txt
EXP=/data/local/tmp/llama-exp-layerskip-logits-diag/full

mkdir -p "$EXP"

echo "===== CPU target-only: cooling 60 seconds ====="
sleep 60

./llama-layerskip \
    -m "$MODEL" \
    -f "$PROMPT" \
    -n 128 \
    -c 2048 \
    -b 512 \
    -ub 128 \
    -t 4 \
    --exit-layer 8 \
    --draft-max 4 \
    --target-only \
    --logits-prefix "$EXP/cpu" \
    > "$EXP/cpu.txt" \
    2> "$EXP/cpu.log"

STATUS=$?
echo "cpu exit=$STATUS"
if [ "$STATUS" -ne 0 ]; then
    tail -n 100 "$EXP/cpu.log"
    exit "$STATUS"
fi

grep -E 'target-only generation|failed|error|ASSERT' "$EXP/cpu.log"
ls -lh "$EXP/cpu.f32" "$EXP/cpu.tsv"

echo "===== HTP split-target-only: cooling 60 seconds ====="
sleep 60

./llama-layerskip \
    -m "$MODEL" \
    -f "$PROMPT" \
    -n 128 \
    -c 2048 \
    -b 512 \
    -ub 128 \
    -t 4 \
    --exit-layer 8 \
    --draft-max 4 \
    --split-target-only \
    --tail-device HTP0 \
    --tail-offload-layers 25 \
    --logits-prefix "$EXP/htp" \
    > "$EXP/htp.txt" \
    2> "$EXP/htp.log"

STATUS=$?
echo "htp exit=$STATUS"
if [ "$STATUS" -ne 0 ]; then
    tail -n 100 "$EXP/htp.log"
    exit "$STATUS"
fi

grep -E 'split-target-only generation|failed|error|ASSERT' "$EXP/htp.log"
ls -lh "$EXP/htp.f32" "$EXP/htp.tsv"
sha256sum "$EXP/cpu.txt" "$EXP/htp.txt"
echo "===== Logit diagnostic completed ====="
