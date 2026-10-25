#!/system/bin/sh
set -eu
CPU=/data/local/tmp/llama-completion-residency-ufs
MODEL=/data/local/tmp/layerskip-llama3-8b-q4-20261020/allshard-q4.gguf
BASE=/data/local/tmp/llama-exp-humaneval-full-budgets-linear-20260929
EXP=/data/local/tmp/llama-exp-q4-target-nearpeak-cpu8-20261025
if pidof llama-batch-gain-plannedinitial llama-batch-gain llama-completion llama-cli >/dev/null;then echo Concurrent-inference;exit 1;fi
mkdir -p "$EXP/runners" "$EXP/logs" "$EXP/results" "$EXP/memory" "$EXP/status"
test ! -e "$EXP/status/started";touch "$EXP/status/started"
echo $$ > "$EXP/runner.pid"
trap 'rc=$?;echo "$rc" > "$EXP/status/run.exit"' EXIT
sha256sum "$CPU/llama-completion" "$CPU"/*.so > "$EXP/deployment.sha256"
sha256sum "$MODEL" > "$EXP/model.sha256"
grep -q '^aea79ab3c536fd4d1c38862117bacf1df9d96fdf09c91c745995b01654a65c33 ' "$EXP/model.sha256"
for I in $(seq 0 19);do ID=$(printf '%03d' "$I");sha256sum "$BASE/prompts/human_eval_$ID.txt";done > "$EXP/prompts.sha256"
export LD_LIBRARY_PATH="$CPU"
unset LD_PRELOAD LLAMA_QUANT_DIRECT LLAMA_Q4_PREPACK_DIR LLAMA_MIX_COMPACT_DIR LLAMA_HTP_IO_CPU LLAMA_MODEL_IO_CPU LLAMA_MODEL_RESIDENCY_IO_CPU
echo 'Llama3-8B EXACT allshard group32Q4 Early/Tail/embedding/head same source SHA as prior heterogeneous20. CPU Target8threads noaffinity, noDraft/tree/NPU. mmap single-layer streaming plus fixed resident suffix selected by existing residency allocator. Match sampled owned2.2-2.3GiB, not hardOSbudget or all-filecache limit. Greedy fixed128 ignoreEOS, context2048. CPU eval127/time excludes prefill; record separately128/time for generation-count comparison. First run no60s; subsequent60s.'
FIRST=1
run_case(){
 CASE=$1;N=$2;BUDGET=$3;ID=$4
 if [ "$FIRST" = 0 ];then echo Cooling60seconds;sleep 60;fi;FIRST=0
 echo "===== $CASE budgetMiB=$BUDGET =====";dumpsys battery | grep -E 'level:|temperature:'
 "$CPU/llama-completion" -m "$MODEL" -f "$BASE/prompts/human_eval_$ID.txt" -n "$N" -c 2048 -b 512 -ub 128 -t 8 -dev none -ngl 0 --load-mode mmap --model-resident-budget-mib "$BUDGET" --runtime-reserve-mib 512 --temp 0 --top-k 1 --ignore-eos --no-display-prompt --no-warmup --perf > "$EXP/results/$CASE.txt" 2> "$EXP/logs/$CASE.log" &
 CHILD=$!;echo "$CHILD" > "$EXP/active.pid"
 sh "$BASE/peak_monitor.sh" "$CHILD" "$EXP/memory/$CASE" run save-peak > "$EXP/logs/$CASE.monitor.log" 2>&1 &
 MON=$!;set +e;wait "$CHILD";S=$?;wait "$MON";set -e
 echo "$S" > "$EXP/status/$CASE.exit";echo "exit=$S"
 if [ "$S" != 0 ];then tail -n 50 "$EXP/logs/$CASE.log";exit 1;fi
 grep -E 'model residency|prompt eval time|eval time|total time' "$EXP/logs/$CASE.log"
 grep -q 'model residency cached layers' "$EXP/logs/$CASE.log"
 cat "$EXP/memory/$CASE/summary.txt"
 PEAK=$(awk -F= '/max sampled RSS \+ DMA-BUF - DMA-BUF RSS=/{split($2,a," ");print a[1]}' "$EXP/memory/$CASE/summary.txt")
 test -n "$PEAK"
 touch "$EXP/status/$CASE.completed"
}
BUDGET=1700;FOUND=0
for TRY in 1 2 3 4 5 6;do
 run_case probe${TRY}_budget${BUDGET}_n16 16 "$BUDGET" 000
 if awk -v p="$PEAK" 'BEGIN{exit !(p>=2.15&&p<=2.308)}';then FOUND=1;break;fi
 if awk -v p="$PEAK" 'BEGIN{exit !(p>2.308)}';then BUDGET=$((BUDGET-128));else BUDGET=$((BUDGET+128));fi
done
if [ "$FOUND" != 1 ];then echo Cannot-match-short-peak-stop;exit 1;fi
echo "$BUDGET" > "$EXP/chosen-budget-mib.txt"
echo "SHORTMATCH budgetMiB=$BUDGET peakGiB=$PEAK; full128 peaks checked separately"
for I in $(seq 0 19);do
 ID=$(printf '%03d' "$I")
 run_case h${ID}_target_t8_n128 128 "$BUDGET" "$ID"
 if ! awk -v p="$PEAK" 'BEGIN{exit !(p>0&&p<=2.308)}';then echo 'Full-run-peak-exceeds2.308GiB; preserve result and stop before next question';exit 1;fi
 echo "PROGRESS completed=$((I+1))/20"
done
touch "$EXP/status/completed"
echo 'COMPLETE same-Q4 CPU8 nearpeak Target20; quality requires scoring saved output'
