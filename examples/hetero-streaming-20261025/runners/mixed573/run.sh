#!/system/bin/sh
set -eu
DIR=/data/local/tmp/mixed-q23-phone573-20261024
STAGED=/data/local/tmp/mixed-q23-staged573-20261024
NEW=/data/local/tmp/q23-no-mid-logits-cpu8-20261025
EXP=/data/local/tmp/llama-exp-q23-no-mid-phone20-cpu8-20261025
Q3=/data/local/tmp/layerskip-q3g32-8b-20261022
Q4=/data/local/tmp/layerskip-llama3-8b-q4-20261020
BASE=/data/local/tmp/llama-exp-humaneval-full-budgets-linear-20260929
if pidof llama-batch-gain llama-batch-gain-plannedinitial llama-tree-diag llama-cli probe convert >/dev/null;then echo Concurrent-inference;exit 1;fi
mkdir -p "$EXP/runners" "$EXP/logs" "$EXP/results" "$EXP/status" "$EXP/memory"
test ! -e "$EXP/status/started";touch "$EXP/status/started"
echo $$ > "$EXP/runner.pid"
trap 'rc=$?;echo "$rc" > "$EXP/status/run.exit"' EXIT
cd "$DIR"
sha256sum -c /data/local/tmp/llama-exp-mixed-q23-phone573-20261024/deployment.sha256 > "$EXP/logs/deployment-check.log"
sha256sum -c /data/local/tmp/llama-exp-mixed-q23-phone573-20261024/weights.sha256 > "$EXP/logs/weights-check.log"
sha256sum "$STAGED/libstaged-q23.so" "$Q3/initial-profile.tsv" > "$EXP/route-input.sha256"
sha256sum "$NEW/libllama.so" > "$EXP/new-library.sha256"
test "$(sha256sum "$NEW/libllama.so" | awk '{print $1}')" = "$(awk '{print $1}' /data/local/tmp/llama-exp-q23-no-mid-logits-cpu8-20261025/new-library.sha256)"
for I in $(seq 0 19);do ID=$(printf '%03d' "$I");test -s "$BASE/prompts/human_eval_$ID.txt";sha256sum "$BASE/prompts/human_eval_$ID.txt";done > "$EXP/prompts.sha256"
export LD_LIBRARY_PATH="$NEW:$DIR" ADSP_LIBRARY_PATH="$DIR/dsp" DSP_LIBRARY_PATH="$DIR/dsp"
unset LLAMA_KEEP_BOUNDARY_LOGITS
export GGML_HEXAGON_DEVICES=1 GGML_HEXAGON_MBUF=256 GGML_HEXAGON_VMEM=2048 GGML_HEXAGON_PROFILE=0
unset LD_PRELOAD LLAMA_DEBUG_TREE_RECORD LLAMA_DEBUG_TREE_REPLAY LLAMA_DEBUG_VERIFY_PAD LLAMA_DEBUG_STOP_ROUNDS LLAMA_Q2_GROUP32_DIR LLAMA_Q3_GROUP32_DIR LLAMA_MIX_G32_DIR LLAMA_Q4_SPLIT_DIR LLAMA_Q4_CONTIGUOUS LLAMA_GAIN_MEASURED_PROFILE LLAMA_HTP_IO_CPU LLAMA_SSD_OUTCOMES LLAMA_TREE_SCALING_MODE LLAMA_TREE_SCALING_NODES LLAMA_TREE_SCALING_ROUNDS LLAMA_BATCH_EMPIRICAL_PROFILE LLAMA_BATCH_NUCLEUS_P LLAMA_GAIN_VERIFY_ONLY_COST LLAMA_GAIN_FIXED_VERIFY_COST LLAMA_GAIN_DRAFT_ONLY_COST
export LLAMA_MIX_COMPACT_DIR="$DIR/weights" LLAMA_BATCH_GAIN=cpu LLAMA_HTP_DIRECT_LOAD=1 LLAMA_HTP_RESIDENT_DIRECT=1 LLAMA_CPU_EARLY_ANON=1 LLAMA_QUANT_DIRECT=1
export LLAMA_PIPELINE_TIMELINE=1 LLAMA_CLUSTER_TRACE=1 LLAMA_EARLY_DIAG=1 LLAMA_TAIL_QUEUE_READERS=1 LLAMA_STI_SLOT_REUSE=1
export LLAMA_Q4_PREPACK_DIR="$Q4/prepack4" LLAMA_Q4_TENSOR_READERS=7 LLAMA_HTP_READ_ROUTE_TRACE=1 LLAMA_TREE_INITIAL_PROFILE="$Q3/initial-profile.tsv"
export STAGED_Q23_MODE=pipe STAGED_READ_WORKERS=7 STAGED_COPY_WORKERS=1
echo 'Frozen573Q3/195Q2 Tail, Q4 Early8+embedding/head. HumanEval000-019 fixed128 ignoreEOS; CPU8 ONLY, no NPU Draft. Same original dynamic batched gain tree top4/depth10/cap63; unchanged Q3 initial profile (not newly CPU8 calibrated), online EMA retained. All24 Tail layers streamed Q2 A/B. Direct=bufferedpread finalHTP+source eviction; staged=O_DIRECT ordinaryDRAM7read+1copy, double8MiB/caller max112MiB. Same post-compute async eviction hook. Alternating within-prompt route order; first inference no60s, subsequent60s cooldown. No forced tree/output equality, no hard OS memory cap or quality-pass claim.'
FIRST=1;DONE=0
echo 'NEW VARIANT: same573 weights/frontend/CPU/DSP, ONLY libllama context allocator updated. Remove23unused boundary vocab outputs immediately; Early/final logits and boundary hidden retained. Not old CPU8 baseline, not a new quantization or numerical kernel fix.'
for I in $(seq 0 19);do
 ID=$(printf '%03d' "$I")
 if [ "$((I % 2))" = 0 ];then ROUTES='direct staged';else ROUTES='staged direct';fi
 for ROUTE in $ROUTES;do
  CASE=h${ID}_${ROUTE}_t8_n128
  if [ "$FIRST" = 0 ];then echo 'Cooling60 seconds';sleep 60;fi;FIRST=0
  unset LD_PRELOAD
  if [ "$ROUTE" = staged ];then export LD_PRELOAD="$STAGED/libstaged-q23.so";fi
  echo "===== $CASE =====";dumpsys battery | grep -E 'level:|temperature:'
  ./llama-batch-gain-plannedinitial -m "$Q4/allshard-q4.gguf" -f "$BASE/prompts/human_eval_$ID.txt" -n 128 -c 2048 -b 512 -ub 128 -t 8 --exit-layer 8 --draft-max 10 --tail-device HTP0 --tail-offload-layers 25 --tail-layer-stream --tail-stream-prefetch --tail-stream-cache-layers 0 --tail-stream-cluster-layers 1 --tail-stream-slots 1 --memory-budget-mib 9216 --runtime-reserve-mib 1536 --tail-budget-prefetch --tail-budget-queue-layers 2 --tail-budget-allocator-mib 3600 --ignore-eos --tree-gain-cost --tree-max-nodes 63 --tree-top-k 4 --tree-verify-base-ms 645.508296 --tree-node-ms 8.476174 --tree-leaf-ms 0 > "$EXP/results/$CASE.txt" 2> "$EXP/logs/$CASE.log" &
  CHILD=$!;echo "$CHILD" > "$EXP/active.pid";unset LD_PRELOAD
  sh "$BASE/peak_monitor.sh" "$CHILD" "$EXP/memory/$CASE" run save-peak > "$EXP/logs/$CASE.monitor.log" 2>&1 &
  MON=$!;set +e;wait "$CHILD";S=$?;wait "$MON";set -e
  echo "$S" > "$EXP/status/$CASE.exit";echo "exit=$S"
  if [ "$S" != 0 ];then tail -n 45 "$EXP/logs/$CASE.log";exit 1;fi
  grep -q '^tail HTP preload failures = 0' "$EXP/logs/$CASE.log"
  grep -q '^mixg32 after-compute eviction ' "$EXP/logs/$CASE.log"
  grep -q '^q4 async_evict summary .*failures=0' "$EXP/logs/$CASE.log"
  grep -q 'CPU Early anonymous read = 72 tensors, 981729280 bytes' "$EXP/logs/$CASE.log"
  grep -q '^layerskip generation 128 tokens ' "$EXP/logs/$CASE.log"
  DROPS=$(grep -c '^boundary output range=' "$EXP/logs/$CASE.log")
  test "$DROPS" -eq 23
  echo "Confirmed unused boundary vocab outputs dropped=$DROPS"
  grep -q 'htp-actions=0' "$EXP/logs/$CASE.log"
  if grep -E 'GGML_ASSERT|mixcompact invalid|NaN|batch-bridge: ERROR|^STAGED_.*ok=0' "$EXP/logs/$CASE.log";then exit 1;fi
  if [ "$ROUTE" = staged ];then
   grep -q '^STAGED_POOLS read=7 copy=1 ' "$EXP/logs/$CASE.log"
   PRELOADS=$(awk '/^tail HTP preload calls =/{print $NF}' "$EXP/logs/$CASE.log")
   CALLS=$(grep -c '^STAGED_Q23 call=' "$EXP/logs/$CASE.log")
   test "$CALLS" -eq "$((PRELOADS * 7))"
   echo "Confirmed staged calls=$CALLS expected=$((PRELOADS * 7))"
  else
   if grep -q '^STAGED_INTERVAL ' "$EXP/logs/$CASE.log";then echo Wrong-route;exit 1;fi
  fi
  grep -E '^layerskip generation|^n_drafted|^n_accept|^n_verify|^n_bonus|^accept =|^phase draft|^phase tail|^phase correction|^pipeline generation|^tail budget generation|^batch-gain summary' "$EXP/logs/$CASE.log"
  cat "$EXP/memory/$CASE/summary.txt"
  touch "$EXP/status/$CASE.completed"
  DONE=$((DONE+1));echo "PROGRESS completed=$DONE/40"
 done
 if cmp -s "$EXP/results/h${ID}_direct_t8_n128.txt" "$EXP/results/h${ID}_staged_t8_n128.txt";then SAME=1;else SAME=0;fi
 echo "PAIR h$ID output_exact=$SAME diagnostic_only=1" >> "$EXP/pairs.tsv"
done
touch "$EXP/status/completed"
echo 'COMPLETE no-mid-logits frozen573 CPU8 HumanEval20 direct/staged 40 runs; output equality diagnostic only, quality not scored here'
