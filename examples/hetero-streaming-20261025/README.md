# Current streaming experiment snapshots (2026-10-25)

This is an experiment archive, not a replacement of the repository's normal examples or a new CMake target. No models, credentials, calibration datasets or generated candidate code are included. Original experiments remain unchanged.

## Entry points

| Component | Code | Measured runner |
| --- | --- | --- |
| CPU gain-tree frontend, shared by both routes | `pipeline/frontend/frontend.cpp`, `batch-gain-run.h`, `queue.cpp` | `runners/q4/run.sh` |
| Buffered pread into final HTP buffers | `pipeline/host/prepacked-read.h`, `llama-context.cpp` | same runner, direct cases |
| O_DIRECT into DRAM, then asynchronous copy | `pipeline/copy/staged-preload.cpp` | same runner, staged cases |
| NPU weight registration and execution | `pipeline/backend/ggml-hexagon.cpp`, `pipeline/backend/htp/` | same libraries and DSP |
| Frozen573 mixed Q2/Q3 variant | `pipeline/mixed-host/`, `pipeline/format/`, `pipeline/mixed-copy/` | `runners/mixed573/run.sh` |
| CPU autoregressive Target, measured single-layer loader | `runtime/target/`, `runners/target/run.sh` | 13 cached layers, budget2212MiB |

AllQ4: Llama3-8B canonical shard Q4_0 group32, Early8/Tail24/embedding/head all Q4. Mixed573: Q4 Early8/aux; Tail573 Q3 plus195 Q2. Both use CPU8 gain tree top4/depth10/cap63, all24Tailstreamed, Q2 A/B, seven tensor coordinators. Copy route uses seven O_DIRECT workers plus one copy worker. No NPU Draft is selected. The23 unused intermediate vocab buffers are removed; hidden/KV/Early/final outputs remain.

Q4 sources in `pipeline/host` defer source-cache eviction until successful layer decode synchronization. Shared buffer lifetime is independent: do not overwrite a slot while its NPU work is in flight. `mixg32` queue names in the host are also used by the Q4 postcompute-eviction hook; this does not imply Q4 weights were changed into mixed weights.

## Important Target source limitation

The exact measured Target executable and libraries are preserved in `runtime/target`, with the SHA identities from the completed experiment. The local current `llama.cpp/tools/completion/completion.cpp` has since changed to two-layer asynchronous prefetch. It is saved ONLY under `target/current-source-reference`; it is NOT the exact source of the measured single-layer binary, and must not be used to claim reproduction of the reported0.712t/s and2.219GiB peak. A hash-matched original single-layer source snapshot has not yet been located. No source was fabricated or silently substituted. Replaying the measured Target uses the frozen runtime and original runner; rebuilding the current reference requires a new validation run.

## Runtime and model identities

`runtime/pipeline` contains the measured frontend, CPU/HTP libraries and DSP; its `libllama.so` is the current Q4 no-mid-logits plus postcompute-eviction build. `runtime/mixed-host/libllama.so` is the measured no-mid-logits573 variant. `runtime/copy` contains the measured Q4 and mixed copying interposers. `runtime/target` contains the separate exact CPU Target runtime. These are Android ARM64/v79 files, not Windows executables.

The GGUF is deliberately not copied. Q4 SHA256:

```text
aea79ab3c536fd4d1c38862117bacf1df9d96fdf09c91c745995b01654a65c33
```

Model, prepack, mixed sidecars and prompt paths remain the original paths in the runners. Runners are archived verbatim, including no-overwrite guards and deployment checks; do not run them against an existing completed output directory. They also refer to the already installed phone runtime paths. This archive is not an automatic replacement deployment. For a new run, use a new experiment directory and update runtime paths together, then verify the hashes. Do not copy only an executable while mixing its ABI with arbitrary library versions.

## Building

The pipeline sources depend on private llama extension APIs and a frozen Android build of the original experiment repository. The build scripts are retained, not advertised as standalone builds against any upstream llama.cpp revision. Required base:

```text
C:/Users/zhuxu/Desktop/llama.cpp/tmp-tail-budget-queue-20261008
```

Mount that read-only as `/queue` in `ghcr.io/snapdragon-toolchain/arm64-android:v0.7`. Mount `pipeline` as `/work` when using `build-host.sh` or `build-backend.sh`. The original host relink script also needs the frozen `models/llama.cpp.o` object, archived as `pipeline/host/model-llama.o`. Base cached objects are external prerequisites, not reimplemented here. `build-dsp.sh` requires the original Hexagon6.6/tool19 v79 SDK/runtime dependencies. It is an advanced rebuild recipe; do not overwrite the measured runtime before validating rebuilt outputs.

For frontend or copy builds, mount the corresponding subdirectory as `/work`. Frontend's original script expects the matching shared libraries in that same work directory; provide the archived runtime there in a separate temporary build directory. The copy build script must be run with the matching interposer source; do not substitute the mixed interposer for Q4. Private absolute `/queue/src` includes are preserved for provenance.

## Validation and result references

Run `./verify.ps1` in PowerShell to check every archived file against `files.sha256.csv`. Provenance contains the original experiment hash manifests; these use phone paths and differ from the portable archive file list. SHA checks confirm archived identities, not that a rebuild has been numerically revalidated.

`runners/*/RESULTS.md` and `summary.json` record the measured data. Source-level declarations and shell syntax are checked on packaging; no20-question inference rerun is performed just to move code. Full quality-scoring outputs and large raw logs remain in the original experiment workspace, not this examples archive. The main report is `experiment/zx/2026-10-25—分片实验.md` in the destination repository.
