# Packaging validation

- Eight Target runtime files match the original completed-run SHA256 manifest exactly.
- The archived frontend runtime matches the existing planned-initial source snapshot's companion binary SHA256.
- Frontend source was independently compiled and linked with Android NDK r29 against the archived runtime. Output: `frontend-rebuild-check`. This does not replace the measured executable.
- Q4 host sources are copied from the exact no-mid-logits/postcompute-eviction build used for the completed20-question suite. Mixed host sources are copied from the exact no-mid-logits573 build, not a diagnostic frontend.
- Source and binary provenance are distinct. Target current-source-reference is later than the measured single-layer runtime and is NOT claimed hash-equivalent or behavior-equivalent.
- All archive files are listed by relative path in `files.sha256.csv`. Run `verify.ps1` after copying.
- No model inference is launched by packaging or verification. Archived original runners retain their no-overwrite checks and existing phone dependencies.
