# Optimized context tiers, 2026-11-06

The saved [configuration snapshot](../../configs/colab-optimized-20261106.json) records the current Magpie defaults and all three scoped boot profiles. The default remains **512K**; 256K is the faster tier and 1M is optional/experimental. Runtime source in `llm.py` and `integrations/magpie_colab/config.py` remains authoritative. The snapshot contains no credentials or deployment-specific addresses.

All measurements use G4, ISTA IQ2_XXS + AtomicChat, images and DSpark enabled, draft depth 2 and the same cached b11963 binary. Complete-code tests use max reasoning effort, temperature 0.6, top_p 1, seed 42 and a 32,768 output budget. Rates include reasoning. First token means the start of reasoning; first visible means the start of the actual answer. Cold timings start with a loaded model, excluding GPU allocation, download and loading.

## Parameters and observed memory headroom

| Setting | 256K | 512K | 1M |
|---|---:|---:|---:|
| Per-request input-plus-output window | 262,144 | 524,288 | 1,048,576 |
| Input budget with 32,768 output reserved, including templates/history | 229,376 | 491,520 | 1,015,808 |
| Expert cache MiB | 73,000 | 73,000 | 68,864 |
| UBATCH | 1,024 | 512 | 512 |
| DYN | 166 | 166 | 80 |
| Index query chunk MiB | 0 | 0 | 128 |
| Decoder slots | 2 | 1 | 1 |
| Minimum available host RAM | 3.72 GiB | 3.79 GiB | 2.52 GiB |
| Minimum free VRAM | 2.12 GiB | 2.09 GiB | 2.02 GiB |

All tiers accept four in-flight clients including queueing. This does not mean four simultaneous full-window decodes. The 256K layout allocates two 262,144-token slots; 512K allocates one 524,288-token slot. Their total context allocations are equal, helping explain their similar memory headroom.

These overrides apply only to the patched ISTA/image/DSpark combination. Other model and feature combinations retain their existing layouts. The memory minima are observed in different qualification sessions, not identical-load guarantees or long-term reliability certification.

## Complete short code

The same fixed LRU task is repeated three times at each tier.

| Measurement | 256K | 512K | 1M |
|---|---:|---:|---:|
| Complete generation rate, mean | 121.46 tok/s | 104.54 tok/s | 95.45 tok/s |
| Generated tokens per answer | 4,427 | 5,326 | 4,174 |
| First visible answer, mean | 30.96 s | 44.10 s | 35.65 s |
| Complete task time, mean | 36.71 s | 51.21 s | 44.03 s |
| Approximately 131K-input cold first token, low-effort math probe | 47.32 s | 55.16 s | 56.33 s |

Different output trajectories and lengths affect task time. For example, the 1M short task finishes before the 512K task while generating fewer tokens at a lower generation rate. This is not evidence that the 1M engine is faster. The 128K math probes are low-effort prefill diagnostics, not complete-code throughput tests.

## Long-input complete code

| Measurement | 256K | 512K | 1M |
|---|---:|---:|---:|
| Actual input tokens | 229,093 | 480,093 | 1,000,125 |
| Cold complete-code generation | not measured | 89.79 tok/s | 73.85 tok/s |
| Prefix-hit complete-code generation | 108.70 tok/s | 91.23 tok/s | 74.37 tok/s |
| Cold-input first token | 85.62 s, raw probe only | 250.53 s | 719.21 s |
| Cold complete-code first visible answer | not measured | 284.35 s | 771.58 s |
| Prefix-hit first token | 0.17 s | 0.30 s | 0.59 s |
| Prefix-hit complete-code first visible answer | 19.75 s | 34.09 s | 33.22 s |

The 256K cold probe generated only 512 reasoning tokens and did not complete the code task. Its 85.62-second first-token time and the later 108.70 tok/s complete-code rate are separate requests; do not combine them into one cold completed-task result.

The tiers have different long-input lengths and some fixture differences: the 1M prompt also embeds three records. Generated answer lengths differ. These measurements show each tier's tested long-input behavior, not a same-input cross-tier A/B experiment. The approximately 12-minute first-token time is for an actual million-token cold input, not every request configured with a 1M window. Tests are synthetic filler plus one toy code task, not realistic long-document quality evaluations.

## Default recommendation

Keep 512K as the capacity-balanced default for a workflow that needs roughly 500K total context. It provides twice the single-request window of 256K with similar observed RAM/VRAM headroom, at about 14% lower short-code generation rate in these tests.

Prefer 256K for speed and short-request concurrency when the entire input, including history and tools, fits within roughly 229K after reserving 32K output. If ordinary usage consistently fits that budget, 256K is the stronger performance-default candidate.

Keep 1M as a manual experimental choice when the 512K input budget is insufficient. Its short-code generation rate is lower and host-memory headroom is tighter. Existing measurements do not prove 512K universally optimal; same-length long-input A/B tests and actual workload sizes would be needed for that stronger conclusion.

Detailed evidence: [256K](colab-256k-20261106.md), [512K](colab-512k-dynamic-20261106.md), [1M](colab-1m-dynamic-20261106.md).
