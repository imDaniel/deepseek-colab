# G4 512K dynamic expert-cache tuning, 2026-11-06

This experiment compares the existing 70,430/512/110 expert-cache/UBATCH/DYN profile with a larger dynamic reservation. It measures complete code generation as well as actual long-input requests. The context window is 524,288 input-plus-generated tokens per request, the output limit is 32,768, and there is one decoder slot. Four accepted clients share that slot through queueing.

The optional 1M profile was subsequently [tuned separately](colab-1m-dynamic-20261106.md). References below to an unchanged 66,000/512/50 1M profile describe the state during this 512K experiment; the 512K configuration and measurements remain unchanged.

## Conditions

ISTA IQ2_XXS + AtomicChat, image input enabled, DSpark enabled at draft depth 2, max reasoning effort, the same cached b11963 binary (SHA-256 `fe01a69712365f81331b71e9e1409f27182a92b808fcbe123df5f1c48de10aa8`). DMA remains disabled as required by the image path. We did not change weights, quantization, routed expert count, C++/CUDA code, draft depth or the decoder slot count. No new compiled release is required.

All profiles use fresh, separate Colab G4 sessions. Short code/prose probes use temperature 0, top_p 1, seed 42 and 512 output tokens, repeated three times. They finish by length during reasoning and are raw generation probes, not completed tasks. Complete LRU code tests allow 32,768 output tokens: temperature 0.6 runs three times in full qualification and once in each screen; temperature 0 runs once. Production sampling defaults are unchanged.

The fixed prompts are listed in [the earlier benchmark](colab-1m-20261106.md#prompts). The long prompt is `Benchmark fixed-480k. Read the repeated filler: ` followed by 480,000 repetitions of ` a`, then `\nEnd of filler. ` and the identical code prompt. The engine reports **480,093 input tokens**. Each full qualification first runs a cold complete-code request, then an identical prefix-cache-hit request, and finally three temperature-0/512-token raw probes. These are synthetic capacity/performance fixtures; they do not establish real-document reasoning quality.

Cold first-token timings start with an already loaded model and include prompt processing. Allocation, weight download and startup are additional. Generation rates include reasoning tokens. First token, first visible answer and total task time are separate measurements.

## Screens

The baseline also received full qualification; the three candidates below received shorter screens before selecting one for a fresh full qualification.

| Cache MiB / UBATCH / DYN | Raw short code tok/s | Raw short prose tok/s | 131,138-input cold first token | Complete code T0.6 tokens / tok/s | Complete code T0 tokens / tok/s | Minimum available RAM |
|---|---:|---:|---:|---:|---:|---:|
| 70430 / 512 / 110 | 102.53 | 84.44 | 56.12 s | 4,174 / 99.06 (mean of 3) | 4,224 / 101.91 | 3.738 GiB |
| 70430 / 512 / 166 | 106.53 | 89.37 | 55.80 s | 5,326 / 101.73 | 3,862 / 106.17 | 1.977 GiB |
| 73000 / 512 / 166 | 109.47 | 91.49 | 55.04 s | 5,326 / 105.03 | 3,862 / 109.72 | 4.187 GiB |
| 73000 / 512 / 200 | 111.08 | 92.95 | 54.82 s | 5,288 / 106.34 | 3,862 / 110.56 | 3.210 GiB |

Increasing DYN alone gives less host-memory headroom. Increasing the total GPU expert cache as well preserves the resident hot-expert set. At cache 73,000, DYN 200 adds about 1.2% complete-code throughput over DYN 166 in the screen, while reducing host RAM headroom by about 1 GiB. We selected 73,000/512/166 for the full qualification rather than adopting the largest reservation.

Main GPU compute buffers stay at 4,618.26 MiB and host compute buffers at 940.86 MiB. The baseline pins 7,814 experts (62.32 GiB), while the selected profile pins 7,839 (62.44 GiB). Actual dynamic slots per group change from `110 107 110 110 110 110 53 53` to `166 111 166 166 166 166 55 55`; group caps mean not every group receives the nominal DYN value. This is expert residency management, separate from the prompt-prefix cache used by warm requests.

## Complete-code qualification

| Measurement | Existing 70430 / 512 / 110 | Selected 73000 / 512 / 166 | Observed change |
|---|---:|---:|---:|
| Short input, complete code T0.6, mean of 3 | 99.06 tok/s | 104.54 tok/s | +5.53% |
| Short input, complete code T0, once | 101.91 tok/s | 108.71 tok/s | +6.67% |
| 480,093 input, cold complete code T0.6, once | 87.11 tok/s | 89.79 tok/s | +3.08% |
| Same 480,093 input, prefix-cache-hit complete code T0.6, once | 88.13 tok/s | 91.23 tok/s | +3.52% |
| 131,138-input cold first token | 56.12 s | 55.16 s | 1.71% less time |

All complete-code requests ended with `stop`, produced visible code, and passed basic capacity, update, recency and eviction checks. Across both full qualifications and three screens, all 18 completed examples passed those checks. This is one toy LRU task, not a broad coding-quality evaluation.

### Task time and output length

| Complete-code case | Existing tokens / first token / first visible / total | Selected tokens / first token / first visible / total |
|---|---|---|
| Cold 480,093-input request | 4,143 / 251.93 s / 292.49 s / 299.49 s | 3,635 / 250.53 s / 284.35 s / 291.01 s |
| Prefix-cache-hit repeat | 3,949 / 0.305 s / 37.73 s / 45.11 s | 3,768 / 0.300 s / 34.09 s / 41.60 s |

The short temperature-0.6 examples generated 4,174 tokens on the baseline versus 5,326 on the selected profile. Thus higher generation throughput did **not** make that short task finish sooner: mean total time increased from 42.43 to 51.21 seconds. Temperature 0 generated 4,224 versus 3,862 tokens, with total time decreasing from 41.48 to 35.56 seconds. Both long requests finished sooner on the candidate, but their output lengths also differ; that entire latency difference cannot be attributed to engine speed.

The separate cached-long temperature-0/512-token raw probe regressed from 88.87 to 87.14 tok/s (about 1.95%). Those probes contain only unfinished reasoning and follow the complete-code requests. The selected profile is an observed improvement for these complete-code cases, not an improvement for every decode workload.

Even with fixed seed and temperature, output hashes differ across some cache configurations, and the cold versus warm complete-code requests differ within each full qualification. The numerical source of those differences was not localized. Cross-family comparisons therefore measure observed workload throughput, not strictly identical-token kernel speed or proven bit-identical generation. Draft acceptance and output trajectory can also change throughput. Separate GPUs and small sample counts further limit attribution, especially for the approximately 3% long-input differences.

## Capacity, queueing and resources

Full qualification also checks two waves each of four short queued code/prose clients (1,024 output tokens per client), a 523,701-input three-record retrieval plus arithmetic request, 1024×1024 and 4096×4096 red-image requests, and another image request after long-context tests. Resource sampling runs every five seconds.

| Check | Existing profile | Selected profile |
|---|---:|---:|
| Four queued code clients, combined two-wave throughput | 99.56 tok/s | 112.44 tok/s |
| Four queued prose clients, combined two-wave throughput | 89.74 tok/s | 94.65 tok/s |
| 523,701-input retrieval, first token / total | 283.54 / 284.87 s | 276.75 / 278.07 s |
| Retrieval records and arithmetic | all correct | all correct |
| Image probes, before and after long input | passed | passed |
| Minimum free VRAM | 4,710 MiB | 2,142 MiB |
| Minimum available RAM | 3.738 GiB | 3.793 GiB |
| Resource samples / peak full-memory PSI avg60 | 219 / 0 | 216 / 0 |

Queue throughput is total generated tokens divided by total wave time, not the average of per-client speeds. These short requests all end by length in reasoning; this does not establish four completed answers or four simultaneous full-window decodes. The single decoder slot and four-in-flight admission limit remain unchanged.

The near-full prompt places the records at roughly 10%, 50% and 90% of repeated filler and asks for JSON containing `kiwi:482917`, `cobalt:691438`, `maple:275604` and `math:63`. Both profiles return all four values correctly, with 115/118 generated tokens. This verifies this synthetic case within the input-plus-output capacity, not realistic long-document quality. The image probes are separate requests, not full-text-plus-image or concurrent multi-image qualification. No OOM or sustained memory pressure was observed during these limited runs; long-term reliability is not established.

## Applied scope and evidence

The selected override applies only to **524,288 + ISTA + AtomicChat + images + DSpark**. It retains UBATCH 512, one decoder slot, the 32,768 output limit and four-request admission. Default context remains 512K. The 256K 73,000/1024/166 and experimental 1M 66,000/512/50 profiles are unchanged. Abiray, unpatched ISTA and disabled-image/draft combinations retain their prior profiles. This parameter-only change uses the existing compiled cache.

All 62 offline regression tests passed, including default boot values, scoped fallback, lifecycle, context/output limits and request admission. See [sanitized machine-readable results](colab-512k-dynamic-20261106.json) for the screens, both full qualifications, hashes, timings, buffers and basic code checks. Infrastructure, credentials and raw generated reasoning are excluded. The retained 1024-image result is the post-long-input probe; the earlier 1024 probe also passed in both run logs.

This is a modest observed complete-code improvement, approximately 5.5% on the short case and 3–3.5% on the 480K cases. Differing token trajectories, the raw-long regression, separate G4 sessions and single cold/warm long samples prevent a claim of universal or strictly causal speedup.
