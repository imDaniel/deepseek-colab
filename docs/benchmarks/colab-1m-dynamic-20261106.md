# G4 1M dynamic expert-cache tuning, 2026-11-06

This experiment revisits the optional 1,048,576-token profile using the dynamic expert-cache approach from the [512K experiment](colab-512k-dynamic-20261106.md). The previous 1M profile is 66,000 MiB expert cache / UBATCH 512 / DYN 50, without index query chunking. The default context remains 512K throughout.

## Fixed conditions and workloads

All configurations use Colab G4, ISTA IQ2_XXS + AtomicChat, images enabled, DSpark at draft depth 2, one decoder slot and the same cached b11963 binary (SHA-256 `fe01a69712365f81331b71e9e1409f27182a92b808fcbe123df5f1c48de10aa8`). UBATCH stays **512** in every measured profile. DMA stays disabled for the image path. No weights, quantization, routed-expert count, C++/CUDA code or sampling defaults were changed. The existing compiled binary already supports `LLAMA_DSV41_IDX_CHUNK_MIB`.

The total window is **1,048,576 input-plus-generated tokens**. Complete code requests reserve at most 32,768 output tokens. Four accepted clients may queue for the single decoder slot. These tests use same-runtime loopback HTTP, not public Magpie end-to-end timings. Cold first-token time begins with a loaded model and includes prompt processing; allocation, downloads and startup are additional.

The [exact benchmark script](colab-1m-dynamic-bench-20261106.py) runs inside an already loaded, matching G4 runtime. It does not allocate a VM. Its SHA-256 is `e5fde9c8da63aff9c4641c9d463d4e1a3b7ebaa18572c61f899a613a4273feca`, matching all main recorded runs.

- Short code and speculative-decoding explanation prompts are identical to [the earlier prompts](colab-1m-20261106.md#prompts). Temperature 0, top_p 1, seed 42, max effort and a 512-token budget, three repeats each, measure raw generation. All end by length in reasoning, not as completed tasks.
- Complete short LRU code uses max effort, temperature 0.6, top_p 1, seed 42 and a 32,768-token budget. It runs three times in each full qualification and once per screen. A temperature-0 complete-code diagnostic runs once per profile.
- The long prompt contains exactly 1,000,000 repetitions of ` a`, three records at approximately 10%, 50% and 90%, then the identical LRU code question. The engine reports **1,000,125 input tokens**. A cold complete-code request is followed by an identical prefix-cache-hit request, both at temperature 0.6/max effort/32,768 output budget.
- An ordinary Chat follow-up keeps the original user message, the visible code answer without prior reasoning, and a new filler-plus-retrieval question. The model tokenizer determines filler length while reserving a 4,096-token output budget and a small header margin. It checks over 1,043,000 input tokens, reuse of over 999,000 cached tokens, all three record codes and `9 * 7 = 63`. Reasoning effort remains max so the leading template prefix stays unchanged. This is a cached append, not a cold near-full request.
- Three temperature-0/512-token long raw probes follow, then two waves each of four short code/prose clients at 1,024 output tokens per client. Full qualification also checks 1024×1024 and 4096×4096 red images before long tests and another 1024 image afterward. Resources are sampled every five seconds.

The exact long prefix is `Benchmark fixed-million-records-v2. Read the repeated filler and records: `, followed by 100,000 filler tokens, `\nRecord kiwi code: 482917.\n`, 400,000 filler tokens, `\nRecord cobalt code: 691438.\n`, 400,000 filler tokens, `\nRecord maple code: 275604.\n`, 100,000 filler tokens, then `\nEnd of filler. ` and the code prompt. The follow-up asks for a JSON object containing the earlier three record codes and `math:63`. These synthetic fixtures do not establish realistic long-document quality.

### Baseline provenance and cache controls

An incomplete initial million-token prefill was discarded when reviewing the follow-up check revealed that changing reasoning effort changes a leading template string and invalidates prefix reuse. The benchmark was corrected before any included long result, and all main workloads were rerun with the script above. The same baseline GPU and loaded weights were retained; the new long prefix prevents reuse of the discarded partial prompt. Its included cold long request caches only 27 boilerplate tokens. Discarded runtime still counts toward actual test usage.

The retained runtime did cache the fixed 32K/128K math probes when those were repeated. Their repeat timings are therefore **warm**, not cold. The cold-math comparison uses the two completed pre-correction math probes from that same GPU/profile and identical request text: 32,834 / 131,138 input tokens, only 27 cached each, first token at 16.2978 / 57.4936 seconds. Those references and their earlier script hash are explicitly separate in the JSON. All main complete-code and long-input comparisons use the final script hash above.

Other profiles use fresh G4 sessions. Hardware, clock and cache-history variation remain limitations; clocks were not locked, and the retained baseline had additional discarded warm-up. Resource differences and small speed changes are observations, not perfectly isolated causal effects.

## Screens

| Cache MiB / UBATCH / DYN / index chunk MiB | Complete short code T0.6 | Complete short code T0 | 131,138-input cold first token | Minimum available host RAM | Minimum free VRAM |
|---|---:|---:|---:|---:|---:|
| 66000 / 512 / 50 / 0, full baseline | 88.77 tok/s, mean of 3 | 91.53 tok/s | 57.49 s, cold reference above | 1.046 GiB | 2,908 MiB |
| 68864 / 512 / 50 / 128 | 91.03 tok/s | 93.84 tok/s | 57.08 s | 4.667 GiB | 2,286 MiB |
| 68864 / 512 / 110 / 128 | 97.51 tok/s | 100.16 tok/s | 55.92 s | 1.721 GiB | 2,280 MiB |
| 68864 / 512 / 80 / 128 | 94.94 tok/s | 97.74 tok/s | 56.37 s | 3.024 GiB | 2,282 MiB |

All temperature-0.6 short-code outputs contain 4,174 tokens with matching output hash `b2b42912cc89d55e`. All temperature-0 short-code outputs contain 4,224 tokens with matching hash `5aef3d2edad01718`. They stop normally with visible code and pass basic LRU checks. The candidate screens are shorter than full qualification and do not include million-input requests; their resource minima must not be presented as full-long-context qualification.

At the same cache size and batch, DYN 110 is about 2.7% faster on this complete-code screen than DYN 80, but leaves about 1.3 GiB less host RAM. We selected DYN 80 for fresh full qualification, keeping UBATCH 512. Selecting the largest dynamic reservation solely from short-prompt tok/s would ignore the limited host-memory headroom.

## Memory allocation

Index chunking reduces the main GPU compute buffer from **9,226.26 to 7,178.26 MiB**, freeing 2 GiB without reducing UBATCH. The host compute buffer remains 1,836.86 MiB. Increasing total expert cache then moves more weights from host RAM to the GPU, while DYN divides the cache between fixed hot experts and adaptive misses.

| Configuration | Pinned hot experts / size | Actual dynamic slots per group |
|---|---|---|
| 66000 / DYN 50 | 7,638 / 60.98 GiB | `50 50 50 50 50 50 50 50` |
| 68864 / DYN 50 | 7,984 / 63.75 GiB | `50 50 50 50 50 50 50 50` |
| 68864 / DYN 110 | 7,626 / 60.81 GiB | `110 104 110 110 110 110 52 52` |
| 68864 / DYN 80 | 7,800 / 62.24 GiB | `80 80 80 80 80 80 52 52` |

The selected balance retains more hot experts than the baseline while increasing dynamic capacity. The expert cache is separate from the prompt-prefix cache used by warm requests. This experiment did not repeat the earlier UBATCH-256/128 strategies, whose million-input cold-prefill penalties were substantial.

## Full qualification

| Measurement | Existing 66000 / 512 / 50 / index 0 | Selected 68864 / 512 / 80 / index 128 | Observed change |
|---|---:|---:|---:|
| Complete short code T0.6, mean of 3 | 88.77 tok/s | 95.45 tok/s | +7.52% |
| Complete short code T0, once | 91.53 tok/s | 98.01 tok/s | +7.08% |
| Actual 1,000,125-input complete code, cold, once | 69.38 tok/s | 73.85 tok/s | +6.44% |
| Identical million-input request, prefix cache hit, once | 68.94 tok/s | 74.37 tok/s | +7.88% |
| Million-input cold first token | 701.59 s | 719.21 s | 2.51% more time |
| 131,138-input cold first token | 57.49 s, reference above | 56.33 s | 2.02% less time |

All 18 complete-code examples across the baseline, three screens and selected full qualification stop normally, produce visible code and pass basic capacity, update, recency and eviction checks after restrictive AST/manual review. These are toy LRU checks, not a broad coding-quality evaluation.

### Generation rate versus task time

| Complete million-input code | Existing tokens / first token / first visible / total | Selected tokens / first token / first visible / total |
|---|---|---|
| Cold | 4,482 / 701.59 s / 757.89 s / 766.19 s | 4,572 / 719.21 s / 771.58 s / 781.12 s |
| Prefix cache hit | 3,338 / 0.607 s / 41.30 s / 49.02 s | 3,042 / 0.595 s / 33.22 s / 41.50 s |

Generation rates include reasoning. The short complete-code outputs match, supporting a useful same-output comparison for that fixed case. The long outputs differ across profiles and between cold/warm runs within each profile. The source of those trajectory differences was not localized; this is not proof of bit-identical generation or broad quality equivalence.

The cold million-input task takes **about 15 seconds longer overall**, despite the higher generation rate, because prompt processing dominates and first-token time increases by 17.6 seconds. Its generated length also increases slightly. The warm task finishes about 7.5 seconds sooner, but part of that difference comes from generating fewer tokens. Its whole task-time improvement must not be attributed solely to engine speed. Each complete long condition is a single sample in a separate G4 session; the percentages are observed results, not universal or statistically established gains.

## Capacity, queueing and resources

| Check | Existing profile | Selected profile |
|---|---:|---:|
| Cached million-input raw T0/512-token generation, mean of 3 | 72.57 tok/s | 77.00 tok/s |
| Four queued short code clients, combined two-wave throughput | 89.83 tok/s | 96.43 tok/s |
| Four queued short prose clients, combined two-wave throughput | 76.62 tok/s | 85.62 tok/s |
| Near-full appended input / reused prefix | 1,044,260 / 1,000,121 tokens | 1,044,260 / 1,000,121 tokens |
| Appended retrieval: first token / total | 50.77 / 52.96 s | 53.33 / 55.24 s |
| Three records and arithmetic | all correct | all correct |
| Image checks before and after long input | passed | passed |
| Minimum available host RAM | 1.046 GiB | 2.525 GiB |
| Minimum free VRAM | 2,908 MiB | 2,064 MiB |
| Resource samples / peak full-memory PSI avg60 | 263 / 0 | 272 / 0 |

The raw long probes end by length with no visible answer. Their approximately 6.1% throughput increase is not a completed-task measurement, and their hashes differ across profiles. Queue throughput is total generated tokens divided by total wave time. All four-client probes end by length in reasoning, so they do not demonstrate four completed answers or four simultaneous full-window decodes. The single decoder slot and four-in-flight admission policy are unchanged.

The appended near-full probes reserve 4,096 output tokens and actually generate 144/140 tokens. Both return all records and arithmetic correctly. They demonstrate prefix reuse and this synthetic near-full capacity case; their first-token times are for roughly 44K additional input after a cached million-token prefix. They must not be labeled cold 1,044K first-token times. Separate image requests do not establish combined million-text-plus-multiple-image performance.

All requests in qualification complete their HTTP/SSE streams successfully, and no OOM failure or sustained memory pressure was observed. Resource minima include different runtime/cache histories, and this limited-duration test does not establish long-term reliability. The saved 1024-image field in JSON is the final post-long-input request; the initial 1024 probe also passed in both run logs.

## Applied scope and decision

The selected profile is applied only to **1,048,576 + ISTA + AtomicChat + images + DSpark**: cache **68,864 MiB**, UBATCH **512**, DYN **80**, index-query chunk **128 MiB**. The default remains **524,288**. Its 73,000/512/166 profile and the 256K 73,000/1024/166 profile are unchanged. Other models, feature combinations and contexts retain their previous layouts. The optional 1M tier remains experimental, with one decoder slot, four accepted requests including queueing and a 32,768 output limit.

This is a balanced generation-throughput and host-memory improvement for the tested workload, with a small cold-prefill penalty. It does not improve every latency measure: the complete cold million-input task and the appended retrieval both take slightly longer. The source forwards the index parameter only for the scoped profile, and **63 offline regression tests pass**, including boot-variable propagation, fallback isolation, request limits and lifecycle checks. No new compiled release is needed.

See [sanitized results](colab-1m-dynamic-20261106.json) for all screens, both full qualifications, output hashes, timing counters, resource summaries and basic code checks. The [benchmark script](colab-1m-dynamic-bench-20261106.py) records the exact protocol. Private infrastructure, credentials and generated reasoning are excluded. The observed 6–8% complete-code generation gains are finite-sample results, with the trajectory, session-history and hardware limitations above.
