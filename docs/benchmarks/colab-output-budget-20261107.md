# Output-budget comparison — 2026-11-07

## Deployment policy

The 512K context now defaults to 131,072 generated tokens. The 1M context has a
configured target default of 262,144, pending release; its current effective
default and public maximum remain 131,072. The 256K context defaults to 65,536.
All budgets include reasoning, visible text and tool calls. Input and output
share the context window. No public flag enables 256K output.

Both Magpie catalogs and the local OMP model override advertise 131,072 output
tokens. The OMP 512K model has a scoped compaction threshold of 360,000 tokens.
This preserves room for output and subsequent tool results. Running client
sessions may need to refresh or reselect the model to adopt updated metadata.

## Method

One isolated G4 session ran the same [SQLite job-queue task](colab-output-budget-20261107-task.md)
at four output caps: 32,768 / 65,536 / 131,072 / 262,144. The last cap was used
only on the isolated engine loopback endpoint. The public controller and both
Magpie relays were not used for performance timing and retained the 128K cap.

The engine and weights match the production 512K profile: ISTA with the
AtomicChat patch, image support and DSpark enabled, one decoding slot,
UBATCH=512, MoE cache 73,000 MiB, DYN=166, DMA=0, draft maximum 2. The GPU was an
NVIDIA RTX PRO 6000 Blackwell Server Edition reporting 97,887 MiB. The cached
engine commit is `39febd2`; the engine API reports `b1-39febd2` for this
cached build. Sampling was effort `max`, temperature 0.6,
top-p 1 and seed 42, one request at a time, one trial per cap.

The slot-erase endpoint returned HTTP 501 on this engine/model combination.
Before each timed request, an identical
one-token request primed the prompt prefix. Thus reported first-token latency
is a **prefix-hit latency**, not a cold-prefill measurement. First token means
either reasoning or visible content; first visible content is reported
separately. Decode rate is `(completion_tokens - 1) / (last_token_time -
first_token_time)`, including reasoning. Memory was sampled every five seconds.

The [harness](colab-output-budget-20261107-run.py) stores counts and hashes of
reasoning, without saving the raw reasoning text. An independent
[16-case checker](colab-output-budget-20261107-check.py) was prepared for visible
implementations, with grading restricted to an unprivileged, network-isolated
Linux sandbox, a read-only host, 1 GiB memory, one CPU and a 120-second limit.
No implementation was returned in this run, so the checker was not executed.
A missing visible implementation is ungradable, rather than 16 executed test
failures.

## Results

| Output cap | Generated tokens | Finish | Elapsed | First token, prefix hit | First visible | Generated tok/s | Visible code |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| 32K | 32,768 | `length` | 5:22.497 | 0.030 s | None | 101.61 | None; ungradable |
| 64K | 65,536 | `length` | 10:24.427 | 0.025 s | None | 104.96 | None; ungradable |
| 128K | 131,072 | `length` | 21:18.000 | 0.025 s | None | 102.56 | None; ungradable |
| 256K | 262,144 | `length` | 44:13.017 | 0.025 s | None | 98.81 | None; ungradable |

All four requests reached their exact cap and ended with `length`. All
generated output was classified by the engine as reasoning; no visible answer
or executable implementation was returned. The transport completed normally.
There was therefore no candidate to run through the 16-case checker, and no
code-correctness score or code-quality ranking can be assigned. These rates
measure reasoning-token generation, **not complete-code generation speed**.

Increasing the cap in this task extended the wait without yielding a final
answer. The short prefix-hit first-token times do not mean a usable answer
arrived quickly. This does not identify the cause of the prolonged reasoning;
it establishes that a larger total cap alone did not resolve it in these four
trials. Keep 256K output closed. A separate thinking-budget experiment would
be a more useful next control than simply raising the total cap again.

| Output cap | Minimum GPU free, MiB | Minimum host available, GiB |
| --- | ---: | ---: |
| 32K | 2,504 | 5.140 |
| 64K | 2,504 | 5.009 |
| 128K | 2,504 | 4.876 |
| 256K | 2,504 | 4.593 |

No out-of-memory or transport error was reported in these four trials. Sampled
free memory is not a guarantee of peak headroom or sustained reliability.
See the [sanitized measurements](colab-output-budget-20261107.json) for hashes,
exact counts, sampling and runtime settings.

## Configuration verification and cleanup

- All 73 Linux offline tests passed, including rejection before admission or
  GPU startup. A Responses `n_predict` extension or multiple simultaneous
  limit fields cannot override the public cap.
- The controller's live Chat Completions and Responses checks returned HTTP
  400 for 256K budgets, including Responses compatibility extensions, while
  production remained off. Both Magpie output overrides are 131,072; the
  HomeLab public model catalog advertises the same ceiling. OMP also reports a 524,288 context and 131,072
  output, with low/high/max reasoning choices.
- The 1M target remains 262,144 with a pending flag; its current effective
  default and cap are 131,072. This benchmark did not enable it publicly.
- Raw artifacts were collected before shutdown. The owned Colab session was
  verified absent from the CLI session list; its temporary controller and
  state were removed. Production credentials and settings were retained.

## Limits

This is one synthetic, short-input correctness task, not a long-input,
concurrency or agent-tool benchmark. One seed and one trial cannot establish
general quality differences. A larger output cap permits more generation; it
does not guarantee longer useful reasoning, completion or correct code.
The fixed seed does not establish identical speculative-decoding trajectories.
Thinking output and visible output consume the same cap. The engine's effort
score of 100 for `max` is an effort instruction, not a separate token budget
or a guarantee that tokens are reserved for the final answer.

The pinned engine supports a separate `reasoning_budget_tokens` request field
(alias `thinking_budget_tokens`), with a default of -1, and can force the
thinking end sequence when a finite budget is exhausted. See its
[request translation](https://github.com/ggml-org/llama.cpp/blob/39febd2adb947b70da35e95f9351b3bf58ad170a/tools/server/server-common.cpp#L1517),
[default](https://github.com/ggml-org/llama.cpp/blob/39febd2adb947b70da35e95f9351b3bf58ad170a/common/common.h#L294)
and [sampler setup](https://github.com/ggml-org/llama.cpp/blob/39febd2adb947b70da35e95f9351b3bf58ad170a/common/sampling.cpp#L314).
This experiment did not set such a budget or force a thinking end. Compatibility,
answer quality and propagation through both relays require a separate test
before adopting that control; source support alone does not establish those
results. Production reasoning behavior was not changed by this comparison.
