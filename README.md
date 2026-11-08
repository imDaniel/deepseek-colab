<div align="center">

<h1>DeepSeek V4.1 Flash on one Colab GPU</h1>

<p><b>The uncensored 249 GB model as your own OpenAI-compatible endpoint, at over 100 tokens a second.</b><br>
One command, one Google Colab G4, no server of your own.<br>
Runs on a Google AI Pro subscription: 200 compute units every month, about 9 per hour of use.</p>

https://github.com/user-attachments/assets/6ab03b08-6992-4fc6-8d6a-9c809c8c6336

</div>

## Quick start

You need a Google AI Pro subscription (200 Colab compute units a month) or Colab Pro (100 a month), Python 3.9 or newer and an SSH client. It runs on Linux, macOS and Windows (PowerShell or cmd; type `python` if `python3` is not found).

```bash
git clone https://github.com/imDaniel/deepseek-colab.git deepseek-colab && cd deepseek-colab
python3 llm.py up
```

The first run offers to install Google's Colab CLI, walks you through the Google login and asks for a few defaults (image input, context size, reasoning effort, idle stop; Enter keeps each one). Then it starts a G4 GPU and shows each step until your endpoint is ready:

```text
Starting a Colab G4 GPU · about 4 minutes

   0:06  ✓  G4 GPU allocated
   0:12  ✓  image encoder downloaded (1 GB)
   0:19  ✓  draft model downloaded (8 GB)
   0:21  ✓  uncensored tensors downloaded (1.7 GB)
   0:55  ✓  model downloaded (138 GB)
   2:19  ✓  llama.cpp built
   2:34  ✓  model loaded onto the GPU
   2:41  ✓  RAM freed, cold experts pinned
   3:22  ✓  Engram table loaded (110 GB)
   3:23  ✓  warmup answer done

✓ Ready after 3:24

  Base URL   http://127.0.0.1:18080/v1
  API key    sk-ds41-3f9c...
  Models     deepseek-v4.1-flash-sec (effort max)
  Context    524,288 tokens per request · image input on · one decoder slot

  Try it     python3 llm.py ask "Hello!"
  opencode   python3 llm.py opencode [project folder]
  Stop       python3 llm.py down

  ● GPU running   idle 2:13 · stops in 7:47   ran 6:31 · about 0.97 compute units
```

The last line stays live: GPU on or off, the countdown to the automatic stop, the cost so far. Paste the base URL and key into any OpenAI-compatible app: Open WebUI, the OpenAI SDKs, curl. `python3 llm.py opencode` starts opencode already connected, in any shell. After 10 minutes without requests the GPU stops by itself, and the next request starts it again. `python3 llm.py down` stops everything right away.

## Choose the model

```bash
python3 llm.py up --model original  # project default: ISTA weights, with optional AtomicChat patch
python3 llm.py up --model abiray    # Abiray's pre-abliterated IQ2_XXS weights
```

The selection applies to the local endpoint and its automatic restarts. Stop the endpoint with `python3 llm.py down` before switching. `python3 llm.py settings` can save a default, and `LLM_MODEL=original` or `LLM_MODEL=abiray` can select a model for unattended `up` or `serve`; the command-line option takes precedence. The original model retains the existing uncensor setting. Abiray already includes its uncensor edit, so the separate AtomicChat patch is not applied. Both variants use the existing shared Engram and DSpark draft files.

Abiray was tested on a fresh G4 with 32K text-only context and DSpark: the model loaded, and non-streaming and streaming requests passed. With a 1M allocation and image input enabled, it correctly identified a test image and answered a synthetic 1,000,054-token prompt in 674 seconds. The project Uncensored mode was tested on a fresh G4 and successfully applied the AtomicChat patch. Sustained operation, long-context accuracy on realistic material and image quality with Abiray have not been established; the project's published performance numbers apply only to the original model.

The Magpie integration defaults to 1,048,576 context tokens and 262,144 output tokens. Switching to 524,288 context selects 131,072 output; 262,144 context selects 65,536 output. The selected output budget is both the default and the advertised maximum, includes reasoning, and shares the context window with input. Up to four requests are accepted including backend queueing; additional requests receive HTTP 429. Invalid or excessive budgets are rejected by the controller before GPU startup. See [the integration guide](integrations/magpie_colab/README.md) for model-catalog synchronization, compaction and logs. The CLI retains its legacy context options.

The startup scripts try to restore a pinned G4 binary from this repository's release. They verify its digest, build inputs, GPU, driver, CUDA and runtime environment before use, and fall back to compiling from source if the cache cannot be used.

## What you get

The table below records the original reference layouts. The selected default is 1M; see [the latest 1M measurements](docs/benchmarks/colab-1m-dynamic-20261106.md) for complete-code speed, long-input latency and limitations.

| | Images on, 262K reference | Text only, up to 131K (`--fast`) |
| --- | --- | --- |
| Decode speed | 105.6 tok/s | 118.7 tok/s |
| Context per request | 262,144 tokens | 131,072 tokens |
| Image input | yes | no |
| First token, 8K / 32K prompt | 9.9 / 27.8 s | 8.8 / 23.4 s |
| Follow-up turns (prompt cache) | about 1 s | about 1 s |

Image input and context size are separate settings. Text only up to 131K gets the faster reference layout. Contexts up to 262K use two decoder slots; 512K and experimental 1M use one, with up to four accepted clients including queueing. Tool calling and separate reasoning output are supported. The reference speeds above average six fixed answers at reasoning effort max and do not describe the newer scoped cache profiles for patched ISTA with images and DSpark; see the benchmarks below.

## What it costs

This project costs about **9 Colab compute units per hour** (8.9 measured) while the GPU runs.

| Plan | Compute units every month | GPU hours |
| --- | --- | --- |
| Google AI Pro | 200 | about 22 |
| Colab Pro | 100 | about 11 |

Each start spends about 4 minutes (about 0.6 units) on downloading and loading. When you're not using it, nothing runs and nothing is billed.

## How it works

<img src="docs/assets/architecture.svg" width="820" alt="Your computer runs llm.py, a local OpenAI endpoint, connected through an SSH tunnel to a Google Colab G4 VM. The GPU (96 GB) holds attention, dense layers, about 7,800 hot experts, the draft model, the KV cache and the image encoder. RAM (190 GB) holds the 110.6 GB Engram table and all other experts, which are copied to the GPU on a miss. Hugging Face supplies about 260 GB per start.">

- **The model.** ISTA-DASLab's 2-bit GSQ-RCO quantization of DeepSeek V4.1 Flash, which matched the original on their AIME, GPQA and LiveCodeBench runs. At every start, AtomicChat's 82 uncensored tensors are spliced in.
- **The memory trick.** Each token uses 6 of the 384 experts in a layer, and some experts are used far more than others. The hottest 7,800 live on the GPU. The rest wait in RAM and cross PCIe only on a miss.
- **The speed.** A small draft model guesses two tokens ahead and the big model checks both in one pass. On top of that, llama.cpp gets 22 patches: fused kernels, an expert cache that runs on the GPU, copies aligned to whole memory lines. Every one was checked to produce the same tokens as the build before it, within floating-point noise.

<img src="docs/assets/speed.svg" width="820" alt="Decode speed by milestone: first run on IQ2_XS 43.5 tokens per second, GPU expert cache 70, draft model 87, fused kernels 100, device cache lookup 110, aligned copies 118.7. With images and 262K context it runs at 105.6.">

## Read next

- [Saved configuration](configs/colab-optimized-20261106.json) and [three-tier comparison](docs/benchmarks/colab-context-tiers-20261106.md): current defaults, scoped boot parameters, complete-code timings and memory headroom.
- [Repository privacy](docs/privacy.md): credential exclusions, secret scanning and Git identity review.
- [SETUP.md](SETUP.md): every platform, settings, connecting apps, troubleshooting.
- [AGENT_START_HERE.md](AGENT_START_HERE.md): point your AI coding agent here and it can set things up and help you.
- [docs/how-it-works.md](docs/how-it-works.md): the memory layout, the boot sequence and every patch.
- [512K dynamic-cache results](docs/benchmarks/colab-512k-dynamic-20261106.md): the current default profile, complete-code generation, cold/warm 480K inputs and resource checks.
- [1M dynamic-cache results](docs/benchmarks/colab-1m-dynamic-20261106.md): the optional experimental profile, complete-code and actual million-input performance, memory and prefix reuse.
- [G4 context tuning results](docs/benchmarks/colab-1m-20261106.md): the earlier 512K cache change and measured 1M throughput, latency and limits.
- [256K tuning results](docs/benchmarks/colab-256k-20261106.md): two-slot expert-cache tuning, complete-code checks, queued clients and prefill tradeoffs.

## Credits

Built on DeepSeek V4.1 Flash by DeepSeek, the GSQ-RCO GGUF by ISTA-DASLab, uncensored tensors by AtomicChat, the DSpark draft GGUF by unsloth, llama.cpp by ggml-org and Google's Colab CLI. The model files come straight from their Hugging Face pages, under their licenses.

Uncensored means the model refuses less often. What you do with it is up to you.

## What's next

DeepSeek V4.1 Flash is the first model. More models are planned, each tuned for Colab the same way, so that this repo becomes a hub: pick a model at the start, run `up`, and get fast inference from the same optimized kernels and memory layout.
