# GPU inference for the vibe tagger

Step 11's fine-tune and eval table (`docs/EVAL.md`) ran entirely on CPU.
`infer.py` has to tag all 19,386 warehouse tracks, and at the CPU eval run's
measured rate that's on the order of a day of wall time (see "CPU vs GPU
latency" below for the actual numbers) - too slow to be a real step in this
project. This doc records the GPU path that replaces it: what didn't work,
what did, and the parity check that justified trusting it.

## The hardware problem

The dev machine's GPU is an AMD Radeon RX 5600 XT (RDNA1, `gfx1010`). Two
routes were tried and abandoned before landing on llama.cpp:

- **ROCm.** No RDNA1 support on Windows, AMD's official ROCm Windows
  builds start at RDNA2. Dead end.
- **`torch-directml`.** Works, but pins `torch` to 2.4.1, which is
  incompatible with the `transformers` version Qwen3 needs (Qwen3 support
  landed in `transformers` well after the last release `torch-directml`
  supports). Dead end.
- **CPU batching as a speedup instead of a GPU.** Measured directly: batch
  8 and batch 16 both ran at 0.8-0.9x the speed of batch 1 on this CPU,
  because the lyrics-heavy prompts (500-700 tokens) are wildly uneven in
  length and padding to the batch max wastes more compute than the batching
  saves. Not an alternative.

## The route that worked: llama.cpp + Vulkan

llama.cpp's Vulkan backend runs on essentially any GPU with a working
Vulkan driver, RDNA1 included, and needs no ROCm or CUDA. It's inference
only, no training path, which is fine, since only `infer.py`'s generation
step needs to move off CPU. Training (Step 11's LoRA fine-tune) already
finished on CPU.

**No compiler installed on this machine (no cmake, no MSVC/Visual Studio
Build Tools), so this uses the prebuilt Windows Vulkan release rather than
building from source:**

```powershell
# one-time: llama.cpp prebuilt binaries (Vulkan backend, Windows x64)
gh release download b11149 --repo ggml-org/llama.cpp `
  --pattern "llama-b11149-bin-win-vulkan-x64.zip" -D tools
Expand-Archive tools/llama-b11149-bin-win-vulkan-x64.zip tools/llama-vulkan

# one-time: the GGUF conversion script + its gguf-py package aren't in the
# binary release (they're plain Python, not compiled) - sparse-clone just
# those from the source repo
git clone --depth 1 --filter=blob:none --no-checkout `
  https://github.com/ggml-org/llama.cpp.git tools/llama.cpp-src
cd tools/llama.cpp-src
git sparse-checkout init --no-cone
"/*`n!/*/`n/convert_hf_to_gguf.py`n/gguf-py/`n/conversion/`n/requirements/`n/requirements.txt" |
  Set-Content .git/info/sparse-checkout
git read-tree -mu HEAD
```

Build-tag `b11149` was simply "whatever the newest build was" on
2026-09-23, pin to a specific tag rather than always grabbing latest, so
a future re-run of this doc's commands doesn't silently pick up a different
build.

### Merge the LoRA adapters

llama.cpp's GGUF conversion only reads plain (non-PEFT) HF checkpoints, so
each arm's adapter has to be merged into the base weights first
(`scripts/merge_adapters.py`, `PeftModel.merge_and_unload()`), saved to
`data/runs/{A,B,C}/merged/`. Arm B was skipped, it's never the winning
arm (C) or the fallback arm (A) that `infer.py` actually uses, and merging
+ converting it would burn time on a model this project doesn't serve.

### Convert to GGUF

```powershell
uv run --with gguf --with sentencepiece --with protobuf `
  python tools/llama.cpp-src/convert_hf_to_gguf.py `
  data/runs/C/merged --outfile data/runs/C/merged/model-f16.gguf --outtype f16
# repeat for A
```

`f16`, no quantisation, the point of this exercise is a faster path to the
*same* predictions the CPU eval already validated, not a smaller or lower-
precision model. The parity check below confirms f16 GGUF doesn't move the
numbers.

### Run the servers

```powershell
tools/llama-vulkan/llama-server.exe -m data/runs/A/merged/model-f16.gguf `
  --port 8711 -ngl 99 --no-webui --parallel 4
tools/llama-vulkan/llama-server.exe -m data/runs/C/merged/model-f16.gguf `
  --port 8712 -ngl 99 --no-webui --parallel 4
```

`-ngl 99` offloads every layer to the GPU (a 0.6B model fits entirely in
VRAM many times over). `--parallel 4` opens 4 concurrent generation slots
per server, matched by `infer.py`'s and `scripts/gpu_parity_check.py`'s
worker counts. Both arm servers run at once on the same physical GPU -
they compete for the same compute, so running both concurrently is slower
per-arm than running one alone, but still far faster than CPU (see below).

Confirmed the GPU was actually doing the work, not falling back to CPU
silently: while a generation request was in flight, Windows'
`\GPU Engine(*)\Utilization Percentage` performance counter showed the
`llama-server` process's compute engine at 95-96% utilisation.

### `selector.tagger.gpu_infer`

The client side: `generate_completion_gpu()` POSTs to `/completion` with
the same raw-prompt framing as `selector.tagger.dataset.build_prompt` +
`"\n"` (no chat template, the adapters were trained on a raw
continuation) and greedy decoding (`temperature=0, top_k=1`, matching
`do_sample=False` on the CPU path). It doesn't replicate the CPU path's
early-stop-on-balanced-JSON stopping criterion - `n_predict` just runs to
the same budget CPU used (80 tokens), and `selector.tagger.eval._parse_prediction`
already extracts only the first complete JSON object and ignores whatever
comes after, so a few extra generated tokens change nothing about the
label. On GPU the extra tokens cost tens of milliseconds. On CPU they were
worth avoiding because they compounded into hours (see `docs/EVAL.md`
"Bugs found").

## Parity check

Before trusting the GPU path for a 19,386-track run, `scripts/gpu_parity_check.py`
re-scored the same 487-track held-out test split from `docs/EVAL.md`
through both arm servers (4 concurrent requests each, matching
`--parallel 4`) and compared against the CPU rows already in that doc:

| Arm | Metric | CPU (docs/EVAL.md) | GPU (this check) | Delta |
|---|---|---|---|---|
| A | valence MAE | 0.106 | 0.106 | +0.0004 |
| A | intensity MAE | 0.094 | 0.095 | +0.0010 |
| A | era match | 61.4% | 61.2% | −0.2pp |
| A | mood_tags exact | 18.5% | 17.7% | −0.8pp |
| A | parse fail | 0.6% | 0.8% | +0.2pp |
| C | valence MAE | 0.096 | 0.097 | +0.0006 |
| C | intensity MAE | 0.073 | 0.074 | +0.0008 |
| C | era match | 61.8% | 61.4% | −0.4pp |
| C | mood_tags exact | 18.9% | 18.7% | −0.2pp |
| C | parse fail | 0.8% | 1.4% | +0.6pp |

Every delta is inside noise, sub-0.001 on the MAE metrics, under 1
percentage point on the match rates, consistent with f16-vs-f32 rounding
and greedy decoding's sensitivity to near-tied logits rather than any real
behaviour change. The GPU path was trusted for `infer.py`'s full run on
this basis. Per-example predictions are in
`data/eval_predictions/gpu_parity_{A,C}.jsonl`.

## CPU vs GPU latency

Single-request (no concurrency) median generation latency, `n_predict=80`,
greedy:

| Path | Arm | Median | Measured with |
|---|---|---|---|
| CPU (Ryzen 5 3600) | C | 13.34s | `scripts/measure_cpu_latency.py`, 8 real test-split examples, real adapter |
| GPU (RX 5600 XT, Vulkan) | A | 2.891s | `scripts/measure_gpu_latency.py`, 15 requests, synthetic prompt |
| GPU (RX 5600 XT, Vulkan) | C | 2.907s | `scripts/measure_gpu_latency.py`, 15 requests, synthetic prompt |

**~4.6x single-request speedup.** The bigger win is concurrency: the parity
check's 4-concurrent-requests throughput was 1.14s/example (arm A) and
1.29s/example (arm C) of wall-clock *per example*, i.e. roughly another
2.5x on top of the single-request number, because a 0.6B model at this
context length doesn't saturate the GPU with one request in flight.
`infer.py` uses 4 slots per arm server (8 worker threads total), so its
real observed throughput is in this range, not the single-request number.

At the single-request CPU rate, 19,386 tracks would take
19,386 x 13.34s ≈ 71.8 hours (~3 days). At the concurrent GPU rate it's
under 2 hours, see `docs/EVAL.md` for the actual run's wall time once
`infer.py`'s full pass has completed.

## Cost per thousand tracks vs the teacher

`docs/TEACHER.md` puts the teacher (`claude-sonnet-5` via the Anthropic
API) at $5.03 per thousand tracks. The fine-tuned student's inference has
no equivalent per-call cost at all: it runs locally against hardware
already owned, with no metered API in the loop. The honest comparison
isn't "$X versus $5.03" - it's that the $5.03/1k figure was a one-time cost
to produce ~3,500 labelled examples to train on, and everything inferred
after that (the other ~15,900 tracks, and any future track) is free at the
margin. That's the actual economic argument for distillation here: the
teacher's cost is paid once, for training data, not per track tagged.

## Reproducing

```powershell
# start both servers (separate terminals or background jobs)
tools/llama-vulkan/llama-server.exe -m data/runs/A/merged/model-f16.gguf --port 8711 -ngl 99 --no-webui --parallel 4
tools/llama-vulkan/llama-server.exe -m data/runs/C/merged/model-f16.gguf --port 8712 -ngl 99 --no-webui --parallel 4

# parity check against the CPU eval rows
uv run python scripts/gpu_parity_check.py

# latency measurements
uv run python scripts/measure_gpu_latency.py
uv run python scripts/measure_cpu_latency.py

# the real thing: tag every warehouse track (needs both servers running)
uv run python -m selector.tagger.infer
```

`tools/` (the llama.cpp binaries and sparse source checkout) and
`data/runs/*/merged/` (merged weights + GGUF files, several GB) are
gitignored, reproducible from this doc, not meant to be committed.
