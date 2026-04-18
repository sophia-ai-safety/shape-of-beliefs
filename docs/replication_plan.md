# Shape of Beliefs — Replication Master Plan

**Repo:** <<https://github.com/sophia-ai-safety/shape-of-beliefs>
**Paper:** Sarfati et al., *The Shape of Beliefs* (arXiv:2602.02315)
**Model:** Llama-3.2-1B
**Last updated:** living document — update as the project evolves

---

## The big picture in one paragraph

This is a mech-interp replication. The workflow: take Llama-3.2-1B (frozen, pretrained), feed it sequences of numbers sampled from Gaussian distributions, capture its internal activations at every layer, train linear probes on those activations, and make figures showing the geometry of how beliefs are encoded. No model training, no fine-tuning, just forward passes and analysis. The heavy part is I/O — ~50GB of cached activations — not compute. Colab Pro+ with any GPU is plenty.

---

## The four environments

The project moves between four places. Knowing which one you're in is most of what "understanding the workflow" means.

| Environment | Role | Persistence |
|---|---|---|
| **GitHub (your fork)** | Source of truth, lives forever | Permanent |
| **Cursor on your laptop** | Read, edit, commit, push | Permanent (local) |
| **Colab VM (rented GPU)** | Runs pipeline, dies after session | Ephemeral |
| **Google Drive** | Holds 50GB outputs | Permanent |

Flow: GitHub ↔ Laptop (clone/push), Laptop → Colab VM (clone on boot), Colab VM → Drive (writes outputs).

**Mental model:** Cursor is your workshop, GitHub is your filing cabinet, Colab is your rented workstation, Drive is your external hard drive. The repo is the thread that ties them all together.

---

## The five-stage pipeline from the paper

| Stage | What it does | Compute profile |
|---|---|---|
| 1. Generate sequences | Sample numbers from N(μ, σ) distributions | CPU, fast |
| 2. Extract activations | Forward passes through Llama-3.2-1B, cache hidden states at every layer | GPU, slow, ~50GB output |
| 3. Train LF probes | Fit linear field probes on cached activations | CPU, cheap |
| 4. Steering runs | Interventions (linear vs manifold-aware) via new forward passes | GPU, medium |
| 5. Make figures | Matplotlib plots from cached data | CPU, fast |

Stage 2 is why you need Colab at all. Everything else can run on a laptop. The 50GB is because you're caching 2048-dimensional float activations from every layer across thousands of tokens × many (μ, σ) combinations.

---

## Where each stage runs

**Rule of thumb:** anything that touches Llama runs on Colab, anything else can go either place.

| Stage | Primary home | Notes |
|---|---|---|
| 1. Generate sequences | Laptop or Colab | Pure NumPy, runs anywhere |
| 2. Extract activations | Colab VM (GPU) | Writes ~50GB to Drive |
| 3. Train LF probes | Laptop | Reads from Drive, CPU fine |
| 4. Steering runs | Colab VM (GPU) | New forward passes needed |
| 5. Figures | Laptop | Notebooks, cheap |

Once stage 2 is done and activations are cached on Drive, you can do almost everything else on your laptop without Colab.

---

## Roadmap — seven steps from fork to reproducible artifact

### Step 1 — Fork and set up locally ✓ DONE

- Fork at `sandguine/shape-of-beliefs`
- Cloned to laptop
- `uv venv` and `uv pip install -e .` ran without errors
- Can open files in Cursor

### Step 2 — Understand the code ← CURRENT STEP

**Goal:** produce a technical report of what the pipeline actually does.
**How:** run the Cursor agent prompt (saved in this doc, see Appendix A) against the repo.
**Deliverable:** a markdown report pasted back into the Claude chat covering, for each script:

- What it does in 2-3 sentences
- CLI arguments (copy from argparse)
- Every hardcoded file path (read or write)
- How it selects GPU/CPU
- Which HF model string it loads
- Whether HF auth is needed
- Dependencies from pyproject.toml

**Why:** without this, every "fix" on Colab is a guess. (Reminder: I guessed `--output-dir` existed as a flag; it doesn't. This step prevents that class of error.)

### Step 3 — Make scripts Colab-friendly

**Goal:** modify scripts so they work in a Colab environment.
**Likely changes:**

- Add `--output-dir` (or similar) flags so outputs can go to Drive
- Handle HF auth (Llama is gated — need accepted license + `HF_TOKEN`)
- Make sure code doesn't assume CUDA blindly
- Any other gotchas surfaced by step 2's report

**Tool:** Claude Code or Cursor's agent on your laptop. PR-sized changes.

### Step 4 — Write the Colab notebook

**Goal:** create `notebooks/run_on_colab.ipynb` inside the repo.
**Contents:**

- Cell 1: mount Google Drive
- Cell 2: `!git clone` your fork, `cd` in, install deps with `uv`
- Cell 3: set HF token
- Cell 4: run stage 1 (sequences)
- Cell 5: run stage 2 (activations) — the big one
- Cell 6: run stage 3 (probes) — optional, can do locally
- Cell 7: quick sanity check — plot one figure

**Final touches:** add Open-in-Colab badge to the README, commit, push.

### Step 5 — Run the pipeline end-to-end once

**Goal:** verify it works.
**How:**

- Start with a tiny config (e.g. 10 sequences, 2 μ values) to catch bugs cheaply
- Scale up to paper settings
- Confirm 50GB lands in Drive and files look reasonable

### Step 6 — Reproduce key figures

**Goal:** regenerate Figures 1-6 from the paper and visually compare.
**How:** use the notebooks in `figures/` against your cached data. Debug discrepancies.

### Step 7 — Make it reproducible for others

**Goal:** someone else can click the Open-in-Colab badge and produce the figures without emailing you.
**Requires:**

- Clear README with exact steps
- Cached data accessible somewhere public (Drive share link, GCS bucket, or Hugging Face Hub)
- Sensible defaults baked into the notebook

---

## Tool-to-task cheat sheet

| Situation | Use this |
|---|---|
| Reading code to understand it | Cursor on laptop |
| Editing multiple files | Cursor's agent or Claude Code on laptop |
| Asking "what does this function do" | Cursor's inline chat on laptop |
| Running a forward pass through Llama | Colab VM |
| Writing the notebook itself | Cursor on laptop, test in Colab |
| Linear probe training | Either; laptop is fine |
| Making figures from cached data | Laptop is faster |

---

## Key facts to remember

- **Llama-3.2-1B is small.** Hidden size 2048, ~16 layers. FP16 weights ~2GB. Fits on any Colab GPU including free T4.
- **Llama is gated.** You need to accept Meta's license on Hugging Face and have `HF_TOKEN` set, or the download will 401.
- **Colab disks are ephemeral.** Never save anything important only to `/content/`. Always write to `/content/drive/MyDrive/...` (after mounting).
- **The 50GB is mostly storage, not compute.** The bottleneck is I/O (writing activations to disk), not GPU flops.
- **No training is involved.** No backprop, no fine-tuning. Just frozen forward passes and linear probe fitting.

---

## Appendix A — Cursor agent prompt for Step 2

Paste this into Cursor (Cmd+I / Ctrl+I) with the repo open:

> I need you to produce a technical report about this repository so I can share it with someone helping me run it on Google Colab. Don't modify any code. Just read and report. Be precise — quote exact lines where relevant, and use file paths.
>
> For each of these files, give me:
>
> 1. `generate_sequences.py`
> 2. `sequences_to_activations.py`
> 3. `linear_field_probes.py`
> 4. `scripts/generate_all.sh` (and anything else in `scripts/`)
> 5. `pyproject.toml`
> 6. `.python-version`
> 7. The README.md (full contents)
>
> For each Python file, report:
>
> - What it does in 2-3 sentences.
> - Its CLI arguments / entry point (is it argparse, hardcoded constants, or something else?).
> - Every hardcoded file path or directory it reads from or writes to. I especially need to know: does it write to a path like `./data/`, `data/`, an absolute path, or something configurable?
> - Every place it calls `.to("cuda")`, `torch.device(...)`, `device=`, or similar — how it decides whether to use GPU.
> - Which Hugging Face model it loads (exact string passed to `from_pretrained`), and whether it uses AutoModelForCausalLM, HookedTransformer (TransformerLens), or something else.
> - Any environment variables it reads.
> - Approximate size of data it produces, if inferable from the code.
> - Is the model identifier hardcoded or configurable? Does the code call `huggingface_hub.login()` or rely on a `HF_TOKEN` env variable?
>
> For `scripts/generate_all.sh`, paste the full contents verbatim and tell me the order of commands and what each stage produces.
>
> For `pyproject.toml`, paste the `[project]`, `[project.dependencies]`, and `[tool.uv]` sections verbatim.
>
> Also check the `app/`, `probes/`, `figures/`, and `utils/` directories — list files in each with a one-line description.
>
> Finally, run `du -sh data/ figures/ probes/ token_subset/ 2>/dev/null` and report the output.
>
> Output the report as a single markdown document. If you're not sure what something does, say so rather than guessing.

---

## Appendix B — Open questions / decisions to make

*Add things here as they come up and get resolved.*

- [ ] Which Drive folder path to use for outputs? (suggestion: `/content/drive/MyDrive/shape-of-beliefs-data/`)
- [ ] Should cached activations eventually go on GCS bucket or stay on Drive for sharing?
- [ ] Do we need to rent an A100 session specifically, or is L4/T4 enough? (probably T4 is fine for 1B model)
- [ ] Do we want to replicate ALL figures or just the key ones?

---

## Appendix C — Change log

- **Initial version:** set up four-environment model, five-stage pipeline, seven-step roadmap. Step 1 confirmed done. Step 2 prompt prepared for Cursor agent.
