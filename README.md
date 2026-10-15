<div align="center">

# CoEvoX: When Harnesses Evolve, Which Models Should Follow?

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b)](#)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CoEvo--RSI-yellow)](https://huggingface.co/collections/nullptr04/coevo-rsi-6ad087479a4180c79495cdc9)

</div>

## Introduction

<img src="assets/coevox-framework.png" alt="CoEvoX framework" width="100%">

**CoEvoX** is a framework for **model–harness co-evolution** of LLM agents. It addresses the **brain–body mismatch problem**: an evolving harness (the body) can demand behaviors that its model (the brain) has not learned. It can also restrict capabilities that the model already has.

CoEvoX tackles this with **bidirectional historical inheritance** over a shared evolutionary archive of past harnesses, checkpoints, and execution outcomes. Each round, a meta-agent performs three steps:

1. **Harness editing.** Editors build new harnesses from historical designs they select from the archive.
2. **Model inheritance.** For each new harness, the meta-agent selects or merges checkpoints from the archive.
3. **Adaptation and promotion.** The best model–harness pair is adapted with SFT on its own rejection-sampled trajectories, and only validated improvements are promoted.

To diagnose interface mismatch, we introduce **TIMR** (Turn-level Interface Mismatch Rate). With this procedure, **Qwen3.5-9B** becomes competitive with much larger models on three domains:

- multi-agent failure attribution;
- role-aware text-to-SQL;
- smart-home instruction following.

<p align="center">
  <img src="assets/coevox-teaser.png" alt="Brain–body alignment" width="60%">
</p>

## Main Results

**Failure attribution** (agent / step accuracy, %). The test benchmarks are held out from both training and evolutionary selection.

| System | W&W Hand Who | W&W Hand When | W&W Algo. Who | W&W Algo. When | W&W Pro Who | W&W Pro When | TraceElephant Who | TraceElephant When |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| DeepSeek-V4-Pro-0813 | 60.34 | 10.34 | 65.87 | 22.22 | 61.77 | 70.40 | 56.62 | 31.96 |
| Qwen3.8-Max | 50.00 | 12.07 | 64.29 | 11.11 | 65.03 | **72.40** | 47.03 | 32.42 |
| FALAT + MiniMax-2.5 | 66.30 | 25.90 | **68.00** | **46.00** | – | – | – | – |
| Self-Harness | 56.90 | 24.14 | 56.35 | 38.89 | 47.94 | 54.11 | 49.77 | 7.76 |
| GSPO | 44.83 | 20.69 | 57.94 | 42.86 | 62.00 | 63.43 | 47.03 | **38.36** |
| Qwen3.5-9B | 37.93 | 15.52 | 50.00 | 31.75 | 57.20 | 49.66 | 25.57 | 4.11 |
| **CoEvoX (Qwen3.5-9B)** | **68.39** | **29.31** | 62.17 | 44.71 | **69.43** | 57.39 | **70.78** | 28.92 |

**Role-aware text-to-SQL** (RBAC, %). BIRD is out-of-distribution and is never used for training. For Viol., lower is better.

| System | Spider Safe-EX | Spider AC-F1 | Spider Safe-Deny | Spider Viol.↓ | BIRD Safe-EX | BIRD AC-F1 | BIRD Safe-Deny | BIRD Viol.↓ |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| DeepSeek-V4-Pro-0813 | 78.33 | 94.90 | 95.65 | 2.04 | 64.16 | **90.39** | 90.35 | 7.00 |
| OpenCode + DeepSeek-V4.1-Flash | 79.58 | 95.52 | 94.85 | 2.41 | **74.86** | 83.06 | 84.04 | 5.90 |
| Pi + MiniMax-M3 | 80.26 | **95.85** | 96.33 | 1.72 | 70.81 | 86.80 | 88.60 | 8.70 |
| Qwen3.5-9B | 76.91 | 94.72 | 92.95 | 3.41 | 53.52 | 87.00 | 85.54 | 9.25 |
| **CoEvoX (Qwen3.5-9B)** | **80.85** | 95.51 | **96.64** | **1.57** | 60.15 | 90.24 | **91.26** | **5.65** |

**Smart-home instruction following** (HomeBench multi-device, %). VM denotes valid instructions and IM denotes invalid ones.

| System | VM Succ. | VM F1 | IM Succ. | IM F1 |
| --- | --: | --: | --: | --: |
| DeepSeek-V4-Pro-0813 | 75.10 | 89.18 | 83.51 | 90.83 |
| GLM-5.3 | 70.61 | 86.65 | 64.95 | 81.42 |
| Qwen3.8-Max | 53.88 | 79.26 | **84.54** | 91.24 |
| Qwen3.5-9B | 10.61 | 40.15 | 62.89 | 78.71 |
| **CoEvoX (Qwen3.5-9B)** | **75.51** | **89.23** | **84.54** | **91.80** |

The paper contains the full baseline tables and the ablation, cost, and inheritance analyses. The figure below shows ten rounds of co-evolution on Who&When Algo. Agent and step accuracy rise while TIMR stays at or below 0.22%.

<p align="center">
  <img src="assets/whowhen-evolution-sketch.png" alt="Co-evolution on Who&When" width="75%">
</p>

## Repository Structure

```text
CoEvoX/
|-- src/
|   |-- meta_evolver/      # co-evolution loop: archive, Curator/Editors, model inheritance, promotion
|   |-- domains/           # task plugins: who_when/, rbac/, homebench/
|   |-- training/          # rejection sampling and full-parameter SFT (ms-swift)
|   |-- diagnostics/       # TIMR / interface-mismatch statistics
|   |-- runner/            # model client, vLLM lifecycle, pair evaluation
|   `-- traces/, utils/
|-- harnesses/             # initial and final evolved harness per domain
|-- configs/               # base.yaml, who_when/rbac/homebench.yaml, rbac_environment.yaml
|-- scripts/               # co-evolution and pair-evaluation entry points
|-- docs/                  # per-domain task and harness design (who_when, rbac, homebench)
|-- assets/                # figures
|-- ms-swift/              # training framework (cloned during setup)
`-- tests/
```

## Setup

```bash
cd CoEvoX
git clone https://github.com/modelscope/ms-swift.git ms-swift
git -C ms-swift checkout 1a7977b7f36025b0bb7528ec121cd3feac3032e3

conda create -n coevox python=3.10 -y
conda activate coevox
python -m pip install -e '.[train,dev]'
python -m pip install -e ms-swift
cp .env.example .env

# RBAC task execution and SQL evaluation run in a separate environment
conda env create --prefix .venvs/rbac --file configs/rbac_environment.yaml
```

Fill in the following variables in `.env`:

- the meta-agent endpoint (`OPENAI_BASE_URL`, `OPENAI_API_KEY`);
- the task-model endpoint (`POLICY_BASE_URL`);
- `MODEL_PATH`, which points to your local Qwen3.5-9B weights.

The meta-agent runs through the Codex CLI, which must be available on `PATH`.

Place the datasets under `data/`, following the paths in `configs/{who_when,rbac,homebench}.yaml`. The supported datasets are Who&When, Who&When Pro, TraceElephant, RBAC-Text2SQL (Spider and BIRD), and HomeBench.

## Co-Evolution

```bash
# Dry run (no GPU, weights, or API calls)
ROUNDS=1 bash scripts/run.sh --dry-run

# Run co-evolution for one domain
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_whowhen.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_rbac.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_homebench.sh

# Resume a round
bash scripts/run.sh --domain who_when --resume <round_id>
```

Harness and model inheritance are configured independently in `configs/base.yaml`. Each side is set to either `agent` (the meta-agent decides) or `predefined` (fixed rules).

## Evaluating a Model–Harness Pair

```bash
# Evaluate the pair promoted in a round of your own co-evolution run (vLLM is managed automatically)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_evolved_full_test.sh <round_id>

# Evaluate a model already served at POLICY_BASE_URL with a harness (initial_harness or generated)
bash scripts/run_whowhen_full_test.sh   generated
bash scripts/run_rbac_full_test.sh      generated
bash scripts/run_homebench_full_test.sh generated
```

Each domain ships two harnesses under `harnesses/<domain>/`: `initial_harness/` and the final evolved harness `generated/` (the default `harness.active` in each config). Candidates from new co-evolution runs are written to `candidates/`. Results are written to `output/<domain>/full_test/`. Each evaluation also reports TIMR in `mismatch.json`.

## Acknowledgments

This repo builds on **[ms-swift](https://github.com/modelscope/ms-swift)** for supervised fine-tuning and **[vLLM](https://github.com/vllm-project/vllm)** for model serving.
