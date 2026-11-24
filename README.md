# Testing the Tester: Why Holistic LLM Judges Reward Broken Answer Keys—and How Contrastive Boundary Pinning Fixes Them

[![License: Apache-2.0 / CC-BY-4.0](https://img.shields.io/badge/License-Apache_2.0_%2F_CC--BY--4.0-blue.svg)](./LICENSE)
[![Runs: 400 Live Generations](https://img.shields.io/badge/Live_Benchmark-400_Runs_(80_Cases_%C3%97_5_Arms)-0f766e.svg)](./data/runs-400-five-arm.json)
[![Latin Square: Exact 5x5](https://img.shields.io/badge/Latin_Square-%5B16%2C16%2C16%2C16%2C16%5D-b45309.svg)](./src/verify-paper1.ts)

This repository contains the complete frozen evaluation transcripts (`400` live generations across `80` SHA-256-verified verbatim cases and `5` experimental arms), the standalone TypeScript psychometric & statistical verification CLI (`npm run verify`), the Register-Decoupled Contrastive Boundary Pinning (`CBP`) prompt engines and live API replication harness (`src/reference-prompts.ts`, `src/replicate-live.ts`, `scripts/run-live-5arm.ts`), and the LaTeX manuscript (`paper/main.tex` & `paper/main.pdf`) for:

> **Testing the Tester: Why Holistic LLM Judges Reward Broken Answer Keys—and How Contrastive Boundary Pinning Fixes Them**

---

## 1. Quick Start: Frozen Artifact Verification vs. Live API Replication

### Mode A — Frozen Artifact Verification (`$0.00` API Spend, `< 1s`, 100% Deterministic)

Recompute every number in **Table 2 (Combined 400-Run Leaderboard)**, **Table 3 (4-Subtrack Breakdown)**, and **Table 4 (5×5 Cyclic Latin-Square Slot Exposure & Positional Bias)** directly from the frozen transcripts in `data/runs-400-five-arm.json`:

```bash
npm install
npm run verify
npm test
```

### Mode B — Live API Replication (`src/reference-prompts.ts` & `npm run replicate:live`)

Inspect or execute the exact system prompts and JSON schemas for `arm-0-single-shot`, `arm-1-alfa-checklist`, `arm-3-cbp-rationale-stem`, and `arm-4-cbp-factored-2x2`:

```bash
# Dry-run inspection (prints prompt schemas, decoding temperatures, and expected SE at $0.00 cost):
npm run replicate:live

# Live API execution using your own OpenRouter or OpenAI key:
OPENROUTER_API_KEY=sk-or-... npm run replicate:live -- --live
```

#### Exact Decoding Hyperparameters & Expected Binomial Standard Error (`±3.2%` SE)
- **Producer Model:** `google/gemini-2.5-flash` (`temperature = 0.35` for Arms 0–2; `temperature = 0.25` for Pass-1 generation in Arms 3–4; `temperature = 0.20` for Pass-2 monotonic repair).
- **Blind Cold-Solver (`Solver B`), Unprimed Novice Guesser & 5×5 Cyclic Latin-Square Judge:** `openai/gpt-4o-mini` at `temperature = 0.0` with deterministic 32-bit FNV-1a option shuffling (`src/solver-harness.ts`).
- **Expected Stochastic Variance Across Live Re-Generations:** Because producer generation uses non-zero sampling temperature (`0.25–0.35`) over $N = 60$ MCQ cases ($N = 20$ per MCQ sub-track), the binomial standard error on overall Cold-Solve Valid Key rate ($p \in [91.7\%, 95.0\%]$) is:
  $$\text{SE}_{60} = \sqrt{\frac{p(1-p)}{60}} \approx \pm 3.2\% \quad (\pm 1.9\text{ items out of }60),$$
  and $\text{SE}_{20} \approx \pm 4.9\%$ ($\pm 1.0$ item out of $20$) on an individual 20-case sub-track.

---

## 2. Register-Decoupled Contrastive Boundary Pinning (`CBP`) & Factored 2×2 Stems

Standard rubric-driven prompting (e.g., inference-time ALFA checklists) conditions generation on abstract evaluative adjectives without enforcing an eliminative boundary witness in the question stem or controlling academic-register test-wiseness cues. **Register-Decoupled Contrastive Boundary Pinning (`CBP`)** decomposes pre-generation planning into five structural slots realized in `ASD-STE100` Simplified Technical English (`PlainHumanPlanSchema` in `src/reference-prompts.ts`):

1. **Contrastive Confusion Locus ($\mathcal{C}$ · `whyTwoAnswersLookRightAtFirst`):** Identifies the pair of competing hypotheses or outcomes that exhibit high surface similarity under rapid reading.
2. **Passage-Grounded Entailment ($\mathcal{G}$ · `whatActuallyHappens`):** Extracts the governing causal mechanism and entailment strictly entailed by the source passage.
3. **Register-Decoupled Invariant ($\mathcal{R}_{\text{STE}}$ · `everydayExplanation`):** Reformulates $\mathcal{G}$ in `ASD-STE100` Simplified Technical English (`0` academic meta-jargon or textbook giveaway buzzwords) so surface test-wiseness cannot separate the key from distractors by lexical register.
4. **Eliminative Boundary Witness ($w^\star$ · `oneFactInQuestionThatSettlesIt`):** Pins the exact numerical value, laboratory finding, code line, or boundary condition embedded inside `questionStem` that uniquely satisfies $\mathcal{G}$ and falsifies all three distractors.
5. **Structured Fault-Injection Taxonomy ($\mathcal{F}_1\dots\mathcal{F}_5$ · `threeCommonMistakes`):** Binds each distractor to a distinct cognitive fault class (`TRUE_GENERALLY_DOESNT_PROVE_IT_HERE`, `REVERSED_THE_DIRECTION_OR_SIGN`, `MISSED_ONE_KEY_DETAIL_IN_SETUP`, `USED_RULE_WHERE_REQUIREMENT_FAILS`, `WENT_WITH_HABIT_AGAINST_GIVEN_FACTS`) paired with its falsifying witness $w^\star$.

### The 5 Experimental Arms (`Arms 0..4` × `80` Verbatim Cases = `400` Live Runs)

| Arm | Slug | Method Label & Architecture Summary | ALFA Checklist? | Monotonic Solver Gate? | Factored 2×2 Stem? |
| :-: | :--- | :--- | :-: | :-: | :-: |
| **0** | `arm-0-single-shot` | Single-Shot Monolithic Baseline | No | No | No (`0%`) |
| **1** | `arm-1-alfa-checklist` | Inference-Time 6-Attribute ALFA Checklist (`arXiv:2502.14860`) | Yes | No | No (`0%`) |
| **2** | `arm-2-checklist-unguarded-pass2` | ALFA Checklist + Unguarded Pass-2 Self-Repair Rewrite (`80/80` rewritten) | Yes | No | No (`0%`) |
| **3** | `arm-3-cbp-rationale-stem` | **Contrastive Boundary Pinning — Unfactored Rationale Stem** (speakers state predictions *with* `"because..."` rationales in stem) | **No (`0`)** | **Yes** | No (`0%`) |
| **4** | `arm-4-cbp-factored-2x2` | **Contrastive Boundary Pinning — Factored 2×2 Stem** (prediction-only stem + crossed `Speaker {1,2} × Reason {R1,R2}` option grid) | **No (`0`)** | **Yes** | **Strict (`100%`)** |

---

## 3. Combined 80-Case Leaderboard (400 Runs · Exact 5×5 Cyclic Latin Square `[16, 16, 16, 16, 16]`)

| Arm Slug | Judge Rubric | 1st Wins | Valid Key % (60 MCQs) | Novice Foil Bite % | Discrim. D | STE Clarity | Debative % | Factored 2×2 % | Repaired / Rolled Back | Cost (USD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `arm-0-single-shot` | **86.52** | 29/80 (36.3%) | 76.7% | **23.3%** | +0.000 | 94.5 | 0% | 0% | 0 / 0 | $0.0627 |
| `arm-1-alfa-checklist` | 85.91 | 17/80 (21.3%) | 88.3% | 16.7% | +0.050 | 95.2 | 0% | 0% | 0 / 0 | $0.1536 |
| `arm-2-checklist-unguarded-pass2` | 85.45 | 16/80 (20.0%) | 85.0% | 20.0% | +0.050 | 94.1 | 0% | 0% | 80 / 0 | $0.2245 |
| `arm-3-cbp-rationale-stem` | 84.73 | 11/80 (13.8%) | **95.0%** | 8.3% | +0.033 | 96.5 | **100%** | 0% | 5 / 3 | $0.2023 |
| `arm-4-cbp-factored-2x2` | 83.60 | 7/80 (8.8%) | **91.7%** | **15.0%** | **+0.067** | **97.4** | **100%** | **100%** | 4 / 5 | $0.2029 |

### Key Empirical Findings
1. **Rubric–Validity Inversion & Contrastive Boundary Pinning (`CBP`) vs. ALFA Checklists:** Holistic 5-way LLM judging awards its `#1` mean rubric score (`86.52`, winning `29/80` cases) to `arm-0-single-shot` despite that baseline suffering a `23.3%` broken or ambiguous key rate (`76.7%` valid keys, $D = +0.000$). Without any ALFA checklist attributes, `CBP` gated by `Solver B` lifts Cold-Solve Valid Key rate from `76.7%` (`arm-0-single-shot`), `85.0%` (`arm-2-checklist-unguarded-pass2`), and `88.3%` (`arm-1-alfa-checklist`) to **`95.0%` (`57/60` in `arm-3-cbp-rationale-stem`)** and **`91.7%` (`55/60` in `arm-4-cbp-factored-2x2`)** ($\text{SE}_{60} = \pm 3.2\%$). On clinical `MedQA-USMLE`, `arm-3-cbp-rationale-stem` reaches `100.0%` (`20/20`) and `arm-4-cbp-factored-2x2` reaches `95.0%` (`19/20`), compared to `80.0%` for `arm-1-alfa-checklist` and `65.0%` for unguarded Pass-2 (`arm-2`).
2. **Resolving the In-Stem Spoiler Paradox via Factored 2×2 Stems (`arm-4-cbp-factored-2x2`):** When speakers state competing predictions *with* `"because..."` rationales inside `questionStem` (`arm-3-cbp-rationale-stem`), unprimed novices exploit the in-stem rationales to guess the key, collapsing Novice Foil-Bite rate to `8.3%` (`0.0%` on Track 2A; $D = +0.033$). Factoring causal rationales out of `questionStem` via `sanitizePredictionOnlyStem` and crossing `Speaker {1,2} × Reason {R1,R2}` in the 4 options (`arm-4-cbp-factored-2x2`) lifts Novice Foil-Bite rate by **`+80.7%` relative (`8.3% -> 15.0%` overall, and `0.0% -> 25.0%` on Track 2A)** and **doubles overall Item Discrimination $D$ from `+0.033` to `+0.067`** (`D = +0.100` across both Track 2A and Track 2B) while achieving the highest `ASD-STE100` clarity score (`97.4/100`, `13.0` words/sentence, `0.00` jargon).
3. **5×5 Cyclic Latin-Square Judging Exposes `67.5%` Slot 1–2 Positional Bias:** Even at `T = 0` under exact `[16, 16, 16, 16, 16]` slot balance (`[4, 4, 4, 4, 4]` per sub-track), Candidate Slots 1 and 2 capture `54/80` (`67.5%`) of all first-place LLM judge wins (`21` in Slot 1, `33` in Slot 2 vs. `4` in Slot 5).
