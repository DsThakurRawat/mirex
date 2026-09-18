# MIREX 2026 AI-Generated Music Detection — Implementation Plan v2

**Status:** Research-complete master plan. This document is the specification for implementing agents. It supersedes v1 entirely.
**Date:** 2026-08-08. Task page: https://music-ir.org/mirex/wiki/2026:AI-Generated_Music_Detection

---

## 0. Executive summary (the thesis)

The evidence from 2024–2026 literature is unambiguous: **in-domain AI-music detection is a solved problem (≥99% F1 / <2% EER), and every published system fails out-of-domain** — Suno-trained detectors get F1 0.63–0.78 on Udio, IRCAM's commercial detector caught 3/50 Boomy tracks before patching, SpecTTTra scored **48.9% EER (random)** on the MIREX 2025 WildSVDD test while scoring 1.75% on its home dataset, and simple resampling to 22.05 kHz fooled IRCAM entirely. The ranking metric — **macro-averaged AUROC across hidden strata (generator family × compression × excerpt length × vocal/instrumental × difficulty)** — is engineered by the organizers (who ran SVDD 2024 and MIREX 2025 and know these failure modes intimately) to punish exactly this brittleness.

Therefore this plan is built on three pillars, in priority order:

1. **Generator-family coverage through data, not architecture novelty.** The six test-set generator families are *named* (Suno, Udio, Mureka, MiniMax, YuE, ACE-Step). Two are open-source (we self-generate unlimited exact-decoder data for ~$0.01–0.10/track); two have massive public dumps (Suno: 116k v5 tracks + 50k version-labeled; Udio: 132k); two are API-accessible (Mureka, MiniMax — we generate ~2k tracks each for a few hundred dollars). This converts a "zero-shot" problem into a mostly-covered problem, with a real-music-only anomaly branch as insurance for unseen versions.
2. **Confound-symmetric training.** Published detectors largely detect *delivery pipelines*, not AI-ness (a sample-rate rule alone achieved 83% precision on a Suno/Udio/MSD corpus). One randomized delivery-chain simulator (codec, bitrate, resample, loudness, excerpting) is applied identically to both classes; contaminated data (Song Describer Dataset ⊂ MTG-Jamendo split-0 test) is quarantined; every dataset-source shortcut is audited before training.
3. **A complementary-hypothesis ensemble selected by leave-one-generator-out (LOGO) worst-case AUROC.** Five branches with decorrelated failure modes — speech-SSL (wav2vec2-XLS-R+AASIST, the SVDD-winning recipe), music-SSL (MERT), long-context spectrogram transformer, an interpretable physics/fakeprint branch (Fourier deconvolution peaks + shift-invariant log-frequency CNN), and a real-only anomaly detector — fused by stacked logistic regression trained on LOGO folds so fusion weights cannot overfit seen generators.

The 24 h single-GPU inference budget (~8+ s/track for any plausible test-set size) makes this ensemble comfortably feasible. The 4 submission slots are used as a risk ladder (§12).

---

## 1. Verified task facts and rules

- Binary classification, **full tracks**, WAV 44.1/48 kHz mono/stereo in, one score in [0,1] per track out (CSV `track_id,ai_generated_score`).
- Positive class = fully AI-generated; negative = human-made. Partial fakes are out of scope this year.
- **Primary metric: macro-averaged AUROC across hidden evaluation strata**; secondary: pooled AUROC, AUPRC, EER, balanced accuracy, F1, FPR on human music, FNR by generator, diagnostics by vocal/instrumental, compression, excerpt length, difficulty.
- Hidden AI side: **Suno, Udio, Mureka, MiniMax, YuE, ACE-Step**. Hidden real side: **derived from the Song Describer Dataset (SDD) validation split** → ~2-min, 320 kbps MP3-sourced Jamendo CC music (indie/electronic-leaning, non-commercial masters).
- **External data, pretrained models, and self-generated data are explicitly allowed.** Everything must be disclosed in the technical report.
- **Forbidden:** SDD validation split for anything (training, validation, model selection, prompt tuning, threshold tuning); any use of hidden eval data.
- Submission: Docker container, standardized inference interface, ≤24 h wall-clock on one GPU for the full hidden set, ≤4 system versions per team, 2–4 page ISMIR-format technical report + compute declaration. Failure on >5% of items → excluded. External API calls at inference discouraged/prohibited — **the container must be fully offline**.
- Precedent: same captains (Yixiao Zhang, You "Neil" Zhang) ran MIREX 2025 Song Deepfake Detection (test = WildSVDD + SONICS; best baseline was wav2vec2+AASIST trained on both domains: 6.14/20.82/2.05% EER; SpecTTTra trained on SONICS alone: 48.92/43.96/1.75% — the canonical cross-domain collapse). You Zhang's research lineage is one-class learning for anti-spoofing (OC-Softmax, SAMO). Expect the eval to be built by people who *specifically test for* generator-holdout generalization and robustness perturbations.

---

### 1.1 Source wording — verified against the task wiki, 2026-08-26

Every claim in §1 was re-checked against
https://music-ir.org/mirex/wiki/2026:AI-Generated_Music_Detection on
2026-08-26 and confirmed. The organizers' verbatim wording is recorded here so
the provenance is checkable without re-fetching the page:

| Claim in §1 | Task-page wording |
|---|---|
| Six generator families | "Suno", "Udio", "Mureka", "MiniMax", "YuE", "ACE-Step" — introduced as **candidates**: "The AI-generated portion will be constructed from multiple music generation systems, where licensing and evaluation conditions permit." |
| Primary metric | "Macro-averaged AUROC across hidden evaluation strata. AUROC is used because different real-world applications may require different operating thresholds." |
| Hidden real side | "The real-music portion of the hidden evaluation set will be sampled from the SDD validation split" |
| Hidden AI side | "the other evaluation examples will be newly generated" |
| Submission | CSV `track_id,ai_generated_score`, scores in [0, 1]; "four versions" per team |
| Runtime | "within a 24-hour wall-clock budget on a single GPU"; over budget or failing on >5% of items ⇒ "excluded from the primary ranking" |
| External data | "Participants may use external datasets and pre-trained models"; "Participants may use public, private, synthetic, or self-constructed data for training" |
| Forbidden | "Participants must not use any part of the hidden evaluation set for training, validation, model selection, prompt tuning, or threshold tuning"; specifically "participants must not use the SDD validation split for training, validation, model selection, prompt tuning, and threshold tuning" |
| Provided resources | "We plan to provide a training dataset to make the task easier to enter"; "The final training dataset will be announced before the submission phase"; "We plan to provide a baseline model and checkpoint to help participants get started." |
| Key dates | Registration deadline, submission deadline, and submission platform are all still **TBD** on the page. |

**Three amendments this forces to §0/§1 as written:**

1. **The family list is a candidate list, not a closed set.** §0 pillar 1 calls
   the six families "*named*" and treats coverage as a nearly-closed problem.
   "Where licensing and evaluation conditions permit" means a family can be
   dropped *or added* before evaluation. This raises the standing of branch E
   (real-only anomaly) — the one branch whose accuracy does not depend on
   having trained on the family — and argues against tuning fusion weights so
   hard on the six that a seventh family collapses the macro average.
2. **The hidden AI audio is freshly generated**, not sampled from the public
   dumps. Training on the Suno v5 / Udio v1.5 dumps therefore carries a
   version-drift gap against whatever those services emit at evaluation time.
   The version-drift proxy (train ≤v4.5 → test v5) is consequently not
   optional colour for the paper — it is the only estimate we will have of
   that gap.
3. **SunoCaps is on the organizers' candidate list and is absent from our
   registry.** We carry `suno_audio` (`humair025/suno-audio`) instead, which
   is a different dataset. Either add SunoCaps or note the deliberate
   substitution in the disclosure table (§8 of the paper).

**Deliberate over-restriction to declare.** The rule bans only the SDD
*validation split*; we quarantine SDD audio entirely (metadata-only). Given
SDD ⊂ MTG-Jamendo split-0 test and that the hidden real pool is drawn from SDD
validation, this is the safe reading — but it is stricter than the letter of
the rule and should be stated as a choice in the technical report, not implied
to be a requirement.

**Not yet in this plan:** the organizer-provided training set and baseline
model. Registry stubs now exist (`mirex_provided`, `mirex_baseline` in
`data_fetch.py`, kind `pending`) plus `MirexProvidedBaseline` in
`baselines.py`. Both must be picked up when announced, and the provided
training set must clear our own quarantine gate before use.

---

## 2. Metric math — what actually determines ranking

**AUROC is rank-based:** AUROC = P(score(fake) > score(real)) computed per stratum. Consequences:

- Monotone score transforms are irrelevant per stratum ⇒ within-stratum *ranking quality* is everything for the primary metric; absolute calibration only matters for secondary metrics (EER, F1, balanced accuracy) — so we calibrate anyway, but never at the expense of ranking.
- **Macro-averaging makes the worst generator family dominate.** If per-family AUROC is {0.99, 0.99, 0.99, 0.99, 0.99, 0.70}, macro = 0.941 and a uniform 0.96 system wins. Expected per-family AUROC is roughly concave in effort spent per family ⇒ optimal allocation pushes the *minimum* up, not the mean. **All model selection, checkpointing, fusion weighting, and ablation decisions use worst-stratum and macro AUROC on our internal harness (§5), never pooled accuracy.**
- Strata also include compression/excerpt-length/difficulty conditions ⇒ a model whose ranking collapses under MP3-64k or on 30 s excerpts loses entire strata. Robustness conditions are first-class strata in our internal harness, not an afterthought.
- FPR on human music is reported separately and the real class is *known* to be Jamendo-flavored CC music: bedroom production, lo-fi, electronic, synth-heavy human tracks are the FP danger zone. We mine hard negatives accordingly (§4.3).

---

## 3. Failure modes of prior art (what we must not repeat)

Each of these is documented; each maps to a design decision in this plan:

| # | Documented failure | Evidence | Our countermeasure |
|---|---|---|---|
| F1 | Cross-generator collapse | Suno→Udio F1 0.63–0.78; Boomy 6–24% recall (TISMIR arms-race); Encodec→DAC transfer ≈0% (Deezer ICASSP'25); FakeMusicCaps open-set: Suno dumped into "real" | Cover all 6 named families with data (§4.2); LOGO selection (§5); anomaly branch (§6.E) |
| F2 | Pipeline-confound shortcuts | Sample-rate rule alone = 83% precision; Suno=192kbps/48k, Udio=320kbps/48k vs human ~141kbps (TISMIR) | Delivery-chain simulator applied symmetrically (§4.4); confound audit (§4.5) |
| F3 | Fragility to cheap transforms | Pitch ±2st → 66%; MP3-64k → 58–74%; resample-22.05k fools IRCAM; high-pass 8kHz → everything flagged AI | Aug-heavy training incl. pitch/stretch/transcode; shift-invariant log-frequency branch (§6.C); robustness strata in harness |
| F4 | "Real as default" bias | Deezer CNN: real class stays 99%+ under all manipulations; unseen artifacts → silently labeled human | Anomaly branch scores *distance from real music manifold* (§6.E); fusion learns per-branch abstention (§7) |
| F5 | Test-set contamination / leakage | SDD ⊂ MTG-Jamendo split-0 TEST; AIME's real half is Jamendo; SONICS leak-safe only at (lyrics,style) level | Quarantine protocol (§4.1) with artist-level blocklists, enforced in code before any training |
| F6 | Short-context ceiling | SpecTTTra +25% F1 from 5s→120s (SONICS); full-song structure is signal | Long-context branch (§6.B) + track-level aggregation (§8) |

---

## 4. Data plan

### 4.1 Quarantine protocol (do this FIRST, before any download touches a training folder)

Build a machine-readable blocklist, enforced by the dataset builder:
1. All 706 SDD track IDs (Zenodo 10072001) — full dataset, not just the 546-track validated subset (safest reading of the rule).
2. All MTG-Jamendo **split-0 test** track IDs (SDD's parent pool — the organizers may draw additional real tracks from it).
3. All artists appearing in SDD, blocked at artist level across all of MTG-Jamendo (artist IDs in `audio_metadata.tsv`).
4. AIME's 500 real tracks and Echoes' ~310 FMA bona fide tracks deduped against our real pools.
5. Cross-dump dedup on the AI side (Udio scrapes overlap heavily; Suno link-dumps vs audio-dumps): dedupe by platform UUID first, then audio fingerprint (chromaprint) across everything.
- **Acceptance check:** an automated report proving zero ID/artist/fingerprint overlap between any training file and the blocklist.

### 4.2 AI-side training pool (~450k+ tracks before balancing)

| Source | What | Why |
|---|---|---|
| **Muse** (HF `bolshyC/Muse`, MIT) | 116k full Suno **v5** songs (CN+EN) | The closest match to eval-time Suno; the single most valuable public set |
| **Suno Audio Dataset** (HF `humair025/suno-audio`, MIT, 213 GB) | 49.7k tracks with per-track `model_name` | Version-stratified Suno (v3→v4.5) for version-drift robustness |
| **SONICS** (Kaggle/HF, CC BY-NC) | 49k Suno v2–v3.5 + Udio-32/130 fakes | Older-version coverage; note: was MIREX 2025 test material — legal to train on per current 2026 wiki, but keep it swappable in case a banned-list appears |
| **Udio dump** (HF `blanchon/udio_dataset`, 626 GB, ~132k tracks) | Udio v1/v1.5 | The only large Udio source (Udio is streaming-only since the Oct 2025 UMG settlement — no fresh data is coming; version gap covered by anomaly branch) |
| **Echoes** (HF `Octavian97/Echoes`, CC-BY-SA) | 4.5k tracks, 10–12 systems incl. **ACE-Step, Suno v5, Udio, DiffRhythm, Riffusion, Stable Audio, SongGen, Mubert** | Semantically aligned to bona fide FMA tracks — the anti-shortcut dataset; hardest published benchmark (15.3% EER); transfers best outward |
| **Self-generated: ACE-Step v1-3.5B** (open, Apache-2.0) | **20k full songs** — RTF ~27× on A100: 4-min song in ~20 s; ≈$50–150 total | Exact decoder match to a named test family (DCAE + ADaMoSHiFiGAN vocoder) |
| **Self-generated: YuE-7B** (open) | **5k songs** — ~$0.03/3-min song on cloud L40S/H100 spot; ≈$150–400 | Exact match to a named family; its X-Codec is 16 kHz-native with Vocos-hallucinated content >8 kHz — a strong learnable fingerprint |
| **API-generated: Mureka (V6/O1+ current)** | **~2k songs**, budget $200–500 | Named family, no public detection-scale corpus exists; MeLoDy-cascade + Stable-Audio-lineage diffusion decoder |
| **API-generated: MiniMax Music (1.5/2.5 via official API or Replicate)** | **~2k songs**, budget $100–300 | Named family, zero public data, undocumented architecture |
| **Diversity fillers:** self-generated DiffRhythm (~10 s/song!), Stable Audio Open, MusicGen, JASCO; plus FakeMusicCaps + AIME + MoM/CLAM subsets | ~30–60k items | Decoder-family diversity (EnCodec/DAC/mel-vocoder/latent-VAE lineages) — these are our LOGO holdout currency, not just training mass |

**Prompt strategy for self-generation** (this is where most teams will be lazy — we won't): sample genre × mood × instrumentation × language × vocal/instrumental from a broad taxonomy (use MTG-Jamendo tag distribution as the *target* distribution, since the real eval class is Jamendo-flavored); lyrics from an LLM in multiple languages + instrumental-only generations (vocal/instrumental is an eval stratum); vary duration 30 s–5 min; vary inference settings (steps, CFG, seeds) — artifact intensity varies with sampler settings, and the eval "difficulty" stratum likely encodes exactly this.

### 4.3 Real-side training pool (~250k tracks after filtering)

| Source | What | Notes |
|---|---|---|
| **MTG-Jamendo minus quarantine** (~50k tracks, 320 kbps MP3, full-length) | **The domain-match core** — same platform, mastering culture, and codec provenance as the hidden real class | Highest-weight real data |
| **FMA-full** (879 GB, 106k full tracks; or FMA-large 93 GB/30 s as fallback) | Genre-diverse CC full tracks | Prefer full for excerpt-length strata |
| **MusicNet** (~330 classical, 34 h) + additional classical/jazz CC | Genre coverage where AI music is rare | |
| **Hard negatives (FP danger zone):** lo-fi/bedroom/synthwave/vaporwave/8-bit/hyperpop human tracks; heavily-compressed masters; old digitized recordings; mined from FMA/Jamendo tags + targeted CC crawls | The "human music that looks AI" tail | FPR-on-human is a reported metric; SubmitHub's checker over-flags human EDM at 62 — that's the trap |

Class balance: keep effective 1:1 per training batch via sampling weights, but preserve *within-class* source diversity quotas so no single source dominates (cap any single dataset at ~35% of its class per epoch).

### 4.4 Delivery-chain simulator (the single most important preprocessing component)

One randomized augmentation pipeline, **identically distributed across both classes**, applied on-the-fly at training time (and in stronger form for robustness strata of the internal harness):

`source audio → [optional pitch ±2 st | time-stretch 0.9–1.1] → [optional EQ tilt / 10-band ±4 dB] → [optional reverb/noise floor, SNR 25–45 dB] → [resample: 22.05/32/44.1/48 kHz round-trip] → [codec: none/MP3/AAC/Opus/Vorbis @ 32–320 kbps] → [loudness normalize to U(-20,-7) LUFS] → [mono-fold p=0.15] → [random excerpt: 10 s–full] → 44.1 kHz float WAV`

Rationale: the hidden real class arrives as 320 kbps-MP3-lineage WAV; the hidden AI class arrives through unknown platform transcodes, and the organizers explicitly stratify by compression and excerpt length. This pipeline simultaneously (a) destroys F2 shortcuts, (b) trains in F3 robustness, (c) mirrors eval conditions. Probabilities tuned so ~20% of samples pass through clean (never let the model believe "clean = one class").

### 4.5 Confound audit (gate before full training)

Train a deliberately dumb probe (logistic regression / small MLP) on **non-content features only** — container bitrate history estimate, spectral rolloff frequency, LUFS, duration, stereo width, silence statistics — to predict the class label on the *post-simulator* training distribution. **Gate: probe AUROC must be <0.60.** If higher, a shortcut survives; find it and fix the simulator. Re-run after every data-mix change.

---

## 5. Internal evaluation harness (build BEFORE any model training)

The harness is our replica of the hidden eval and the sole arbiter of every decision.

- **Strata grid:** {each generator family × version} × {clean, MP3-128, MP3-64, Opus-48, resample-22.05, pitch±1st} × {30 s, 60 s, 120 s, full} × {vocal, instrumental}, real side drawn from held-out Jamendo-like + FMA pools (quarantine-respecting), passed through the same condition grid.
- **LOGO protocol:** for each of the 6 named families, train candidate on the other 5 (+ fillers), evaluate on the held-out family. Report the vector of per-family AUROCs, macro mean, and **min**. Additionally hold out entire *decoder lineages* (e.g., all mel-vocoder systems) as a stress variant.
- **Version-drift proxy:** train on Suno ≤v4.5, test on v5 (Muse); train on Udio-32, test on Udio-130. This estimates our exposure to unreleased versions (Suno v5.x, Udio "v2", MiniMax 2.5) that the test set may contain.
- **Frozen dev set:** a fixed ~20k-track stratified sample with locked membership; never trained on; all reported numbers come from it. Fusion/calibration use separate LOGO folds (§7).
- Metrics reported exactly as MIREX will: macro-AUROC (primary), per-stratum table, pooled AUROC, EER, AUPRC, FPR@human, FNR@family.

**Acceptance:** harness reproduces published baseline behavior (e.g., SONICS SpecTTTra checkpoint should score near-perfect on its home stratum and collapse on WildSVDD-like/unseen strata) before we trust it.

---

## 6. Model portfolio (complementary hypotheses, decorrelated failures)

All branches output per-chunk logits, aggregated per §8, fused per §7. Train every branch with the delivery-chain simulator; optimizer SAM (sharpness-aware minimization — documented DG gains for deepfake detection) where it fits the budget; mixup on logits as regularizer.

### A. Speech-SSL branch — wav2vec2-XLS-R (300M) + AASIST head
The SVDD-2024-winning and MIREX-2025-best-baseline recipe. Fine-tune top layers with layer-weighted aggregation (SLS-style sensitive layer selection); RawBoost + our simulator as augmentation; 4–10 s chunks. Catches vocal-deepfake cues (phonation, sibilance shimmer, formant unnaturalness). Optionally a WavLM twin for cheap ensemble diversity (proven complementary in SVDD top systems).

### B. Music-SSL branch — MERT-330M + shallow transformer head
Music-domain pretraining (pitch/rhythm/timbre tokens) — proven complement to speech SSL (SingGraph; CLAM's dual-stream design). 10–30 s chunks. Catches production/performance unnaturalness: quantized timing, absent sidechain ducking dynamics, uniform mastering, structure-level repetition.

### C. Physics/artifact branch (interpretable, near-free, dangerous to skip)
1. **Fourier fakeprint features:** time-averaged log-spectrum ("fakeprint"), lower spectral envelope removed; deconvolution upsamplers provably emit peak combs at frequencies f = k · (f_out / ∏ strides above layer l) — weight-independent, architecture-determined (ISMIR 2025 best paper; 10-parameter LR hits ~100% on Suno v3.5/Udio). Feed peak-comb energies + rolloff/cutoff descriptors (YuE's 16 kHz limit, Suno's 12–16 kHz drift, 8–16 kHz haze band statistics) to gradient-boosted trees.
2. **Shift-invariant CNN:** log-frequency STFT remapping turns pitch/speed shifts into translations; cross-correlation filter + max-pool gives transposition invariance (Deezer ISMIR 2026 design) — this is the answer to F3 without giving up the physics signal.
3. **High-resolution linear-STFT CNN on the 4–24 kHz band** (n_fft ≥ 4096): band-limit boundaries, vocoder haze, checkerboard textures live here.

### D. Long-context structure branch — 120 s spectrogram transformer
SpecTTTra-α-style (time/freq axis tokenization, ~20M params) or ConvNeXt-T on 120 s mels, trained on *our* diverse pool (its published failure was its training data, not its architecture). Catches song-level cues: Udio's ~32 s window seams and extend-boundary style drift, loop-heavy structure, outro fades, energy uniformity.

### E. Real-only anomaly branch (zero-shot insurance against F1/F4)
Trained with **zero fake data** ⇒ immune to generator-coverage gaps by construction; this is what fires on Udio-v2/Suno-v5.5/MiniMax-2.5 material we cannot obtain.
1. **Normalizing flow over frozen MERT/XLS-R features** of real music only (MusicDET recipe — ICML 2026 — beats discriminative detectors on unseen generators); score = negative log-likelihood.
2. **OC-Softmax embedding model** (task captain's own lineage): compact real-music embedding ball with margin; score = distance from center. Loss: L = (1/N) Σᵢ log(1 + exp(α(m_{yᵢ} − ŵᵀx̂ᵢ)(−1)^{yᵢ})) with m₀ (real, tight) > m₁ (fake, pushed out); at inference only the real-margin geometry matters.
3. Optional cheap variant: **codec re-encoding distance** — re-encode input through public codecs (EnCodec-24k, DAC-44k) at high bitrate; audio already living near a codec manifold reconstructs anomalously well; score = −spectral distance. Include only if it survives the harness.

**Branch-budget note:** A+B+D are ~1 GPU-week-class fine-tunes each on 4×A100/H100 or a few weeks on 2×4090; C trains in hours; E days. If compute is short, priority order is A, C, B, E, D (C is nearly free and covers the most strata per dollar; D is the first to cut).

---

## 7. Fusion and calibration

- Per-branch track scores → z-normalize per branch → **stacked logistic regression trained exclusively on LOGO folds**: for fold g, stack inputs are branch scores produced by branch-models trained *without* family g, targets are family-g vs real. This forces fusion weights to encode "how much do I trust each branch on a generator it never saw" — the exact question the hidden test asks. Regularize toward equal weights (ridge, strong λ); fallback submission uses plain rank-averaging (provably monotone-safe for AUROC, zero overfit risk).
- **Per-branch abstention handling:** branches A–D output ~0.5-ish confidence when artifact-free (F4); branch E outputs a continuous anomaly score always. The stacker sees all raw scores and learns this pattern; verify on harness that removing any single branch degrades macro-AUROC by <2 points (no single point of failure).
- Final monotone calibration (isotonic or Platt) on pooled LOGO predictions for the secondary threshold metrics; monotone ⇒ AUROC untouched.

## 8. Track-level aggregation

Chunk logits z₁…z_n per track (chunk lengths per branch as in §6): s_track = (1−λ)·mean(z) + λ·mean(top-k(z)), with λ, k tuned on the harness per branch. Rationale: mean is robust for long tracks; top-k preserves sensitivity when artifacts are localized (Udio seams, shimmer bursts) and for 30 s excerpts (where n is small, top-k≈mean automatically — well-behaved across the excerpt-length strata). Evaluate also logit-of-mean-prob variant; pick per-branch by harness macro-AUROC.

## 9. Test-time augmentation (TTA)

For the TTA submission slot: average each branch's score over {identity, MP3-128 round-trip, resample-32k round-trip, ±0.5 st pitch}. Motivation: TISMIR showed single transforms flip decisions; averaging over the transform group stabilizes ranking. Verify on harness that TTA never hurts clean strata by >0.5 pt.

---

## 10. Submission engineering

- **Runtime budget:** all five branches + TTA ≈ well under 5 s/track on one A100 (SSL forward passes dominate). 24 h / e.g. 10k tracks = 8.6 s/track → fits with ≥2× margin. Rehearse on a full-scale synthetic test directory (10k WAVs, mixed lengths up to 8 min) inside the exact container; measure p99 per-track latency and peak VRAM; enforce a hard per-track timeout with a mid-prior (0.5) fallback score so the >5%-failure exclusion rule can never trigger.
- **Container:** offline-only (no API calls), pinned weights baked into the image, deterministic seeds, input tolerance tests: 44.1 & 48 kHz, mono & stereo, 10 s–10 min durations, silent/corrupt-file handling.
- **Four submission slots (risk ladder):**
  1. Full stacked ensemble (primary).
  2. Rank-average ensemble, no stacker (protects against stacker overfit).
  3. Best single model per harness (likely A or B) — protects against ensemble-level bugs.
  4. Full ensemble + TTA (robustness play, in case hidden strata are perturbation-heavy).

## 11. Compute, storage, timeline

- **Storage:** ~3–4 TB NVMe (raw ~2.5–3 TB + features/checkpoints). Trim option: FMA-large instead of FMA-full → ~1.8 TB.
- **Compute:** target 4×A100/H100 for ~2–3 weeks of training total (or 2×4090 with longer wall-clock and gradient checkpointing); generation budget: ACE-Step+YuE ≈ $200–550 cloud spot; Mureka+MiniMax APIs ≈ $300–800. Total cash budget besides GPUs: **under $1.5k**.
- **Timeline (deadline TBD; assume ~mid-Oct 2026, ISMIR is Nov 8–12):**
  - Wk 1–2: quarantine + downloads + dedup + simulator + confound audit + **harness** + baselines reproduced (fakeprint LR, wav2vec2+AASIST MIREX-2025 baseline, SONICS checkpoint).
  - Wk 2–4: self-generation campaign (ACE-Step, YuE, DiffRhythm; Mureka/MiniMax API) — runs unattended in parallel.
  - Wk 3–6: train branches A–E; weekly harness reports (macro + min stratum).
  - Wk 6–7: fusion, calibration, TTA, ablations.
  - Wk 7–8: container hardening, full-scale runtime rehearsal, 4 submissions, technical report.
  - Slack: 2+ weeks if deadline is later.

## 12. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hidden set uses generator versions newer than any obtainable data (Udio "v2", Suno v5.x, MiniMax 2.5) | High | Anomaly branch (E); version-drift proxy in harness; API-generate as late as possible before deadline |
| Suno watermarking (announced 2026-08-06) changes signal landscape | Low for this cycle (test audio likely pre-dates) | Don't rely on watermarks at all; if a public watermark detector appears, evaluate as an extra feature and disclose |
| Organizers ban SONICS or other public sets late | Medium | Data mix is modular; retrain path without any single dataset is rehearsed (source-quota training makes this cheap) |
| Udio dump is v1/v1.5 only, test may be newer | High | Udio's diffusion+window-seam signature may persist; E-branch; long-context branch catches seam structure |
| Real-class domain surprise (non-Jamendo human sources) | Medium | FMA/MusicNet/hard-negative diversity in real pool; FPR monitoring stratum |
| Stacker overfit | Medium | LOGO-fold stacking + ridge; rank-average fallback slot |
| Container DQ (>5% failures / timeout) | Low | Hard timeouts + fallback scores + full-scale rehearsal |
| Dataset licensing (NC clauses) | Low for MIREX research use | Track licenses in the disclosure table from day 1; MIT/CC0 sources prioritized (Muse is MIT) |

## 13. Paper plan

- **Venue ladder:** MIREX 2026 technical report (required, 2–4 pp) → ISMIR 2027 full paper; ICASSP 2027 as fast-turnaround alternative; TISMIR for an extended version with the confound-audit methodology.
- **Contributions (each independently defensible):**
  1. **Confound-audit protocol** + evidence of shortcut prevalence across public datasets (the <0.60 probe gate as a reusable standard).
  2. **LOGO macro-AUROC selection** as the correct objective for generator-generalization, with measured gaps between iid and LOGO selection.
  3. **Complementarity analysis**: physics branch vs SSL branches vs real-only anomaly — which strata each wins, with the decorrelation evidence.
  4. **A released corpus**: our self-generated Mureka/MiniMax/YuE/ACE-Step tracks with full prompt/settings metadata (fills the exact gap Echoes identified: provider diversity drives transfer).
- Title candidate: *"Coverage, Confounds, and Complementarity: What It Actually Takes to Detect AI-Generated Music in the Wild."*

## 14. Execution phases for implementing agents (with acceptance gates)

1. **P0 — Rules & scaffolding:** repo layout, config system, blocklist builder. Gate: quarantine report shows zero overlap (§4.1).
2. **P1 — Data acquisition:** downloads, dedup, metadata DB (per-track: source, generator, version, license, fingerprint). Gate: dataset census report matches §4.2/4.3 targets.
3. **P2 — Simulator + audit:** delivery-chain simulator with unit tests (bit-exact reproducibility given seed). Gate: confound probe AUROC <0.60 (§4.5).
4. **P3 — Harness:** strata grid, LOGO runner, frozen dev set, metric suite. Gate: reproduces published baseline collapse patterns (§5).
5. **P4 — Baselines:** fakeprint-LR, wav2vec2+AASIST (MIREX-2025 recipe), SONICS checkpoint evaluated on harness. Gate: harness numbers logged; these are the floor.
6. **P5 — Generation campaign:** ACE-Step/YuE/DiffRhythm local; Mureka/MiniMax API; prompt taxonomy per §4.2. Gate: per-generator spectrogram QA sweep (verify cutoffs/artifacts empirically — community-measured numbers in this plan must be re-verified on our own generations).
7. **P6 — Branch training:** A→C→B→E→D priority. Gate per branch: beats all baselines on harness macro-AUROC AND min-stratum.
8. **P7 — Fusion/calibration/TTA.** Gate: ensemble beats best branch on macro AND min; no-single-point-of-failure check (§7).
9. **P8 — Packaging:** Docker, rehearsal at full scale, 4 submission variants, disclosure table, technical report. Gate: p99 latency × plausible test size < 18 h; all input-tolerance tests pass.

## 15. Key references (all verified this session)

- Task: music-ir.org/mirex/wiki/2026:AI-Generated_Music_Detection ; 2025 precedent: .../2025:Song_Deepfake_Detection (+Results)
- SONICS/SpecTTTra: arXiv:2408.14080, github.com/awsaf49/sonics
- Deezer: arXiv:2501.10111 (ICASSP'25, deepfake-detector code+weights); arXiv:2506.19108 (Fourier fakeprints, ISMIR'25 best paper, github.com/deezer/ismir25-ai-music-detector); arXiv:2607.27454 (shift-invariant robustness, ISMIR'26)
- SVDD 2024: arXiv:2408.16132 (challenge), arXiv:2406.02438 (CtrSVDD); winners' recipes (Fosafer, NBU_MISL, I2R-ASTAR); SingGraph arXiv:2406.03111; OC-Softmax arXiv:2010.13995; SAMO arXiv:2211.02718
- Generalization evidence: TISMIR 10.5334/tismir.254 (arms race); Echoes arXiv:2603.23667; ArtifactBench arXiv:2604.16254; MusicDET arXiv:2605.18072; Sofia arXiv:2606.16612; CLAM/MoM arXiv:2512.00621; FakeMusicCaps arXiv:2409.10684; SAM-for-deepfakes arXiv:2506.11532
- Generators: YuE arXiv:2503.08638 (X-Codec 16 kHz + Vocos upsampler); ACE-Step arXiv:2506.00045 (DCAE + ADaMoSHiFiGAN); Mureka/MusiCoT arXiv:2503.19611 (MeLoDy cascade + Stable-Audio-lineage decoder); Suno watermarking: techcrunch.com 2026-08-06
- Data: HF bolshyC/Muse, humair025/suno-audio, blanchon/udio_dataset, Octavian97/Echoes, disco-eth/AIME; SDD Zenodo 10072001 (⊂ MTG-Jamendo split-0 test — QUARANTINE); FMA github.com/mdeff/fma; MTG-Jamendo github.com/MTG/mtg-jamendo-dataset
