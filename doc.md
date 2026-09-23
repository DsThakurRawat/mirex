# The Landscape of Voice, Speech, Audio & Music AI/ML Competitions, Challenges, and Conferences (2026 Edition)

## TL;DR
- **MIREX is not the only game in town — there is a large, active ecosystem of MIREX-like shared-task/evaluation campaigns.** The closest analogues are DCASE (acoustic events), Blizzard/VoiceMOS (TTS), the Voice Conversion Challenge/SVCC, ASVspoof (anti-spoofing/deepfake), CHiME (robust ASR), the Sound Demixing Challenge (music separation), the Odyssey/MSP-Podcast emotion challenges, ADReSS/ADReSSo (health), and standing leaderboards like SUPERB/ML-SUPERB/HEAR.
- **MIREX itself was revived**: after a dormant period it now runs as a community-led "Future MIREX" effort (Queen Mary University of London prominent), with MIREX 2025 held alongside ISMIR 2025 in Daejeon, Korea — the tradition continues, plus a new 2025 Automatic Music Transcription (AMT) Challenge.
- **As of late August 2026, the challenges you can still act on** are mostly tied to 2026/2027 conference cycles: BirdCLEF+ 2026 just closed (June 2026), DCASE 2026 results are out, and ASVspoof/CHiME‑9/ICASSP‑2026 cycles have concluded, so your next actionable windows are **ISMIR/MIREX 2026 (Abu Dhabi, Nov 2026, late-breaking demos due 25 Sept 2026)**, the **India-focused NCC 2026 (IIT Hyderabad, Feb–Mar 2027 cycle already dated 26 Feb–1 Mar 2026)**, and the **2027 ICASSP/Interspeech/DCASE challenge cycles** that open in late 2026.

## Key Findings

MIREX (Music Information Retrieval Evaluation eXchange) belongs to a broad family of "evaluation-as-a-service" campaigns in audio ML: organizers release common datasets and a hidden test set, participants submit systems, and results are presented at an affiliated conference. The pattern repeats across every subfield. The most direct structural analogues to MIREX — annual, conference-affiliated, task-based, with published overview papers — are **DCASE, ASVspoof, CHiME, the Blizzard Challenge, the Voice Conversion Challenge, DIHARD, ZeroSpeech, IWSLT, ComParE, the ICASSP Signal Processing Grand Challenges, and the Sound Demixing Challenge.**

A crucial finding for your primary question: **MIREX did not die.** It was revived as a community-run "Future MIREX" effort with strong involvement from Queen Mary University of London's UKRI Centre for Doctoral Training in AI and Music. MIREX 2025 ran and presented at ISMIR 2025 in Daejeon, Korea, adding new tasks such as Music Reasoning QA and an Expressive Piano Performance Rendering contest (RenCon) alongside classic tasks (beat tracking, key detection). Note: "MIREX Next Generation" is a distinct, older (2011–2013) University of Illinois project led by J. Stephen Downie and should not be confused with the current revival.

## Comparison Summary Table

| Challenge / Benchmark | Focus area | Affiliated venue | Status (Aug 2026) | Typical deadline window |
|---|---|---|---|---|
| **MIREX / Future MIREX** | Music information retrieval (beat, key, transcription, music QA) | ISMIR | Active (revived; MIREX 2025 ran) | Mid-year; results at ISMIR (Nov) |
| **AMT Challenge** | Multi-instrument music transcription | ISMIR-adjacent | Active (2025 edition) | 2025 cycle |
| **DCASE Challenge** | Acoustic scenes & events, anomaly detection, sound separation | DCASE Workshop | Active; DCASE 2026 concluded | Tasks launch ~Apr, deadline ~15 Jun |
| **Sound Demixing (SDX/MDX)** | Music/cinematic source separation | ISMIR / AIcrowd | Intermittent (2021, 2023 editions) | Winter–spring when run |
| **Singing Voice Deepfake Detection (SVDD)** | AI-singer detection | IEEE SLT / Interspeech | Active (2024 inaugural) | Spring–summer |
| **Blizzard Challenge** | Text-to-speech synthesis | Speech Synthesis Workshop (SSW) | Active (2025 = Bildts) | Data ~spring; workshop late year |
| **Voice Conversion Challenge / SVCC** | Voice/singing-voice conversion | Interspeech / ASRU / SSW | Intermittent (VCC last 2020; SVCC 2023) | When run |
| **VoiceMOS Challenge** | Predicting MOS/quality of synthetic speech | Interspeech / ASRU | Active (2022–2024) | Spring–summer |
| **ASVspoof / SASV** | Anti-spoofing, audio deepfake detection | Interspeech / ASVspoof Workshop | Active (ASVspoof 5 = 2024) | Multi-phase; ~1–2 yr cadence |
| **VoxSRC** | Speaker recognition & diarization | Interspeech | Discontinued (2019–2023) | Was summer |
| **NIST SRE / LRE / OpenASR / OpenSAT** | Speaker/language recognition, low-resource ASR | NIST-run | Active (periodic) | Announced per cycle |
| **CHiME** | Far-field / multi-talker robust ASR | ICASSP/Interspeech satellite | Active (CHiME-9 = 2026) | Data ~late prior year; workshop mid-year |
| **DIHARD** | Speaker diarization ("hard" conditions) | Interspeech | Dormant (last DIHARD III, 2020–21) | Was winter |
| **ZeroSpeech** | Zero-resource / unsupervised speech | Interspeech / NeurIPS | Intermittent | When run |
| **IWSLT** | Spoken language translation | ACL (IWSLT) | Active (2026 = 23rd edition) | Data late year; papers spring |
| **ComParE** | Computational paralinguistics | Interspeech / ACM MM | Active (annual since 2009) | Spring |
| **Odyssey / MSP-Podcast SER** | Speech emotion recognition | Odyssey / Interspeech | Active (2024 Odyssey; 2025 Interspeech) | Winter–spring |
| **AVEC** | Audio/visual emotion | ACM MM | Discontinued (last ~2019) | — |
| **ICASSP SP Grand Challenges (DNS, AEC, etc.)** | Speech enhancement, echo cancellation, etc. | ICASSP | Active (2026 SPGCs ran) | Proposals ~mid prior year |
| **ADReSS / ADReSSo / ADReSS-M** | Alzheimer's/dementia from speech | Interspeech / ICASSP | Active periodically (2020, 2021, 2023) | Spring |
| **DiCOVA / COVID sound** | Respiratory disease from sound | Interspeech | Dormant (2021 peak) | — |
| **MADASR** | Low-resource Indian-language ASR | IEEE ASRU | Active (MADASR 2.0 = ASRU 2025) | Mid-year |
| **BirdCLEF+ / LifeCLEF** | Bioacoustics species ID | CLEF / Kaggle | Active (BirdCLEF+ 2026 ran) | Entry ~May, final ~Jun |
| **DSTC** | Dialogue system technology | Various | Active (long-running) | Rolling |
| **SUPERB / ML-SUPERB / HEAR** | Speech/audio representation benchmarks | Interspeech/SLT/NeurIPS | Active (leaderboards) | Rolling |

## Details

### 1. Music Information Retrieval (the MIREX family)

**MIREX / Future MIREX** — the original evaluation campaign, coordinated since 2005 by IMIRSEL at the University of Illinois (J. Stephen Downie), evaluating over 1,068 algorithms across 23 unique MIR task categories (melody extraction, beat tracking, chord estimation, cover-song ID, etc.). Because music recordings cannot be redistributed for copyright reasons, MIREX pioneered the "submit your executable, we run it on the hidden corpus" model. After a dormant period it was revived as a community-led "Future MIREX" effort (Queen Mary University of London's AIM CDT prominent), with MIREX 2025 running in conjunction with ISMIR 2025 (Daejeon, Korea) and introducing new tasks including Music Reasoning QA and RenCon (expressive piano performance rendering). Official site: music-ir.org/mirex. Contact: future-mirex@googlegroups.com.

**AMT Challenge 2025** — a 2025 Automatic Music Transcription challenge (multi-instrument) that explicitly builds on MIREX's transcription lineage (drum transcription, polyphonic piano transcription). Documented in an arXiv overview paper ("Advancing Multi-Instrument Music Transcription: Results from the 2025 AMT Challenge").

**Sound Demixing Challenge (SDX) / Music Demixing (MDX)** — music source separation (separating a song into vocals/bass/drums/other) plus cinematic sound separation (dialogue/effects/music). Sponsored by Sony Group Corporation, hosted on AIcrowd. Editions: Music Demixing Challenge 2021 (MDX'21) and SDX'23 (added a cinematic track plus "robust MSS" using the SDXDB23_LabelNoise and SDXDB23_Bleeding datasets). Overview papers appear in TISMIR and Frontiers in Signal Processing; it follows the older SiSEC MUS tradition. Independent leaderboards with prize pools and open tracks. URL: aicrowd.com.

**Singing Voice Deepfake Detection (SVDD) Challenge** — detecting AI-generated singing voices; two tracks (CtrSVDD controlled, WildSVDD in-the-wild, built on the SingFake dataset). Inaugural edition held with IEEE SLT 2024: per Zhang et al. (arXiv 2408.16132), "For the CtrSVDD track, we received submissions from 47 teams, with 37 surpassing our baselines and the top team achieving a 1.65% equal error rate" (baselines B01/B02 achieved 12.03% and 11.16% EER). URL: svddchallenge.org.

Other music-specific efforts: MedleyDB-based tasks, MIDI/symbolic music generation challenges, and the AI Song Contest (a creative rather than benchmark competition).

### 2. Speech & Speaker (recognition, diarization, robustness)

**CHiME** — the flagship far-field/multi-talker robust ASR series since 2011. CHiME-9 (2026) is held jointly with the HSCMA workshop as an ICASSP 2026 satellite in Barcelona (workshop 4 May 2026). CHiME-9 Task 1 (MCoRec — Multi-Modal Context-aware Recognition) targets transcription of overlapping conversations using audio+video; organizers include teams from KIT, Meta, and CMU (Thai-Binh Nguyen, Katerina Zmolikova, Pingchuan Ma, Christian Fuegen, Shinji Watanabe, Alexander Waibel). URL: chimechallenge.org.

**VoxSRC (VoxCeleb Speaker Recognition Challenge)** — ran annually 2019–2023 at Interspeech (Oxford VGG + partners: Jaesung Huh, Joon Son Chung, Arsha Nagrani, Andrew Zisserman et al.), covering speaker verification and diarization "in the wild." Now **discontinued**; a retrospective was published in IEEE/ACM TASLP (2024). Data and evaluation lists remain public. URL: mm.kaist.ac.kr/datasets/voxceleb/voxsrc.

**NIST SRE / LRE / OpenASR / OpenSAT** — U.S. NIST-run periodic evaluations for speaker recognition, language recognition, low-resource ASR, and speech analytics. Unlike VoxSRC, NIST does not release all training data publicly. Announced per cycle at nist.gov.

**DIHARD** — "hard" speaker diarization (audiobooks, restaurants, clinical interviews). Last major edition was DIHARD III (2020–2021); appears **dormant**.

### 3. Speech Synthesis & Voice Conversion

**Blizzard Challenge** — the premier TTS evaluation, running since 2005, decided by large listening tests. The 2025 edition was the nineteenth. Per the official overview (Do, Coler, Dijkstra et al., ISCA Archive), it "focused on speech synthesis for Bildts, a low-resource language variety in the Netherlands," providing "a single-speaker 7-hour Bildts dataset" (from one male speaker, Jan de Groot of Omrop Fryslân) with an optional zero-shot task; "seven teams participated in the main task and five in the zero-shot task." (The overview paper cites ~6,000 native speakers of Bildts; the official SSW13/Fryske Akademy materials cite ~10,000 first- and second-language speakers.) Co-located with the Speech Synthesis Workshop 2025 (SSW13) in Leeuwarden. URL: blogs.helsinki.fi/ssw13-2025.

**Voice Conversion Challenge (VCC)** — ran 2016, 2018, and 2020; the singing-voice spinoff **SVCC 2023** ran at ASRU 2023. VCC is currently intermittent/dormant with no announced 2025/2026 edition.

**VoiceMOS Challenge** — predicting mean-opinion-score (MOS) quality of synthetic/converted speech; editions 2022–2024 (e.g., the T05 system for VoiceMOS 2024), associated with Interspeech/ASRU. Related toolkit: VERSA (a versatile evaluation toolkit for speech, audio, and music).

### 4. Anti-Spoofing & Deepfake Detection

**ASVspoof** — the anchor series for spoofing/deepfake countermeasures, born from an Interspeech 2013 special session. **ASVspoof 5 (2024)** was the fifth edition. Per Wang et al. (arXiv 2502.08857 / Computer Speech & Language Vol. 95, 2025), the database was "generated in a crowdsourced fashion... from ~2,000 speakers (cf. ~100 earlier)" and "contains attacks generated with 32 different algorithms," with adversarial attacks "incorporated for the first time"; the results overview (arXiv 2408.08739) provides "an overview of the ASVspoof 5 challenge results for the submissions of 53 participating teams." New metrics support both stand-alone countermeasures and spoofing-robust ASV (SASV). The organizers explicitly "outline a road-map for the future of ASVspoof." URL: asvspoof.org.

**SASV (Spoofing-Aware Speaker Verification)** — integrates anti-spoofing with speaker verification; first ran 2022 and is now folded into ASVspoof 5 metrics.

**ADD (Audio Deepfake Detection Challenge)** — a China-based series extending attack scenarios (TTS, VC, replay) in realistic conditions.

### 5. General Audio, Acoustic Events & Bioacoustics

**DCASE (Detection and Classification of Acoustic Scenes and Events)** — the leading general-audio campaign, run by a Steering Group (coordinators Annamaria Mesaros, Tampere University; Romain Serizel, LORIA), with an affiliated DCASE Workshop (the eleventh edition accompanies DCASE 2026). Per the DCASE Steering Group (dcase.community, 19 January 2026), for DCASE 2026 it "selected... seven tasks" — including domain-agnostic incremental learning for sound classification, noise-aware unsupervised anomalous sound detection for machine condition monitoring (Task 2), and spatial semantic segmentation of sound scenes (Task 4). The 2026 schedule: task descriptions 1 Feb, challenge launch 1 Apr, evaluation data 1 Jun, deadline 15 Jun, results 30 Jun. URL: dcase.community.

**BirdCLEF+ / LifeCLEF** — bioacoustic species identification hosted on Kaggle within the LifeCLEF lab (working notes at CLEF). BirdCLEF+ 2026 focuses on Brazil's Pantanal wetlands with 234 multi-taxon classes (birds, amphibians, insects, reptiles); timeline: start 11 Mar 2026, entry/team-merger deadline 27 May 2026, final submission 3 Jun 2026, scored by macro-averaged ROC-AUC over hidden 5-second soundscape windows with a CPU-only inference budget. URL: kaggle.com/competitions/birdclef-2026.

### 6. Spoken Language Understanding, Translation & Dialogue

**IWSLT (International Conference on Spoken Language Translation)** — the premier speech-translation evaluation campaign, with a 22+ year track record; venue of ACL/ISCA/ELRA's SIGSLT. The 23rd edition (IWSLT 2026) was co-located with ACL 2026 in San Diego (July 2026), covering ten tasks (offline/simultaneous ST, low-resource/dialectal ST, speech generation, instruction-following speech processing, evaluation metrics), with 30+ teams. The call for 2026 tasks closed 30 Sept 2025. URL: iwslt.org.

**DSTC (Dialog System Technology Challenge)** — long-running dialogue-systems challenge series (originally Dialog State Tracking), with multiple tracks per edition. **Alexa Prize** (Amazon) SocialBot / TaskBot / SimBot university challenges are industry-run conversational-AI competitions.

### 7. Health & Paralinguistics

**ComParE (Computational Paralinguistics Challenge)** — held at Interspeech every year since 2009 (organizer Björn Schuller and colleagues), covering states/traits from the speech signal (emotion, breathing, masks, COVID cough/speech, requests/complaints). It has also appeared at ACM Multimedia (the 15th ComParE was at ACM MM 2023). Uses standardized openSMILE/BoAW/auDeep/DeepSpectrum baselines.

**Odyssey / MSP-Podcast Speech Emotion Recognition** — the Odyssey 2024 Emotion Recognition Challenge (organized by Carlos Busso's lab) used the MSP-Podcast corpus with categorical (8-class) and attribute (arousal/valence/dominance) tasks. It continued as the "SER in Naturalistic Conditions Challenge" at Interspeech 2025 (MSP-Podcast, 324+ hours of naturalistic conversational speech). URLs: github.com/msplabresearch.

**ADReSS / ADReSSo / ADReSS-M** — Alzheimer's dementia detection from spontaneous speech, organized by Saturnino Luz, Fasih Haider, Brian MacWhinney and colleagues. ADReSS at Interspeech 2020 (34 teams), ADReSSo at Interspeech 2021 (speech-only; 237-sample dataset), ADReSS-M at ICASSP 2023 (multilingual, English→Greek transfer, published in IEEE Open Journal of Signal Processing). Datasets openly available.

**DiCOVA / COVID sound challenges** — respiratory-disease detection from cough/breath/speech, peaked ~2021; now largely dormant.

**Bridge2AI Voice** — an NIH-funded U.S. effort building a large, ethically-sourced voice-as-biomarker dataset for clinical AI (a major dataset initiative rather than a competition).

**AVEC (Audio/Visual Emotion Challenge)** — ran at ACM Multimedia through ~2019; now discontinued.

### 8. Benchmarks & Leaderboards (standing, not annual)

**SUPERB (Speech processing Universal PERformance Benchmark)** — a leaderboard for self-supervised speech models across many tasks (ASR, speaker ID, emotion, etc.), with the S3PRL toolkit; a SUPERB challenge ran at SLT 2023 (12 submissions, adding memory/compute efficiency evaluation). **SUPERB-SG** extends to generation/semantics. **ML-SUPERB** covers 143 languages (ASR + language ID); the 2023 ML-SUPERB Challenge extended this. URL: superbbenchmark.org.

**HEAR (Holistic Evaluation of Audio Representations)** — NeurIPS 2021 benchmark evaluating audio representations across speech, music, and environmental sound. Newer benchmarks: DASB (discrete audio tokens), X-ARES, AV-SUPERB (audio-visual), MARBLE (music), and the ICME 2025 Audio Encoder Capability Challenge.

### 9. India-Specific Efforts (highlighted for the user)

India has become a major hub for low-resource/multilingual speech AI, largely driven by the government's National Language Translation Mission (Bhashini, MeitY) and academic labs:

- **MADASR (Multi-lingual, multi-dialect ASR) Challenge** — the closest Indian analogue to a MIREX-style shared task. MADASR 1.0 ran at ASRU 2023 (Bengali, Bhojpuri; ~850 hours, four tracks). **MADASR 2.0 at IEEE ASRU 2025** introduced a subset of the RESPIN corpus — over 1,200 hours of read speech across 8 Indian languages (Bengali, Bhojpuri, Chhattisgarhi, Kannada, Magahi, Maithili, Marathi, Telugu) and 33 dialects, evaluated by WER, CER, language-ID accuracy and dialect-ID accuracy across four tracks (arXiv 2511.15418; accepted Proc. ASRU 2025). Organized by SPIRE/SPRING Lab (S. Umesh), IIT Madras. URL: sites.google.com/view/respinasrchallenge2025.
- **IndicSUPERB** (AI4Bharat, IIT Madras) — a 6-task speech-understanding benchmark across 12 Indian languages, including the Kathbath dataset (1,684 hours). GitHub: AI4Bharat/IndicSUPERB.
- **SPRING-INX** (SPRING Lab, IIT Madras) — ~2,000 hours of legally sourced, manually transcribed speech across 10 Indian languages (NLTM/MeitY-funded).
- **Project Vaani** (IISc + Google/ARTPARK) — an India-representative multimodal dataset; ~31,270 hours of audio and ~2,067 hours of transcribed speech open-sourced (plus ~289K images), across 165+ districts.
- **IndicVoices / IndicVoices-R** (AI4Bharat) — inclusive multilingual speech datasets (IndicVoices-R released at the NeurIPS 2024 Datasets & Benchmarks track).
- **Bhashini** — the national AI language platform (MeitY); runs pilots and hackathons rather than a single benchmark (e.g., multilingual support at Maha Kumbh 2025).
- **Gram Vaani ASR Challenge** — spontaneous telephone-speech Hindi in regional variations (Interspeech 2022).

Indian research venues: **NCC (National Conference on Communications)** — the 32nd edition is at IIT Hyderabad, 26 Feb–1 Mar 2026 (Joint Telematics Group of the IITs/IISc; co-organized with IIT Bhilai, IIIT Hyderabad, IIIT Naya Raipur; proceedings on IEEE Xplore) — has active speech/signal tracks. Official: ee.iith.ac.in/NCC2026. **WiSSAP (Winter School on Speech and Audio Processing)** — an advanced school under ISCA's SIG-ILSP/IndSCA; the most recent confirmed edition was WiSSAP 2024 at KL University, Andhra Pradesh (9–12 Dec 2024, "Multilingual Speech Processing of Indic Languages"); no 2025/2026 edition could be verified. Indian institutions also run national hackathons under Bhashini and the Smart India Hackathon.

### 10. Research Conferences, Workshops & Journals

**Conferences / workshops:**
- **ISMIR** — the music-IR flagship. ISMIR 2026 (27th) is in Abu Dhabi, UAE (Nov 8–12, 2026), theme "Crossroads"; the paper submission deadline was 27 April 2026, late-breaking demos due 25 Sept 2026 (limited to 75 posters). ISMIR 2027 will be in London (Sept 2027). URL: ismir2026.ismir.net.
- **Interspeech** — the largest speech conference (ISCA). Interspeech 2026 is in Sydney, Australia (28 Sep–1 Oct 2026; some listings show 27 Sep opening), theme "Diversity & Equity – Speaking Together"; paper deadline 25 Feb 2026, with a new "Long Paper" track. Interspeech 2027 = São Paulo, Brazil (29 Aug–2 Sep 2027). URL: interspeech2026.org.
- **ICASSP** — IEEE flagship on acoustics/speech/signal processing. ICASSP 2026 is in Barcelona (spring 2026) and hosts the SP Grand Challenges (2026 SPGCs included the RASE radar-speech and Hyper-Object challenges). Paper deadlines are typically September/October of the prior year. Grand-Challenge top teams present 2-page papers and may extend to IEEE OJ-SP.
- **IEEE ASRU** (Automatic Speech Recognition & Understanding) — biennial; ASRU 2025 hosted MADASR 2.0.
- **IEEE SLT** (Spoken Language Technology) — biennial; SLT 2024 hosted SVDD.
- **IEEE WASPAA** — Workshop on Applications of Signal Processing to Audio and Acoustics (biennial, Mohonk, NY).
- **Odyssey** — The Speaker and Language Recognition Workshop (ISCA), which hosts the emotion-recognition challenges.
- **DCASE Workshop** — accompanies the DCASE Challenge.
- **SMC** (Sound and Music Computing), **DAFx** (Digital Audio Effects), **NIME** (New Interfaces for Musical Expression), **AES** conventions/conferences, **EUSIPCO**, **ICA**, **Forum Acusticum** — music/audio-engineering and acoustics venues.
- **SSW** (Speech Synthesis Workshop) — hosts Blizzard.
- **SPSC** (Security and Privacy in Speech Communication symposium).
- **Machine Learning for Audio workshops** at NeurIPS/ICML/ICLR; audio/music tracks at **ACM Multimedia**; speech-related tracks at **ACL/EMNLP**.

**Journals:**
- **IEEE/ACM TASLP** (Transactions on Audio, Speech, and Language Processing) — the field's leading journal.
- **Computer Speech & Language** (Elsevier).
- **Speech Communication** (Elsevier).
- **TISMIR** (Transactions of the ISMIR) — open access, music IR.
- **JAES** (Journal of the Audio Engineering Society).
- **EURASIP Journal on Audio, Speech, and Music Processing** (open access).
- **IEEE Open Journal of Signal Processing (OJ-SP)** — publishes ICASSP Grand Challenge and ADReSS-M overviews.

## Recommendations

**If you want to participate right now (late Aug 2026), the realistic near-term targets are:**
1. **DCASE 2027 and ICASSP 2027 Grand Challenges** — watch dcase.community (call for task proposals appears ~Dec; challenge launches ~Feb–Apr) and the ICASSP 2027 site. These are the most beginner-accessible, with open data, baselines, and clear leaderboards.
2. **BirdCLEF+ 2027 on Kaggle** — if you like bioacoustics/general audio ML; expect a Mar–Jun 2027 window. Kaggle's CPU-only inference constraint makes it approachable without massive compute.
3. **MIREX / ISMIR 2026 (Abu Dhabi, Nov 2026)** — since MIREX is your anchor interest, join the Future MIREX Google group (future-mirex@googlegroups.com) and consider the ISMIR 2026 late-breaking demo track (deadline 25 Sept 2026) if you have preliminary music-AI results.
4. **Interspeech 2027 challenges (São Paulo)** — the ComParE, MSP-Podcast/emotion, and various special-session challenges reopen annually; watch for calls in late 2026/early 2027.
5. **India-specific: MADASR follow-ups** — if low-resource Indian-language ASR is your interest, follow SPRING/SPIRE Lab IIT Madras and AI4Bharat; the next ASRU cycle (2027) is the likely venue. Attend **NCC 2026 at IIT Hyderabad (26 Feb–1 Mar 2026)** to network locally.

**Staged plan:**
- *Stage 1 (now–Oct 2026):* Pick ONE subfield matching your interest (music vs. speech vs. general audio). Reproduce a baseline from a recently-closed challenge (DCASE 2026, BirdCLEF+ 2026, ASVspoof 5) using its public data — these remain downloadable and are excellent practice.
- *Stage 2 (Nov 2026–Feb 2027):* Enter a live 2027-cycle challenge with open registration. Threshold that should change your plan: if you beat the official baseline on the development set, commit to a full submission; if not, pivot to a standing benchmark (SUPERB/HEAR/ML-SUPERB) where you can iterate without a fixed deadline.
- *Stage 3 (2027):* Convert a strong submission into a system-description paper at the affiliated workshop (DCASE/ASVspoof/CHiME/Odyssey), which is the standard publication incentive.

**Decision thresholds:** Choose competition vs. standing benchmark based on compute. If you have limited GPUs, prefer Kaggle challenges (CPU inference caps level the field) and frozen-feature benchmarks (SUPERB) over CHiME/ASVspoof (which reward large-scale training).

## Caveats
- **Timelines shift year to year.** Exact 2027 dates were not all announced as of late August 2026; treat forward-looking dates as projections based on historical cadence, and verify on official sites before relying on them.
- **"Active vs. dormant" is a judgment call.** Several series (VCC, DIHARD, AVEC, DiCOVA, ZeroSpeech) have no announced current edition but could return; they are flagged as intermittent/dormant based on the absence of recent editions, not an official cancellation.
- **MIREX dormancy window unconfirmed.** The precise years MIREX paused before its community revival could not be pinned to a primary source; the confirmed facts are that it ran since 2005 and that MIREX 2025 ran under the Future MIREX effort at ISMIR 2025 (Daejeon, Korea).
- **A DCASE 2026 "421 submissions" figure** appeared in a site snippet but could not be independently corroborated; treat it as unverified. The Blizzard 2025 Bildts speaker count differs between sources (~6,000 in the overview paper vs. ~10,000 on the official SSW13 site).
- **WiSSAP 2025/2026 unverified.** The most recent confirmed Winter School on Speech and Audio Processing was December 2024 (KL University); a later edition may exist without a strong web presence.
- **Some arXiv identifiers referenced here carry 2026 date-stamps** (e.g., DCASE 2026, ASVspoof 5 revisions, IWSLT 2026, AMT 2025), consistent with the current date; a few are preprints/working notes rather than final published papers.
- **Beware predatory "conference" listings**, especially numerous fake "National Conference on Communications" events in India; the authoritative NCC is the IIT/IISc Joint Telematics Group event on IEEE Xplore.
- This report prioritizes breadth per your request; for any single challenge you decide to enter, consult its official site for binding rules, licensing, and current deadlines.