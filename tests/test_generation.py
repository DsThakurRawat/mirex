"""Tests for the self-generation campaign package (plan P5, §4.2).

No network, no GPU, no API keys: taxonomy determinism, per-backend prompt
rendering, ledger resume semantics, retry/backoff behavior, honest client
failures without credentials, and metadata-DB registration.
"""
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generation.base import (BACKEND_FAMILY, FatalGenerationError,  # noqa: E402
                             GenerationError, GenerationJob, JobLedger,
                             register_output, with_retries)
from generation.minimax_client import MiniMaxClient  # noqa: E402
from generation.mureka_client import MurekaClient  # noqa: E402
from generation.prompts import (GENRE_TAGS, LANGUAGE_WEIGHTS,  # noqa: E402
                                MOOD_TAGS, PromptTaxonomy, RENDER_BACKENDS)
from metadata_db import MetadataDatabase  # noqa: E402

LANGS = {lang for lang, _ in LANGUAGE_WEIGHTS}


# --- taxonomy -------------------------------------------------------------
def test_taxonomy_deterministic_same_seed():
    a = list(PromptTaxonomy(seed=42).sample(100))
    b = list(PromptTaxonomy(seed=42).sample(100))
    assert a == b


def test_taxonomy_differs_across_seeds_and_indices():
    a = list(PromptTaxonomy(seed=42).sample(100))
    c = list(PromptTaxonomy(seed=43).sample(100))
    assert a != c
    assert len({json.dumps(s, sort_keys=True) for s in a}) == 100


def test_taxonomy_index_stability():
    """Style i must not depend on how many styles were drawn before it."""
    t = PromptTaxonomy(seed=42)
    assert t.style(57) == list(t.sample(1, start=57))[0]
    assert t.style(57) == list(t.sample(100))[57]


def test_taxonomy_field_validity():
    genre_vocab = {g for g, _ in GENRE_TAGS}
    mood_vocab = {m for m, _ in MOOD_TAGS}
    styles = list(PromptTaxonomy(seed=42).sample(300))
    for s in styles:
        assert 1 <= len(s["genre_tags"]) <= 3
        assert set(s["genre_tags"]) <= genre_vocab
        assert 1 <= len(s["mood_tags"]) <= 2
        assert set(s["mood_tags"]) <= mood_vocab
        assert 1 <= len(s["instrumentation"]) <= 3
        assert s["language"] in LANGS
        assert 30 <= s["duration_s"] <= 300
        assert isinstance(s["vocal"], bool)
        if s["vocal"]:
            assert s["vocal_gender"] in ("female", "male")
        else:
            assert s["vocal_gender"] is None
    inst_frac = sum(1 for s in styles if not s["vocal"]) / len(styles)
    assert 0.30 < inst_frac < 0.50      # target ~40% instrumental (plan §4.2)


# --- rendering ------------------------------------------------------------
@pytest.mark.parametrize("backend", RENDER_BACKENDS)
def test_render_nonempty_backend_specific(backend):
    t = PromptTaxonomy(seed=42)
    vocal = next(s for s in t.sample(50) if s["vocal"])
    instr = next(s for s in t.sample(50) if not s["vocal"])
    for style in (vocal, instr):
        r = t.render(style, backend)
        if backend == "yue":
            assert r["genre"].strip()
            assert r["lyrics"].strip()          # YuE is lyrics-to-song
            assert "[verse]" in r["lyrics"]
        else:
            assert r["prompt"].strip()
            assert any(g in r["prompt"] for g in style["genre_tags"])
    rv = t.render(vocal, backend)
    if backend == "ace_step":
        assert "[verse]" in rv["lyrics"]
        assert t.render(instr, backend)["lyrics"] == "[instrumental]"
    if backend in ("mureka", "minimax"):
        assert "[Verse]" in rv["lyrics"]
        assert len(rv["lyrics"]) >= 1
    if backend == "minimax":
        assert len(rv["lyrics"]) <= 3500 and len(rv["prompt"]) <= 2000


def test_render_unknown_backend_raises():
    t = PromptTaxonomy(seed=42)
    with pytest.raises(ValueError):
        t.render(t.style(0), "suno")


def test_lyrics_provider_hook():
    t = PromptTaxonomy(seed=42, lyrics_provider=lambda style: "[verse]\nLLM")
    style = next(s for s in t.sample(50) if s["vocal"])
    assert t.render(style, "ace_step")["lyrics"] == "[verse]\nLLM"


def test_lyrics_language_and_length():
    t = PromptTaxonomy(seed=42)
    seen = set()
    for s in t.sample(400):
        seen.add(s["language"])
        text = t.lyrics(s)
        assert text.count("[verse]") >= 1 and text.count("[chorus]") >= 1
    assert seen == LANGS                 # all 8 languages exercised


# --- ledger resume --------------------------------------------------------
def _jobs(n, backend="ace_step"):
    t = PromptTaxonomy(seed=42)
    return [GenerationJob(job_id=f"{backend}-{i:06d}", backend=backend,
                          style=t.style(i)) for i in range(n)]


def test_ledger_resume(tmp_path):
    path = tmp_path / "jobs.jsonl"
    jobs = _jobs(5)
    ledger = JobLedger(path)
    assert ledger.pending(jobs) == jobs
    ledger.mark_done(jobs[0].job_id, ["/x/a.wav"])
    ledger.mark_done(jobs[3].job_id, ["/x/b.wav"], meta={"k": 1})
    ledger.mark_failed(jobs[1].job_id, "boom")
    remaining = ledger.pending(jobs)
    assert [j.job_id for j in remaining] == [jobs[1].job_id, jobs[2].job_id,
                                             jobs[4].job_id]
    # Reload from disk: same plan (this is the actual resume path).
    ledger2 = JobLedger(path)
    assert {j.job_id for j in ledger2.pending(jobs)} == \
        {j.job_id for j in remaining}
    assert ledger2.status(jobs[0].job_id) == "done"
    assert ledger2.status(jobs[1].job_id) == "failed"
    # A failed job that later succeeds is no longer pending.
    ledger2.mark_done(jobs[1].job_id, ["/x/c.wav"])
    assert jobs[1].job_id not in {j.job_id for j in ledger2.pending(jobs)}


def test_ledger_ignores_corrupt_lines(tmp_path):
    path = tmp_path / "jobs.jsonl"
    path.write_text('{"job_id": "a", "status": "done"}\nnot json\n')
    assert JobLedger(path).done_ids() == {"a"}


# --- retry / backoff ------------------------------------------------------
def test_with_retries_eventually_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise GenerationError("transient")
        return "ok"

    assert with_retries(flaky, attempts=4, base_delay_s=0.0) == "ok"
    assert calls["n"] == 3


def test_with_retries_exhaustion_raises():
    with pytest.raises(GenerationError, match="All 3 attempts"):
        with_retries(lambda: (_ for _ in ()).throw(GenerationError("x")),
                     attempts=3, base_delay_s=0.0)


def test_with_retries_fatal_not_retried():
    calls = {"n": 0}

    def fatal():
        calls["n"] += 1
        raise FatalGenerationError("bad key")

    with pytest.raises(FatalGenerationError):
        with_retries(fatal, attempts=5, base_delay_s=0.0)
    assert calls["n"] == 1


# --- API clients: honest failure without keys, no network -----------------
def _rendered_job(vocal=True):
    t = PromptTaxonomy(seed=42)
    style = next(s for s in t.sample(50) if s["vocal"] is vocal)
    job = GenerationJob(job_id="test-000000", backend="mureka", style=style)
    return job, style, t


def test_mureka_raises_without_key(monkeypatch):
    monkeypatch.delenv("MUREKA_API_KEY", raising=False)
    client = MurekaClient(api_key="")
    job, style, t = _rendered_job()
    with pytest.raises(FatalGenerationError, match="MUREKA_API_KEY"):
        client.run(job, t.render(style, "mureka"))
    with pytest.raises(FatalGenerationError):
        client.submit("pop", "[Verse]\nla", vocal=True)


def test_minimax_raises_without_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    client = MiniMaxClient(api_key="")
    job, style, t = _rendered_job()
    job.backend = "minimax"
    with pytest.raises(FatalGenerationError, match="MINIMAX_API_KEY"):
        client.run(job, t.render(style, "minimax"))


# --- metadata DB registration (pinned contract) ---------------------------
def test_register_output_contract(tmp_path):
    db = MetadataDatabase(db_path=tmp_path / "meta.db")
    t = PromptTaxonomy(seed=42)
    job = GenerationJob(job_id="ace_step-000007", backend="ace_step",
                        style=t.style(7))
    settings = {"infer_step": 27, "guidance_scale": 7.5}
    ids = register_output(db, job, [tmp_path / "ace_step-000007.wav"],
                          settings, generator_version="v1-3.5B")
    assert ids == ["gen_ace_step:ace_step-000007"]
    rows = db.fetch("source_dataset=?", ("gen_ace_step",))
    assert len(rows) == 1
    row = rows[0]
    assert row["is_ai"] == 1
    assert row["generator_family"] == BACKEND_FAMILY["ace_step"] == "ace-step"
    assert row["generator_version"] == "v1-3.5B"
    extra = json.loads(row["extra_json"])
    assert extra["style"] == job.style
    assert extra["settings"] == settings
    # Multi-choice outputs (Mureka) get suffixed unique track_ids.
    job2 = GenerationJob(job_id="mureka-000001", backend="mureka",
                         style=t.style(1))
    ids2 = register_output(db, job2, ["/x/a.mp3", "/x/b.mp3"], {})
    assert ids2 == ["gen_mureka:mureka-000001#0", "gen_mureka:mureka-000001#1"]
    assert len(db.fetch("source_dataset=?", ("gen_mureka",))) == 2
