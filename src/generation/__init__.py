"""Self-generation campaign package (plan §4.2 prompt strategy, phase P5).

Produces exact-decoder training data for four of the six named hidden-test
generator families:

- ACE-Step v1-3.5B  (local GPU)   -> ace_step_runner.AceStepRunner
- YuE-7B            (local GPU)   -> yue_runner.YueRunner
- Mureka            (paid API)    -> mureka_client.MurekaClient
- MiniMax Music     (paid API)    -> minimax_client.MiniMaxClient

Prompt sampling targets the MTG-Jamendo tag distribution (the hidden real
class is Jamendo-flavored), varies language, vocal/instrumental, duration and
sampler settings, and is fully deterministic given ``config.SEED`` so the
campaign is reproducible for the paper (§13, contribution 4).

Entry point::

    cd src && python -m generation.campaign --backend ace_step --count 20000

Every finished track is registered in ``metadata_db.MetadataDatabase`` with
``source_dataset="gen_<backend>"``, ``is_ai=1`` and the full style + sampler
settings in ``extra_json``.
"""
from generation.base import GenerationJob, JobLedger  # noqa: F401
from generation.prompts import PromptTaxonomy  # noqa: F401
