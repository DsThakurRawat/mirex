"""Prompt taxonomy for the self-generation campaign (plan §4.2, phase P5).

Target distribution = MTG-Jamendo tag vocabulary, because the hidden real
class is "Jamendo-flavored" CC music (plan §1, §4.2): we want our AI tracks to
cover the same genre/mood/instrument space so the detector cannot use topic or
genre as a shortcut (plan F2).

Tag constants below are embedded from the MTG-Jamendo dataset repository
(github.com/MTG/mtg-jamendo-dataset), file
``stats/raw_30s_cleantags_50artists/{genre,mood_theme,instrument}.tsv``
(fetched 2026-08-08): top 60 of 87 genre tags, top 40 of 56 mood/theme tags,
and the most frequent instrument tags, each with its track count used as a
sampling weight.

Determinism: every style is derived from ``random.Random(f"{seed}:{index}")``,
so style ``i`` is stable regardless of how many styles are drawn before it,
across processes and platforms. Lyrics are template-based (multi-verse, eight
languages) with no LLM dependency at runtime; plug an LLM later via the
``lyrics_provider`` hook.
"""
from __future__ import annotations

import random
from typing import Callable, Iterator, Optional

# --- MTG-Jamendo genre tags: top 60 by track count (§4.2) -----------------
GENRE_TAGS: list[tuple[str, int]] = [
    ("electronic", 16480), ("soundtrack", 8094), ("pop", 7805),
    ("ambient", 7570), ("rock", 6865), ("classical", 5602),
    ("easylistening", 4833), ("experimental", 3941), ("alternative", 3761),
    ("chillout", 3678), ("dance", 2827), ("hiphop", 2657), ("indie", 2632),
    ("folk", 2498), ("orchestral", 2434), ("jazz", 2371), ("lounge", 2259),
    ("newage", 2202), ("techno", 2179), ("poprock", 2172), ("house", 2169),
    ("world", 1887), ("popfolk", 1808), ("trance", 1528),
    ("instrumentalpop", 1477), ("metal", 1435), ("downtempo", 1431),
    ("atmospheric", 1395), ("triphop", 1343), ("funk", 1283),
    ("reggae", 1245), ("blues", 1082), ("progressive", 1078),
    ("electropop", 1003), ("rap", 993), ("singersongwriter", 776),
    ("punkrock", 729), ("symphonic", 712), ("latin", 704),
    ("industrial", 696), ("synthpop", 635), ("minimal", 629),
    ("instrumentalrock", 598), ("psychedelic", 590), ("country", 584),
    ("club", 579), ("rnb", 568), ("dubstep", 547), ("fusion", 512),
    ("darkambient", 509), ("soul", 504), ("drumnbass", 501),
    ("hardrock", 490), ("disco", 447), ("dub", 445), ("ethno", 429),
    ("deephouse", 427), ("breakbeat", 382), ("postrock", 377),
    ("grunge", 375),
]

# --- MTG-Jamendo mood/theme tags: top 40 by track count -------------------
MOOD_TAGS: list[tuple[str, int]] = [
    ("happy", 1657), ("film", 1502), ("energetic", 1357),
    ("relaxing", 1350), ("emotional", 1271), ("melodic", 1213),
    ("dark", 1202), ("epic", 982), ("dream", 951), ("love", 909),
    ("inspiring", 877), ("sad", 749), ("meditative", 742),
    ("uplifting", 693), ("advertising", 673), ("motivational", 635),
    ("deep", 635), ("romantic", 627), ("christmas", 623),
    ("documentary", 612), ("corporate", 609), ("positive", 539),
    ("summer", 505), ("space", 503), ("background", 496), ("fun", 480),
    ("soundscape", 480), ("soft", 465), ("ambiental", 460), ("calm", 457),
    ("children", 456), ("adventure", 448), ("upbeat", 444),
    ("melancholic", 441), ("slow", 437), ("commercial", 428),
    ("drama", 424), ("movie", 413), ("action", 407), ("ballad", 334),
]

# --- MTG-Jamendo instrument tags (top of 40, human-readable forms) --------
INSTRUMENT_TAGS: list[tuple[str, int]] = [
    ("piano", 7750), ("synthesizer", 7496), ("drums", 6120),
    ("bass", 5726), ("guitar", 4804), ("electric guitar", 4241),
    ("keyboard", 2044), ("acoustic guitar", 1947), ("violin", 1837),
    ("drum machine", 1764), ("electric piano", 1282), ("strings", 1278),
    ("cello", 1114), ("sampler", 1070), ("percussion", 908),
    ("saxophone", 891), ("trumpet", 873), ("flute", 859),
    ("classical guitar", 721), ("orchestra", 640), ("harp", 589),
    ("rhodes", 543), ("brass", 518), ("accordion", 416),
]

# --- Language mix (§4.2: "lyrics ... in multiple languages") --------------
# zh is up-weighted: Muse (Suno v5) is CN+EN and Mureka/MiniMax are
# Chinese-market platforms, so the hidden AI strata plausibly skew zh+en.
LANGUAGE_WEIGHTS: list[tuple[str, float]] = [
    ("en", 0.38), ("zh", 0.14), ("es", 0.10), ("fr", 0.09), ("de", 0.08),
    ("it", 0.07), ("pt", 0.07), ("ja", 0.07),
]

INSTRUMENTAL_PROB = 0.40      # vocal/instrumental is an eval stratum (§1)
DURATION_RANGE_S = (30, 300)  # §4.2: vary duration 30 s–5 min
VOCAL_TIMBRES = ["bright", "airy", "warm", "deep", "soft"]

LYRICS_THEMES = ["love", "loss", "night city", "ocean", "freedom", "memory",
                 "rain", "open road", "stars", "home", "dance", "seasons"]

# --- Per-language stanza templates and word banks -------------------------
# Slots: {theme} {noun} {place} {feeling}. Deliberately simple: these are
# generation prompts, not poetry; an LLM can replace them via lyrics_provider.
_LANG_BANK: dict[str, dict[str, list[str]]] = {
    "en": {
        "verse": [
            "I walk through {place} with this {noun} in my hands",
            "We talked about {theme} till the morning came",
            "The {noun} keeps its silence while the {feeling} light fades",
            "Every shadow in {place} still knows my name",
            "A {feeling} wind carries what we could not say",
        ],
        "chorus": [
            "Oh, {theme} is calling me tonight",
            "Hold on to the {noun}, hold on to the light",
            "We are {feeling}, we are wide awake",
        ],
        "nouns": ["photograph", "old guitar", "paper map", "broken clock",
                  "silver key", "unsent letter"],
        "places": ["the empty streets", "the harbor", "the winter fields",
                   "the neon town", "the quiet hills"],
        "feelings": ["restless", "tender", "burning", "fearless", "fading"],
        "themes": {
            "love": "love", "loss": "what we lost",
            "night city": "the night city", "ocean": "the ocean",
            "freedom": "freedom", "memory": "memory", "rain": "the rain",
            "open road": "the open road", "stars": "the stars",
            "home": "home", "dance": "the dance",
            "seasons": "the turning seasons"},
    },
    "es": {
        "verse": [
            "Camino por {place} con {noun} en la mano",
            "Hablamos de {theme} hasta el amanecer",
            "La {noun} guarda silencio y la luz se apaga",
            "Cada sombra de {place} sabe mi nombre",
            "Un viento {feeling} se lleva lo que no dije",
        ],
        "chorus": [
            "Oh, {theme} me llama esta noche",
            "Guarda la {noun}, guarda la luz",
            "Estamos despiertos, seguimos aquí",
        ],
        "nouns": ["fotografía", "guitarra vieja", "mapa", "llave de plata",
                  "carta", "melodía"],
        "places": ["las calles vacías", "el puerto", "los campos",
                   "la ciudad de neón", "las colinas"],
        "feelings": ["inquieto", "tierno", "ardiente", "valiente", "lejano"],
        "themes": {
            "love": "el amor", "loss": "lo que perdimos",
            "night city": "la ciudad de noche", "ocean": "el mar",
            "freedom": "la libertad", "memory": "la memoria",
            "rain": "la lluvia", "open road": "la carretera",
            "stars": "las estrellas", "home": "el hogar",
            "dance": "el baile", "seasons": "las estaciones"},
    },
    "fr": {
        "verse": [
            "Je marche dans {place} avec {noun} dans les mains",
            "Nous parlions de {theme} jusqu'au matin",
            "La {noun} se tait pendant que la lumière s'éteint",
            "Chaque ombre de {place} connaît mon nom",
            "Un vent {feeling} emporte nos silences",
        ],
        "chorus": [
            "Oh, {theme} m'appelle ce soir",
            "Garde la {noun}, garde la lumière",
            "Nous sommes éveillés, encore debout",
        ],
        "nouns": ["photographie", "vieille guitare", "carte",
                  "clé d'argent", "lettre", "mélodie"],
        "places": ["les rues vides", "le port", "les champs d'hiver",
                   "la ville néon", "les collines"],
        "feelings": ["fiévreux", "tendre", "brûlant", "sans peur",
                     "lointain"],
        "themes": {
            "love": "l'amour", "loss": "ce qu'on a perdu",
            "night city": "la ville la nuit", "ocean": "l'océan",
            "freedom": "la liberté", "memory": "la mémoire",
            "rain": "la pluie", "open road": "la grande route",
            "stars": "les étoiles", "home": "la maison",
            "dance": "la danse", "seasons": "les saisons"},
    },
    "de": {
        "verse": [
            "Ich gehe durch {place}, {noun} in der Hand",
            "Wir sprachen über {theme} bis zum Morgen",
            "Die {noun} schweigt, während das Licht verglüht",
            "Jeder Schatten hier kennt meinen Namen",
            "Ein {feeling} Wind trägt fort, was wir verschwiegen",
        ],
        "chorus": [
            "Oh, {theme} ruft mich heute Nacht",
            "Halt die {noun} fest, halt das Licht",
            "Wir sind hellwach, wir bleiben hier",
        ],
        "nouns": ["Fotografie", "alte Gitarre", "Landkarte",
                  "silberne Uhr", "Melodie", "Erinnerung"],
        "places": ["die leeren Straßen", "den alten Hafen",
                   "die Winterfelder", "die Neonstadt",
                   "die stillen Hügel"],
        "feelings": ["rastloser", "zarter", "brennender", "furchtloser",
                     "verblassender"],
        "themes": {
            "love": "die Liebe", "loss": "das Verlorene",
            "night city": "die nächtliche Stadt", "ocean": "das Meer",
            "freedom": "die Freiheit", "memory": "die Erinnerung",
            "rain": "der Regen", "open road": "die offene Straße",
            "stars": "die Sterne", "home": "das Zuhause",
            "dance": "der Tanz", "seasons": "die Jahreszeiten"},
    },
    "it": {
        "verse": [
            "Cammino per {place} con {noun} tra le mani",
            "Parlavamo di {theme} fino al mattino",
            "La {noun} resta muta mentre la luce svanisce",
            "Ogni ombra di {place} conosce il mio nome",
            "Un vento {feeling} porta via i silenzi",
        ],
        "chorus": [
            "Oh, {theme} mi chiama stanotte",
            "Tieni la {noun}, tieni la luce",
            "Siamo svegli davvero, ancora qui",
        ],
        "nouns": ["fotografia", "vecchia chitarra", "mappa",
                  "chiave d'argento", "lettera", "melodia"],
        "places": ["le strade vuote", "il porto", "i campi d'inverno",
                   "la città al neon", "le colline"],
        "feelings": ["inquieto", "tenero", "ardente", "coraggioso",
                     "lontano"],
        "themes": {
            "love": "l'amore", "loss": "ciò che abbiamo perso",
            "night city": "la città di notte", "ocean": "il mare",
            "freedom": "la libertà", "memory": "la memoria",
            "rain": "la pioggia", "open road": "la strada aperta",
            "stars": "le stelle", "home": "casa", "dance": "il ballo",
            "seasons": "le stagioni"},
    },
    "pt": {
        "verse": [
            "Caminho por {place} com {noun} nas mãos",
            "Falamos de {theme} até o amanhecer",
            "A {noun} fica em silêncio e a luz se apaga",
            "Cada sombra de {place} sabe o meu nome",
            "Um vento {feeling} leva o que não disse",
        ],
        "chorus": [
            "Oh, {theme} me chama esta noite",
            "Guarda a {noun}, guarda a luz",
            "Estamos acordados, seguimos aqui",
        ],
        "nouns": ["fotografia", "violão velho", "mapa", "chave de prata",
                  "carta", "melodia"],
        "places": ["as ruas vazias", "o porto", "os campos",
                   "a cidade de néon", "as colinas"],
        "feelings": ["inquieto", "terno", "ardente", "sem medo",
                     "distante"],
        "themes": {
            "love": "o amor", "loss": "o que perdemos",
            "night city": "a cidade à noite", "ocean": "o mar",
            "freedom": "a liberdade", "memory": "a memória",
            "rain": "a chuva", "open road": "a estrada",
            "stars": "as estrelas", "home": "o lar", "dance": "a dança",
            "seasons": "as estações"},
    },
    "zh": {
        "verse": [
            "我走过{place}手里握着{noun}",
            "我们谈着{theme}直到天亮",
            "{noun}沉默着灯光渐渐熄灭",
            "{place}的影子都记得我的名字",
            "{feeling}的风带走没说出的话",
        ],
        "chorus": [
            "哦{theme}在今夜呼唤我",
            "抓紧{noun}抓紧那道光",
            "我们醒着我们还在这里",
        ],
        "nouns": ["照片", "旧吉他", "地图", "停摆的钟", "钥匙", "信"],
        "places": ["空荡的街道", "港口", "冬天的田野", "霓虹的城市",
                   "安静的山丘"],
        "feelings": ["不安", "温柔", "炽热", "无畏", "渐远"],
        "themes": {
            "love": "爱", "loss": "失去的一切", "night city": "夜色的城市",
            "ocean": "海", "freedom": "自由", "memory": "记忆",
            "rain": "雨", "open road": "远方的路", "stars": "星光",
            "home": "家", "dance": "舞步", "seasons": "四季"},
    },
    "ja": {
        "verse": [
            "{place}を歩く{noun}を抱いて",
            "朝が来るまで{theme}を語った",
            "{noun}は黙って光は消えてく",
            "{place}の影が僕の名を知ってる",
            "{feeling}風が言葉をさらっていく",
        ],
        "chorus": [
            "ああ{theme}が今夜呼んでいる",
            "{noun}を離さないで光を離さないで",
            "僕らは目を覚ましてまだここにいる",
        ],
        "nouns": ["写真", "古いギター", "地図", "止まった時計", "鍵",
                  "手紙"],
        "places": ["空っぽの街", "港", "冬の野原", "ネオンの街", "静かな丘"],
        "feelings": ["落ち着かない", "優しい", "燃える", "恐れない",
                     "消えゆく"],
        "themes": {
            "love": "愛", "loss": "失くしたもの", "night city": "夜の街",
            "ocean": "海", "freedom": "自由", "memory": "記憶",
            "rain": "雨", "open road": "果てない道", "stars": "星",
            "home": "帰る場所", "dance": "ダンス", "seasons": "巡る季節"},
    },
}

RENDER_BACKENDS = ("ace_step", "yue", "mureka", "minimax")


def _weighted_sample(rng: random.Random, tagged: list[tuple[str, int]],
                     k: int) -> list[str]:
    """k distinct tags, probability proportional to MTG-Jamendo track count."""
    chosen: list[str] = []
    tags = [t for t, _ in tagged]
    weights = [w for _, w in tagged]
    guard = 0
    while len(chosen) < k and guard < 100:
        tag = rng.choices(tags, weights=weights, k=1)[0]
        if tag not in chosen:
            chosen.append(tag)
        guard += 1
    return chosen


class PromptTaxonomy:
    """Deterministic sampler over the MTG-Jamendo-shaped style space (§4.2).

    Parameters
    ----------
    seed:
        Campaign seed (use ``config.SEED``). Style ``i`` is a pure function
        of ``(seed, i)``.
    lyrics_provider:
        Optional hook ``callable(style_dict) -> str`` returning full lyrics
        with ``[verse]``/``[chorus]`` section labels; plugs in an LLM later
        without touching the campaign (plan §4.2 "lyrics from an LLM").
        When None, the built-in template generator is used.
    """

    def __init__(self, seed: int,
                 lyrics_provider: Optional[Callable[[dict], str]] = None):
        self.seed = seed
        self.lyrics_provider = lyrics_provider

    # --- style sampling ---------------------------------------------------
    def style(self, index: int) -> dict:
        """The deterministic style dict for campaign index ``index``."""
        rng = random.Random(f"{self.seed}:{index}")
        genre_tags = _weighted_sample(
            rng, GENRE_TAGS, rng.choices([1, 2, 3], [0.35, 0.45, 0.20])[0])
        mood_tags = _weighted_sample(
            rng, MOOD_TAGS, rng.choices([1, 2], [0.55, 0.45])[0])
        vocal = rng.random() >= INSTRUMENTAL_PROB
        instr_pool = INSTRUMENT_TAGS
        instrumentation = _weighted_sample(
            rng, instr_pool, rng.choices([1, 2, 3], [0.30, 0.45, 0.25])[0])
        language = rng.choices([l for l, _ in LANGUAGE_WEIGHTS],
                               [w for _, w in LANGUAGE_WEIGHTS], k=1)[0]
        lo, hi = DURATION_RANGE_S
        return {
            "style_id": index,
            "genre_tags": genre_tags,
            "mood_tags": mood_tags,
            "instrumentation": instrumentation,
            "language": language,
            "vocal": vocal,
            "vocal_gender": rng.choice(["female", "male"]) if vocal else None,
            "vocal_timbre": rng.choice(VOCAL_TIMBRES) if vocal else None,
            "duration_s": int(round(rng.uniform(lo, hi) / 5.0) * 5),
            "lyrics_theme": rng.choice(LYRICS_THEMES),
            "seed": rng.randrange(2**31),
        }

    def sample(self, n: int, start: int = 0) -> Iterator[dict]:
        """Yield ``n`` deterministic styles for indices ``start..start+n-1``."""
        for i in range(start, start + n):
            yield self.style(i)

    # --- lyrics -----------------------------------------------------------
    def lyric_sections(self, style: dict) -> list[tuple[str, list[str]]]:
        """Template lyrics as ``[('verse', lines), ('chorus', lines), ...]``.

        Verse count scales with duration (~1 section pair per 90 s); the
        chorus is sampled once and repeated, like a real song.
        """
        lang = style["language"] if style["language"] in _LANG_BANK else "en"
        bank = _LANG_BANK[lang]
        rng = random.Random(f"lyrics:{style['seed']}")
        theme = bank["themes"].get(style["lyrics_theme"],
                                   style["lyrics_theme"])

        def fill(template: str) -> str:
            return template.format(theme=theme,
                                   noun=rng.choice(bank["nouns"]),
                                   place=rng.choice(bank["places"]),
                                   feeling=rng.choice(bank["feelings"]))

        chorus = [fill(t) for t in bank["chorus"]]
        n_pairs = max(1, min(3, int(style["duration_s"]) // 90 + 1))
        sections: list[tuple[str, list[str]]] = []
        for _ in range(n_pairs):
            verse = [fill(t) for t in rng.sample(bank["verse"], 4)]
            sections.append(("verse", verse))
            sections.append(("chorus", list(chorus)))
        return sections

    def lyrics(self, style: dict, label_case: str = "lower") -> str:
        """Full lyric text with bracketed section labels.

        Uses ``lyrics_provider`` when configured (LLM hook), else templates.
        ``label_case``: 'lower' -> ``[verse]`` (YuE/ACE-Step convention),
        'title' -> ``[Verse]`` (Mureka/MiniMax docs convention).
        """
        if self.lyrics_provider is not None:
            return self.lyrics_provider(style)
        blocks = []
        for label, lines in self.lyric_sections(style):
            tag = f"[{label.title()}]" if label_case == "title" else f"[{label}]"
            blocks.append(tag + "\n" + "\n".join(lines))
        return "\n\n".join(blocks)

    # --- per-backend rendering (§4.2) --------------------------------------
    def render(self, style: dict, backend: str) -> dict:
        """Render a style into backend-specific prompt strings.

        Returns
        -------
        dict with keys per backend:
          - ``ace_step``: ``prompt`` (comma tag string), ``lyrics``
            (``[instrumental]`` for instrumental tracks).
          - ``yue``: ``genre`` (space-separated genre/instrument/mood/gender/
            timbre token line for genre.txt), ``lyrics`` (double-newline
            separated ``[verse]``/``[chorus]`` sections for lyrics.txt).
          - ``mureka`` / ``minimax``: natural-language ``prompt`` + ``lyrics``.
        """
        if backend not in RENDER_BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; "
                             f"expected one of {RENDER_BACKENDS}")
        g, m, ins = (style["genre_tags"], style["mood_tags"],
                     style["instrumentation"])
        vocal = style["vocal"]
        gender = style.get("vocal_gender") or ""
        timbre = style.get("vocal_timbre") or ""

        if backend == "ace_step":
            tags = g + m + ins
            tags.append(f"{gender} vocals" if vocal else "instrumental")
            return {"prompt": ", ".join(tags),
                    "lyrics": self.lyrics(style) if vocal
                    else "[instrumental]"}

        if backend == "yue":
            # YuE genre.txt format: genre/instrument/mood/gender/timbre
            # tokens separated by spaces; YuE is lyrics-to-song, so lyrics
            # are always rendered (instrumental strata come from the other
            # backends).
            tokens = g + ins + m
            tokens += ([gender, f"{timbre} vocal", "vocal"] if vocal
                       else ["instrumental"])
            return {"genre": " ".join(tokens),
                    "lyrics": self.lyrics(style)}

        # Natural-language prompt for the API platforms.
        desc = (f"A {', '.join(m)} {', '.join(g)} track "
                f"featuring {', '.join(ins)}")
        desc += (f", expressive {timbre} {gender} vocals."
                 if vocal else ", purely instrumental, no vocals.")
        if backend == "mureka":
            # Mureka docs use compact tag-ish prompts ("r&b, slow, ...").
            prompt = ", ".join(m + g + ins)
            prompt += f", {gender} vocal" if vocal else ", instrumental"
            return {"prompt": prompt,
                    "lyrics": self.lyrics(style, label_case="title")
                    if vocal else ""}
        # minimax: lyrics hard limit 3500 chars, prompt 2000 (verified docs).
        lyr = self.lyrics(style, label_case="title") if vocal else ""
        return {"prompt": desc[:2000], "lyrics": lyr[:3500]}
