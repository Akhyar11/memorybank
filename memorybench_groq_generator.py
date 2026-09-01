#!/usr/bin/env python3
"""
MemoryBench-ID / Groq Conversation Generator
=============================================

Purpose
-------
Generate synthetic Indonesian multi-turn conversations for training/evaluating
a tiny memory-augmented language model.

Design principles
-----------------
1. Python owns the ground truth:
   - facts
   - memory WRITE / UPDATE events
   - target query
   - target answer
   - difficulty
2. Groq/Qwen only produces natural-language conversation.
3. Groq Structured Outputs are used so the generated conversation is valid JSON.
4. The generator is resumable and writes JSONL incrementally.
5. Rate-limit headers are read from every response.
6. No multi-account / quota bypass behavior is implemented.
7. Local augmentation can turn one semantic episode into many training examples.

Recommended first model:
    qwen/qwen3.8-27b

Environment:
    GROQ_API_KEY=your_key_here

Install:
    pip install groq

Examples:
    python memorybench_groq_generator.py --episodes 10 --output data/test.jsonl
    python memorybench_groq_generator.py --episodes 1000 --output data/train.jsonl
    python memorybench_groq_generator.py --episodes 1000 --output data/train.jsonl --difficulty 4
    python memorybench_groq_generator.py --episodes 100 --output data/test.jsonl --language mixed
    python memorybench_groq_generator.py --estimate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from groq import Groq
except ImportError:
    print("Missing dependency. Run: pip install groq", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen/qwen3.8-27b"

# These are safety estimates for scheduling, not claims about the user's
# actual Groq limits. The script also reads x-ratelimit-* response headers.
DEFAULT_TPM_ESTIMATE = 8000
DEFAULT_RPD_ESTIMATE = 1000
DEFAULT_MAX_OUTPUT_TOKENS = 900

# We intentionally keep individual generations small. This is much more
# efficient for a memory dataset than asking for 10K+ token conversations.
MIN_TURNS = 8
MAX_TURNS = 24

ROLES = ("user", "assistant")


# ---------------------------------------------------------------------------
# Synthetic knowledge base
# ---------------------------------------------------------------------------

FACT_POOLS: Dict[str, List[str]] = {
    "name": [
        "Akhyar", "Bima", "Dimas", "Fajar", "Raka", "Andi", "Rian", "Arif",
        "Bagas", "Bayu", "Citra", "Dewi", "Nadia", "Rina", "Sinta", "Maya",
        "Nisa", "Tari", "Alya", "Putri", "Kevin", "Ayu", "Reza", "Sari",
        "Dito", "Gilang", "Tika", "Jihan", "Budi", "Yoga", "Agus", "Vina"
    ],
    "age": [str(x) for x in range(18, 45)],
    "city": [
        "Solo", "Surakarta", "Sukoharjo", "Karanganyar", "Boyolali", "Klaten",
        "Wonogiri", "Yogyakarta", "Semarang", "Jakarta", "Bandung", "Surabaya",
        "Malang", "Medan", "Makassar", "Denpasar", "Bogor", "Depok",
        "Tangerang", "Bekasi", "Bantul", "Sleman", "Cimahi", "Padang", "Palembang",
        "Balikpapan", "Samarinda", "Manado", "Jakarta Selatan", "Jakarta Pusat"
    ],
    "hobby": [
        "berenang", "bersepeda", "lari", "membaca", "memasak", "fotografi",
        "bermain gim", "menggambar", "memancing", "mendaki gunung",
        "bermain musik", "menulis", "thrifting", "nongkrong di coffee shop",
        "mabar e-sports", "koleksi photocard", "maraton drakor", "jalan-jalan",
        "baking", "berkebun", "workout di gym", "main futsal"
    ],
    "food": [
        "apel", "pizza", "sate", "nasi goreng", "mie ayam", "bakso",
        "soto", "rendang", "pecel", "gado-gado", "pisang", "roti",
        "seblak", "mie gacoan", "nasi padang", "martabak", "ayam geprek",
        "dimsum", "sushi", "burger", "nasi uduk", "ketoprak"
    ],
    "drink": [
        "kopi", "teh", "air putih", "jus jeruk", "cokelat panas",
        "susu", "es teh", "jus alpukat", "es matcha latte", "boba",
        "kopi susu gula aren", "thai tea", "es teler", "wedang jahe", "es kelapa"
    ],
    "pet": [
        "kucing", "anjing", "ikan", "kelinci", "burung", "hamster",
        "tidak punya hewan peliharaan", "kura-kura", "iguana", "sugar glider",
        "ayam pelung"
    ],
    "color": [
        "biru", "merah", "hijau", "hitam", "putih", "ungu", "kuning",
        "abu-abu", "oranye", "pink", "coklat", "navy", "maroon", "tosca"
    ],
    "job": [
        "pengembang perangkat lunak", "mahasiswa", "guru", "desainer",
        "fotografer", "akuntan", "barista", "penulis", "teknisi",
        "analis data", "UI/UX designer", "admin sosmed", "driver ojol",
        "content creator", "digital marketer", "HRD", "arsitek", "perawat",
        "wirausaha", "freelancer", "videografer"
    ],
    "favorite_subject": [
        "matematika", "bahasa Inggris", "informatika", "fisika", "sejarah",
        "biologi", "ekonomi", "bahasa Indonesia", "seni", "sosiologi",
        "geografi", "kimia", "olahraga", "PKn"
    ],
    "weekend_activity": [
        "berolahraga", "menonton film", "bermain gim", "jalan-jalan",
        "membaca buku", "bersepeda", "memasak", "bertemu teman",
        "pergi ke mall", "kulineran", "rebahan", "nonton konser",
        "camping", "bantu orang tua bersih-bersih"
    ],
    "transport": [
        "sepeda", "motor", "mobil", "bus", "kereta", "berjalan kaki",
        "KRL", "MRT", "ojek online", "TransJakarta", "angkot", "LRT"
    ],
}

FACT_LABELS = {
    "name": "nama",
    "age": "umur",
    "city": "tempat tinggal",
    "hobby": "hobi",
    "food": "makanan favorit",
    "drink": "minuman favorit",
    "pet": "hewan peliharaan",
    "color": "warna favorit",
    "job": "pekerjaan",
    "favorite_subject": "pelajaran favorit",
    "weekend_activity": "aktivitas akhir pekan",
    "transport": "transportasi",
}

FACT_QUERY_TEMPLATES = {
    "name": [
        "Siapa nama saya?",
        "Kamu masih ingat nama saya?",
        "Nama saya siapa?",
        "Saya tadi memperkenalkan diri sebagai siapa?",
        "Eh, kamu masih ingat namaku nggak?",
        "Tadi di awal aku ngenalin diri sebagai siapa hayo?",
        "Btw, ingat namaku kan?",
        "Kalau boleh ngetes, namaku siapa coba?"
    ],
    "age": [
        "Berapa umur saya?",
        "Kamu ingat umur saya?",
        "Saya sekarang berusia berapa?",
        "Eh, aku tadi bilang umurku berapa ya?",
        "Inget nggak umurku sekarang berapa?",
        "Coba tebak, berdasarkan obrolan tadi, umurku berapa?"
    ],
    "city": [
        "Saya tinggal di mana?",
        "Kamu ingat saya tinggal di kota mana?",
        "Sekarang saya berdomisili di mana?",
        "Tempat tinggal saya di mana?",
        "Tadi aku bilang asalku dari mana hayo?",
        "Inget nggak aku domisili di mana sekarang?",
        "Eh, kota tempat tinggalku apa ya tadi?"
    ],
    "hobby": [
        "Apa hobi saya?",
        "Kamu masih ingat hobi saya?",
        "Saya biasanya suka melakukan apa?",
        "Inget nggak sih hobi utamaku apa?",
        "Kalau lagi luang, biasanya aku ngapain?",
        "Eh, hobi kesukaanku apa ya tadi aku bilangnya?"
    ],
    "food": [
        "Apa makanan favorit saya?",
        "Saya suka makanan apa?",
        "Kamu ingat makanan yang saya sukai?",
        "Masih ingat nggak makanan kesukaanku apa?",
        "Tadi aku bilang lagi pengen makan apa?",
        "Menu favoritku apa coba?"
    ],
    "drink": [
        "Apa minuman favorit saya?",
        "Saya biasanya suka minum apa?",
        "Minuman yang saya sukai apa?",
        "Eh, inget nggak minuman andalanku apa?",
        "Kalau nongkrong aku biasanya pesen minum apa hayo?"
    ],
    "pet": [
        "Saya punya hewan peliharaan apa?",
        "Kamu ingat soal hewan peliharaan saya?",
        "Di rumah saya punya hewan apa?",
        "Inget nggak peliharaanku di rumah apa?",
        "Tadi aku cerita punya hewan apa ya?"
    ],
    "color": [
        "Apa warna favorit saya?",
        "Saya paling suka warna apa?",
        "Kamu ingat warna kesukaan saya?",
        "Warna favoritku apa coba?",
        "Masih inget nggak warna kesukaanku?"
    ],
    "job": [
        "Pekerjaan saya apa?",
        "Saya bekerja sebagai apa?",
        "Kamu ingat pekerjaan saya?",
        "Ingat nggak aku kerja jadi apa?",
        "Tadi aku bilang profesiku apa ya?",
        "Eh, kerjaanku sekarang apa coba?"
    ],
    "favorite_subject": [
        "Pelajaran favorit saya apa?",
        "Saya paling suka pelajaran apa?",
        "Inget nggak pelajaran sekolah yang paling aku suka?",
        "Mata pelajaran favoritku apa hayo?"
    ],
    "weekend_activity": [
        "Biasanya saya melakukan apa saat akhir pekan?",
        "Apa kegiatan akhir pekan saya?",
        "Tadi aku cerita kalau weekend biasanya ngapain?",
        "Inget nggak kegiatan rutinku tiap akhir pekan?"
    ],
    "transport": [
        "Biasanya saya pergi menggunakan apa?",
        "Transportasi yang biasa saya gunakan apa?",
        "Kendaraan andalanku kalau pergi-pergi apa ya?",
        "Eh, inget nggak aku sering kemana-mana naik apa?"
    ],
}

UPDATE_TEMPLATES = {
    "name": [
        "Oh iya, sekarang saya lebih suka dipanggil {value}.",
        "Mulai sekarang panggil saya {value}.",
        "Ada perubahan kecil, nama panggilan saya sekarang {value}.",
        "Eh btw, mending panggil aku {value} aja mulai sekarang.",
        "Oh ya, biar akrab panggil aku {value} ya sekarang."
    ],
    "city": [
        "Sekarang saya sudah pindah ke {value}.",
        "Belakangan ini saya tinggal di {value}.",
        "Saya baru saja pindah dan sekarang berdomisili di {value}.",
        "Fyi aja, aku baru aja pindah kos ke {value}.",
        "Oh ya, sekarang aku udah menetap di {value} loh.",
        "Eh info aja nih, aku baru aja pindahan ke {value}."
    ],
    "hobby": [
        "Sekarang saya lebih sering {value}.",
        "Akhir-akhir ini hobi saya berubah, saya lebih suka {value}.",
        "Belakangan saya mulai menekuni {value}.",
        "Eh, sekarang aku lagi keranjingan {value} nih.",
        "Btw, belakangan hobi lamaku kegeser, sekarang lebih sering {value}.",
        "Gara-gara liat sosmed, sekarang aku jadi suka {value}."
    ],
    "food": [
        "Sekarang saya lebih suka {value}.",
        "Belakangan makanan favorit saya berubah menjadi {value}.",
        "Eh, ngomong-ngomong seleraku berubah, sekarang lebih suka {value}.",
        "Btw, belakangan aku lagi kecanduan banget makan {value}.",
        "Lidah nggak bisa bohong, sekarang menu favoritku ganti jadi {value}."
    ],
    "drink": [
        "Sekarang saya lebih sering minum {value}.",
        "Belakangan saya lebih suka {value}.",
        "Btw sekarang aku lagi demen banget sama {value}.",
        "Lagi ngurangin yang aneh-aneh, sekarang lebih sering pesen {value}."
    ],
    "pet": [
        "Sekarang saya punya {value}.",
        "Ada perubahan soal hewan peliharaan saya, sekarang saya punya {value}.",
        "Oh ya, soal peliharaan, sekarang aku ngerawat {value} di rumah.",
        "Btw, aku baru aja adopsi {value} lho.",
        "Eh, sekarang di rumahku udah ada {value} baru."
    ],
    "color": [
        "Sekarang saya lebih suka warna {value}.",
        "Belakangan warna favorit saya berubah menjadi {value}.",
        "Nggak tau kenapa, belakangan aku lagi suka banget sama warna {value}.",
        "Lagi nyoba gaya baru, sekarang aku identik sama warna {value}."
    ],
    "job": [
        "Sekarang pekerjaan saya berubah menjadi {value}.",
        "Belakangan saya bekerja sebagai {value}.",
        "Eh, update dikit, aku sekarang udah resign dan kerja jadi {value}.",
        "Btw, aku sekarang udah beralih profesi jadi {value} loh.",
        "Alhamdulillah baru aja keterima kerja baru sebagai {value}."
    ],
    "favorite_subject": [
        "Sekarang pelajaran yang paling saya sukai adalah {value}.",
        "Gara-gara gurunya asik, sekarang aku lebih suka pelajaran {value}.",
        "Btw, sekarang minatku udah ganti, aku lebih suka {value}."
    ],
    "weekend_activity": [
        "Sekarang akhir pekan saya biasanya diisi dengan {value}.",
        "Btw akhir-akhir ini tiap weekend aku malah sibuk {value}.",
        "Lagi bosen sama rutinitas lama, sekarang weekend biasanya {value}."
    ],
    "transport": [
        "Sekarang saya lebih sering menggunakan {value}.",
        "Btw sekarang buat ke mana-mana aku lebih sering naik {value}.",
        "Lagi nyari yang praktis aja, belakangan seringnya pake {value}."
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    key: str
    value: str
    introduced_at: int = -1
    updated_at: int = -1


@dataclass
class MemoryEvent:
    step: int
    operation: str  # WRITE | UPDATE
    key: str
    value: str
    previous_value: Optional[str] = None


@dataclass
class GenerationStats:
    requested: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    api_calls: int = 0
    estimated_output_tokens: int = 0
    last_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def stable_id(*parts: str) -> str:
    raw = "||".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def approx_tokens(text: str) -> int:
    # Conservative approximation for scheduling. Exact usage is read from
    # Groq response usage when available.
    return max(1, len(text) // 4)


def choose_unique(rng: random.Random, items: List[str], n: int) -> List[str]:
    n = min(n, len(items))
    return rng.sample(items, n)


def choose_language(rng: random.Random, language: str) -> str:
    if language in ("id", "en"):
        return language
    return "id" if rng.random() < 0.85 else "mixed"


# ---------------------------------------------------------------------------
# Ground-truth episode generation
# ---------------------------------------------------------------------------

def make_facts(
    rng: random.Random,
    difficulty: int,
) -> Tuple[Dict[str, Fact], List[str]]:
    all_keys = list(FACT_POOLS.keys())

    # More facts at higher difficulty.
    n_facts = {
        1: 3,
        2: 4,
        3: 6,
        4: 8,
        5: 10,
    }[difficulty]

    keys = choose_unique(rng, all_keys, n_facts)
    facts: Dict[str, Fact] = {}

    for key in keys:
        facts[key] = Fact(key=key, value=rng.choice(FACT_POOLS[key]))

    # name and city are useful anchors and should usually exist.
    if "name" not in facts:
        facts["name"] = Fact("name", rng.choice(FACT_POOLS["name"]))
    if "city" not in facts and difficulty >= 2:
        facts["city"] = Fact("city", rng.choice(FACT_POOLS["city"]))

    return facts, list(facts.keys())


def choose_target(
    rng: random.Random,
    facts: Dict[str, Fact],
) -> str:
    preferred = [
        k for k in ("name", "city", "hobby", "food", "age")
        if k in facts
    ]
    if preferred and rng.random() < 0.75:
        return rng.choice(preferred)
    return rng.choice(list(facts.keys()))


def maybe_create_update(
    rng: random.Random,
    facts: Dict[str, Fact],
    difficulty: int,
) -> Optional[Tuple[str, str, str]]:
    if difficulty < 3:
        return None

    update_probability = {
        3: 0.35,
        4: 0.60,
        5: 0.80,
    }[difficulty]

    if rng.random() > update_probability:
        return None

    candidates = [
        k for k in facts.keys()
        if k in UPDATE_TEMPLATES and len(FACT_POOLS[k]) > 1
    ]
    if not candidates:
        return None

    key = rng.choice(candidates)
    old = facts[key].value
    possible = [v for v in FACT_POOLS[key] if v != old]
    new = rng.choice(possible)

    return key, old, new


def build_memory_plan(
    rng: random.Random,
    difficulty: int,
) -> Dict[str, Any]:
    facts, fact_keys = make_facts(rng, difficulty)
    target_key = choose_target(rng, facts)

    update = maybe_create_update(rng, facts, difficulty)

    # Keep target updated if the target is the updated fact; otherwise update
    # another fact to create a useful distractor.
    if update and rng.random() < 0.55:
        target_key = update[0]

    # Approximate event positions. These are semantic positions for the
    # generated conversation, not forced exact turn numbers.
    turns = {
        1: 8,
        2: 10,
        3: 12,
        4: 16,
        5: 22,
    }[difficulty]

    events: List[MemoryEvent] = []

    # Each fact must be introduced at some point.
    intro_steps = sorted(
        rng.sample(range(1, max(2, turns - 2)), len(facts))
    )

    for step, key in zip(intro_steps, facts.keys()):
        facts[key].introduced_at = step
        events.append(
            MemoryEvent(
                step=step,
                operation="WRITE",
                key=key,
                value=facts[key].value,
            )
        )

    if update:
        key, old, new = update
        min_update_step = max(facts[key].introduced_at + 1, turns // 2)
        max_update_step = max(min_update_step + 1, turns - 2)
        update_step = rng.randint(min_update_step, max_update_step)
        facts[key].updated_at = update_step
        events.append(
            MemoryEvent(
                step=update_step,
                operation="UPDATE",
                key=key,
                value=new,
                previous_value=old,
            )
        )
        facts[key].value = new

    # Query should happen after the target's final event.
    target_events = [e for e in events if e.key == target_key]
    target_step = max((e.step for e in target_events), default=1)
    query_step = max(target_step + 2, turns - 1)

    return {
        "facts": facts,
        "fact_keys": fact_keys,
        "target_key": target_key,
        "target_value": facts[target_key].value,
        "events": events,
        "turns": turns,
        "query_step": query_step,
    }


# ---------------------------------------------------------------------------
# Groq structured-output schema
# ---------------------------------------------------------------------------

DIALOGUE_SCHEMA = {
    "type": "object",
    "properties": {
        "conversation": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant"],
                    },
                    "text": {
                        "type": "string",
                    },
                },
                "required": ["role", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["conversation"],
    "additionalProperties": False,
}


def build_system_prompt(language: str) -> str:
    if language == "en":
        lang_instruction = "Write the conversation in natural English."
    elif language == "mixed":
        lang_instruction = (
            "Write primarily in natural Indonesian. Occasional natural English "
            "words are allowed, but do not switch languages excessively."
        )
    else:
        lang_instruction = (
            "Write in highly natural, everyday conversational Indonesian (Bahasa Indonesia Sehari-hari). "
            "Vary the persona and tone greatly between episodes: sometimes polite (saya/anda), "
            "sometimes casual (aku/kamu), and sometimes informal/slang (gue/lu, cowok/cewek bumi). "
            "Use natural particles like 'sih', 'nih', 'dong', 'deh', 'lho', 'ya', 'kok'. "
            "Make it sound exactly like real humans chatting on WhatsApp or Telegram, "
            "with realistic emotions, reactions, typos, or sudden topic changes."
        )

    return f"""
You are a high-quality synthetic conversational data generator for a research
dataset about persistent conversational memory.

{lang_instruction}

Your job is to turn a hidden memory plan into a natural multi-turn dialogue.

IMPORTANT:
- The hidden facts are ground truth.
- Do not output the hidden facts as a list.
- Introduce facts naturally across the conversation.
- Do not mention that you are generating a dataset.
- Do not mention "memory", "ground truth", "target", "event", or "dataset".
- Add realistic small talk and semantic distractors.
- Avoid repeating the same sentence structure.
- The user should naturally reveal personal facts.
- The assistant should respond naturally and briefly.
- If a fact is updated, the NEW value must become the current value.
- Do not introduce a conflicting value unless the plan explicitly contains an update.
- The final conversation should feel like a real casual chat.
- The target fact must be inferable from the conversation.
- Do not ask the final memory query yourself.
- Do not answer the target query explicitly at the end.
- Use only the supplied facts; do not invent personal facts that could conflict
  with the supplied facts.
""".strip()


def build_user_prompt(
    plan: Dict[str, Any],
    language: str,
    rng: random.Random,
) -> str:
    facts = plan["facts"]

    visible_facts = "\n".join(
        f"- {key}: {fact.value}"
        for key, fact in facts.items()
    )

    update_lines = []
    for event in plan["events"]:
        if event.operation == "UPDATE":
            template = rng.choice(UPDATE_TEMPLATES[event.key])
            phrase = template.format(value=event.value)
            update_lines.append(
                f"- At a later point, {event.key} changes to '{event.value}'. "
                f"Use a natural transition like: \"{phrase}\""
            )

    updates = "\n".join(update_lines) if update_lines else "- No updates."

    distractor_budget = {
        1: "light",
        2: "light",
        3: "moderate",
        4: "strong",
        5: "very strong",
    }[max(1, min(5, plan.get("difficulty", 3)))]

    return f"""
Create a natural {plan['turns']}-turn conversation.

FACTS TO INTRODUCE NATURALLY:
{visible_facts}

UPDATES:
{updates}

DIALOGUE REQUIREMENTS:
- Start naturally; do not dump all facts immediately.
- Write using casual, everyday Indonesian (bahasa gaul/sehari-hari) like 'aku/kamu/gue/lu/eh/btw'.
- Spread the facts across the conversation.
- Include {distractor_budget} unrelated or semantically adjacent discussion.
- Vary how facts are expressed.
- The user should be the source of most personal facts.
- The assistant should acknowledge and continue the conversation.
- Do not include a final question asking for the target fact.
- Do not provide a final summary.
- Do not explicitly enumerate the facts.
- Do not expose hidden metadata.

The target memory key is: {plan['target_key']}
The current target value is: {plan['target_value']}

The target key/value is only provided so you can make sure it appears
naturally somewhere in the dialogue. Never mention the words "target key"
or "target value" in the conversation.

Return ONLY the required JSON structure.
""".strip()


# ---------------------------------------------------------------------------
# Groq client and rate-aware API
# ---------------------------------------------------------------------------

class GroqGenerator:
    def __init__(
        self,
        model: str,
        max_output_tokens: int,
        temperature: float,
        max_retries: int = 6,
    ):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Set it in your environment before running the generator."
            )

        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        self.last_headers: Dict[str, str] = {}

    def _extract_headers(self, response: Any) -> Dict[str, str]:
        headers: Dict[str, str] = {}

        # Groq SDK exposes headers on the response object in current versions.
        raw = getattr(response, "headers", None)
        if raw:
            try:
                for k, v in raw.items():
                    headers[str(k).lower()] = str(v)
            except Exception:
                pass

        self.last_headers = headers
        return headers

    @staticmethod
    def _seconds_from_reset(value: Optional[str]) -> float:
        if not value:
            return 0.0

        value = value.strip()

        # Common examples: "7.66s", "2m59.56s", "1h2m3s".
        total = 0.0
        m = re.fullmatch(
            r"(?:(\d+(?:\.\d+)?)h)?"
            r"(?:(\d+(?:\.\d+)?)m)?"
            r"(?:(\d+(?:\.\d+)?)s)?",
            value,
        )
        if m:
            if m.group(1):
                total += float(m.group(1)) * 3600
            if m.group(2):
                total += float(m.group(2)) * 60
            if m.group(3):
                total += float(m.group(3))
            return total

        try:
            return float(value)
        except ValueError:
            return 0.0

    def _respect_headers(self, estimated_tokens: int) -> None:
        headers = self.last_headers
        remaining = headers.get("x-ratelimit-remaining-tokens")
        reset = headers.get("x-ratelimit-reset-tokens")

        if remaining is None:
            return

        try:
            remaining_int = int(float(remaining))
        except ValueError:
            return

        # Avoid starting a request when the previous response tells us there
        # is not enough minute-level token budget.
        if remaining_int < max(estimated_tokens, 256):
            seconds = self._seconds_from_reset(reset)
            if seconds > 0:
                time.sleep(min(seconds + 0.5, 120.0))

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        estimated_tokens: int,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        self._respect_headers(estimated_tokens)

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "memory_dialogue",
                            "strict": True,
                            "schema": DIALOGUE_SCHEMA,
                        },
                    },
                    temperature=self.temperature,
                    max_completion_tokens=self.max_output_tokens,
                    reasoning_effort="none",
                )

                headers = self._extract_headers(response)

                content = response.choices[0].message.content or "{}"
                result = json.loads(content)

                # Keep usage if exposed by the SDK.
                usage = getattr(response, "usage", None)
                if usage is not None:
                    try:
                        result["_usage"] = {
                            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0)),
                            "completion_tokens": int(
                                getattr(usage, "completion_tokens", 0)
                            ),
                            "total_tokens": int(getattr(usage, "total_tokens", 0)),
                        }
                    except Exception:
                        pass

                return result, headers

            except Exception as exc:
                last_error = exc
                message = str(exc).lower()

                # Try to respect retry-after when available on exception.
                retry_after = None
                response = getattr(exc, "response", None)
                if response is not None:
                    raw_headers = getattr(response, "headers", None)
                    if raw_headers:
                        try:
                            retry_after = raw_headers.get("retry-after")
                            self.last_headers = {
                                str(k).lower(): str(v)
                                for k, v in raw_headers.items()
                            }
                        except Exception:
                            pass

                if retry_after:
                    try:
                        wait = float(retry_after) + 0.5
                    except ValueError:
                        wait = 2.0
                elif "429" in message or "rate limit" in message:
                    wait = min(2 ** attempt, 90)
                else:
                    wait = min(1.5 * attempt, 15)

                if attempt < self.max_retries:
                    time.sleep(wait)
                else:
                    break

        raise RuntimeError(
            f"Groq generation failed after {self.max_retries} attempts: "
            f"{last_error}"
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_episode(
    episode: Dict[str, Any],
    plan_facts: Dict[str, Fact],
    min_turns: int,
    max_turns: int,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    
    conversation = episode.get("conversation", [])
    if not isinstance(conversation, list):
        return False, ["conversation is not a list"]

    if not (min_turns <= len(conversation) <= max_turns):
        errors.append(f"turn_count={len(conversation)} outside [{min_turns}, {max_turns}]")

    previous_role = None
    for i, msg in enumerate(conversation):
        if not isinstance(msg, dict):
            errors.append(f"turn {i}: not an object")
            continue

        role = msg.get("role")
        text = clean_text(str(msg.get("text", "")))

        if role not in ROLES:
            errors.append(f"turn {i}: invalid role={role}")

        if not text:
            errors.append(f"turn {i}: empty text")

        if previous_role == role:
            errors.append(f"turn {i}: consecutive role={role}")

        previous_role = role

    # Query dan answer harus tersedia
    if not episode.get("query"):
        errors.append("missing query")
    if not episode.get("answer"):
        errors.append("missing answer")

    # Target harus tersedia
    if not episode.get("target") or not episode["target"].get("key") or not episode["target"].get("value"):
        errors.append("missing target key or value")
    else:
        # Answer harus cocok dengan current memory state
        target_value = episode["target"]["value"]
        if episode.get("answer") and episode["answer"] != target_value:
            errors.append(f"answer '{episode['answer']}' != current state '{target_value}'")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Query / answer construction
# ---------------------------------------------------------------------------

def build_query(
    rng: random.Random,
    target_key: str,
    language: str,
) -> str:
    if language == "en":
        english = {
            "name": ["What is my name?", "Do you remember my name?"],
            "age": ["How old am I?", "Do you remember my age?"],
            "city": ["Where do I live?", "Which city do I live in?"],
            "hobby": ["What is my hobby?", "Do you remember my hobby?"],
            "food": ["What is my favorite food?", "What food do I like?"],
            "drink": ["What is my favorite drink?", "What do I like to drink?"],
            "pet": ["What pet do I have?", "Do I have a pet?"],
            "color": ["What is my favorite color?", "Which color do I like?"],
            "job": ["What is my job?", "What do I do for work?"],
        }
        values = english.get(target_key, ["What do you remember about me?"])
        return rng.choice(values)

    values = FACT_QUERY_TEMPLATES.get(
        target_key,
        ["Kamu masih ingat informasi tentang saya yang tadi?"],
    )
    return rng.choice(values)


# ---------------------------------------------------------------------------
# Local augmentation
# ---------------------------------------------------------------------------

def paraphrase_query_local(
    rng: random.Random,
    target_key: str,
    base_query: str,
    n: int,
) -> List[str]:
    # This intentionally uses deterministic local templates. The Groq seed
    # remains the expensive/high-quality language generation step.
    candidates = list(FACT_QUERY_TEMPLATES.get(target_key, []))
    candidates = [q for q in candidates if q != base_query]

    rng.shuffle(candidates)
    return [base_query] + candidates[: max(0, n - 1)]


def make_training_examples(
    episode: Dict[str, Any],
    rng: random.Random,
    augmentation: int,
) -> List[Dict[str, Any]]:
    target_key = episode["target"]["key"]
    target_value = episode["target"]["value"]
    base_query = episode["query"]

    queries = paraphrase_query_local(
        rng,
        target_key,
        base_query,
        max(1, augmentation),
    )

    examples = []

    for idx, query in enumerate(queries):
        ex_id = stable_id(
            episode["episode_id"],
            "aug",
            str(idx),
            query,
            target_value,
        )

        examples.append(
            {
                "sample_id": ex_id,
                "episode_id": episode["episode_id"],
                "conversation": episode["conversation"],
                "query": query,
                "answer": target_value,
                "task": "memory_recall",
                "target_memory": episode["target"],
                "memory_events": episode["memory_events"],
                "difficulty": episode["difficulty"],
            }
        )

    return examples


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MemoryBenchGenerator:
    def __init__(
        self,
        output_path: Path,
        model: str,
        language: str,
        difficulty: int,
        max_output_tokens: int,
        temperature: float,
        augmentation: int,
        seed: int,
        max_retries: int,
    ):
        self.output_path = output_path
        self.model = model
        self.language = language
        self.difficulty = difficulty
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.augmentation = augmentation
        self.seed = seed
        self.rng = random.Random(seed)

        self.api = GroqGenerator(
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            max_retries=max_retries,
        )

        self.stats = GenerationStats()

    def _load_existing_ids(self) -> set:
        ids = set()

        if not self.output_path.exists():
            return ids

        with self.output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    episode_id = obj.get("episode_id")
                    if episode_id:
                        ids.add(episode_id)
                except json.JSONDecodeError:
                    # Ignore a corrupted trailing line; new data will still
                    # be appended. The user can remove it manually if needed.
                    continue

        return ids

    def _make_episode_id(self, index: int, plan: Dict[str, Any]) -> str:
        # Include random state so resumed generation never depends on Python's
        # process-local hash randomization.
        seed_material = (
            f"{self.seed}:{index}:{plan['target_key']}:"
            f"{plan['target_value']}:{uuid.uuid4().hex}"
        )
        return "ep_" + hashlib.sha1(seed_material.encode()).hexdigest()[:16]

    def _build_episode(
        self,
        index: int,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        episode_id = self._make_episode_id(index, plan)

        language = choose_language(self.rng, self.language)

        # The prompt gets difficulty only through the plan itself.
        plan["difficulty"] = self.difficulty

        system_prompt = build_system_prompt(language)
        user_prompt = build_user_prompt(plan, language, self.rng)

        estimated_tokens = min(
            self.max_output_tokens,
            max(300, plan["turns"] * 22),
        )

        result, headers = self.api.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            estimated_tokens=estimated_tokens,
        )

        conversation = result.get("conversation", [])
        usage = result.pop("_usage", {})

        # Sort events by semantic step.
        events = sorted(
            plan["events"],
            key=lambda e: (e.step, e.operation),
        )

        target_event = max(
            [e for e in events if e.key == plan["target_key"]],
            key=lambda e: e.step,
        )

        episode = {
            "episode_id": episode_id,
            "schema_version": "memorybench-id/0.2",
            "language": language,
            "model": self.model,
            "conversation": [
                {
                    "role": m.get("role"),
                    "text": clean_text(m.get("text", "")),
                }
                for m in conversation
            ],
            "query": build_query(
                self.rng,
                plan["target_key"],
                language,
            ),
            "answer": plan["target_value"],
            "task": "memory_recall",
            "target": {
                "key": plan["target_key"],
                "value": plan["target_value"],
                "last_event": asdict(target_event),
            },
            "facts": [
                {
                    "key": key,
                    "value": fact.value,
                    "introduced_at": fact.introduced_at,
                    "updated_at": fact.updated_at,
                }
                for key, fact in plan["facts"].items()
            ],
            "memory_events": [asdict(e) for e in events],
            "difficulty": {
                "level": self.difficulty,
                "turns_requested": plan["turns"],
                "num_facts": len(plan["facts"]),
                "num_updates": sum(
                    1 for e in events if e.operation == "UPDATE"
                ),
                "memory_distance": max(
                    0,
                    plan["query_step"] - target_event.step,
                ),
            },
            "generation": {
                "seed": self.seed,
                "index": index,
                "usage": usage,
                "rate_limit_headers": headers,
            },
        }

        valid, errors = validate_episode(
            episode=episode,
            plan_facts=plan["facts"],
            min_turns=max(MIN_TURNS, plan["turns"] - 5),
            max_turns=plan["turns"] + 5,
        )

        if not valid:
            episode["validation_errors"] = errors
            raise ValueError(
                "Validation failed: " + "; ".join(errors)
            )

        return episode

    def _append_jsonl(self, obj: Dict[str, Any], path: Optional[Path] = None) -> None:
        if path is None:
            path = self.output_path
            
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    obj,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    def run(self, episodes: int, start_index: int = 0) -> GenerationStats:
        existing = self._load_existing_ids()

        self.stats.requested = episodes

        print("=" * 72)
        print("MemoryBench-ID / Groq Generator v0.2")
        print("=" * 72)
        print(f"Model       : {self.model}")
        print(f"Difficulty  : {self.difficulty}")
        print(f"Language    : {self.language}")
        print(f"Output      : {self.output_path}")
        print(f"Episodes    : {episodes}")
        print(f"Augment     : {self.augmentation}")
        print(f"Seed        : {self.seed}")
        print("=" * 72)

        rejected_path = self.output_path.with_name("rejected.jsonl")
        report_path = self.output_path.parent / "validation_report.json"

        max_attempts = episodes * 3
        attempts = 0
        index = max(start_index, len(existing))

        print(f"Resuming from index {index} (found {len(existing)} existing episodes)...")

        while self.stats.completed < episodes and attempts < max_attempts:
            attempts += 1
            
            # Ground truth is created locally.
            plan = build_memory_plan(
                self.rng,
                self.difficulty,
            )

            # Since episode IDs are random, resume primarily relies on the
            # existing JSONL being preserved. We still retain the check hook.
            try:
                episode = self._build_episode(index, plan)

                if episode["episode_id"] in existing:
                    self.stats.skipped += 1
                    index += 1
                    continue

                self._append_jsonl(episode)
                existing.add(episode["episode_id"])

                self.stats.completed += 1
                self.stats.api_calls += 1
                index += 1

                usage = episode["generation"].get("usage", {})
                self.stats.estimated_output_tokens += int(
                    usage.get(
                        "completion_tokens",
                        approx_tokens(
                            " ".join(
                                x["text"]
                                for x in episode["conversation"]
                            )
                        ),
                    )
                )

                # Optional local augmentation file.
                if self.augmentation > 1:
                    aug_path = self.output_path.with_name(
                        self.output_path.stem + "_samples.jsonl"
                    )

                    examples = make_training_examples(
                        episode,
                        self.rng,
                        self.augmentation,
                    )

                    with aug_path.open("a", encoding="utf-8") as f:
                        for example in examples:
                            f.write(
                                json.dumps(
                                    example,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )

                print(
                    f"[{self.stats.completed:>6}/{episodes}] "
                    f"{episode['episode_id']} | "
                    f"target={episode['target']['key']}="
                    f"{episode['target']['value']} | "
                    f"turns={len(episode['conversation'])} | "
                    f"events={len(episode['memory_events'])}"
                )

            except KeyboardInterrupt:
                print("\nInterrupted. Progress is already saved.")
                break

            except ValueError as exc:
                self.stats.failed += 1
                self.stats.last_error = str(exc)
                # This was a validation failure, save it to rejected.jsonl.
                if 'episode' in locals() and isinstance(episode, dict):
                    self._append_jsonl(episode, rejected_path)
                print(f"[REJECTED {index}] {exc}", file=sys.stderr)
                index += 1

            except Exception as exc:
                self.stats.failed += 1
                self.stats.last_error = str(exc)
                print(f"[FAILED {index}] {exc}", file=sys.stderr)
                index += 1

        print("\nFinished.")
        
        # Save Validation Report
        report = {
            "Generated": self.stats.completed + self.stats.failed,
            "Valid": self.stats.completed,
            "Rejected": self.stats.failed,
            "Validation rate": f"{(self.stats.completed / max(1, self.stats.completed + self.stats.failed)) * 100:.2f}%",
            "Details": asdict(self.stats)
        }
        
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(json.dumps(report, indent=2, ensure_ascii=False))

        return self.stats


# ---------------------------------------------------------------------------
# Estimate helper
# ---------------------------------------------------------------------------

def print_estimate() -> None:
    print(
        """
MemoryBench-ID rough capacity estimator
----------------------------------------

Using the Groq limits shown in your current account:

    qwen/qwen3.8-27b
    RPM  = 30
    RPD  = 1,000
    TPM  = 8,000
    TPD  = 2,000,000

These are only the limits you supplied; the generator itself reads response
headers at runtime and does not hard-code them as authorization to exceed them.

If one generated episode uses approximately:

    300 output tokens -> 300,000 tokens/day at 1,000 requests/day
    500 output tokens -> 500,000 tokens/day
    700 output tokens -> 700,000 tokens/day
    1,000 output tokens -> 1,000,000 tokens/day

The practical request ceiling is 1,000 requests/day before considering retries,
validation failures, and other API usage.

Recommended first experiment:

    10 episodes
    difficulty=2
    max-output-tokens=700
    augmentation=5

Then inspect the JSONL manually before scaling.
"""
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MemoryBench-ID synthetic conversations using Groq."
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of Groq-generated semantic episodes.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/memorybench_train.jsonl"),
        help="Output JSONL path.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Groq model. Default: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--difficulty",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=2,
        help="Difficulty level 1-5.",
    )

    parser.add_argument(
        "--language",
        choices=["id", "en", "mixed"],
        default="id",
        help="Conversation language.",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum completion tokens per episode.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature.",
    )

    parser.add_argument(
        "--augmentation",
        type=int,
        default=1,
        help="Number of local query variants written to *_samples.jsonl.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260830,
        help="Random seed.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting episode index for bookkeeping.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum API retries per episode.",
    )

    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print a quota/capacity estimate and exit.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.estimate:
        print_estimate()
        return

    if args.episodes <= 0:
        raise SystemExit("--episodes must be > 0")

    if not 0.0 <= args.temperature <= 1.0:
        raise SystemExit("--temperature must be between 0 and 1")

    if args.max_output_tokens < 128:
        raise SystemExit("--max-output-tokens should be >= 128")

    if args.augmentation < 1:
        raise SystemExit("--augmentation must be >= 1")

    generator = MemoryBenchGenerator(
        output_path=args.output,
        model=args.model,
        language=args.language,
        difficulty=args.difficulty,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        augmentation=args.augmentation,
        seed=args.seed,
        max_retries=args.max_retries,
    )

    generator.run(
        episodes=args.episodes,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
