"""
extraction.py
─────────────────────────────────────────────────────────────────────────
- PDF -> text -> chunks
- Groq LLM extraction of cruise-domain entities + relationships from each chunk
"""

import io
import re
import json
import PyPDF2
from groq import Groq

print("[extraction.py] Module loading...")

# LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MODEL = "llama-3.1-8b-instant"

EXTRACT_SYSTEM_PROMPT = """You are a cruise-industry knowledge graph extraction expert.
Extract cruise-specific entities and relationships from the given text chunk.

Return ONLY valid JSON (no markdown fences, no commentary):
{
  "entities": [
    {"id": "lowercase_snake_case_id", "label": "Display Name",
     "type": "CruiseLine|Ship|Port|Destination|Excursion|Amenity|Cabin|Restaurant|Activity|Review|Passenger|Policy|Package|Other",
     "description": "concise 1-2 sentence description"}
  ],
  "relationships": [
    {"source": "source_entity_id", "target": "target_entity_id",
     "relation": "UPPER_SNAKE_CASE_RELATION", "description": "brief description"}
  ]
}

Guidelines:
- entity ids must be unique, lowercase, snake_case
- relation types: OPERATES, HOMEPORTS_AT, VISITS_PORT, OFFERS_EXCURSION, HAS_AMENITY,
  HAS_CABIN_TYPE, HAS_RESTAURANT, RATED_BY, RECOMMENDED_FOR, PART_OF_ITINERARY,
  INCLUDES_IN_PACKAGE, SUITABLE_FOR, LOCATED_IN, PRICED_AT, DEPARTS_FROM, ARRIVES_AT, etc.
- extract every cruise-relevant entity and relation found in the text
- be thorough and precise
- return ONLY the raw JSON object, nothing else"""


# ── PDF → Text ──────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> tuple:
    print("[extraction.extract_text_from_pdf] Reading PDF bytes...")
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text()
        if t:
            pages_text.append(t.strip())
        print(f"[extraction.extract_text_from_pdf]   Page {i+1}/{len(reader.pages)} -> {len(t) if t else 0} chars")
    full_text = "\n\n".join(pages_text)
    print(f"[extraction.extract_text_from_pdf] ✓ Done. Total pages={len(reader.pages)}, total chars={len(full_text)}")
    return full_text, len(reader.pages)


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 200) -> list:
    print(f"[extraction.chunk_text] Chunking text (size={chunk_size}, overlap={overlap})...")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.6:
                end = start + last_period + 1
                chunk = text[start:end]
        chunk = chunk.strip()
        if len(chunk) > 60:
            chunks.append(chunk)
        start = end - overlap
    print(f"[extraction.chunk_text] ✓ Produced {len(chunks)} chunks.")
    return chunks


# ── LLM Extraction ────────────────────────────────────────────────────────
def extract_entities_relations(client: Groq, chunk: str, chunk_idx: int = 0) -> dict:
    print(f"[extraction.extract_entities_relations] Calling Groq LLM ({LLM_MODEL}) for chunk #{chunk_idx}...")
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract entities and relationships:\n\n{chunk}"}
            ],
            temperature=0.1,
            max_tokens=2500,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[extraction.extract_entities_relations] ❌ Groq API call failed: {e}")
        return {"entities": [], "relationships": []}

    parsed = _safe_parse_json(raw)
    n_e, n_r = len(parsed.get("entities", [])), len(parsed.get("relationships", []))
    print(f"[extraction.extract_entities_relations] ✓ Chunk #{chunk_idx}: extracted {n_e} entities, {n_r} relationships.")
    return parsed


def _safe_parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip("`").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    print("[extraction._safe_parse_json] ⚠️ Failed to parse LLM output as JSON. Returning empty result.")
    return {"entities": [], "relationships": []}


def select_seed_entities(client: Groq, query: str, entity_list: list, top_n: int = 6) -> list:
    """Ask the LLM to pick the most relevant entity IDs for a user query (used alongside vector search)."""
    print(f"[extraction.select_seed_entities] Asking LLM to identify up to {top_n} relevant entities for query: '{query}'")
    brief = json.dumps([{"id": e["id"], "label": e["label"], "type": e["type"]} for e in entity_list[:200]])
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content":
                f'Query: "{query}"\nEntities: {brief}\nReturn ONLY a JSON array of up to {top_n} most relevant entity IDs: ["id1","id2",...]'}],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[extraction.select_seed_entities] ❌ Groq API call failed: {e}")
        return []

    text = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
    try:
        ids = json.loads(text)
        if isinstance(ids, list):
            print(f"[extraction.select_seed_entities] ✓ LLM selected entities: {ids}")
            return ids
    except Exception:
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                ids = json.loads(match.group())
                print(f"[extraction.select_seed_entities] ✓ LLM selected entities (regex fallback): {ids}")
                return ids
            except Exception:
                pass
    print("[extraction.select_seed_entities] ⚠️ Could not parse LLM seed entity response.")
    return []
