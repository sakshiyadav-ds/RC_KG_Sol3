"""
graphrag_pipeline.py
─────────────────────────────────────────────────────────────────────────
Orchestrates the full GraphRAG workflow (Solution 3):

INGESTION:
  PDF → text → chunks → Groq LLM extraction (entities + relations)
      → embeddings (backend from embeddings.py) → Neo4j graph + vector store

QUERY (Hybrid Retrieval):
  question
    → embed question                                (embeddings.py)
    → Neo4j vector kNN over :Entity               (vector retrieval)
    → Neo4j vector kNN over :Chunk                (vector retrieval)
    → LLM seed-entity selection cross-check        (LLM-assisted)
    → Neo4j Cypher multi-hop graph traversal        (graph retrieval)
    → assemble context (graph + chunks)
    → Groq LLM answer synthesis

Returns embedding_backend + embedding_dim in every result dict so app.py
can display which model was active for each ingestion/query.
"""

import time
from groq import Groq

import embeddings
from neo4j_manager import Neo4jManager
from extraction import (
    extract_text_from_pdf, chunk_text,
    extract_entities_relations, select_seed_entities,
)

print("[graphrag_pipeline.py] Module loading...")

CHATBOT_SYSTEM_PROMPT = """
You are an expert Royal Caribbean cruise advisor.

Answer questions using the provided information about ships, itineraries,
destinations, excursions, dining, amenities, cabins, packages, and passenger
experiences.

Guidelines:
- Be accurate, helpful, and conversational.
- Do not mention internal systems, databases, graphs, vectors, or retrieval methods.
- Do not provide Cypher query 
- For recommendations, explain why the option is suitable.
- Use markdown, bullet points and clear formatting.
- Compare options when relevant.
- Include details such as price, duration, location, amenities, or difficulty when available.
- If the information is unavailable, say so clearly.
"""


class GraphRAGPipeline:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 groq_api_key: str):
        print("\n" + "=" * 70)
        print("[GraphRAGPipeline.__init__] Initialising pipeline...")
        print(f"  Embedding backend : {embeddings.get_backend_name()}")
        print(f"  Embedding dim     : {embeddings.get_embedding_dim()}")
        print("=" * 70)
        self.db          = Neo4jManager(neo4j_uri, neo4j_user, neo4j_password)
        self.groq_client = Groq(api_key=groq_api_key)
        self.db.setup_schema()
        print("[GraphRAGPipeline.__init__] ✅ Pipeline ready.\n")

    def close(self):
        self.db.close()

    # ═════════════════════════════════════════════════════════════════════
    # INGESTION
    # ═════════════════════════════════════════════════════════════════════
    def ingest_pdf(self, pdf_bytes: bytes, source_name: str,
                   chunk_size: int = 1800, max_chunks: int = 20,
                   progress_cb=None) -> dict:
        t_start = time.time()
        emb_backend = embeddings.get_backend_name()
        emb_dim     = embeddings.get_embedding_dim()
        print("\n" + "█" * 70)
        print(f"[ingest_pdf] START: {source_name}  "
              f"(backend={emb_backend}, dim={emb_dim})")
        print("█" * 70)

        if progress_cb: progress_cb("Extracting text from PDF...", 0, 1)
        full_text, page_count = extract_text_from_pdf(pdf_bytes)
        if not full_text.strip():
            raise ValueError("No extractable text found (PDF may be image-based).")

        chunks = chunk_text(full_text, chunk_size=chunk_size)[:max_chunks]
        print(f"[ingest_pdf] {len(chunks)} chunks to process (cap={max_chunks}).")

        total_entities = total_relations = 0
        all_entities   = []
        all_relations  = []

        for i, chunk in enumerate(chunks):
            if progress_cb:
                progress_cb(f"Chunk {i+1}/{len(chunks)} — LLM extraction...", i, len(chunks))
            print(f"\n[ingest_pdf] ── Chunk {i+1}/{len(chunks)} ──")

            # LLM extraction
            extracted = extract_entities_relations(self.groq_client, chunk, chunk_idx=i)
            entities  = extracted.get("entities", [])
            relations = extracted.get("relationships", [])

            # Embed + write entities
            if entities:
                ent_texts = [f"{e.get('label','')}: {e.get('description','')}"
                             for e in entities]
                ent_vecs  = embeddings.embed_batch(ent_texts)
                for ent, vec in zip(entities, ent_vecs):
                    self.db.upsert_entity(
                        entity_id=ent.get("id", ""),
                        label=ent.get("label", ent.get("id", "")),
                        etype=ent.get("type", "Other"),
                        description=ent.get("description", ""),
                        embedding=vec,
                        source_doc=source_name,
                    )

            # Write relations
            for rel in relations:
                src, tgt = rel.get("source", ""), rel.get("target", "")
                if src and tgt:
                    self.db.upsert_relationship(
                        src, tgt,
                        rel.get("relation", "RELATED_TO"),
                        rel.get("description", ""),
                    )

            # Embed + write chunk
            chunk_id  = f"{source_name}_chunk_{i}"
            chunk_vec = embeddings.embed_text(chunk)
            self.db.upsert_chunk(
                chunk_id, chunk, chunk_vec, source_name,
                [e.get("id") for e in entities if e.get("id")],
            )

            total_entities += len(entities)
            total_relations += len(relations)
            all_entities.extend(entities)
            all_relations.extend(relations)

        if progress_cb:
            progress_cb("Waiting for Neo4j vector indexes...", len(chunks), len(chunks))
        self.db.wait_for_indexes()

        stats   = self.db.get_graph_stats()
        elapsed = time.time() - t_start
        print(f"\n[ingest_pdf] ✅ DONE: {source_name}  {elapsed:.1f}s  "
              f"{total_entities} entities  {total_relations} relations")
        print("█" * 70 + "\n")

        return {
            "source":            source_name,
            "page_count":        page_count,
            "chunks_processed":  len(chunks),
            "entities_added":    total_entities,
            "relations_added":   total_relations,
            "elapsed_seconds":   round(elapsed, 1),
            "embedding_backend": emb_backend,
            "embedding_dim":     emb_dim,
            "graph_stats":       stats,
            "sample_entities":   all_entities[:20],
            "sample_relations":  all_relations[:12],
        }

    # ═════════════════════════════════════════════════════════════════════
    # HYBRID QUERY
    # ═════════════════════════════════════════════════════════════════════
    def query(self, question: str, vector_top_k: int = 8,
              hops: int = 2, chunk_top_k: int = 4) -> dict:
        t_start     = time.time()
        emb_backend = embeddings.get_backend_name()
        emb_dim     = embeddings.get_embedding_dim()
        reasoning   = []

        print("\n" + "▓" * 70)
        print(f"[query] \"{question}\"  (backend={emb_backend}, dim={emb_dim})")
        print("▓" * 70)

        # Step 1 — Embed question
        print("[query] Step 1 — Embedding question...")
        q_vec = embeddings.embed_text(question)
        reasoning.append(f"🔢 Question embedded ({emb_backend.upper()}, {emb_dim}d)")

        # Step 2 — Vector search: entities
        print("[query] Step 2 — Vector kNN over :Entity...")
        vec_entities = self.db.vector_search_entities(q_vec, top_k=vector_top_k)
        reasoning.append(f"🔍 Vector search found {len(vec_entities)} similar entities in Neo4j")

        # Step 3 — Vector search: chunks
        print("[query] Step 3 — Vector kNN over :Chunk...")
        vec_chunks = self.db.vector_search_chunks(q_vec, top_k=chunk_top_k)
        reasoning.append(f"📄 Vector search retrieved {len(vec_chunks)} relevant text chunks")

        # Step 4 — LLM seed-entity cross-check
        print("[query] Step 4 — LLM seed-entity cross-check...")
        all_ents_brief = self.db.list_all_entities_brief(limit=300)
        llm_seed_ids   = select_seed_entities(
            self.groq_client, question, all_ents_brief, top_n=6
        )
        seed_ids = list(dict.fromkeys(
            [e["id"] for e in vec_entities] + llm_seed_ids
        ))
        seed_ids = [s for s in seed_ids if s]
        reasoning.append(
            f"🎯 Combined seed set: {len(seed_ids)} entities "
            f"(vector ∪ LLM-selected)"
        )

        if not seed_ids:
            return {
                "answer": (
                    "I couldn't find relevant information in the knowledge graph. "
                    "Please make sure PDFs have been ingested first."
                ),
                "reasoning_log":     reasoning,
                "subgraph":          {"nodes": [], "edges": []},
                "vector_entities":   [],
                "vector_chunks":     [],
                "stats":             {},
                "embedding_backend": emb_backend,
                "embedding_dim":     emb_dim,
                "elapsed_seconds":   round(time.time() - t_start, 1),
            }

        # Step 5 — Graph traversal
        print(f"[query] Step 5 — Cypher graph traversal ({hops} hops)...")
        subgraph = self.db.expand_subgraph(seed_ids, hops=hops, limit=80)
        reasoning.append(
            f"🕸️ Cypher {hops}-hop traversal → "
            f"{len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges "
            f"(what plain vector search cannot do)"
        )

        # Step 6 — LLM synthesis
        print("[query] Step 6 — Groq LLM synthesis...")
        graph_ctx = self._format_graph_context(subgraph)
        chunk_ctx = "\n\n---\n\n".join(c["text"] for c in vec_chunks)

        user_prompt = f"""Question: {question}

=== ENTITIES (vector similarity search) ===
{self._format_entity_list(vec_entities)}

=== KNOWLEDGE GRAPH SUBGRAPH ({hops}-hop Cypher traversal) ===
{graph_ctx}

=== SUPPORTING TEXT CHUNKS (vector similarity search) ===
{chunk_ctx}

Answer the question using the graph relationships and text above.
Cite specific multi-hop paths where relevant."""

        resp = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": CHATBOT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        answer = resp.choices[0].message.content.strip()
        reasoning.append("💬 Groq LLM synthesised answer from graph + vector context")

        elapsed = time.time() - t_start
        print(f"[query] ✅ Done in {elapsed:.2f}s")
        print("▓" * 70 + "\n")

        return {
            "answer":            answer,
            "reasoning_log":     reasoning,
            "seed_ids":          seed_ids,
            "subgraph":          subgraph,
            "vector_entities":   vec_entities,
            "vector_chunks":     vec_chunks,
            "stats": {
                "vector_entities_found": len(vec_entities),
                "vector_chunks_found":   len(vec_chunks),
                "subgraph_nodes":        len(subgraph["nodes"]),
                "subgraph_edges":        len(subgraph["edges"]),
                "hops":                  hops,
            },
            "embedding_backend": emb_backend,
            "embedding_dim":     emb_dim,
            "elapsed_seconds":   round(elapsed, 2),
        }

    # ── Helpers ───────────────────────────────────────────────────────────
    def _format_entity_list(self, entities: list) -> str:
        lines = [
            f"- {e['label']} ({e['type']}) [score={e.get('score',0):.3f}]: "
            f"{e.get('description','')}"
            for e in entities
        ]
        return "\n".join(lines) or "(none)"

    def _format_graph_context(self, subgraph: dict) -> str:
        lines = ["NODES:"]
        for n in subgraph["nodes"][:40]:
            lines.append(f"  - {n['label']} ({n['type']}): {n.get('description','')}")
        lines.append("\nRELATIONSHIPS:")
        for e in subgraph["edges"][:40]:
            lines.append(f"  - {e['source']} --[{e['relation']}]--> {e['target']}")
        return "\n".join(lines)

    # ── Pass-through utils ────────────────────────────────────────────────
    def get_graph_stats(self) -> dict:  return self.db.get_graph_stats()
    def get_full_graph(self, limit=300): return self.db.get_full_graph(limit=limit)
    def clear_database(self) -> int:    return self.db.clear_all()
