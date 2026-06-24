"""
neo4j_manager.py
─────────────────────────────────────────────────────────────────────────
All Neo4j interaction: schema setup, vector index creation,
node/relationship writes, vector similarity search, and multi-hop
Cypher graph traversal for GraphRAG retrieval.

Graph model:
  (:Entity)  — typed knowledge graph nodes with `embedding` vector property
  (:Chunk)   — raw text chunk nodes with `embedding` vector property
  Relationships — dynamic typed edges between :Entity nodes
  [:MENTIONS] — :Chunk → :Entity

Vector indexes (Neo4j native):
  entity_embedding_index  on :Entity(embedding)
  chunk_embedding_index   on :Chunk(embedding)

The embedding dimension is read dynamically from embeddings.get_embedding_dim()
at schema-setup time, so switching backends (OpenAI 1536d vs mpnet 768d vs
minilm 384d) just requires calling embeddings.set_backend() BEFORE connecting,
then clearing and re-ingesting the graph.
"""

import time
from neo4j import GraphDatabase
import embeddings  # imported as module so get_embedding_dim() is always current

print("[neo4j_manager.py] Module loading...")


class Neo4jManager:
    def __init__(self, uri: str, user: str, password: str):
        print(f"[Neo4jManager] Connecting to {uri} ...")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._verify_connection()

    def _verify_connection(self):
        self.driver.verify_connectivity()
        print("[Neo4jManager] ✅ Connection verified.")

    def close(self):
        print("[Neo4jManager] Closing driver.")
        self.driver.close()

    # ── Schema / Index Setup ──────────────────────────────────────────────
    def setup_schema(self):
        """Create constraints + vector indexes. Idempotent — safe to call every run.
        Reads the current embedding dimension from embeddings.get_embedding_dim()."""
        dim = embeddings.get_embedding_dim()
        print(f"[Neo4jManager.setup_schema] Setting up schema (embedding_dim={dim})...")
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            s.run("CREATE CONSTRAINT chunk_id_unique  IF NOT EXISTS FOR (c:Chunk)  REQUIRE c.id IS UNIQUE")
            print("  ✓ Uniqueness constraints ensured.")

            # Drop + recreate vector indexes if dimension changed
            for idx_name, label, prop in [
                ("entity_embedding_index", "Entity", "embedding"),
                ("chunk_embedding_index",  "Chunk",  "embedding"),
            ]:
                # Check existing index dimension
                existing = s.run(
                    "SHOW INDEXES WHERE name = $name",
                    name=idx_name
                ).data()
                if existing:
                    try:
                        opts = existing[0].get("options", {})
                        existing_dim = (opts.get("indexConfig", {})
                                           .get("vector.dimensions"))
                        if existing_dim and int(existing_dim) != dim:
                            print(f"  ⚠️  Dropping {idx_name} (was {existing_dim}d, need {dim}d)...")
                            s.run(f"DROP INDEX {idx_name} IF EXISTS")
                    except Exception:
                        pass  # can't read dim, leave it

                s.run(f"""
                    CREATE VECTOR INDEX {idx_name} IF NOT EXISTS
                    FOR (n:`{label}`) ON (n.{prop})
                    OPTIONS {{indexConfig: {{
                        `vector.dimensions`: {dim},
                        `vector.similarity_function`: 'cosine'
                    }}}}
                """)
                print(f"  ✓ Vector index ensured: {idx_name} (dim={dim}, cosine)")
        print("[Neo4jManager.setup_schema] Schema setup complete.\n")

    def wait_for_indexes(self, timeout: int = 60):
        print("[Neo4jManager.wait_for_indexes] Waiting for vector indexes...")
        with self.driver.session() as s:
            s.run("CALL db.awaitIndexes($t)", t=timeout)
        print("[Neo4jManager.wait_for_indexes] ✓ Indexes online.\n")

    # ── Reset ─────────────────────────────────────────────────────────────
    def clear_all(self) -> int:
        print("[Neo4jManager.clear_all] Deleting ALL nodes and relationships...")
        with self.driver.session() as s:
            r = s.run("MATCH (n) DETACH DELETE n RETURN count(n) AS deleted")
            count = r.single()["deleted"]
        print(f"[Neo4jManager.clear_all] ✓ Deleted {count} nodes.\n")
        return count

    # ── Write: Entities ───────────────────────────────────────────────────
    def upsert_entity(self, entity_id: str, label: str, etype: str,
                       description: str, embedding: list, source_doc: str = "") -> bool:
        with self.driver.session() as s:
            s.run("""
                MERGE (e {id: $id})
                ON CREATE SET
                    e.label       = $label,
                    e.type        = $etype,
                    e.description = $description,
                    e.embedding   = $embedding,
                    e.source_doc  = $source_doc,
                    e.created_at  = timestamp()
                ON MATCH SET
                    e.description = CASE
                        WHEN size($description) > size(coalesce(e.description,''))
                        THEN $description ELSE e.description END,
                    e.embedding   = $embedding
            """, id=entity_id, label=label, etype=etype, description=description,
                 embedding=embedding, source_doc=source_doc)
        return True

    def upsert_relationship(self, source_id: str, target_id: str,
                             relation: str, description: str = "") -> bool:
        rel_type = "".join(c if c.isalnum() else "_" for c in relation.upper())
        with self.driver.session() as s:
            s.run("MERGE (a:Entity {id:$id}) ON CREATE SET a.label=$id, a.type='Other'", id=source_id)
            s.run("MERGE (b:Entity {id:$id}) ON CREATE SET b.label=$id, b.type='Other'", id=target_id)
            s.run(f"""
                MATCH (a {{id:$sid}}), (b {{id:$tid}})
                MERGE (a)-[r:{rel_type}]->(b)
                ON CREATE SET r.description=$desc, r.relation_name=$rel
            """, sid=source_id, tid=target_id, desc=description, rel=relation)
        return True

    # ── Write: Chunks ─────────────────────────────────────────────────────
    def upsert_chunk(self, chunk_id: str, text: str, embedding: list,
                      source_doc: str, entity_ids: list = None) -> bool:
        with self.driver.session() as s:
            s.run("""
                MERGE (c:Chunk {id:$id})
                ON CREATE SET c.text=$text, c.embedding=$embedding,
                              c.source_doc=$source_doc, c.created_at=timestamp()
                ON MATCH SET  c.text=$text, c.embedding=$embedding
            """, id=chunk_id, text=text, embedding=embedding, source_doc=source_doc)
            for eid in (entity_ids or []):
                s.run("""
                    MATCH (c:Chunk {id:$cid}), (e:Entity {id:$eid})
                    MERGE (c)-[:MENTIONS]->(e)
                """, cid=chunk_id, eid=eid)
        return True

    # ── Vector Search (Neo4j native kNN) ──────────────────────────────────
    def vector_search_entities(self, query_embedding: list, top_k: int = 8) -> list:
        print(f"[Neo4jManager] Vector kNN over :Entity (top_k={top_k})...")
        with self.driver.session() as s:
            rows = s.run("""
                CALL db.index.vector.queryNodes('entity_embedding_index', $top_k, $emb)
                YIELD node, score
                RETURN node.id AS id, node.label AS label, node.type AS type,
                       node.description AS description, score
                ORDER BY score DESC
            """, top_k=top_k, emb=query_embedding).data()
        print(f"[Neo4jManager] ✓ Found {len(rows)} entities via vector search.")
        for r in rows[:4]:
            print(f"    • {r['label']} ({r['type']})  score={r['score']:.4f}")
        return rows

    def vector_search_chunks(self, query_embedding: list, top_k: int = 5) -> list:
        print(f"[Neo4jManager] Vector kNN over :Chunk (top_k={top_k})...")
        with self.driver.session() as s:
            rows = s.run("""
                CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $emb)
                YIELD node, score
                RETURN node.id AS id, node.text AS text,
                       node.source_doc AS source_doc, score
                ORDER BY score DESC
            """, top_k=top_k, emb=query_embedding).data()
        print(f"[Neo4jManager] ✓ Found {len(rows)} chunks via vector search.")
        return rows

    # ── Graph Traversal (the "graph" half of GraphRAG) ─────────────────────
    def expand_subgraph(self, seed_ids: list, hops: int = 2, limit: int = 60) -> dict:
        print(f"[Neo4jManager] Graph traversal: {hops}-hop from {len(seed_ids)} seeds...")
        cypher = f"""
            MATCH (seed) WHERE seed.id IN $seed_ids
            MATCH path = (seed)-[*0..{hops}]-(connected)
            WITH collect(DISTINCT connected) AS nodes_list
            UNWIND nodes_list AS n
            WITH collect(DISTINCT n) AS all_nodes
            UNWIND all_nodes AS a
            MATCH (a)-[r]-(b) WHERE b IN all_nodes
            RETURN
                [x IN all_nodes |
                  {{id: x.id, label: x.label, type: x.type, description: x.description}}
                ] AS nodes,
                collect(DISTINCT {{
                    source: startNode(r).id,
                    target: endNode(r).id,
                    relation: type(r),
                    description: r.description
                }})[..{limit}] AS relationships
            LIMIT 1
        """
        try:
            with self.driver.session() as s:
                record = s.run(cypher, seed_ids=seed_ids, hops=hops, limit=limit).single()
                if not record:
                    return {"nodes": [], "edges": []}
                nodes = record["nodes"]
                edges = record["relationships"]
        except Exception as e:
            print(f"[Neo4jManager] ❌ Traversal failed: {e}")
            return {"nodes": [], "edges": []}

        seen, unique_nodes = set(), []
        for n in nodes:
            if n["id"] not in seen:
                seen.add(n["id"])
                unique_nodes.append(n)

        print(f"[Neo4jManager] ✓ Subgraph: {len(unique_nodes)} nodes, {len(edges)} edges.")
        return {"nodes": unique_nodes, "edges": edges}

    def get_entity_neighbors_cypher(self, entity_id: str, hops: int = 1) -> list:
        with self.driver.session() as s:
            return s.run("""
                MATCH (e {id:$id})-[r]-(n)
                RETURN type(r) AS relation, n.label AS neighbor_label,
                       n.type AS neighbor_type, n.id AS neighbor_id
                LIMIT 25
            """, id=entity_id).data()

    # ── Stats / Introspection ─────────────────────────────────────────────
    def get_graph_stats(self) -> dict:
        with self.driver.session() as s:
            n_entities = s.run("MATCH (e) RETURN count(e) AS c").single()["c"]
            n_chunks   = s.run("MATCH (c:Chunk)  RETURN count(c) AS c").single()["c"]
            n_rels     = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            types      = {r["type"]: r["cnt"] for r in s.run("""
                MATCH (e) RETURN e.type AS type, count(*) AS cnt ORDER BY cnt DESC
            """)}
            # Return current embedding backend info too
            emb_backend = embeddings.get_backend_name()
            emb_dim     = embeddings.get_embedding_dim()
        stats = {
            "entities": n_entities, "chunks": n_chunks,
            "relationships": n_rels, "type_breakdown": types,
            "embedding_backend": emb_backend, "embedding_dim": emb_dim,
        }
        print(f"[Neo4jManager.get_graph_stats] {stats}")
        return stats

    def get_full_graph(self, limit: int = 300) -> dict:
        with self.driver.session() as s:
            node_list = s.run(f"""
                MATCH (e)
                RETURN
                    coalesce(e.id, elementId(e)) AS id,
                    coalesce(e.label, head(labels(e)), 'Unknown') AS label,
                    coalesce(e.type, head(labels(e)), 'Unknown') AS type,
                    coalesce(e.description, '') AS description
                LIMIT {limit}
            """).data()
            node_ids  = [n["id"] for n in node_list]
            edge_list = s.run("""
                MATCH (a)-[r]->(b)
                WHERE a.id IN $ids AND b.id IN $ids
                RETURN a.id AS source, b.id AS target, type(r) AS relation
                LIMIT 600
            """, ids=node_ids).data()
        return {"nodes": node_list, "edges": edge_list}

    def list_all_entities_brief(self, limit: int = 500) -> list:
        with self.driver.session() as s:
            return s.run(f"""
                MATCH (e)
                RETURN e.id AS id, e.label AS label, e.type AS type
                LIMIT {limit}
            """).data()
