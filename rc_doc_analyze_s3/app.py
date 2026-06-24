"""
app.py  —  Royal Caribbean GraphRAG Chatbot  (Solution 3)
─────────────────────────────────────────────────────────────────────────
Hybrid retrieval pipeline:
  PDF upload → text → chunks → Groq LLM entity/relation extraction
             → embeddings (OpenAI / mpnet / MiniLM) → Neo4j vector store
  Query time: embed question → Neo4j vector kNN → Cypher graph traversal
             → Groq LLM answer synthesis

Key change from original:
  • Embedding model is selectable in the sidebar:
      - OpenAI text-embedding-3-small (1536d)  — best quality, needs API key
      - all-mpnet-base-v2             (768d)   — strong local, no API key
      - all-MiniLM-L6-v2             (384d)   — fast local,   no API key
  • Selected backend is wired into both embeddings.py and neo4j_manager.py
    before the pipeline is initialised, so Neo4j vector indexes are always
    created at the correct dimension.
  • OpenAI API key field appears only when OpenAI backend is chosen.
  • Ingestion progress shows embedding backend + dimension in the log.
  • Graph Stats card shows active backend + dim.
"""

import os
import time
import traceback

import streamlit as st

print("\n" + "=" * 70)
print("[app.py] Royal Caribbean GraphRAG Chatbot starting...")
print("=" * 70)

st.set_page_config(
    page_title="Royal Caribbean GraphRAG",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── base ── */
.stApp                { background-color: #FAFBFC; }
.main .block-container{ padding-top: 1.5rem; max-width: 1100px; }
h1, h2, h3            { color: #003087; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background-color: #F4F6F9;
    border-right: 1px solid #E5E7EB;
}

/* ── chat bubbles ── */
.stChatMessage {
    background-color: white;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
}

/* ── metric cards ── */
.metric-box  { background:white; border:1px solid #E5E7EB; border-radius:8px;
               padding:10px 14px; text-align:center; }
.metric-num  { font-size:22px; font-weight:700; color:#003087; }
.metric-lbl  { font-size:11px; color:#6B7280; text-transform:uppercase; }

/* ── status pills ── */
.status-ok   { color:#059669; font-weight:600; }
.status-bad  { color:#DC2626; font-weight:600; }

/* ── embedding badge ── */
.emb-badge {
    display:inline-block; padding:3px 10px; border-radius:12px;
    font-size:12px; font-weight:600; margin:4px 0;
}
.emb-openai { background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; }
.emb-mpnet  { background:#F0FDF4; color:#166534; border:1px solid #BBF7D0; }
.emb-minilm { background:#FFF7ED; color:#9A3412; border:1px solid #FED7AA; }

/* ── step trace in answer debug ── */
.hop-line { font-family:'Courier New',monospace; font-size:12px;
            background:#F1F5F9; padding:3px 8px; border-radius:4px;
            margin:2px 0; display:block; }

/* ── buttons ── */
.stButton > button { border-radius:8px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "pipeline":          None,
        "connected":         False,
        "embed_backend":     "minilm",   # active backend name
        "messages":          [],
        "debug_log":         [],
        "ingestion_results": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def log(msg: str):
    print(msg)
    st.session_state.debug_log.append(msg)
    if len(st.session_state.debug_log) > 600:
        st.session_state.debug_log = st.session_state.debug_log[-600:]


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚢 GraphRAG Setup")
    st.caption("Hybrid Vector + Graph chatbot — Royal Caribbean")

    # ── 1. Neo4j ─────────────────────────────────────────────────────────
    st.markdown("### 1️⃣ Neo4j Connection")
    neo4j_uri  = st.text_input(
        "Neo4j URI",
        value=os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687"),
        help="Neo4j Aura: neo4j+s://<id>.databases.neo4j.io  |  Local: bolt://localhost:7687",
    )
    neo4j_user = st.text_input("Username", value=os.environ.get("NEO4J_USER", "neo4j"))
    neo4j_pass = st.text_input("Password", type="password",
                                value=os.environ.get("NEO4J_PASSWORD", ""))

    # ── 2. Groq ───────────────────────────────────────────────────────────
    st.markdown("### 2️⃣ Groq API Key")
    groq_key = st.text_input(
        "Groq API Key", type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        placeholder="gsk_...",
        help="Free key at console.groq.com",
    )

    # ── 3. Embedding model ────────────────────────────────────────────────
    st.markdown("### 3️⃣ Embedding Model")
    st.caption("Choose before connecting. Changing after connecting requires clearing the graph and re-ingesting.")

    backend_options = {
        "mpnet":  "🟢 all-mpnet-base-v2              (768d  — strong local)",
        "minilm": "🟠 all-MiniLM-L6-v2               (384d  — fast local)",
    }
    chosen_backend = st.selectbox(
        "Embedding backend",
        options=list(backend_options.keys()),
        format_func=lambda k: backend_options[k],
        index=list(backend_options.keys()).index(
            st.session_state.embed_backend
        ),
        help="OpenAI needs an API key below. Both local options run fully offline.",
    )

    # Show OpenAI key input only when needed
    openai_key = ""
    if chosen_backend == "openai":
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            placeholder="sk-...",
            help="Required for text-embedding-3-small",
        )

    # Dim preview
    dim_map = {"openai": 1536, "mpnet": 768, "minilm": 384}
    badge_cls = {"openai": "emb-openai", "mpnet": "emb-mpnet", "minilm": "emb-minilm"}
    st.markdown(
        f'<span class="emb-badge {badge_cls[chosen_backend]}">'
        f'  {chosen_backend.upper()}  ·  {dim_map[chosen_backend]}d</span>',
        unsafe_allow_html=True,
    )

    # ── Connect / Disconnect ──────────────────────────────────────────────
    col_conn, col_disc = st.columns(2)
    connect_clicked    = col_conn.button("🔌 Connect",    use_container_width=True)
    disconnect_clicked = col_disc.button("⏏ Disconnect", use_container_width=True)

    if connect_clicked:
        missing = []
        if not neo4j_uri:  missing.append("Neo4j URI")
        if not neo4j_user: missing.append("Username")
        if not neo4j_pass: missing.append("Password")
        if not groq_key:   missing.append("Groq API Key")
        if chosen_backend == "openai" and not openai_key:
            missing.append("OpenAI API Key (needed for OpenAI embeddings)")

        if missing:
            st.error(f"Please fill in: {', '.join(missing)}")
        else:
            try:
                log(f"[app] Connecting — backend={chosen_backend} dim={dim_map[chosen_backend]}...")
                with st.spinner("Initialising embedding model and connecting to Neo4j..."):
                    # Wire embedding backend BEFORE importing neo4j_manager
                    import importlib
                    import embeddings as emb_mod
                    emb_mod.set_backend(chosen_backend, openai_api_key=openai_key)
                    # Force neo4j_manager to re-read the updated EMBEDDING_DIM
                    import neo4j_manager as njm
                    importlib.reload(njm)
                    # Now build pipeline
                    from graphrag_pipeline import GraphRAGPipeline
                    pipeline = GraphRAGPipeline(neo4j_uri, neo4j_user, neo4j_pass, groq_key)

                st.session_state.pipeline      = pipeline
                st.session_state.connected     = True
                st.session_state.embed_backend = chosen_backend
                log(f"[app] ✅ Connected. backend={chosen_backend}, dim={dim_map[chosen_backend]}")
                st.success("Connected successfully!")
            except Exception as e:
                st.session_state.connected = False
                log(f"[app] ❌ Connection failed: {e}")
                st.error(f"Connection failed: {e}")
                with st.expander("Full traceback"):
                    st.code(traceback.format_exc())

    if disconnect_clicked and st.session_state.pipeline:
        try:
            st.session_state.pipeline.close()
        except Exception:
            pass
        st.session_state.pipeline  = None
        st.session_state.connected = False
        log("[app] Disconnected.")
        st.info("Disconnected.")

    st.divider()

    # ── Connection status ─────────────────────────────────────────────────
    if st.session_state.connected:
        st.markdown('<span class="status-ok">● Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-bad">● Not connected</span>', unsafe_allow_html=True)

    st.divider()

    # ── 4. PDF Ingestion ──────────────────────────────────────────────────
    st.markdown("### 4️⃣ Ingest PDFs")
    uploaded_files = st.file_uploader(
        "Upload Royal Caribbean PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    chunk_size         = st.slider("Chunk size (chars)",  1000, 3000, 1800, 100)
    max_chunks_per_pdf = st.slider("Max chunks per PDF",  2,    30,   12,   1)

    ingest_clicked = st.button(
        "📥 Ingest PDFs into Neo4j",
        use_container_width=True,
        disabled=not st.session_state.connected,
    )

    if ingest_clicked:
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            pipeline     = st.session_state.pipeline
            progress_bar = st.progress(0, text="Starting ingestion...")
            for f_idx, f in enumerate(uploaded_files):
                log(f"[app] Ingesting {f_idx+1}/{len(uploaded_files)}: {f.name}")
                pdf_bytes = f.read()

                def _progress_cb(step_label, current, total, _f=f):
                    pct = int((current / max(total, 1)) * 100)
                    progress_bar.progress(pct, text=f"{_f.name}: {step_label}")
                    log(f"[app]   {step_label} ({current}/{total})")

                try:
                    result = pipeline.ingest_pdf(
                        pdf_bytes,
                        source_name=f.name,
                        chunk_size=chunk_size,
                        max_chunks=max_chunks_per_pdf,
                        progress_cb=_progress_cb,
                    )
                    st.session_state.ingestion_results.append(result)
                    log(f"[app] ✅ {f.name}: {result['entities_added']} entities, "
                        f"{result['relations_added']} relations, "
                        f"embedding={result.get('embedding_backend','?')} "
                        f"({result.get('embedding_dim','?')}d)")
                except Exception as e:
                    log(f"[app] ❌ Ingestion failed for {f.name}: {e}")
                    st.error(f"Failed to ingest {f.name}: {e}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())

            progress_bar.progress(100, text="✅ All files processed.")
            st.success(f"Ingested {len(uploaded_files)} file(s).")

    st.divider()

    # ── Graph stats ───────────────────────────────────────────────────────
    if st.session_state.connected:
        st.markdown("### 📊 Graph Stats")
        try:
            stats = st.session_state.pipeline.get_graph_stats()
            c1, c2 = st.columns(2)
            c1.markdown(
                f"<div class='metric-box'><div class='metric-num'>{stats['entities']}</div>"
                f"<div class='metric-lbl'>Entities</div></div>",
                unsafe_allow_html=True,
            )
            c2.markdown(
                f"<div class='metric-box'><div class='metric-num'>{stats['relationships']}</div>"
                f"<div class='metric-lbl'>Relationships</div></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Chunks: {stats['chunks']}")

            # Active embedding model badge
            emb_b = stats.get("embedding_backend", st.session_state.embed_backend)
            emb_d = stats.get("embedding_dim", dim_map.get(emb_b, "?"))
            st.markdown(
                f'<span class="emb-badge {badge_cls.get(emb_b, "")}">  '
                f'{emb_b.upper()}  ·  {emb_d}d  ·  active</span>',
                unsafe_allow_html=True,
            )

            if stats["type_breakdown"]:
                with st.expander("Entity type breakdown"):
                    for t, c in stats["type_breakdown"].items():
                        st.write(f"**{t}**: {c}")
        except Exception as e:
            st.caption(f"Stats unavailable: {e}")

        if st.button("🗑️ Clear entire graph", width="stretch"):
            try:
                deleted = st.session_state.pipeline.clear_database()
                log(f"[app] Cleared {deleted} nodes.")
                st.session_state.ingestion_results = []
                st.session_state.messages          = []
                st.success(f"Cleared {deleted} nodes.")
            except Exception as e:
                st.error(f"Clear failed: {e}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════
st.title("🚢 Royal Caribbean GraphRAG Chatbot")
st.caption(
    "Hybrid retrieval: **Neo4j Vector Index** (semantic kNN) + "
    "**Cypher Graph Traversal** (multi-hop reasoning) + **Groq LLM** synthesis"
)

if not st.session_state.connected:
    # ── Landing / onboarding ─────────────────────────────────────────────
    st.info("👈 Fill in the sidebar — Neo4j, Groq, pick an embedding model — then click Connect.")

    st.markdown("---")
    st.subheader("How this GraphRAG system works")

    step_cols = st.columns(4)
    steps = [
        ("1. Ingest", "PDFs → text → chunks → **Groq LLM** extracts entities & relationships → embedded → written to **Neo4j** as a typed graph + vector store"),
        ("2. Vector Search", "Your question is **embedded** and matched against entity/chunk vectors using Neo4j's native `db.index.vector.queryNodes` — retrieves semantically closest nodes"),
        ("3. Graph Traversal", "From vector-matched entities, a **multi-hop Cypher query** (`MATCH (a)-[*1..2]-(b)`) expands the surrounding subgraph — the step plain vector search **cannot** do"),
        ("4. Answer Synthesis", "Groq LLM generates a **grounded, cited answer** from graph structure + retrieved text chunks — not a hallucination"),
    ]
    for col, (title, body) in zip(step_cols, steps):
        with col:
            st.markdown(f"**{title}**")
            st.markdown(body)

    st.markdown("---")
    st.subheader("Embedding model guide")
    emb_rows = [
        ["all-mpnet-base-v2",             "768",  "Local (no key)",   "⭐⭐ Good",  "Strong general-purpose model, runs on CPU"],
        ["all-MiniLM-L6-v2",             "384",  "Local (no key)",   "⭐ Fast",    "Lightweight, quick to load, good for demos"],
    ]
    st.table({
        "Model":       [r[0] for r in emb_rows],
        "Dims":        [r[1] for r in emb_rows],
        "Requires":    [r[2] for r in emb_rows],
        "Quality":     [r[3] for r in emb_rows],
        "Notes":       [r[4] for r in emb_rows],
    })

else:
    tab_chat, tab_graph, tab_log = st.tabs([
        "💬 Chat",
        "🕸️ Graph Explorer",
        "📋 Ingestion History",
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — CHAT
    # ════════════════════════════════════════════════════════════════════
    with tab_chat:

        # ── Retrieval config (collapsible) ────────────────────────────
        with st.expander("⚙️ Retrieval settings", expanded=False):
            rc1, rc2, rc3 = st.columns(3)
            vector_top_k  = rc1.slider("Vector top-k entities",   2, 20, 8,  1)
            chunk_top_k   = rc2.slider("Vector top-k chunks",     1, 10, 4,  1)
            graph_hops    = rc3.slider("Graph traversal hops",    1,  4, 2,  1)
            st.caption(
                "**Vector top-k**: how many semantically similar entities / chunks to retrieve from Neo4j.  "
                "**Graph hops**: how many relationship hops to expand from the matched entities."
            )

        # ── Chat history ──────────────────────────────────────────────
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant" and msg.get("meta"):
                    meta = msg["meta"]

                    with st.expander("🔍 Retrieval trace — how this answer was built"):

                        # Reasoning log
                        st.markdown("**Step-by-step reasoning log:**")
                        for step in meta.get("reasoning_log", []):
                            st.markdown(f"- {step}")

                        # Stats bar
                        st.markdown("**Retrieval stats:**")
                        s_cols = st.columns(5)
                        stat_keys = [
                            ("vector_entities_found", "Vector Entities"),
                            ("vector_chunks_found",   "Vector Chunks"),
                            ("subgraph_nodes",        "Graph Nodes"),
                            ("subgraph_edges",        "Graph Edges"),
                            ("hops",                  "Hops"),
                        ]
                        for col, (k, label) in zip(s_cols, stat_keys):
                            col.metric(label, meta.get("stats", {}).get(k, "—"))

                        st.caption(f"⏱️ Total query time: {meta.get('elapsed_seconds', '?')}s")

                        # Embedding model used
                        if meta.get("embedding_backend"):
                            st.caption(
                                f"🔢 Embedding: **{meta['embedding_backend'].upper()}**  "
                                f"({meta.get('embedding_dim','?')}d)"
                            )

                        # Graph relationships traversed
                        subgraph_edges = meta.get("subgraph", {}).get("edges", [])
                        if subgraph_edges:
                            st.markdown("**Graph relationships traversed (Cypher multi-hop):**")
                            for e in subgraph_edges[:15]:
                                st.markdown(
                                    f'<span class="hop-line">'
                                    f'{e["source"]}  —[{e["relation"]}]→  {e["target"]}'
                                    f'</span>',
                                    unsafe_allow_html=True,
                                )
                            if len(subgraph_edges) > 15:
                                st.caption(f"… and {len(subgraph_edges)-15} more edges")

                        # Vector-matched chunks
                        vector_chunks = meta.get("vector_chunks", [])
                        if vector_chunks:
                            st.markdown("**Top retrieved text chunks (from Neo4j vector search):**")
                            for i, chunk in enumerate(vector_chunks[:3]):
                                with st.expander(
                                    f"Chunk {i+1} — {chunk.get('source_doc','?')}  "
                                    f"(score {chunk.get('score', 0):.3f})"
                                ):
                                    st.text(chunk.get("text", "")[:600])

        # ── Suggested questions ───────────────────────────────────────
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "What excursions are available in Cozumel?",
            "Which ship has the FlowRider surf simulator?",
            "Recommend a snorkeling excursion for families",
            "What dining options are available on Icon of the Seas?",
            "Which itineraries visit both St. Lucia and Barbados?",
            "What do passengers say about the accessible facilities?",
        ]
        sugg_cols = st.columns(3)
        suggestion_clicked = None
        for col, s in zip(list(sugg_cols) * 2, suggestions):
            if col.button(s, use_container_width=True, key=f"sugg_{s[:18]}"):
                suggestion_clicked = s

        # ── Chat input ────────────────────────────────────────────────
        with st.form(key="chat_input_form", clear_on_submit=True):
            ci_col, btn_col = st.columns([5, 1])
            with ci_col:
                user_question = st.text_input(
                    "Ask",
                    placeholder="e.g. Which ships visit Cozumel and what excursions are available?",
                    label_visibility="collapsed",
                )
            with btn_col:
                send_clicked = st.form_submit_button("Send ➤", use_container_width=True)

        final_question = None
        if send_clicked and user_question.strip():
            final_question = user_question.strip()
        elif suggestion_clicked:
            final_question = suggestion_clicked

        if final_question:
            log(f"[app] Question: '{final_question}'")
            st.session_state.messages.append({"role": "user", "content": final_question})

            with st.spinner("🔎 Embedding → Neo4j vector search → graph traversal → Groq synthesis..."):
                try:
                    t0     = time.time()
                    result = st.session_state.pipeline.query(
                        final_question,
                        vector_top_k=vector_top_k,
                        hops=graph_hops,
                        chunk_top_k=chunk_top_k,
                    )
                    elapsed = time.time() - t0
                    log(f"[app] ✅ Answered in {elapsed:.2f}s  "
                        f"(entities={result['stats'].get('vector_entities_found','?')}, "
                        f"chunks={result['stats'].get('vector_chunks_found','?')}, "
                        f"graph_nodes={result['stats'].get('subgraph_nodes','?')})")

                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": result["answer"],
                        "meta":    result,
                    })
                except Exception as e:
                    log(f"[app]: Query failed: {e}")
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": f"Something went wrong: {e}",
                        "meta":    None,
                    })
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
            st.rerun()

        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑️ Clear chat history", key="clear_chat"):
                st.session_state.messages = []
                st.rerun()

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — GRAPH EXPLORER
    # ════════════════════════════════════════════════════════════════════
    with tab_graph:
        st.markdown("### 🕸️ Knowledge Graph")
        st.caption("Entities and relationships extracted from your PDFs and stored in Neo4j.")

        ge_col1, ge_col2 = st.columns([3, 1])
        graph_limit = ge_col1.slider("Max nodes to show", 50, 300, 150, 25)
        if ge_col2.button("🔄 Refresh", use_container_width=True):
            st.rerun()

        try:
            graph_data = st.session_state.pipeline.get_full_graph(limit=graph_limit)
            n_nodes, n_edges = len(graph_data["nodes"]), len(graph_data["edges"])
            st.write(f"Showing **{n_nodes} nodes** · **{n_edges} relationships** (cap={graph_limit})")

            # Type filter
            entity_types = sorted({
                n.get("type") or "Unknown"
                for n in graph_data["nodes"]
                })
            selected_types = st.multiselect(
                "Filter by entity type", entity_types, default=entity_types
            )
            filtered_nodes = [
                n for n in graph_data["nodes"]
                if (n.get("type") or "Unknown") in selected_types
            ] 
            filtered_ids   = {n["id"] for n in filtered_nodes}
            filtered_edges = [
                e for e in graph_data["edges"]
                if e["source"] in filtered_ids and e["target"] in filtered_ids
            ]

            # Render graph
            try:
                from streamlit_agraph import agraph, Node, Edge, Config

                TYPE_COLORS = {
                    "CruiseLine":  "#003087", "Ship":       "#007B8A",
                    "Port":        "#22D3A5", "Destination":"#A78BFA",
                    "Excursion":   "#F59E0B", "Amenity":    "#34D399",
                    "Cabin":       "#F472B6", "Restaurant": "#EF4444",
                    "Activity":    "#3B82F6", "Review":     "#9333EA",
                    "Passenger":   "#EC4899", "Policy":     "#6B7280",
                    "Package":     "#10B981", "Other":      "#9CA3AF",
                }
                nodes = [
                    Node(
                        id=n["id"],
                        label=n["label"][:22],
                        size=20,
                        color=TYPE_COLORS.get(n["type"], "#9CA3AF"),
                        title=f"{n['type']}: {n.get('description','')[:120]}",
                    )
                    for n in filtered_nodes
                ]
                edges = [
                    Edge(source=e["source"], target=e["target"], label=e["relation"])
                    for e in filtered_edges
                ]
                config = Config(
                    width="100%", height=580,
                    directed=True, physics=True, hierarchical=False,
                )
                agraph(nodes=nodes, edges=edges, config=config)

            except ImportError:
                st.warning(
                    "Install `streamlit-agraph` for interactive graph visualisation: "
                    "`pip install streamlit-agraph`"
                )
                st.json({"nodes": n_nodes, "edges": n_edges})

            # Raw tables
            with st.expander("🔍 Raw node / edge data"):
                dn, de = st.columns(2)
                with dn:
                    st.markdown(f"**Nodes ({len(filtered_nodes)})**")
                    st.dataframe(
                        [{"id": n["id"], "label": n["label"], "type": n["type"]}
                         for n in filtered_nodes],
                        use_container_width=True,
                    )
                with de:
                    st.markdown(f"**Edges ({len(filtered_edges)})**")
                    st.dataframe(filtered_edges, use_container_width=True)

        except Exception as e:
            st.error(f"Could not load graph: {e}")

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — INGESTION HISTORY + DEBUG LOG
    # ════════════════════════════════════════════════════════════════════
    with tab_log:
        st.markdown("### 📋 Ingestion History")

        if not st.session_state.ingestion_results:
            st.info("No PDFs ingested yet this session.")
        else:
            # Summary table
            summary_rows = []
            for r in st.session_state.ingestion_results:
                summary_rows.append({
                    "File":              r.get("source", "?"),
                    "Pages":             r.get("page_count", "?"),
                    "Chunks processed":  r.get("chunks_processed", "?"),
                    "Entities added":    r.get("entities_added", "?"),
                    "Relations added":   r.get("relations_added", "?"),
                    "Embedding backend": r.get("embedding_backend", "?"),
                    "Embedding dim":     r.get("embedding_dim", "?"),
                    "Time (s)":          r.get("elapsed_seconds", "?"),
                })
            st.dataframe(summary_rows, use_container_width=True)

            # Detail cards
            for res in st.session_state.ingestion_results:
                with st.expander(
                    f"📄 {res['source']}  —  "
                    f"{res['entities_added']} entities · "
                    f"{res['relations_added']} relations · "
                    f"{res.get('embedding_backend','?').upper()} {res.get('embedding_dim','?')}d"
                ):
                    # Sample entities table
                    sample_ents = res.get("sample_entities", [])
                    if sample_ents:
                        st.markdown("**Sample extracted entities:**")
                        st.dataframe(
                            [{"id": e.get("id"), "label": e.get("label"),
                              "type": e.get("type"), "description": e.get("description", "")[:80]}
                             for e in sample_ents[:12]],
                            use_container_width=True,
                        )
                    # Sample relations table
                    sample_rels = res.get("sample_relations", [])
                    if sample_rels:
                        st.markdown("**Sample extracted relationships:**")
                        st.dataframe(
                            [{"source": r.get("source"), "relation": r.get("relation"),
                              "target": r.get("target")}
                             for r in sample_rels[:10]],
                            use_container_width=True,
                        )
                    # Full JSON
                    with st.expander("Full result JSON"):
                        st.json({k: v for k, v in res.items()
                                 if k not in ("sample_entities", "sample_relations")})

        st.divider()
        st.markdown("### 🖥️ Live Debug Log")
        st.caption("Mirrors all print() statements from the Python backend (also visible in your terminal)")
        col_log, col_clear = st.columns([5, 1])
        with col_clear:
            if st.button("Clear log"):
                st.session_state.debug_log = []
                st.rerun()
        if st.session_state.debug_log:
            st.text_area(
                "Debug output",
                value="\n".join(st.session_state.debug_log[-250:]),
                height=340,
                label_visibility="collapsed",
            )
        else:
            st.caption("No log entries yet.")

print("[app.py] Render cycle complete.\n")
