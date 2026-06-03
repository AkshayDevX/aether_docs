import sys
import os
import json
import tempfile
import streamlit as st
from pyvis.network import Network
from dotenv import load_dotenv

# Ensure parent directory is in python path to load local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from database import GraphDatabase, sanitize_node_id
from embeddings import EmbeddingGenerator
from agent.parser import HierarchicalMarkdownParser
from agent.builder import builder_agent, db as builder_db
from agent.query import query_agent

# Load environment keys
load_dotenv(os.path.join(parent_dir, ".env"))

import re

def render_content_with_mermaid(text: str):
    # Match code blocks starting with ```mermaid or ```graph... and ending with ```
    pattern = re.compile(r"```(mermaid|graph.*?)\n(.*?)```", re.DOTALL | re.IGNORECASE)
    
    last_end = 0
    for match in pattern.finditer(text):
        start, end = match.span()
        # Render preceding markdown text
        prev_text = text[last_end:start].strip()
        if prev_text:
            st.markdown(prev_text)
            
        # Get diagram code
        lang = match.group(1)
        code = match.group(2).strip()
        
        # Prepend lang description if missing from the content itself (e.g. code is just raw connections)
        if not code.startswith(("graph", "flowchart", "sequenceDiagram", "classDiagram", "stateDiagram", "erDiagram", "gantt", "pie", "gitGraph")):
            if lang.startswith("graph") or lang.startswith("flowchart"):
                code = f"{lang}\n{code}"
                
        # Render Mermaid using HTML, CDN JavaScript, and svg-pan-zoom in an iframe
        mermaid_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                    background-color: #0f172a;
                    font-family: sans-serif;
                }}
                #container {{
                    width: 100%;
                    height: 100%;
                    box-sizing: border-box;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    background-color: #0f172a;
                    position: relative;
                }}
                #container svg {{
                    width: 100% !important;
                    height: 100% !important;
                    display: block;
                }}
                .svg-pan-zoom-control {{
                    fill: #38bdf8 !important;
                    fill-opacity: 0.8 !important;
                }}
                .svg-pan-zoom-control:hover {{
                    fill-opacity: 1.0 !important;
                }}
                .svg-pan-zoom-control-background {{
                    fill: #1e293b !important;
                    fill-opacity: 0.5 !important;
                }}
            </style>
        </head>
        <body>
            <div id="container">
                <pre class="mermaid" style="margin: 0; text-align: center; color: white;">
{code}
                </pre>
            </div>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ 
                    startOnLoad: true,
                    theme: 'dark',
                    securityLevel: 'loose',
                    themeVariables: {{
                        background: '#0f172a',
                        primaryColor: '#1e293b',
                        primaryTextColor: '#f8fafc',
                        lineColor: '#38bdf8'
                    }}
                }});

                // Periodically check if Mermaid has rendered and replaced the pre tag with an SVG
                const interval = setInterval(() => {{
                    const svg = document.querySelector('#container svg');
                    if (svg) {{
                        clearInterval(interval);
                        
                        // Style the SVG and parent wrappers to fill the container completely
                        let el = svg;
                        while (el && el.id !== 'container') {{
                            el.style.width = '100%';
                            el.style.height = '100%';
                            el.style.maxWidth = '100%';
                            el.style.maxHeight = '100%';
                            el.style.margin = '0';
                            el.style.padding = '0';
                            el = el.parentElement;
                        }}
                        
                        try {{
                            // Initialize svg-pan-zoom for premium panning & zooming experience
                            window.panZoom = svgPanZoom(svg, {{
                                zoomEnabled: true,
                                controlIconsEnabled: true,
                                fit: true,
                                center: true,
                                minZoom: 0.2,
                                maxZoom: 10,
                                zoomScaleSensitivity: 0.2
                            }});
                            
                            // Adjust zoom slightly on window resize to keep it centered
                            window.addEventListener('resize', () => {{
                                window.panZoom.resize();
                                window.panZoom.fit();
                                window.panZoom.center();
                            }});
                        }} catch (e) {{
                            console.error("svgPanZoom initialization failed: ", e);
                        }}
                    }}
                }}, 50);
            </script>
        </body>
        </html>
        """
        # Render the diagram in a fixed-height container that feels comfortable
        st.iframe(mermaid_html, height=450)
        
        last_end = end
        
    # Render remaining markdown text
    remaining_text = text[last_end:].strip()
    if remaining_text:
        st.markdown(remaining_text)


# 1. Premium CSS Injection
CSS = """
<style>
    /* Sleek background and modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stHeadingContainer h1, .stHeadingContainer h2 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Beautiful glassmorphism stats cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }
    .metric-val {
        font-size: 32px;
        font-weight: 800;
        color: #00C6FF;
    }
    .metric-lbl {
        font-size: 14px;
        color: #8892B0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* scrolling logs panel */
    .log-panel {
        font-family: 'Consolas', monospace;
        font-size: 13px;
        background-color: #0d1117;
        color: #58a6ff;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        height: 250px;
        overflow-y: scroll;
    }
    
    /* Human in the loop conflict approval card styling */
    .conflict-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1.5px solid #ef4444; /* red indicator */
        border-radius: 16px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    .conflict-header {
        font-size: 18px;
        font-weight: bold;
        color: #ef4444;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .similarity-badge {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .node-details {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #00c6ff;
    }
    
    .node-details.existing {
        border-left: 3px solid #10b981;
    }
    
    /* Decrease default top padding in Streamlit sidebar */
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebar"] > div:first-child,
    .stSidebar .sidebar-content {
        padding-top: 1rem !important;
    }
</style>
"""

# Initialize DB connection locally in UI to show stats
db = GraphDatabase()

# Set up Streamlit Page Settings
st.set_page_config(
    page_title="AetherDocs - Technical Knowledge Graph Agent",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom styled aesthetics
st.markdown(CSS, unsafe_allow_html=True)

# 2. Sidebar Navigation and Quick Info
with st.sidebar:
    # Premium Centered Header Section (Logo, Title, and Subtitle)
    logo_path = os.path.join(current_dir, "logo.png")
    if os.path.exists(logo_path):
        import base64
        with open(logo_path, "rb") as f:
            encoded_base64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style="text-align: center; padding: 0 0 10px 0; margin-bottom: 10px;">
                <img src="data:image/png;base64,{encoded_base64}" 
                     style="width: 110px; height: 110px; border-radius: 50%; 
                            border: 2px solid rgba(0, 198, 255, 0.5); 
                            box-shadow: 0 0 30px rgba(0, 198, 255, 0.3); 
                            object-fit: cover; margin-bottom: 15px;
                            transition: transform 0.3s ease;" 
                     alt="AetherDocs Logo" />
                <h1 style="font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 800; margin: 0; 
                           background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    AetherDocs
                </h1>
                <p style="font-family: 'Outfit', sans-serif; font-size: 13px; color: #8892B0; margin: 8px 0 0 0; line-height: 1.4;">
                    Self-Structuring Knowledge Graph Agent
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown("<h2 style='text-align: center; color: #00C6FF;'>🕸️ AetherDocs</h2>", unsafe_allow_html=True)
    st.write("---")
    
    # Dynamic Project Workspace Selector
    st.markdown("### 📁 Active Project Workspace")
    available_projects = db.get_all_projects()
    
    workspace_options = ["All Projects"] + available_projects
    # Ensure active_project in session_state is valid
    if "active_project" not in st.session_state or st.session_state.active_project not in workspace_options:
        st.session_state.active_project = "All Projects"
        
    selected_project = st.selectbox(
        "Select active project workspace:",
        options=workspace_options,
        index=workspace_options.index(st.session_state.active_project),
        label_visibility="collapsed"
    )
    st.session_state.active_project = selected_project
    st.write("---")
    
    # Model Selection Display
    st.markdown("### 🧠 Primary Cognitive Brain")
    st.info("🤖 **DeepSeek-V4-Flash**\n- Base URL: `api.deepseek.com`\n- Context: `1,000,000 Tokens`\n- Hybrid Semantic Embedding: `Local SentenceTransformer`")
    
    # Simple Database Maintenance Tool
    st.markdown("### 🛠️ DB Maintenance")
    if st.button("🚨 Reset Knowledge Graph", type="secondary", use_container_width=True):
        conn = db.get_connection()
        c = conn.cursor()
        c.execute("DELETE FROM edges;")
        c.execute("DELETE FROM nodes;")
        conn.commit()
        conn.close()
        st.success("Knowledge Graph Reset successfully.")
        st.rerun()

# Header banner
st.title("🕸️ AetherDocs Architecture Graph Agent")
st.write("Ingest complex software documentation, extract entities, resolve semantic duplicates with Human-in-the-Loop approval, and perform multi-hop Graph-RAG queries.")
st.write("---")

# Tab Layout
tab_ingest, tab_approval, tab_query = st.tabs([
    "📂 Ingest Documentation",
    "🤝 Merge Approvals (Human-in-the-Loop)",
    "🔍 Search & Explore Context Graph"
])

# Define default thread id for our single-user local graph ingestion session
THREAD_ID = "ingest_thread_session"


# ==========================================
# TAB 1: INGESTION & STATS PANEL
# ==========================================
with tab_ingest:
    st.header("Upload Documentation & Code Specs")
    
    # Row 1: Graph Stats Cards
    col_n, col_e, col_l = st.columns(3)
    
    # Filter stats by selected project workspace
    active_proj = st.session_state.get("active_project", "All Projects")
    project_filter = None if active_proj == "All Projects" else active_proj
    all_nodes = db.get_all_nodes(project=project_filter)
    all_edges = db.get_all_edges(project=project_filter)
    
    labels_counts = {}
    for node in all_nodes:
        labels_counts[node["label"]] = labels_counts.get(node["label"], 0) + 1
        
    lbls_str = ", ".join([f"{k}: {v}" for k, v in labels_counts.items()]) if labels_counts else "None yet"
    
    with col_n:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(all_nodes)}</div><div class="metric-lbl">Total Entities (Nodes)</div></div>', unsafe_allow_html=True)
    with col_e:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(all_edges)}</div><div class="metric-lbl">Total Relationships (Edges)</div></div>', unsafe_allow_html=True)
    with col_l:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{len(labels_counts)}</div><div class="metric-lbl">Unique Node Labels ({lbls_str})</div></div>', unsafe_allow_html=True)
        
    st.write("---")
    
    # Project Workspace Name Input for Ingestion
    st.markdown("### 🏷️ Target Project Workspace")
    default_proj_name = "EventSpine" if active_proj == "All Projects" else active_proj
    ingest_project = st.text_input(
        "Enter target project name to tag these documents:",
        value=default_proj_name,
        placeholder="e.g. EventSpine, AetherDocs, BillingSystem"
    ).strip()
    
    if not ingest_project:
        ingest_project = "Default Project"
        
    st.write("---")
    
    # File Uploader
    uploaded_files = st.file_uploader("Upload technical Markdown specifications or Code READMEs:", type=["md", "txt"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("🚀 Build Architecture Knowledge Graph", type="primary", use_container_width=True):
            all_chunks = []
            
            # Save files to temp directory and parse them
            with tempfile.TemporaryDirectory() as temp_dir:
                for idx, uploaded_file in enumerate(uploaded_files):
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Parse hierarchically
                    file_chunks = HierarchicalMarkdownParser.parse_file(temp_file_path)
                    all_chunks.extend(file_chunks)
                    
            if not all_chunks:
                st.warning("No logical sections or markdown content extracted from files.")
            else:
                st.success(f"Parsed files into **{len(all_chunks)}** structural markdown chunks.")
                
                # Initialize state values for LangGraph thread
                initial_state = {
                    "document_chunks": all_chunks,
                    "current_chunk_idx": 0,
                    "project": ingest_project,
                    "extracted_nodes": [],
                    "extracted_edges": [],
                    "unresolved_merges": [],
                    "approved_merges": {},
                    "rejected_merges": [],
                    "logs": [f"Parsed {len(all_chunks)} chunks for project '{ingest_project}'. Initiating LangGraph Builder..."]
                }
                
                # Setup progress bars
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                log_placeholder = st.empty()
                
                config = {"configurable": {"thread_id": THREAD_ID}}
                
                # We start or reset the LangGraph thread state
                builder_agent.update_state(config, initial_state)
                
                # Run extraction loop
                logs_display = []
                
                # Loop through chunk extraction
                while True:
                    # Fetch current state to see cursor position
                    current_state = builder_agent.get_state(config)
                    cursor = current_state.values.get("current_chunk_idx", 0)
                    total_chunks = len(all_chunks)
                    
                    if cursor >= total_chunks:
                        progress_bar.progress(1.0)
                        status_text.success("All document sections processed! Ingestion complete. ✅")
                        break
                        
                    progress_pct = float(cursor) / float(total_chunks)
                    progress_bar.progress(progress_pct)
                    status_text.info(f"Processing chunk {cursor + 1} of {total_chunks}...")
                    
                    # Stream execution of the agent on the thread
                    # The stream will pause if it hits the interrupt_before=["human_approval_node"] directive!
                    try:
                        for event in builder_agent.stream(None, config, stream_mode="updates"):
                            # Update logs
                            updated_state = builder_agent.get_state(config)
                            logs_display = updated_state.values.get("logs", [])
                            
                            log_html = "".join([f"<div>{l}</div>" for l in logs_display[-12:]])
                            log_placeholder.markdown(f'<div class="log-panel">{log_html}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error during agent execution: {e}")
                        break
                        
                    # Check if the thread is currently suspended at an interrupt
                    next_state = builder_agent.get_state(config)
                    next_steps = next_state.next
                    unresolved = next_state.values.get("unresolved_merges", [])
                    
                    # If we are paused BEFORE human_approval_node, break and notify the user to resolve conflicts
                    if "human_approval_node" in next_steps or len(unresolved) > 0:
                        status_text.warning(f"Ingestion PAUSED. Flagged **{len(unresolved)}** borderline duplicate concepts requiring merge review!")
                        st.info("👉 Head over to the **'Merge Approvals'** tab to review duplicates and resume extraction!")
                        break
                
                st.rerun()


# ==========================================
# TAB 2: HUMAN-IN-THE-LOOP APPROVALS
# ==========================================
with tab_approval:
    st.header("🤝 Deduplication Merge Approval Center")
    st.write("Review entities extracted by DeepSeek that have high semantic similarity to existing concepts in your Knowledge Graph. Approve or reject merges to prevent graph fragmentation.")
    st.write("---")
    
    config = {"configurable": {"thread_id": THREAD_ID}}
    state_data = builder_agent.get_state(config)
    unresolved_merges = state_data.values.get("unresolved_merges", []) if state_data.values else []
    
    if not unresolved_merges:
        st.success("Zero pending duplicates to review! Your Knowledge Graph is perfectly clean. ✨")
    else:
        st.warning(f"Found **{len(unresolved_merges)}** concepts requiring validation before completing ingestion.")
        
        # Display the active conflict cards
        for idx, conflict in enumerate(unresolved_merges):
            extracted = conflict["extracted"]
            existing = conflict["existing"]
            similarity = conflict["similarity"]
            
            st.markdown(f"""
            <div class="conflict-card">
                <div class="conflict-header">
                    <span>⚠️ Borderline Semantic Entity Collision</span>
                    <span class="similarity-badge">Similarity: {int(similarity * 100)}%</span>
                </div>
                <div class="row">
                    <div class="node-details">
                        <strong>📌 Extracted Node (New):</strong> {extracted['name']}<br/>
                        <small>Label: {extracted['label']} | {extracted['description']}</small>
                    </div>
                    <div class="node-details existing">
                        <strong>✅ Matches Existing Database Node:</strong> {existing['name']}<br/>
                        <small>Label: {existing['label']} | {existing['description']}</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action Buttons in Columns
            col_app, col_rej, col_custom = st.columns(3)
            
            with col_app:
                if st.button(f"🤝 Merge into '{existing['name']}'", key=f"app_{idx}", type="primary", use_container_width=True):
                    # Record approval
                    approved = dict(state_data.values.get("approved_merges", {}))
                    approved[extracted["temp_id"]] = existing["id"]
                    
                    # Remove this resolved item from the state list
                    remaining = [c for i, c in enumerate(unresolved_merges) if i != idx]
                    
                    builder_agent.update_state(
                        config=config,
                        values={"approved_merges": approved, "unresolved_merges": remaining},
                        as_node="entity_resolution_node"
                    )
                    st.toast(f"Merged '{extracted['name']}' into '{existing['name']}'")
                    st.rerun()
                    
            with col_rej:
                if st.button(f"🆕 Keep as Separate Entity", key=f"rej_{idx}", type="secondary", use_container_width=True):
                    # Record rejection
                    rejected = list(state_data.values.get("rejected_merges", []))
                    rejected.append(extracted["temp_id"])
                    
                    remaining = [c for i, c in enumerate(unresolved_merges) if i != idx]
                    
                    builder_agent.update_state(
                        config=config,
                        values={"rejected_merges": rejected, "unresolved_merges": remaining},
                        as_node="entity_resolution_node"
                    )
                    st.toast(f"Saved '{extracted['name']}' as separate node.")
                    st.rerun()
                    
            with col_custom:
                # Custom input for renaming
                new_custom_name = st.text_input("Or input a custom distinct name:", value=extracted["name"], key=f"custom_name_{idx}")
                if st.button("✏️ Rename & Create", key=f"btn_custom_{idx}", use_container_width=True):
                    # If customized to existing, it merges. If new, it creates.
                    new_temp_id = sanitize_node_id(new_custom_name)
                    
                    # Update name in the extracted list
                    extracted_list = list(state_data.values.get("extracted_nodes", []))
                    for node in extracted_list:
                        if sanitize_node_id(node["name"]) == extracted["temp_id"]:
                            node["name"] = new_custom_name
                            
                    rejected = list(state_data.values.get("rejected_merges", []))
                    rejected.append(new_temp_id)
                    
                    remaining = [c for i, c in enumerate(unresolved_merges) if i != idx]
                    
                    builder_agent.update_state(
                        config=config,
                        values={
                            "extracted_nodes": extracted_list,
                            "rejected_merges": rejected,
                            "unresolved_merges": remaining
                        },
                        as_node="entity_resolution_node"
                    )
                    st.toast(f"Renamed entity to '{new_custom_name}' and saved.")
                    st.rerun()
                    
        st.write("---")
        
        # Action to Resume the LangGraph ingestion thread after resolving approvals
        if not unresolved_merges:
            if st.button("⚡ Resume Ingestion Graph", type="primary", use_container_width=True):
                progress_bar = st.progress(0.9)
                status_text = st.empty()
                status_text.info("Resuming extraction stream...")
                
                # Resume execution: passing None triggers the thread checkpointer to resume from current paused node
                try:
                    for event in builder_agent.stream(None, config, stream_mode="updates"):
                        pass
                    
                    status_text.success("Ingestion resumed and completed successfully! ✅")
                except Exception as e:
                    st.error(f"Error resuming graph: {e}")
                    
                st.rerun()


# ==========================================
# TAB 3: EXPLORE & HYBRID SEARCH CHAT
# ==========================================
with tab_query:
    col_header, col_clear = st.columns([0.8, 0.2])
    with col_header:
        st.header("🔍 Search & Explore Context Graph")
    with col_clear:
        if st.button("🧹 Clear Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.toast("Chat history cleared!")
            st.rerun()
            
    st.write("Query your technical documentation using DeepSeek-V4-Flash. The agent locates the target nodes semantically, traverses relationships, and renders a live dynamic sub-graph visualization.")
    st.write("---")
    
    # Initialize message list in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_content_with_mermaid(msg["content"])
            else:
                st.markdown(msg["content"])
            if "graph_html" in msg and msg["graph_html"]:
                st.iframe(msg["graph_html"], height=450)
                
    # New Input Query
    user_query = st.chat_input("Ask a question about the architecture (e.g. 'What tables does Auth Service depend on?')...")
    
    if user_query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display loader
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            log_placeholder = st.empty()
            
            status_placeholder.info("🤖 Spawning Graph-RAG Query Agent...")
            
            # Map existing st.session_state.messages into LangChain message primitives for context history
            from langchain_core.messages import HumanMessage, AIMessage
            agent_messages = []
            
            for p_msg in st.session_state.messages[:-1]: # exclude the current user query just appended
                if p_msg["role"] == "user":
                    agent_messages.append(HumanMessage(content=p_msg["content"]))
                else:
                    agent_messages.append(AIMessage(content=p_msg["content"]))
            
            # Execute Query workflow
            active_proj = st.session_state.get("active_project", "All Projects")
            project_filter = None if active_proj == "All Projects" else active_proj
            
            initial_state = {
                "query": user_query,
                "project": project_filter,
                "retrieved_nodes": [],
                "retrieved_edges": [],
                "answer": "",
                "logs": ["Starting Graph-RAG query execution..."],
                "messages": agent_messages
            }
            
            try:
                # Stream events from compiled LangGraph
                logs_display = []
                for event in query_agent.stream(initial_state, stream_mode="updates"):
                    # Extract active nodes/logs
                    for node_name, values in event.items():
                        logs_display.extend(values.get("logs", []))
                        
                    # Filter unique logs
                    seen = set()
                    unique_logs = [x for x in logs_display if not (x in seen or seen.add(x))]
                    
                    log_html = "".join([f"<div>{l}</div>" for l in unique_logs[-8:]])
                    log_placeholder.markdown(f'<div class="log-panel" style="height:150px;">{log_html}</div>', unsafe_allow_html=True)
                
                result = query_agent.invoke(initial_state)
                
                status_placeholder.empty()
                log_placeholder.empty()
                
                # Display Answer
                render_content_with_mermaid(result["answer"])
                
                # Visualizing the Retrieved Sub-Graph Network in PyVis!
                retrieved_nodes = result["retrieved_nodes"]
                retrieved_edges = result["retrieved_edges"]
                
                graph_html = ""
                if retrieved_nodes:
                    st.write("---")
                    st.subheader("🕸️ Retrieved Architectural Sub-Graph Context")
                    
                    # Create PyVis Network
                    net = Network(height="400px", width="100%", bgcolor="#0f172a", font_color="#f8fafc", directed=True)
                    # Use Barnes-Hut algorithm for smooth gravity physics
                    net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=150)
                    
                    # Add Nodes
                    for node in retrieved_nodes:
                        # Choose color based on label category
                        color_map = {
                            "API": "#00C6FF",
                            "Database": "#10B981",
                            "Component": "#F59E0B",
                            "File": "#8B5CF6",
                            "Developer": "#EC4899",
                            "Library": "#14B8A6"
                        }
                        color = color_map.get(node["label"], "#64748B")
                        
                        net.add_node(
                            node["id"],
                            label=node["name"],
                            title=f"Label: {node['label']}\nDescription: {node['description']}",
                            color=color,
                            size=25 if node["id"] in [n["id"] for n in result["retrieved_nodes"][:4]] else 18
                        )
                        
                    # Add Edges
                    for edge in retrieved_edges:
                        net.add_edge(
                            edge["source"],
                            edge["target"],
                            label=edge["relationship"],
                            title=json.dumps(edge["properties"]),
                            color="#38bdf8",
                            width=2
                        )
                        
                    # Save pyvis file temporarily
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_graph_path = os.path.join(temp_dir, "graph.html")
                        net.save_graph(temp_graph_path)
                        
                        # Read HTML to embed
                        with open(temp_graph_path, "r", encoding="utf-8") as f:
                            graph_html = f.read()
                            
                    st.iframe(graph_html, height=450)
                else:
                    st.info("No nodes were retrieved in local context. The response was synthesized using background models.")
                
                # Append assistant response to in-memory messages state
                msg_entry = {"role": "assistant", "content": result["answer"], "graph_html": graph_html if graph_html else None}
                st.session_state.messages.append(msg_entry)
                st.rerun()
                
            except Exception as e:
                st.error(f"Failed to execute Graph-RAG pipeline: {e}")
