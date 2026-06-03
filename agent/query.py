import os
import json
from typing import Dict, Any

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Project imports
from agent.state import GraphQueryState
from database import GraphDatabase
from embeddings import EmbeddingGenerator

# Initialize database and embeddings
db = GraphDatabase()
embedder = EmbeddingGenerator()

def get_llm() -> ChatDeepSeek:
    """Instantiates the DeepSeek model pointing to the official API endpoints."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY. Please verify your environment setup.")

    return ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0.1,
        extra_body={"thinking": {"type": "disabled"}}
    )


# 1. LangGraph Query Node Functions
def semantic_retrieval_node(state: GraphQueryState) -> Dict[str, Any]:
    """Node: Performs semantic search to locate the most relevant starting nodes in the graph."""
    query = state["query"]
    project_name = state.get("project")
    logs = list(state.get("logs", []))
    
    scope_desc = f" in project '{project_name}'" if project_name else ""
    logs.append(f"Initiating semantic search for starting nodes matching query: '{query}'{scope_desc}...")
    
    # Generate query embedding
    query_vector = embedder.get_embedding(query)
    
    # Retrieve top 4 matching nodes semantically
    semantic_matches = db.semantic_search_nodes(query_vector, top_k=4, project=project_name)
    
    retrieved_nodes = []
    logs.append(f"Semantic search completed. Found {len(semantic_matches)} candidate seed nodes:")
    
    for node, score in semantic_matches:
        retrieved_nodes.append(node)
        logs.append(f" - Seed: '{node['name']}' (Label: '{node['label']}', Score: {score:.2f})")
        
    return {
        "retrieved_nodes": retrieved_nodes,
        "logs": logs
    }


def neighborhood_traversal_node(state: GraphQueryState) -> Dict[str, Any]:
    """Node: Crawls incoming/outgoing edges of seed nodes to capture structural relations."""
    seed_nodes = state["retrieved_nodes"]
    logs = list(state.get("logs", []))
    
    logs.append("Initiating structural sub-graph neighborhood traversal (1-degree connections)...")
    
    retrieved_nodes_map = {node["id"]: node for node in seed_nodes}
    retrieved_edges = []
    
    edge_tracker = set() # Prevent duplicate edges
    
    for seed in seed_nodes:
        seed_id = seed["id"]
        neighbors = db.get_node_neighbors(seed_id)
        
        # Process Outgoing edges & targets
        for conn in neighbors["outgoing"]:
            edge = conn["edge"]
            node = conn["node"]
            
            # Record unique edge
            edge_key = f"{seed_id}-{node['id']}-{edge['relationship']}"
            if edge_key not in edge_tracker:
                edge_tracker.add(edge_key)
                retrieved_edges.append({
                    "source": seed_id,
                    "target": node["id"],
                    "relationship": edge["relationship"],
                    "properties": edge["properties"]
                })
            
            # Record neighbor node
            if node["id"] not in retrieved_nodes_map:
                retrieved_nodes_map[node["id"]] = node
                
        # Process Incoming edges & sources
        for conn in neighbors["incoming"]:
            edge = conn["edge"]
            node = conn["node"]
            
            # Record unique edge
            edge_key = f"{node['id']}-{seed_id}-{edge['relationship']}"
            if edge_key not in edge_tracker:
                edge_tracker.add(edge_key)
                retrieved_edges.append({
                    "source": node["id"],
                    "target": seed_id,
                    "relationship": edge["relationship"],
                    "properties": edge["properties"]
                })
            
            # Record neighbor node
            if node["id"] not in retrieved_nodes_map:
                retrieved_nodes_map[node["id"]] = node
                
    logs.append(f"Traversal complete: Captured structural local context consisting of "
                f"{len(retrieved_nodes_map)} nodes and {len(retrieved_edges)} edge relationships.")
                
    return {
        "retrieved_nodes": list(retrieved_nodes_map.values()),
        "retrieved_edges": retrieved_edges,
        "logs": logs
    }


def answer_synthesis_node(state: GraphQueryState) -> Dict[str, Any]:
    """Node: Serializes retrieved sub-graph context into Markdown and synthesizes an answer using DeepSeek."""
    query = state["query"]
    nodes = state["retrieved_nodes"]
    edges = state["retrieved_edges"]
    logs = list(state.get("logs", []))
    
    logs.append("Serializing sub-graph context and calling DeepSeek-V4-Flash for synthesis...")
    
    # 1. Format nodes section
    nodes_md = []
    for n in nodes:
        props = f" | Metadata: {json.dumps(n['properties'])}" if n['properties'] else ""
        nodes_md.append(f"- **{n['name']}** (ID: `{n['id']}`, Label: `{n['label']}`): {n['description']}{props}")
        
    nodes_context = "\n".join(nodes_md) if nodes_md else "No nodes found in sub-graph context."
    
    # 2. Format edges section
    edges_md = []
    for e in edges:
        props = f" (Details: {json.dumps(e['properties'])})" if e['properties'] else ""
        edges_md.append(f"- `{e['source']}` --[{e['relationship']}]--> `{e['target']}`{props}")
        
    edges_context = "\n".join(edges_md) if edges_md else "No relationship connections found in sub-graph context."
    
    # Formulate complete context prompt
    graph_context = (
        "### STRUCTURED KNOWLEDGE GRAPH CONTEXT\n\n"
        "#### TECHNICAL ENTITIES (NODES):\n"
        f"{nodes_context}\n\n"
        "#### ARCHITECTURAL RELATIONSHIPS (EDGES):\n"
        f"{edges_context}"
    )
    
    # Retrieve past conversation history
    messages = list(state.get("messages", []))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Solutions Architect and Lead AI Engineer.\n"
            "Your task is to answer a developer query by utilizing the provided structured Knowledge Graph context "
            "representing our codebase architecture.\n\n"
            "Explain technical relationships, call-flows, dependencies, and owners clearly based *strictly* on the "
            "graph nodes and edges provided. If the graph context doesn't contain enough information, explain "
            "what was found in the graph, and detail what is missing.\n\n"
            "Represent paths or call chains cleanly using markdown diagrams (e.g. A -> B -> C) where applicable.\n\n"
            "Knowledge Graph Context:\n"
            "------------------------\n"
            "{graph_context}\n"
            "------------------------"
        )),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{query}")
    ])
    
    try:
        llm = get_llm()
        chain = prompt | llm
        
        response = chain.invoke({
            "graph_context": graph_context,
            "history": messages,
            "query": query
        })
        
        answer = response.content
        logs.append("Answer synthesized successfully.")
        
        # Append the new dialog turn to history
        messages.append(HumanMessage(content=query))
        messages.append(AIMessage(content=answer))
        
        return {
            "answer": answer,
            "messages": messages,
            "logs": logs
        }
    except Exception as e:
        error_msg = f"Failed to synthesize response due to error: {e}"
        logs.append(error_msg)
        
        messages.append(HumanMessage(content=query))
        messages.append(AIMessage(content=error_msg))
        
        return {
            "answer": error_msg,
            "messages": messages,
            "logs": logs
        }


# 2. Compile the LangGraph query state machine
query_workflow = StateGraph(GraphQueryState)

# Add Nodes
query_workflow.add_node("semantic_retrieval_node", semantic_retrieval_node)
query_workflow.add_node("neighborhood_traversal_node", neighborhood_traversal_node)
query_workflow.add_node("answer_synthesis_node", answer_synthesis_node)

# Set Entrypoint
query_workflow.set_entry_point("semantic_retrieval_node")

# Define Transitions
query_workflow.add_edge("semantic_retrieval_node", "neighborhood_traversal_node")
query_workflow.add_edge("neighborhood_traversal_node", "answer_synthesis_node")
query_workflow.add_edge("answer_synthesis_node", END)

# Compile Q&A agent (no checkpointer/interrupts needed since Q&A is synchronous and fast)
query_agent = query_workflow.compile()
