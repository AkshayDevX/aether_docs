import os
import re
import json
import numpy as np
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

# Project imports
from agent.state import GraphBuilderState
from database import GraphDatabase, sanitize_node_id
from embeddings import EmbeddingGenerator

# 1. Pydantic Schemas for Structured LLM Extraction
class ExtractedNode(BaseModel):
    name: str = Field(description="Name of the software component, library, file, API endpoint, or developer (e.g., 'Auth Service', 'users.py', 'PostgreSQL', 'Akshay').")
    label: str = Field(description="Unified category. Select one: 'API', 'Database', 'Component', 'File', 'Developer', 'Library', 'Protocol', 'Variable'.")
    description: str = Field(description="A concise one-sentence description explaining its technical function or role.")
    properties: Dict[str, str] = Field(default_factory=dict, description="Metadata details (e.g., {'port': '8080', 'language': 'Python', 'owner': 'Auth Team'}).")

class ExtractedEdge(BaseModel):
    source: str = Field(description="The name of the source node.")
    target: str = Field(description="The name of the destination node.")
    relationship: str = Field(description="Uppercase action label. Select one: 'CALLS', 'IMPORTS', 'DEPENDS_ON', 'WRITES_TO', 'READS_FROM', 'IMPLEMENTS', 'DEVELOPED_BY'.")
    properties: Dict[str, str] = Field(default_factory=dict, description="Additional details (e.g., {'protocol': 'gRPC', 'payload': 'JSON'}).")

class ExtractionResult(BaseModel):
    nodes: List[ExtractedNode] = Field(default_factory=list, description="List of technical components extracted from the text.")
    edges: List[ExtractedEdge] = Field(default_factory=list, description="List of directed relationships connecting the extracted nodes.")


# Initialize shared database and embedding generator
db = GraphDatabase()
embedder = EmbeddingGenerator()

def get_llm() -> ChatDeepSeek:
    """Instantiates the DeepSeek model pointing to the official API endpoints."""
    # Read keys from env
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY. Please verify your environment setup.")
    
    return ChatDeepSeek(
        model="deepseek-v4-flash",
        temperature=0.1,
        extra_body={"thinking": {"type": "disabled"}}
    )


# 2. LangGraph Node Functions
def extract_chunk_node(state: GraphBuilderState) -> Dict[str, Any]:
    """Node: Reads the current chunk and extracts entities/edges using DeepSeek structured output."""
    chunks = state["document_chunks"]
    idx = state["current_chunk_idx"]
    logs = list(state.get("logs", []))
    
    if idx >= len(chunks):
        logs.append("All chunks processed successfully. Exiting extraction loop.")
        return {"logs": logs}
        
    active_chunk = chunks[idx]
    chunk_content = active_chunk["content"]
    chunk_heading = active_chunk["heading"]
    chunk_source = active_chunk["source"]
    
    logs.append(f"Ingesting section '{chunk_heading}' from file '{chunk_source}'...")
    
    # Prompt Setup
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Senior Software Architect. Your task is to extract a technical Knowledge Graph from "
            "the provided codebase specifications or software documentation.\n\n"
            "Identify ALL technical entities (modules, files, database tables, developer roles, variables, API routes) "
            "as nodes, and identify relationships (calls, imports, dependencies, data writes/reads) as directed edges.\n"
            "Ensure you capture only authentic technical connections. Keep node names precise and clear."
        )),
        ("user", "Document context:\n---\nHeading: {heading}\nSource: {source}\n\nContent:\n{content}\n---")
    ])
    
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(ExtractionResult)
        chain = prompt | structured_llm
        
        result = chain.invoke({
            "heading": chunk_heading,
            "source": chunk_source,
            "content": chunk_content
        })
        
        extracted_nodes = [node.model_dump() for node in result.nodes]
        extracted_edges = [edge.model_dump() for edge in result.edges]
        
        logs.append(f"Extraction successful: Found {len(extracted_nodes)} nodes, {len(extracted_edges)} edges.")
        
        return {
            "extracted_nodes": extracted_nodes,
            "extracted_edges": extracted_edges,
            "logs": logs
        }
    except Exception as e:
        logs.append(f"Failed entity extraction on chunk {idx} due to: {e}. Skipping chunk.")
        return {
            "extracted_nodes": [],
            "extracted_edges": [],
            "logs": logs
        }


def entity_resolution_node(state: GraphBuilderState) -> Dict[str, Any]:
    """Node: Runs embedding comparisons to detect duplicate nodes and prepares the approval queues."""
    extracted_nodes = state["extracted_nodes"]
    logs = list(state.get("logs", []))
    
    unresolved_merges = []
    approved_merges = dict(state.get("approved_merges", {}))
    rejected_merges = list(state.get("rejected_merges", []))
    
    project_name = state.get("project", "Default Project")
    logs.append(f"Running entity resolution and semantic duplicate checking for project '{project_name}'...")
    
    for node in extracted_nodes:
        name = node["name"]
        label = node["label"]
        desc = node["description"]
        properties = node["properties"]
        
        # Build project-scoped temp_id
        temp_id = f"{sanitize_node_id(project_name)}_{sanitize_node_id(name)}"
        
        # Don't deduplicate if this temp_id is already explicitly processed in this session
        if temp_id in approved_merges or temp_id in rejected_merges:
            continue
            
        # Check if identical ID exists in SQLite and belongs to the same project
        existing_node = db.get_node(temp_id)
        if existing_node and existing_node.get("properties", {}).get("project") == project_name:
            # Identical primary key id matches -> auto-merge
            approved_merges[temp_id] = temp_id
            logs.append(f"Auto-merged identical node: '{name}' (ID matches: '{temp_id}')")
            continue
            
        # Run semantic search against description in DB, scoped to same project
        node_text = f"Name: {name}\nLabel: {label}\nDescription: {desc}"
        node_vector = embedder.get_embedding(node_text)
        
        # Query nearest matches restricted to this project
        matches = db.semantic_search_nodes(node_vector, top_k=1, project=project_name)
        
        if matches:
            match_node, score = matches[0]
            
            # High Confidence (Deduplicate / Merge automatically)
            if score >= 0.85:
                approved_merges[temp_id] = match_node["id"]
                logs.append(f"High-confidence semantic merge: '{name}' -> existing '{match_node['name']}' (score: {score:.2f})")
                
            # Borderline Confidence (HALT & Ask human)
            elif score >= 0.60:
                # Add metadata to the candidate conflict
                unresolved_merges.append({
                    "extracted": {
                        "temp_id": temp_id,
                        "name": name,
                        "label": label,
                        "description": desc,
                        "properties": properties
                    },
                    "existing": match_node,
                    "similarity": round(score, 2)
                })
                logs.append(f"Flagged borderline duplicate: '{name}' matches existing '{match_node['name']}' (score: {score:.2f})")
            
            # Low Confidence -> mark as unique node
            else:
                rejected_merges.append(temp_id)
        else:
            # Zero database entries -> mark as unique node
            rejected_merges.append(temp_id)
            
    return {
        "unresolved_merges": unresolved_merges,
        "approved_merges": approved_merges,
        "rejected_merges": rejected_merges,
        "logs": logs
    }


def human_approval_node(state: GraphBuilderState) -> Dict[str, Any]:
    """
    Node: Placeholder representing human decision resolution.
    This node is interrupted BEFORE execution. When resumed,
    the approvals/rejections dictionary will have been populated by the UI.
    """
    logs = list(state.get("logs", []))
    logs.append("Human approval choices integrated successfully.")
    
    # Empty unresolved merges as they have now been processed by the user
    return {
        "unresolved_merges": [],
        "logs": logs
    }


def write_to_db_node(state: GraphBuilderState) -> Dict[str, Any]:
    """Node: Commits the resolved nodes and edges to the SQLite database."""
    extracted_nodes = state["extracted_nodes"]
    extracted_edges = state["extracted_edges"]
    approved_merges = state["approved_merges"]
    logs = list(state.get("logs", []))
    
    logs.append("Writing verified nodes and edges to SQLite...")
    
    # 1. Write Nodes
    node_id_mappings = {} # Maps original extracted node name -> finalized database node id
    
    project_name = state.get("project", "Default Project")
    
    for node in extracted_nodes:
        name = node["name"]
        label = node["label"]
        desc = node["description"]
        properties = node["properties"]
        
        # Inject project tag
        properties["project"] = project_name
        
        temp_id = f"{sanitize_node_id(project_name)}_{sanitize_node_id(name)}"
        
        if temp_id in approved_merges:
            # Merged! Point to the existing node id
            resolved_id = approved_merges[temp_id]
            node_id_mappings[name] = resolved_id
            
            # Optionally update description/properties in DB if we want,
            # but keeping existing is standard to prevent database pollution
            logs.append(f"Merged node in graph: '{name}' unified as ID '{resolved_id}'")
        else:
            # Write new node
            node_vector = embedder.get_embedding(f"Name: {name}\nLabel: {label}\nDescription: {desc}")
            success = db.upsert_node(
                node_id=temp_id,
                name=name,
                label=label,
                description=desc,
                properties=properties,
                embedding=node_vector
            )
            node_id_mappings[name] = temp_id
            if success:
                logs.append(f"Created new database node: '{name}' (ID: '{temp_id}')")
                
    # 2. Write Edges
    for edge in extracted_edges:
        source_name = edge["source"]
        target_name = edge["target"]
        relationship = edge["relationship"]
        properties = edge["properties"]
        
        # Inject project tag
        properties["project"] = project_name
        
        # Map source/target names to their final resolved DB node IDs
        # If the LLM referenced a node that wasn't in extracted_nodes for this chunk (rare but possible),
        # fallback to project-scoped sanitized ID
        source_id = node_id_mappings.get(source_name, f"{sanitize_node_id(project_name)}_{sanitize_node_id(source_name)}")
        target_id = node_id_mappings.get(target_name, f"{sanitize_node_id(project_name)}_{sanitize_node_id(target_name)}")
        
        # Verify both target nodes exist in the database (or were just written)
        # to ensure SQLite foreign key constraints succeed
        source_exists = db.get_node(source_id) is not None or source_id in node_id_mappings.values()
        target_exists = db.get_node(target_id) is not None or target_id in node_id_mappings.values()
        
        if source_exists and target_exists:
            success = db.upsert_edge(
                source=source_id,
                target=target_id,
                relationship=relationship,
                properties=properties
            )
            if success:
                logs.append(f"Created relationship edge: '{source_id}' --[{relationship}]--> '{target_id}'")
        else:
            missing = []
            if not source_exists: missing.append(f"Source ID '{source_id}'")
            if not target_exists: missing.append(f"Target ID '{target_id}'")
            logs.append(f"Warning: Skipped edge '{source_name} -> {target_name}' because these node IDs are missing: {', '.join(missing)}")
            
    # Increment the chunk idx to move to next section
    next_idx = state["current_chunk_idx"] + 1
    
    return {
        "current_chunk_idx": next_idx,
        # Clear out raw node/edge state for the next chunk loop
        "extracted_nodes": [],
        "extracted_edges": [],
        "logs": logs
    }


# 3. Router Condition Functions
def resolution_router(state: GraphBuilderState) -> str:
    """Decides if the graph should pause for human approval or go straight to database write."""
    unresolved = state.get("unresolved_merges", [])
    if len(unresolved) > 0:
        return "human_approval_node"
    return "write_to_db_node"


def post_write_router(state: GraphBuilderState) -> str:
    """Decides if the graph should loop to the next chunk or complete the ingestion."""
    chunks = state["document_chunks"]
    idx = state["current_chunk_idx"]
    if idx >= len(chunks):
        return END
    return "extract_chunk_node"


# 4. Compile the LangGraph builder state machine
builder_workflow = StateGraph(GraphBuilderState)

# Add Nodes
builder_workflow.add_node("extract_chunk_node", extract_chunk_node)
builder_workflow.add_node("entity_resolution_node", entity_resolution_node)
builder_workflow.add_node("human_approval_node", human_approval_node)
builder_workflow.add_node("write_to_db_node", write_to_db_node)

# Set Entrypoint
builder_workflow.set_entry_point("extract_chunk_node")

# Define Transitions
builder_workflow.add_edge("extract_chunk_node", "entity_resolution_node")

# Conditional Routing from Entity Resolution
builder_workflow.add_conditional_edges(
    "entity_resolution_node",
    resolution_router,
    {
        "human_approval_node": "human_approval_node",
        "write_to_db_node": "write_to_db_node"
    }
)

builder_workflow.add_edge("human_approval_node", "write_to_db_node")

# Conditional Routing from Write DB back to Loop
builder_workflow.add_conditional_edges(
    "write_to_db_node",
    post_write_router,
    {
        END: END,
        "extract_chunk_node": "extract_chunk_node"
    }
)

# Initialize in-memory checkpoint memory
memory = MemorySaver()

# Compile the agent with state interrupts BEFORE executing human_approval_node!
builder_agent = builder_workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_approval_node"]
)
