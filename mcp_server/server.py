import sys
import os
import json
import re
from typing import Dict, List, Optional

# Ensure parent directory is in python path to load our local database and embedding modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from mcp.server.fastmcp import FastMCP
from database import GraphDatabase, sanitize_node_id
from embeddings import EmbeddingGenerator

# Initialize FastMCP Server
mcp = FastMCP("AetherDocs Graph Server")

# Initialize Database and Embedding Generator
db = GraphDatabase()
embedder = EmbeddingGenerator()

@mcp.tool()
def add_node(name: str, label: str, description: str, properties_json: str = "{}") -> str:
    """
    Creates or updates a node in the technical knowledge graph.
    
    Args:
        name: The distinct name of the node (e.g. 'Authentication Module').
        label: The category/type of the node (e.g. 'Module', 'API', 'Table', 'Developer').
        description: A short details paragraph explaining what it is (used for semantic searches).
        properties_json: JSON string of metadata key-values (e.g. '{"version": "1.0", "owner": "Akshay"}').
    """
    node_id = sanitize_node_id(name)
    try:
        properties = json.loads(properties_json) if properties_json else {}
    except json.JSONDecodeError:
        return f"Error: properties_json is not valid JSON. Received: '{properties_json}'"
    
    # Generate semantic embedding of name and description
    embedding_text = f"Name: {name}\nLabel: {label}\nDescription: {description}"
    embedding_vector = embedder.get_embedding(embedding_text)
    
    success = db.upsert_node(
        node_id=node_id,
        name=name,
        label=label,
        description=description,
        properties=properties,
        embedding=embedding_vector
    )
    
    if success:
        return json.dumps({
            "status": "success",
            "message": f"Successfully created/updated node '{name}'",
            "node_id": node_id
        })
    else:
        return f"Error: Failed to upsert node '{name}' in database."

@mcp.tool()
def add_edge(source_name: str, target_name: str, relationship: str, properties_json: str = "{}") -> str:
    """
    Creates or updates a directed edge (relationship) between two nodes in the graph.
    Both source and target nodes must exist in the database before drawing the edge.
    
    Args:
        source_name: The name of the originating node (e.g. 'Authentication Module').
        target_name: The name of the destination node (e.g. 'User Database').
        relationship: The type of link (e.g. 'DEPENDS_ON', 'CALLS', 'WRITES_TO', 'IMPLEMENTS').
        properties_json: JSON string of metadata key-values.
    """
    source_id = sanitize_node_id(source_name)
    target_id = sanitize_node_id(target_name)
    
    try:
        properties = json.loads(properties_json) if properties_json else {}
    except json.JSONDecodeError:
        return f"Error: properties_json is not valid JSON. Received: '{properties_json}'"
    
    # Ensure both nodes exist
    source_node = db.get_node(source_id)
    target_node = db.get_node(target_id)
    
    missing = []
    if not source_node:
        missing.append(f"Source '{source_name}' (ID: {source_id})")
    if not target_node:
        missing.append(f"Target '{target_name}' (ID: {target_id})")
        
    if missing:
        return f"Error: Cannot create edge. The following nodes do not exist: {', '.join(missing)}"
        
    success = db.upsert_edge(
        source=source_id,
        target=target_id,
        relationship=relationship.upper().strip(),
        properties=properties
    )
    
    if success:
        return json.dumps({
            "status": "success",
            "message": f"Successfully created link: {source_name} --[{relationship.upper()}]--> {target_name}"
        })
    else:
        return f"Error: Failed to create edge in database."

@mcp.tool()
def semantic_node_search(query: str, top_k: int = 5) -> str:
    """
    Finds the most semantically relevant nodes using cosine similarity of vector embeddings.
    
    Args:
        query: The natural language search query (e.g. 'authentication tokens or db users').
        top_k: The maximum number of matches to retrieve.
    """
    # Generate embedding for search term
    query_vector = embedder.get_embedding(query)
    
    # Run similarity search
    results = db.semantic_search_nodes(query_vector, top_k=top_k)
    
    formatted_results = []
    for node, score in results:
        formatted_results.append({
            "node": node,
            "similarity_score": round(score, 4)
        })
        
    return json.dumps(formatted_results, indent=2)

@mcp.tool()
def get_node_neighbors(node_name: str) -> str:
    """
    Retrieves the local sub-graph surrounding a target node (all incoming and outgoing connections).
    
    Args:
        node_name: The name of the node to explore (e.g. 'Authentication Module').
    """
    node_id = sanitize_node_id(node_name)
    
    # Ensure node exists
    node = db.get_node(node_id)
    if not node:
        return f"Error: Node '{node_name}' (ID: {node_id}) does not exist in the database."
        
    neighbors = db.get_node_neighbors(node_id)
    
    output = {
        "center_node": node,
        "outgoing_connections": neighbors["outgoing"],
        "incoming_connections": neighbors["incoming"]
    }
    
    return json.dumps(output, indent=2)

@mcp.tool()
def execute_graph_query(sql_query: str) -> str:
    """
    Executes a direct read-only SQL query against the SQLite Graph database.
    Use this for complex traversals, counting nodes, or multi-hop relationship joins.
    Only SELECT queries are authorized.
    
    Example Schema:
    - Table 'nodes': Columns (id TEXT, name TEXT, label TEXT, description TEXT, properties TEXT)
    - Table 'edges': Columns (id INTEGER, source TEXT, target TEXT, relationship TEXT, properties TEXT)
    
    Args:
        sql_query: The raw SQL SELECT query (e.g., 'SELECT label, count(*) FROM nodes GROUP BY label;').
    """
    try:
        results = db.run_raw_sql(sql_query)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error executing SQL: {str(e)}"

if __name__ == "__main__":
    # Start the server (FastMCP automatically routes stdio by default when executed)
    mcp.run()
