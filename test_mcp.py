import os
import json
import numpy as np

# Load dotenv immediately before importing any internal modules to ensure keys are populated during setup
from dotenv import load_dotenv
load_dotenv()

from database import GraphDatabase
from mcp_server.server import add_node, add_edge, semantic_node_search, get_node_neighbors, execute_graph_query, db, embedder

def run_tests():
    print("==================================================")
    print("STARTING AETHERDOCS MCP SERVER INTERNAL INTEGRATION TESTS")
    print("==================================================")
    
    # 1. Clear any existing test nodes/edges to ensure clean slate
    print("\n1. Initializing database and clearing previous test entries...")
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM edges;")
    cursor.execute("DELETE FROM nodes;")
    conn.commit()
    conn.close()
    print("Database cleared successfully.")

    # 2. Test add_node tool
    print("\n2. Testing 'add_node' tool...")
    node1_res = add_node(
        name="User Authentication API",
        label="API",
        description="Core service handles JWT generation, session authentication, and OAuth login flows.",
        properties_json=json.dumps({"port": 8080, "owner": "Akshay"})
    )
    print(f"Result: {node1_res}")
    node1_data = json.loads(node1_res)
    assert node1_data["status"] == "success"
    assert node1_data["node_id"] == "user_authentication_api"

    node2_res = add_node(
        name="User Database",
        label="Database",
        description="PostgreSQL storage for users records, salted password hashes, and user metadata schema.",
        properties_json=json.dumps({"engine": "PostgreSQL", "encrypted": True})
    )
    print(f"Result: {node2_res}")
    node2_data = json.loads(node2_res)
    assert node2_data["status"] == "success"
    assert node2_data["node_id"] == "user_database"

    # 3. Test add_edge tool
    print("\n3. Testing 'add_edge' tool...")
    edge_res = add_edge(
        source_name="User Authentication API",
        target_name="User Database",
        relationship="DEPENDS_ON",
        properties_json=json.dumps({"latency_ms": 15})
    )
    print(f"Result: {edge_res}")
    edge_data = json.loads(edge_res)
    assert edge_data["status"] == "success"

    # 4. Test semantic_node_search tool
    print("\n4. Testing 'semantic_node_search' tool...")
    search_res = semantic_node_search(query="JWT authentication and session handling", top_k=2)
    print(f"Search Results:\n{search_res}")
    search_data = json.loads(search_res)
    assert len(search_data) > 0
    
    if not embedder.is_mock:
        print("Embeddings are active. Performing strict semantic assertions.")
        assert search_data[0]["node"]["name"] == "User Authentication API", "Expected 'User Authentication API' as top semantic match."
        assert search_data[0]["similarity_score"] > 0.4, "Expected similarity score above threshold."

        # Search for database records
        search_res2 = semantic_node_search(query="encrypted records storage database", top_k=2)
        print(f"Search Results 2:\n{search_res2}")
        search_data2 = json.loads(search_res2)
        assert len(search_data2) > 0
        assert search_data2[0]["node"]["name"] == "User Database", "Expected 'User Database' as top semantic match."
    else:
        print("\n[WARNING] Embeddings generator is operating in MOCK mode (due to API key absence, rate limits, or network timeouts).")
        print("Bypassing strict semantic search sorting assertions as mock vectors are pseudorandom.")

    # 5. Test get_node_neighbors tool
    print("\n5. Testing 'get_node_neighbors' tool...")
    neighbor_res = get_node_neighbors(node_name="User Authentication API")
    print(f"Neighbor Results:\n{neighbor_res}")
    neighbor_data = json.loads(neighbor_res)
    assert neighbor_data["center_node"]["id"] == "user_authentication_api"
    assert len(neighbor_data["outgoing_connections"]) == 1
    assert neighbor_data["outgoing_connections"][0]["node"]["id"] == "user_database"

    # 6. Test execute_graph_query (SQL SELECT) tool
    print("\n6. Testing 'execute_graph_query' tool...")
    sql_res = execute_graph_query(sql_query="SELECT label, count(*) as count FROM nodes GROUP BY label;")
    print(f"SQL Output:\n{sql_res}")
    sql_data = json.loads(sql_res)
    assert len(sql_data) == 2  # API and Database
    
    print("\n==================================================")
    print("ALL MCP SERVER INTEGRATION TESTS PASSED SUCCESSFULLY! [SUCCESS]")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
