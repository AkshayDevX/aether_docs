import os
import sys
import sqlite3
import json
import re
import numpy as np
from typing import List, Dict, Tuple, Optional

def sanitize_node_id(name: str) -> str:
    """Helper to convert any node name to a safe, lower-snake-case identifier."""
    cleaned = re.sub(r'[^a-zA-Z0-9\s\-_]', '', name)
    cleaned = re.sub(r'[\s\-_]+', '_', cleaned)
    return cleaned.strip('_').lower()

class GraphDatabase:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default path relative to this script
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "graph.db")
        
        self.db_path = db_path
        self.initialize_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_db(self):
        """Initializes the database schema for nodes and edges."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Check if the existing nodes table has a UNIQUE constraint on name
        try:
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='nodes';")
            row = cursor.fetchone()
            if row and "UNIQUE" in row[0]:
                print("Migrating database nodes table to remove name UNIQUE constraint...", file=sys.stderr)
                cursor.execute("DROP TABLE IF EXISTS edges;")
                cursor.execute("DROP TABLE IF EXISTS nodes;")
                conn.commit()
        except Exception as e:
            print(f"Error checking table constraints: {e}", file=sys.stderr)
            
        # Create Nodes Table (without UNIQUE constraint on name to support multi-project)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            label TEXT NOT NULL,
            description TEXT,
            properties TEXT,
            embedding BLOB
        );
        """)
        
        # Create Edges Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            relationship TEXT NOT NULL,
            properties TEXT,
            FOREIGN KEY (source) REFERENCES nodes (id) ON DELETE CASCADE,
            FOREIGN KEY (target) REFERENCES nodes (id) ON DELETE CASCADE,
            UNIQUE(source, target, relationship)
        );
        """)
        
        # Create Chat Sessions Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            project TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create Chat Messages Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            graph_html TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
        );
        """)
        
        conn.commit()
        conn.close()

    def upsert_node(self, node_id: str, name: str, label: str, description: str, 
                    properties: Dict, embedding: Optional[np.ndarray] = None) -> bool:
        """Upserts a node into the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        properties_json = json.dumps(properties)
        embedding_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        
        try:
            cursor.execute("""
            INSERT INTO nodes (id, name, label, description, properties, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                label = excluded.label,
                description = excluded.description,
                properties = excluded.properties,
                embedding = COALESCE(excluded.embedding, nodes.embedding);
            """, (node_id, name, label, description, properties_json, embedding_blob))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error upserting node: {e}")
            return False
        finally:
            conn.close()

    def delete_node(self, node_id: str) -> bool:
        """Deletes a node (cascades and deletes associated edges automatically)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM nodes WHERE id = ?;", (node_id,))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting node {node_id}: {e}")
            return False
        finally:
            conn.close()

    def upsert_edge(self, source: str, target: str, relationship: str, properties: Dict) -> bool:
        """Upserts a directed edge between source and target nodes."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        properties_json = json.dumps(properties)
        
        try:
            cursor.execute("""
            INSERT INTO edges (source, target, relationship, properties)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source, target, relationship) DO UPDATE SET
                properties = excluded.properties;
            """, (source, target, relationship, properties_json))
            conn.commit()
            return True
        except sqlite3.IntegrityError as ie:
            print(f"Integrity Error upserting edge {source} -> {target}: {ie}. Ensure nodes exist first.")
            return False
        except Exception as e:
            print(f"Error upserting edge: {e}")
            return False
        finally:
            conn.close()

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Fetches a single node by its ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, label, description, properties FROM nodes WHERE id = ?;", (node_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            res = dict(row)
            res['properties'] = json.loads(res['properties']) if res['properties'] else {}
            return res
        return None

    def get_all_nodes(self, project: str = None) -> List[Dict]:
        """Fetches all nodes without embeddings for lightweight retrieval."""
        conn = self.get_connection()
        cursor = conn.cursor()
        if project:
            cursor.execute("SELECT id, name, label, description, properties FROM nodes WHERE json_extract(properties, '$.project') = ?;", (project,))
        else:
            cursor.execute("SELECT id, name, label, description, properties FROM nodes;")
        rows = cursor.fetchall()
        conn.close()
        
        nodes = []
        for r in rows:
            node = dict(r)
            node['properties'] = json.loads(node['properties']) if node['properties'] else {}
            nodes.append(node)
        return nodes

    def get_all_edges(self, project: str = None) -> List[Dict]:
        """Fetches all edges."""
        conn = self.get_connection()
        cursor = conn.cursor()
        if project:
            cursor.execute("""
            SELECT e.id, e.source, e.target, e.relationship, e.properties 
            FROM edges e
            JOIN nodes n ON e.source = n.id
            WHERE json_extract(n.properties, '$.project') = ?;
            """, (project,))
        else:
            cursor.execute("SELECT id, source, target, relationship, properties FROM edges;")
        rows = cursor.fetchall()
        conn.close()
        
        edges = []
        for r in rows:
            edge = dict(r)
            edge['properties'] = json.loads(edge['properties']) if edge['properties'] else {}
            edges.append(edge)
        return edges

    def get_all_projects(self) -> List[str]:
        """Fetches all unique project names present in the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT DISTINCT json_extract(properties, '$.project') as project FROM nodes;")
            rows = cursor.fetchall()
            projects = []
            for r in rows:
                p = r["project"]
                if p:
                    projects.append(str(p))
            return sorted(projects)
        except Exception as e:
            print(f"Error fetching projects: {e}")
            return []
        finally:
            conn.close()

    def get_node_neighbors(self, node_id: str) -> Dict[str, List[Dict]]:
        """Fetches adjacent nodes and directed edges connected to a target node."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Outgoing edges and target nodes
        cursor.execute("""
        SELECT e.id as edge_id, e.relationship, e.properties as edge_properties,
               n.id as node_id, n.name, n.label, n.description, n.properties as node_properties
        FROM edges e
        JOIN nodes n ON e.target = n.id
        WHERE e.source = ?;
        """, (node_id,))
        outgoing = cursor.fetchall()
        
        # Incoming edges and source nodes
        cursor.execute("""
        SELECT e.id as edge_id, e.relationship, e.properties as edge_properties,
               n.id as node_id, n.name, n.label, n.description, n.properties as node_properties
        FROM edges e
        JOIN nodes n ON e.source = n.id
        WHERE e.target = ?;
        """, (node_id,))
        incoming = cursor.fetchall()
        
        conn.close()
        
        def format_neighbors(rows):
            results = []
            for r in rows:
                results.append({
                    "edge": {
                        "id": r["edge_id"],
                        "relationship": r["relationship"],
                        "properties": json.loads(r["edge_properties"]) if r["edge_properties"] else {}
                    },
                    "node": {
                        "id": r["node_id"],
                        "name": r["name"],
                        "label": r["label"],
                        "description": r["description"],
                        "properties": json.loads(r["node_properties"]) if r["node_properties"] else {}
                    }
                })
            return results

        return {
            "outgoing": format_neighbors(outgoing),
            "incoming": format_neighbors(incoming)
        }

    def semantic_search_nodes(self, query_embedding: np.ndarray, top_k: int = 5, project: str = None) -> List[Tuple[Dict, float]]:
        """
        Retrieves top_k nodes based on cosine similarity of embeddings.
        Computes cosine similarity in memory using NumPy.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        # Retrieve nodes with non-null embeddings
        if project:
            cursor.execute("SELECT id, name, label, description, properties, embedding FROM nodes WHERE embedding IS NOT NULL AND json_extract(properties, '$.project') = ?;", (project,))
        else:
            cursor.execute("SELECT id, name, label, description, properties, embedding FROM nodes WHERE embedding IS NOT NULL;")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
            
        nodes = []
        embeddings = []
        
        for r in rows:
            node = dict(r)
            node['properties'] = json.loads(node['properties']) if node['properties'] else {}
            # Parse embedding blob back to float32 numpy array
            emb_bytes = node.pop('embedding')
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            
            nodes.append(node)
            embeddings.append(emb)
            
        # Convert embeddings list to matrix
        emb_matrix = np.vstack(embeddings) # shape: (num_nodes, emb_dim)
        
        # Standardize query embedding shape
        query_vector = query_embedding.astype(np.float32).flatten()
        
        # Compute Cosine Similarity: (A . B) / (||A|| * ||B||)
        dot_products = np.dot(emb_matrix, query_vector)
        matrix_norms = np.linalg.norm(emb_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        # Guard against divide by zero
        matrix_norms[matrix_norms == 0] = 1e-10
        if query_norm == 0:
            query_norm = 1e-10
            
        similarities = dot_products / (matrix_norms * query_norm)
        
        # Sort results
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in sorted_indices:
            results.append((nodes[idx], float(similarities[idx])))
            
        return results

    def run_raw_sql(self, query: str, params: tuple = ()) -> List[Dict]:
        """Runs a read-only custom SQL query against the database (safeguarded)."""
        query_lower = query.strip().lower()
        if not query_lower.startswith("select"):
            raise ValueError("Only SELECT queries are allowed for raw execution.")
            
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"Error running raw SQL: {e}")
            raise e
        finally:
            conn.close()

    def create_chat_session(self, session_id: str, title: str, project: str) -> bool:
        """Creates a new persistent chat session in the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO chat_sessions (id, title, project)
            VALUES (?, ?, ?);
            """, (session_id, title, project))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error creating chat session: {e}", file=sys.stderr)
            return False
        finally:
            conn.close()

    def get_chat_sessions(self, project: str = None) -> List[Dict]:
        """Fetches all chat sessions from the database, optionally filtered by project."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if project:
                cursor.execute("""
                SELECT id, title, project, created_at 
                FROM chat_sessions 
                WHERE project = ? 
                ORDER BY created_at DESC;
                """, (project,))
            else:
                cursor.execute("""
                SELECT id, title, project, created_at 
                FROM chat_sessions 
                ORDER BY created_at DESC;
                """)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"Error fetching chat sessions: {e}", file=sys.stderr)
            return []
        finally:
            conn.close()

    def delete_chat_session(self, session_id: str) -> bool:
        """Deletes a chat session (cascades and deletes associated messages automatically)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM chat_sessions WHERE id = ?;", (session_id,))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error deleting chat session {session_id}: {e}", file=sys.stderr)
            return False
        finally:
            conn.close()

    def add_chat_message(self, session_id: str, role: str, content: str, graph_html: str = None) -> bool:
        """Appends a new chat message to a persistent chat session."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO chat_messages (session_id, role, content, graph_html)
            VALUES (?, ?, ?, ?);
            """, (session_id, role, content, graph_html))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving chat message for session {session_id}: {e}", file=sys.stderr)
            return False
        finally:
            conn.close()

    def get_chat_messages(self, session_id: str) -> List[Dict]:
        """Fetches all messages belonging to a chat session in chronological order."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
            SELECT role, content, graph_html, created_at 
            FROM chat_messages 
            WHERE session_id = ? 
            ORDER BY id ASC;
            """, (session_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"Error fetching chat messages for session {session_id}: {e}", file=sys.stderr)
            return []
        finally:
            conn.close()
