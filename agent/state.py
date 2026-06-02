from typing import TypedDict, List, Dict, Any, Optional

class GraphBuilderState(TypedDict):
    """
    State managed during the ingestion and construction of the Knowledge Graph.
    """
    # Ingestion tracking
    document_chunks: List[Dict[str, Any]] 
    current_chunk_idx: int              
    
    # Project Workspace Name
    project: Optional[str]              
    
    # Raw extractions from the active chunk
    extracted_nodes: List[Dict[str, Any]]  
    extracted_edges: List[Dict[str, Any]]  
    
    # Entity Resolution (Deduplication) State
    unresolved_merges: List[Dict[str, Any]] 
    approved_merges: Dict[str, str]       
    rejected_merges: List[str]     
    
    # System logs
    logs: List[str]


class GraphQueryState(TypedDict):
    """
    State managed during query answering (Hybrid Graph-RAG).
    """
    query: str                            
    
    # Project Workspace Name
    project: Optional[str]              
    
    # Context aggregation
    retrieved_nodes: List[Dict[str, Any]] 
    retrieved_edges: List[Dict[str, Any]] 
    
    # Final output
    answer: str                           
    logs: List[str]
    messages: List[Any]
