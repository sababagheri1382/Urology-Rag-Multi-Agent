from typing import TypedDict, List, Any, Optional

class AgenticRAGState(TypedDict):
    user_query: str                  
    user_role: str                   
    detected_language: str           
    route_to: str                    
    final_output: str                
    
    english_query: Optional[str]     
    retrieved_responses: Optional[List[Any]] 
    reranked_responses: Optional[List[Any]]  
    rag_response: Optional[str]
    citations: Optional[List[str]]
