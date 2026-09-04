from langgraph.graph import StateGraph, END
from state import AgenticRAGState
from nodes import (
    router_node, translate_to_en_node, retrieval_node, 
    rerank_node, generation_node, translate_to_fa_node, out_of_domain_node
)
from edges import route_question, route_after_rag

workflow = StateGraph(AgenticRAGState)

workflow.add_node("router", router_node)
workflow.add_node("translate_to_en", translate_to_en_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("rerank", rerank_node) 
workflow.add_node("generate", generation_node)
workflow.add_node("translate_to_fa", translate_to_fa_node)
workflow.add_node("out_of_domain", out_of_domain_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_question,
    {
        "translate_to_en": "translate_to_en",
        "retrieve": "retrieve",
        "out_of_domain": "out_of_domain"
    }
)

workflow.add_edge("translate_to_en", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")

workflow.add_conditional_edges(
    "generate",
    route_after_rag,
    {
        "translate_to_fa": "translate_to_fa",
        END: END
    }
)

workflow.add_edge("translate_to_fa", END)
workflow.add_edge("out_of_domain", END)

app = workflow.compile()
