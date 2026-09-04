from typing import Literal, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from state import AgenticRAGState
from config import router_llm, generation_llm, hybrid_retriever, reranker_model
from translator import fa_to_en_chain, en_to_fa_chain

class RouteQuery(BaseModel):
    """Route the user query to the appropriate pipeline and extract metadata."""
    domain: Literal["urology", "out_of_domain"] = Field(
        ..., description="Is the question related to the medical and urology domain?"
    )
    language: Literal["fa", "en"] = Field(
        ..., description="The original language of the user's query ('fa' for Persian or mixed Persian/English, 'en' for strictly English)."
    )

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert medical routing classifier specializing in Urology, Nephrology, Pharmacology, and General Medicine.

### TASK ###
Analyze the user's query and accurately extract two fields: `Domain` and `Language`.

### CLASSIFICATION RULES ###
1. **Domain**:
   - Classify as **`urology`** if the query contains ANY medical, anatomical, physiological, pharmacological, or biochemical terms. This broadly includes:
     * Urology, Nephrology, and Renal physiology (e.g., RTA / Renal Tubular Acidosis).
     * Medications, drugs, and chemicals (e.g., Acetohydroxamic acid, calcium oxalate, uric acid).
     * Urinary symptoms, kidney conditions, prostate & bladder issues, andrology.
     * Diagnostic tests, imaging, and surgeries (e.g., TURP).
     * **Multiple-Choice Questions (MCQ) or exam prompts** containing medical content. If the query starts with "Based on the retrieved context..." but asks a medical/biology question, it MUST be classified as `urology`.
   - Classify as **`out_of_domain`** ONLY if the query is entirely non-medical (e.g., programming, economics, weather, history, math, general chit-chat).

2. **Language**:
   - Classify as **`fa`** if the query contains ANY Persian characters, Persian-English mixed terms, or transliterations.
   - Classify as **`en`** ONLY if the entire query is strictly in English.

### OUTPUT FORMAT ###
You MUST output EXACTLY in this format with NO additional commentary:
Domain: <urology | out_of_domain> | Language: <fa | en>

### EXAMPLES ###
<user_query>Based on the retrieved context, answer the following multiple-choice question... Question: Type 1 (distal) RTA is characterized by which abnormality?</user_query>
Domain: urology | Language: en

<user_query>Acetohydroxamic acid contributes to reducing infection stone formation by:</user_query>
Domain: urology | Language: en

<user_query>سلام، هیدرونفروز گرید ۲ در کلیه چپ یعنی چی؟ آیا باید عمل بشم؟</user_query>
Domain: urology | Language: fa

<user_query>What is the recommended antibiotic prophylaxis for TURP?</user_query>
Domain: urology | Language: en

<user_query>سلام هوا چطوره امروز؟</user_query>
Domain: out_of_domain | Language: fa

<user_query>Write a python script to sort an array.</user_query>
Domain: out_of_domain | Language: en
"""),
    ("human", "<user_query>{question}</user_query>")
])


structured_llm_router = router_llm.with_structured_output(RouteQuery)
router_chain = router_prompt | structured_llm_router


def get_documents_from_vector_db(query: str) -> list:
    print(f">>> [Vector DB] Retrieving real docs for: {query}")
    
    try:
        results = hybrid_retriever.retrieve(query, top_k=20)
    except Exception as e:
        print(f">>> [Vector DB Error] Failed to retrieve documents: {e}")
        return None 
    docs=[]
    for index, item in enumerate(results, start=1):
        text = item.get("text")
        breadcrumb = item.get("breadcrumb") or f"Unknown source {index}"

        if text and text.strip():
            docs.append({
                "text": text.strip(),
                "breadcrumb": breadcrumb.strip()
                if isinstance(breadcrumb, str)
                else str(breadcrumb)
            })
    if not docs:
        print(">>> [Warning] No documents retrieved from Vector DB!")
    
    return docs

def router_node(state: AgenticRAGState):
    print("---NODE: ROUTER---")
    question = state["user_query"]
    
    try:
        result = router_chain.invoke({"question": question})
        print(f"[Router Node Output] raw_content={result}")
        print(f"[Router Node Parsed] route_to={result.domain}, detected_language={result.language}")
        
        return {
            "detected_language": result.language,
            "route_to": result.domain 
        }
    except Exception as e:
        print(f"[Router Node Error] Routing LLM failed: {e}")
        return {
            "detected_language": "error", 
            "route_to": "out_of_domain" 
        }

def translate_to_en_node(state: AgenticRAGState):
    print("---NODE: TRANSLATE TO ENGLISH---")
    
    try:
        english_translation = fa_to_en_chain.invoke({
            "text": state['user_query']
        }).strip()
    except Exception as e:
        print(f"[Translation to EN Error] LLM translation failed: {e}")
        english_translation = state['user_query']

    return {"english_query": english_translation}

def retrieval_node(state: AgenticRAGState):
    print("---NODE: RETRIEVAL---")
    query = state.get("english_query") or state["user_query"]
    docs = get_documents_from_vector_db(query)

    if docs is None:
        return {"retrieved_responses": [{"error": True}]}
    
    return {"retrieved_responses": docs}

def rerank_node(state: AgenticRAGState):
    print("---NODE: RERANKER---")
    query = state.get("english_query") or state["user_query"]
    docs = state.get("retrieved_responses", [])
    
    if not docs:
        print("[Reranker] No docs found to rerank.")
        return {"reranked_responses": []}
        
    if docs and docs[0].get("error"):
        return {"reranked_responses": docs}

    pairs = [[query, doc["text"]] for doc in docs]
    
    top_k = 5
    try:
        scores = reranker_model.predict(pairs)
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]
    except Exception as e:
        print(f"[Reranker Error] Scoring failed: {e}")
        reranked_docs = docs[:top_k]
    
    print(f"[Debug] Reranker kept {len(reranked_docs)} documents out of original docs.")
    return {"reranked_responses": reranked_docs}


class GeneratedAnswer(BaseModel):
    answer: str = Field(
        description="The clean, direct medical answer. Do NOT include source brackets, citations, or references inside this text."
    )
    used_source_indices: List[int] = Field(
        default=[],
        description="List of integer indices (e.g. [1, 2]) corresponding ONLY to the sources strictly used to formulate the answer."
    )


def generation_node(state: AgenticRAGState):
    print("---NODE: GENERATION---")
    query = state.get("english_query") or state["user_query"]
    docs = state.get("reranked_responses", [])
    user_type = state.get("user_role", "doctor")
    detected_language = state.get("detected_language", "fa")

    if docs and docs[0].get("error"):
        answer = "I apologize, but an error occurred while generating the response. Please try again later."
        return {
            "rag_response": answer,
            "final_output": answer,
            "citations": []
        }

    if not docs:
        answer = "I don't know based on the provided documents."
        return {
            "rag_response": answer,
            "final_output": answer,
            "citations": []
        }
    
    context_blocks = []
    doc_lookup = {}  
    
    for index, doc in enumerate(docs, 1):
        breadcrumb = doc.get("breadcrumb") or f"Unknown source {index}"
        text = doc.get("text", "").strip()
        doc_lookup[index] = breadcrumb

        context_blocks.append(
            f"[Source {index}: {breadcrumb}]\n{text}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    tone_instruction = (
        "Use simple, empathetic language suitable for a patient. "
        "Keep the answer to 3-5 short sentences, avoid jargon, and briefly "
        "explain any necessary medical term in plain words."
        if user_type == "patient"
        else "Use professional, clinical, and academic medical terminology "
             "suitable for a doctor. Keep the answer to 3-5 sentences and be "
             "as precise and information-dense as possible."
    )

    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert, highly accurate urology assistant.

### TASK ###
Answer the user's question using ONLY the information inside the <context></context> tags below.

### RULES ###
1. If the answer is not contained within the context, set answer to: "I don't know based on the provided documents." and return an empty source list.
2. {tone_instruction}
3. Do NOT mention words like "the context", "documents", or "sources" in your answer text.
4. Do NOT include inline citations (such as [Source 1], brackets, or footnotes) inside the `answer` string. Keep the answer completely clean and natural.
5. In the `used_source_indices` field, list ONLY the integer numbers of the sources you actually relied upon to answer.

<context>
{context}
</context>"""),
        ("human", "<question>{question}</question>")
    ])

    structured_llm = generation_llm.with_structured_output(GeneratedAnswer)
    rag_chain = rag_prompt | structured_llm
    
    try:
        result: GeneratedAnswer = rag_chain.invoke({
            "context": context,
            "question": query,
            "tone_instruction": tone_instruction
        })
        answer = result.answer.strip()
        
        used_breadcrumbs = []
        for idx in result.used_source_indices:
            if idx in doc_lookup:
                bc = doc_lookup[idx]
                if bc not in used_breadcrumbs:
                    used_breadcrumbs.append(bc)
                    
    except Exception as e:
        print(f"[Generation Error] Failed to generate answer: {e}")
        answer = "I apologize, but an error occurred while generating the response. Please try again later."
        used_breadcrumbs = [] 

    if detected_language == "en" and used_breadcrumbs:
        sources_md = (
            "\n\n---\n"
            "**📚 References & Breadcrumbs:**\n"
            + "\n".join(f"- `{breadcrumb}`" for breadcrumb in used_breadcrumbs)
        )
        final_output = answer + sources_md
    else:
        final_output = answer

    return {
        "rag_response": answer,
        "final_output": final_output,
        "citations": used_breadcrumbs
    }

def translate_to_fa_node(state: AgenticRAGState):
    print("---NODE: TRANSLATE TO PERSIAN---")
    rag_answer = state["rag_response"]
    citations = state.get("citations", [])
    
    try:
        persian_translation = en_to_fa_chain.invoke({
            "text": rag_answer
        })
        if hasattr(persian_translation, "content"):
            persian_translation = persian_translation.content.strip()
        else:
            persian_translation = str(persian_translation).strip()
            
    except Exception as e:
        print(f"[Translation to FA Error] Failed to translate: {e}")
        persian_translation = rag_answer

    if citations:
        sources_md = (
            "\n\n---\n"
            "**📚 مسیر منابع و مراجع:**\n"
            + "\n".join(f"- `{c}`" for c in citations)
        )
        final_output = persian_translation + sources_md
    else:
        final_output = persian_translation

    return {"final_output": final_output}

def out_of_domain_node(state: AgenticRAGState):
    print("---NODE: OUT OF DOMAIN---")
    lang = state.get("detected_language", "fa")
    
    if lang == "error":
        return {"final_output": "عذرخواهی می‌کنم، اما هنگام تولید پاسخ خطایی رخ داد. لطفاً بعداً دوباره تلاش کنید."}
        
    ans = (
        "متاسفم، من یک دستیار تخصصی در حوزه اورولوژی هستم و نمی‌توانم به سوالات خارج از این حوزه پاسخ دهم."
        if lang == "fa"
        else "I apologize, I am a specialized urology assistant and cannot answer questions outside of this domain."
    )    
    return {"final_output": ans}
