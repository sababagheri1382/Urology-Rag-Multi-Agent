import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from retriever import MedicalHybridRetriever

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
qdrant_endpoint = os.getenv("QDRANT_CLUSTER_ENDPOINT")
qdrant_key = os.getenv("QDRANT_API_KEY")

hybrid_retriever = MedicalHybridRetriever(
    qdrant_url=qdrant_endpoint,
    qdrant_api_key=qdrant_key
)

generation_llm = ChatOpenAI(
    model="gpt-4o", 
    api_key=openai_api_key,
    base_url="https://api.gapgpt.app/v1",
    temperature=0,
)

router_llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=openai_api_key,
    base_url="https://api.gapgpt.app/v1",
    temperature=0,
)

translator_llm = ChatOpenAI(
    model="gpt-4o",
    base_url="https://api.gapgpt.app/v1",
    api_key=openai_api_key,
    temperature=0.01,
    max_tokens=512,
    request_timeout=60.0,
    max_retries=2,
)

print("Loading Reranker model... (might take a few seconds on first run)")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
