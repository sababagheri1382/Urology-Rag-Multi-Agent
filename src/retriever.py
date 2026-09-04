import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

qdrant_url = os.getenv("QDRANT_CLUSTER_ENDPOINT")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


class MedicalHybridRetriever:

    def __init__(self,qdrant_url: str,qdrant_api_key: str):

        self.collection_name = "final_rag_collection"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.client = QdrantClient(url=qdrant_url,api_key=qdrant_api_key)

        self.dense_model = SentenceTransformer("BAAI/bge-large-en-v1.5",device=self.device)

        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    def retrieve(self, query:str , top_k: int= 15):

        query_dense_vector = self.dense_model.encode(
            query, convert_to_numpy = True, normalize_embeddings= True
        ).tolist()

        sparse_result = list(self.sparse_model.embed([query]))[0]

        query_sparse_vector = models.SparseVector(
            indices = sparse_result.indices.tolist(),
            values = sparse_result.values.tolist()
        )

        prefetch_dense = models.Prefetch(
            query=query_dense_vector,
            using="dense-vector",
            limit=top_k * 2 
        )
        
        prefetch_sparse = models.Prefetch(
            query=query_sparse_vector,
            using="sparse-vector",
            limit=top_k      
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[prefetch_dense, prefetch_sparse],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True
        )

        retrieved_docs = []
        for point in results.points:
            retrieved_docs.append({
                "chunk_id": point.payload.get("chunk_id", ""),
                "text": point.payload.get("text", ""),
                "breadcrumb": point.payload.get("breadcrumb", ""),
                "score": point.score
            })
            
        return retrieved_docs


# تست
def test_retriever():
    print("Qdrant loading...")
    retriever = MedicalHybridRetriever(qdrant_url=qdrant_url ,qdrant_api_key=qdrant_api_key)
    
    test_query = "A 58-year-old Hispanic female with a history of recurrent urinary tract infections treated three to four times in the past 18 months is seen by her family physician. At present she is asymptomatic. She has no history of nephrolithiasis. Renal ultrasound demonstrates moderate left hydronephrosis and a large density within the renal pelvis with posterior shadowing. A kidney-ureterbladder (KUB) view with tomography reveals a poorly opacified dendritic stone in the renal pelvis and lower pole calyces. Prior urine cultures have Proteus and Klebsiella species. The stone composition of this patient is most likely:calcium oxalate,uric acid,magnesium ammonium phosphate,cystine,hydroxyapatite."

    print(f"\n searching '{test_query}'")
    
    results = retriever.retrieve(test_query, top_k=10)
    
    print(f"\n {len(results)} documents\n")
    
    for i, res in enumerate(results, 1):
        print(f"--- results {i} ---")
        print(f"Score: {res['score']}")
        print(f"chunk id: {res['chunk_id']}")
        print(f"preview: {res['text'][:200]}...") 
        print(f"Breadcrumb: {res.get('breadcrumb', 'N/A')}")
        print("-" * 40)

# اجرای تست
if __name__ == "__main__":
    test_retriever()






    