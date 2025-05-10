# Import client library
from qdrant_client import QdrantClient, models
from tqdm.auto import tqdm
import json
import os

def drop_collection(client, collection_name):
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

def create_collection(client, collection_name, dense_vector_name, dense_model_name, sparse_vector_name):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                dense_vector_name: models.VectorParams(
                    size=client.get_embedding_size(dense_model_name), 
                    distance=models.Distance.COSINE
                )
            },  # size and distance are model dependent
            sparse_vectors_config={
                sparse_vector_name: models.SparseVectorParams()
            },
        )

def ingest_data(client, collection_name, dense_vector_name, dense_model_name, sparse_vector_name, sparse_model_name):
    payload_path = "./data/startups_demo.json"
    documents = []
    metadata = []

    with open(payload_path) as fd:
        # Count total lines first
        total_lines = sum(1 for _ in fd)
        fd.seek(0)  # Reset file pointer to beginning
        
        # Create progress bar for document processing
        with tqdm(total=total_lines, desc="Reading documents") as pbar:
            for line in fd:
                obj = json.loads(line)
                description = obj["description"]
                dense_document = models.Document(text=description, model=dense_model_name)
                sparse_document = models.Document(text=description, model=sparse_model_name)
                documents.append(
                    {
                        dense_vector_name: dense_document,
                        sparse_vector_name: sparse_document,
                    }
                )
                metadata.append(obj)
                pbar.update(1)

                print(documents)

        client.upload_collection(
            collection_name=collection_name,
            vectors=tqdm(documents, desc="Uploading to Qdrant"),
            payload=metadata,
            parallel=4
        )

def main():
    client = QdrantClient(url=os.getenv("QDRANT_ENDPOINT"))

    collection_name = "startups"

    dense_vector_name = "dense"
    dense_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_vector_name = "sparse"
    sparse_model_name = "prithivida/Splade_PP_en_v1"

    drop_collection(client, collection_name)
    create_collection(client, collection_name, dense_vector_name, dense_model_name, sparse_vector_name)
    ingest_data(client, collection_name, dense_vector_name, dense_model_name, sparse_vector_name, sparse_model_name)

if __name__ == '__main__':
    main()