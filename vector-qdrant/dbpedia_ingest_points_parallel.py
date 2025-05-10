from qdrant_client import QdrantClient, models
from datasets import load_dataset
from fastembed import SparseTextEmbedding
import os
import time
from itertools import islice
import uuid
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from more_itertools import chunked
import random

# Force reloading the environment variables
load_dotenv(override=True)

dataset = load_dataset(
    "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M",
    split="train",
    streaming=True
).take(1000000)

client = QdrantClient(os.getenv("QDRANT_HOST"), prefer_grpc=True)

collection_name, shard_number, replication_factor = os.getenv("COLLECTION_NAME"), os.getenv("SHARD_NUMBER"), os.getenv("REPLICATION_FACTOR")
client.delete_collection(collection_name="dbpedia_entities_openai3")

# We create a collection with the following parameters:
# - dense vector size: 1536
# - distance: cosine
# - on_disk: True (we store the dense vector on disk)
# - quantization: BinaryQuantization (we store the BQ vector in RAM)
# - sparse vectors: will be based on the Splade model
# - hnsw_config: HnswConfigDiff (we disable HNSW graph construction, which will allow for faster uploads, and we'll turn it on later)
# - Shard number: 3 (we split the collection into 3 shards, with 3 nodes this typically results in 1 shard per node)
# - replication_factor: 1 (we want to replicate the collection on 1 node)
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE,
            on_disk=True # We store the dense vector on disk
        ),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams()
    },
    quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True), # We store the BQ vector in RAM
    ),
    hnsw_config=models.HnswConfigDiff(
        m=0, # We disable HNSW graph construction, which will allow for faster uploads, and we'll turn it on later
    ),
    shard_number=int(shard_number),
    replication_factor=int(replication_factor)
)

# Add indexing for the user_id field
client.create_payload_index(
    collection_name=collection_name,
    field_name="user_id",
    field_schema="integer"
)

# Sparse embedding model
sparse_model = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1",
    threads=8
)

# Upsert batch of points
def process_and_upload(batch):
    points = []
    for item in batch:
        sparse_embedding = list(sparse_model.embed(item["text"]))[0]
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": item["text-embedding-3-large-1536-embedding"],
                    "sparse": {"indices": sparse_embedding.indices, "values": sparse_embedding.values}
                },
                payload={**{k: v for k, v in item.items() if k != "text-embedding-3-large-1536-embedding"}, "user_id": random.randint(1, 10)}
            )
        )
    client.upsert(
        collection_name=collection_name,
        points=points
    )

# Process and upload points in parallel
batch_size, max_workers = int(os.getenv("BATCH_SIZE")), int(os.getenv("MAX_WORKERS"))
print(f"Batch size: {batch_size}, Max workers: {max_workers}")

def stream_and_ingest(dataset, batch_size, max_workers):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create an iterator of batches from the streaming dataset
        batch_iter = chunked(dataset, batch_size)

        futures = []
        with tqdm(desc="Inserting batches") as pbar:
            for batch in batch_iter:
                # Submit each batch one-by-one
                futures.append(executor.submit(process_and_upload, batch))

                # Control memory usage by throttling futures
                if len(futures) >= max_workers * 2:
                    done, not_done = wait_some(futures)
                    for f in done:
                        f.result()
                        pbar.update(1)
                    futures = not_done

            # Final remaining
            for f in as_completed(futures):
                f.result()
                pbar.update(1)

def wait_some(futures):
    from concurrent.futures import wait, FIRST_COMPLETED
    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
    return list(done), list(not_done)

print(f"Starting to upload points to {collection_name} with shard number {shard_number} and replication factor {replication_factor}")
stream_and_ingest(dataset, batch_size, max_workers)

# Enable HNSW graph construction
client.update_collection(
    collection_name="dbpedia_entities_openai3",
    hnsw_config=models.HnswConfigDiff(
        m=16, # We enable HNSW graph construction
    )
)

# Check the number of vectors in the collection
collection_info = client.get_collection("dbpedia_entities_openai3")
print(f"Number of vectors in collection: {collection_info.points_count}")

# Wait for indexing to complete
while True:
    status = client.get_collection("dbpedia_entities_openai3").status
    print(f"Indexing status: {status}")
    if status == "green":
        print("Indexing complete.")
        break
    time.sleep(5)

client