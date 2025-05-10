from qdrant_client import QdrantClient, models
from datasets import load_dataset
from fastembed import SparseTextEmbedding
import os
import time
from itertools import islice
import uuid
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force reloading the environment variables
os.load_dotenv(override=True)

dataset = load_dataset("Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M", split="train", streaming=True).take(10000)

client = QdrantClient(os.getenv("QDRANT_HOST"), prefer_grpc=True)

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
    collection_name="dbpedia_entities_openai3",
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
    shard_number=os.getenv("SHARD_NUMBER"),
    replication_factor=os.getenv("REPLICATION_FACTOR")
)

# Sparse embedding model
sparse_model = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1",
    threads=8
)

# Yield successive n-sized batches from iterable.
def batched(iterable, n):
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch


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
                payload={k: v for k, v in item.items() if k != "text-embedding-3-large-1536-embedding"}
            )
        )
    client.upsert(
        collection_name="dbpedia_entities_openai3",
        points=points
    )

# Process and upload points in parallel
batch_size, max_workers = os.getenv("BATCH_SIZE"), os.getenv("MAX_WORKERS") # Adjust based on system's capacity
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for batch in batched(dataset, batch_size):
        futures.append(executor.submit(process_and_upload, batch))
    for future in tqdm(as_completed(futures), total=len(futures), desc="Inserting batches"):
        future.result()

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