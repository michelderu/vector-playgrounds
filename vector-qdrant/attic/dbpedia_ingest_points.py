from qdrant_client import QdrantClient, models
from datasets import load_dataset
from fastembed import SparseTextEmbedding
import os

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
# - Shard number: 3 (we split the collection into 3 shards)
client.create_collection(
    collection_name="dbpedia_entities_openai3",
    vectors_config={
        "dense": models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE,
            on_disk=True # We store the dense vector on disk
        ),
    },
    quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True), # We store the BQ vector in RAM
    ),
    sparse_vectors_config={
        "sparse": models.SparseVectorParams()
    },
    hnsw_config=models.HnswConfigDiff(
        m=0, # We disable HNSW graph construction, which will allow for faster uploads, and we'll turn it on later
    ),
    shard_number=3
)

# Load the data in batches
from itertools import islice
import uuid
from tqdm import tqdm

# Yield successive n-sized batches from iterable.
def batched(iterable, n):
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch

sparse_model = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1",
    threads=4
)

batch_size = 100  # Adjust based on your system's capacity
for batch in tqdm(batched(dataset, batch_size), desc="Inserting batches"):
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

    # upload_collection does not support named vectors
    client.upsert(
        collection_name="dbpedia_entities_openai3",
        points=points
    )

# Check the number of vectors in the collection
collection_info = client.get_collection("dbpedia_entities_openai3")
print(f"Number of vectors in collection: {collection_info.points_count}")

# Enable HNSW graph construction
client.update_collection(
    collection_name="dbpedia_entities_openai3",
    hnsw_config=models.HnswConfigDiff(
        m=16, # We enable HNSW graph construction
    ),
)

# Scroll through the collection and print the first vector
points, _ = client.scroll(
    collection_name="dbpedia_entities_openai3",
    limit=1,
    with_payload=True,
    with_vectors=True
)

for point in points:
    print(point)