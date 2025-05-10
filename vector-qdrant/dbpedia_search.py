from qdrant_client import QdrantClient, models
from openai import OpenAI
import os
import time
from fastembed import SparseTextEmbedding
import random

client = QdrantClient(url=os.getenv("QDRANT_HOST"), prefer_grpc=True)
collection_name = os.getenv("COLLECTION_NAME")

query_text = "What about quantum computing?"

dense_model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
dense_vector = dense_model.embeddings.create(input=query_text, model="text-embedding-3-large", dimensions=1536).data[0].embedding

sparse_model = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1",
    threads=4
)
sparse_vector = list(sparse_model.embed(query_text))[0]

### search (deprecated): Dense vector with Binary Quantization + Filtering
print ("### Search: Dense vector with Binary Quantization ###")

start_time = time.perf_counter()
results = client.search(
    collection_name="dbpedia_entities_openai3",
    query_vector=models.NamedVector(name="dense", vector=dense_vector),
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(
            ignore=False,
            rescore=True,
            oversampling=3.0, # Ideal value for openai 1563
        )
    ),
    query_filter=models.Filter( # filter by user_id
        must=[
            models.FieldCondition(
                key='user_id',
                match=models.MatchValue(
                    value=random.randint(1, 10)
                )
            )
        ]
    ),
    limit=5
)
end_time = time.perf_counter()

for result in results:
    print(f"{result.payload['title']} ({result.score})")

print(f"  Time taken to process results: {(end_time - start_time)*1000:.2f}ms")

### query_points: Dense vector + Filtering
print ("\n### Query points: Dense vector ###")

start_time = time.perf_counter()
results = client.query_points(
    collection_name="dbpedia_entities_openai3",
    query=dense_vector,
    using="dense",
    with_payload=True,
    with_vectors=False,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key='user_id',
                match=models.MatchValue(
                    value=random.randint(1, 10)
                )
            )
        ]
    ),
    limit=5
)
end_time = time.perf_counter()

for result in results.points:
    print(f"{result.payload['title']} ({result.score})")

print(f"  Time taken to process results: {(end_time - start_time)*1000:.2f}ms")

### query_points: Sparse vector + Filtering
print ("\n### Query points: Sparse vector ###")

start_time = time.perf_counter()
results = client.query_points(
    collection_name="dbpedia_entities_openai3",
    query=models.SparseVector(**sparse_vector.as_object()), # unpack dictionary into keyword arguments
    using="sparse",
    with_payload=True,
    with_vectors=False,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key='user_id',
                match=models.MatchValue(
                    value=random.randint(1, 10)
                )
            )
        ]
    ),
    limit=5
)
end_time = time.perf_counter()

for result in results.points:
    print(f"{result.payload['title']} ({result.score})")

print(f"  Time taken to process results: {(end_time - start_time)*1000:.2f}ms")

### query_points: Dense vector with Binary Quantization + Filtering
print ("\n### Query points: Dense vector with Binary Quantization ###")

start_time = time.perf_counter()
results = client.query_points(
    collection_name="dbpedia_entities_openai3",
    query=dense_vector,
    using="dense",
    with_payload=True,
    with_vectors=False,
    limit=5,
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(
            ignore=False,
            rescore=True,
            oversampling=3.0,
        )
    ),
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key='user_id',
                match=models.MatchValue(
                    value=random.randint(1, 10)
                )
            )
        ]
    )
)
end_time = time.perf_counter()

for result in results.points:
    print(f"{result.payload['title']} ({result.score})")

print(f"  Time taken to process results: {(end_time - start_time)*1000:.2f}ms")

### query_points: Reciprocal Rank Fusion with Dense vector + Sparse vector + Filtering
print ("\n### Query points: Reciprocal Rank Fusion with Dense vector + Sparse vector ###")

prefetch = [
    models.Prefetch(
      query=dense_vector,
      using="dense",
      limit=20,
    ),
    models.Prefetch(
      query=models.SparseVector(**sparse_vector.as_object()), # unpack dictionary into keyword arguments
      using="sparse",
      limit=20,
    )
]

start_time = time.perf_counter()
results = client.query_points(
    collection_name="dbpedia_entities_openai3",
    prefetch=prefetch,
    query=models.FusionQuery(
        fusion=models.Fusion.RRF,
    ),
    with_payload=True,
    with_vectors=False,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key='user_id',
                match=models.MatchValue(
                    value=random.randint(1, 10)
                )
            )
        ]
    ),
    limit=5
)
end_time = time.perf_counter()

for result in results.points:
    print(f"{result.payload['title']} ({result.score})")

print(f"  Time taken to process results: {(end_time - start_time)*1000:.2f}ms")

### query_points: Reciprocal Rank Fusion with Dense vector and Binary Quantization + Sparse vector + Filtering
print ("\n### Query points: Reciprocal Rank Fusion with Dense vector + Sparse vector and Binary Quantization ###")

prefetch = [
    models.Prefetch(
      query=dense_vector,
      using="dense",
      limit=20,
    ),
    models.Prefetch(
      query=models.SparseVector(**sparse_vector.as_object()), # unpack dictionary into keyword arguments
      using="sparse",
      limit=20,
    )
]

start_time = time.perf_counter()
results = client.query_points(
    collection_name="dbpedia_entities_openai3",
    prefetch=prefetch,
    query=models.FusionQuery(
        fusion=models.Fusion.RRF,
    ),
    with_payload=True,
    with_vectors=False,
    limit=5,
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(
            ignore=False,
            rescore=True,
            oversampling=3.0,
        )
    ),
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key='user_id',
                match=models.MatchValue(
                    value=random.randint(1, 10)
                )
            )
        ]
    )
)
end_time = time.perf_counter()

for result in results.points:
    print(f"{result.payload['title']} ({result.score})")

print(f"  Time taken to process results: {(end_time - start_time)*1000:.2f}ms")

client.close()