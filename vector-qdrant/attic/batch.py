from qdrant_client import QdrantClient, models
import os

client = QdrantClient(url=os.getenv("QDRANT_ENDPOINT"))

client.delete_collection(collection_name="my_collection")   

client.create_collection(
    collection_name="my_collection",
    vectors_config={
        "image": models.VectorParams(
            size=4,                  # 4-dimensional vector
            distance=models.Distance.DOT
        ),
        "text": models.VectorParams(
            size=5,                  # 5-dimensional vector
            distance=models.Distance.COSINE
        ),
    }
)

client.upsert(
    collection_name="my_collection",
    points=[
        models.PointStruct(
            id=1,
            vector={
                "image": [0.9, 0.1, 0.1, 0.2],
                "text": [0.4, 0.7, 0.1, 0.8, 0.1],
            },
            payload={"label": "cat"}
        ),
        models.PointStruct(
            id=2,
            vector={
                "image": [0.9, 0.8, 0.7, 0.6],
                "text": [0.4, 0.3, 0.2, 0.1, 0.0],
            },
            payload={"label": "dog"}
        )
    ]
)

# Does not work
batch = models.Batch(
    ids=[1, 2],
    vectors=[
        {"image": [0.1, 0.2, 0.3, 0.4], "text": [0.9, 0.8, 0.7, 0.6, 0.5]},
        {"image": [0.5, 0.6, 0.7, 0.8], "text": [0.4, 0.3, 0.2, 0.1, 0.0]}
    ],
    payloads=[{"tag": "cat"}, {"tag": "dog"}]
)

client.upsert(
    collection_name="my_collection",
    points=batch
)