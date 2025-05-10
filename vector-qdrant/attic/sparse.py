from fastembed import SparseTextEmbedding
from qdrant_client import models
import time

model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1", threads=4)

start_time = time.perf_counter()
sparse_vectors = list(model.embed("Hello, world!"))[0]
end_time = time.perf_counter()

print(f"Embedding time: {(end_time - start_time)*1000:.2f}ms")
print(sparse_vectors.indices)
print(sparse_vectors.values)

print (sparse_vectors)
print (sparse_vectors.as_object())

print(models.SparseVector(**sparse_vectors.as_object()))