# Vector demo with pgvector using Supbaase

## Create a new project and database
Getting the Supabase DBaaS up an running was very easy:
- Create an account or use your github social
- Create a free organizaion
- Create your project and choose a CSP/region. This creates your database.
- Create the basic access and authorization

## Configuration of the database and vector client
Connect to the database. You'll need the following information:
- Host
- Database name
- Port
- User
- Password
```python
import vecs

DB_CONNECTION = "postgresql://postgres:PASSWORD@db.ucmahhwnsdebdnimuvza.supabase.co:5432/postgres"

# create vector store client
vx = vecs.create_client(DB_CONNECTION)
```

## Configuration of the Vector Index
By default, pgvector performs exact nearest neighbor search, which provides perfect recall.

You can add an index to use approximate nearest neighbor search, which trades some recall for speed. Unlike typical indexes, you will see different results for queries after adding an approximate index.

Supported index types are:
- IVFFlat
- HNSW - added in 0.5.0

Creating the index is straighforward as follows:
```python
# index the collection to be queried by cosine distance
docs.create_index(measure=vecs.IndexMeasure.cosine_distance)
```
Available options for query measure are:
- vecs.IndexMeasure.cosine_distance
- vecs.IndexMeasure.l2_distance
- vecs.IndexMeasure.max_inner_product

### Searching for nearest vectors
Querying for semantically comparable vectors is done easily as well with K=5
```python
docs.query(
    data=[0.4,0.5,0.6],  # required
    limit=5,                     # number of records to return
    filters={},                  # metadata filters
    measure="cosine_distance",   # distance measure to use
    include_value=False,         # should distance measure values be returned?
    include_metadata=False,      # should record metadata be returned?
)
```

## Colab notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1w7wRCLh-Rzcrbn-McejmBen5WEcipbW8#scrollTo=kOzVOcFAhUxx)

[Back to top](../README.md)