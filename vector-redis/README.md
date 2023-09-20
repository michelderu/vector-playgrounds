# Vector demo with Redis

## Configuration of the database
In this case we'll run the Redis DB inside the execution runtime as such
```bash
!curl -fsSL https://packages.redis.io/redis-stack/redis-stack-server-6.2.6-v7.focal.x86_64.tar.gz -o redis-stack-server.tar.gz 
!tar -xvf redis-stack-server.tar.gz
!pip install redis
```
Just make sure you load the SSL libraries before as well
```bash
!wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
!sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
```

## Start the database and get a client connection
Starting the database is done as follows
```bash
!./redis-stack-server-6.2.6-v7/bin/redis-stack-server --daemonize yes
```
And then get a client object
```python
import redis
client = redis.Redis(host = 'localhost', port=6379, decode_responses=True)
  
client.ping()
```

## Creating the KNN Vector Index
Which is done as follows within the snippet `$.description_embeddings`:
```python
from redis.commands.search.field import TagField, TextField, NumericField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

INDEX_NAME = 'idx:bikes_vss'
DOC_PREFIX = 'bikes:'

try:
    # check to see if index exists
    client.ft(INDEX_NAME).info()
    print('Index already exists!')
except:
    # schema
    schema = (
        TextField('$.model', no_stem=True, as_name='model'),
        TextField('$.brand', no_stem=True, as_name='brand'),
        NumericField('$.price', as_name='price'),
        TagField('$.type', as_name='type'),
        TextField('$.description', as_name='description'),
        VectorField('$.description_embeddings',
            'FLAT', {
                'TYPE': 'FLOAT32',
                'DIM': VECTOR_DIMENSION,
                'DISTANCE_METRIC': 'COSINE',
            },  as_name='vector'
        ),
    )

    # index Definition
    definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.JSON)

    # create Index
    client.ft(INDEX_NAME).create_index(fields=schema, definition=definition)
```
‼️ I wasn't able to create an ANN index though. KNN improves relevancy over performance a bit so I wanted to test out the quality of the results using ANN. As ANN improves scalability over quality with just a small impact on quality, it seems like the best index type to use overall.

## Semantic searching using the Vector Index
Below is the code for semantic searching with K=3
```python
query = (
    Query('(*)=>[KNN 3 @vector $query_vector AS vector_score]')
     .sort_by('vector_score')
     .return_fields('vector_score', 'id', 'brand', 'model', 'description')
     .dialect(2)
)
```

## Colab notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_5H4u0OQPSEmW3Bw8ZgV-xuGtQdH3rWO#scrollTo=e8UXmiCozOMG)

[Back to top](../README.md)