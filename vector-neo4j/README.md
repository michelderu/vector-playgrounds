# Vector demo with Neo4j

## Configuration of the cluster
Getting the Aura DBaaS up an running was very easy:
- Just log in with a Google account
- Create your free cluster
- Basic access is created automatically and the details are downloaded to your machine

## Creation of the ANN Vector Index
Creating the index is easy and done as follows
```python
graph.query("""
CALL db.index.vector.createNodeIndex(
  'wikipedia', // index name
  'Chunk',     // node label
  'embedding', // node property
   1536,       // vector size
   'cosine'    // similarity metric
)
""")
```
This will create an ANN vector Index.

## Loading data with embeddings
Loading data in combination with their semanticaly encoded embeddings is done as follows
```python
graph.query("""
UNWIND $data AS row
CREATE (c:Chunk {text: row.text})
WITH c, row
CALL db.create.setVectorProperty(c, 'embedding', row.embedding)
YIELD node
RETURN distinct 'done'
""", {'data': chunks})
```

### Searching for nearest vectors
Querying for semantically comparable vectors is done as follows, with K=3 in this example
```python
embedding = self.embeddings.embed_query("What is the gameplay of Baldur's Gate 3 like?")

vector_search = """
WITH $embedding AS e
CALL db.index.vector.queryNodes('wikipedia',$k, e) yield node, score
RETURN node.text AS result
ORDER BY score DESC
LIMIT 3
"""

context = self.graph.query(
  vector_search, {'embedding': embedding, 'k': 3})
```

## Colab notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_5rBSL8PoYAAxkqT6UjwYZxdooGeql1P#scrollTo=rSsRMik_DUgx)

[Back to top](../README.md)