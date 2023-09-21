# Vector demo with Weaviate

## Configuration of the cluster
Getting the Weaviate DBaaS up an running was not the simplest experience:
- Need to create a personal account, no socials integrations
- Create your cluster, which chooses a CSP and region for you
- Only very basic access and authorization

## Configuration of Vector indexes
- Creating indexes is done by defining a class
- The class also specifies the vectorized and LLM modules to be used
```python
class_obj = {
    "class": "Question",
    "vectorizer": "text2vec-openai",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
    "moduleConfig": {
        "text2vec-openai": {},
        "generative-openai": {}  # Ensure the `generative-openai` module is used for generative queries
    }
}

client.schema.create_class(class_obj)
```
Weaviate makes it easy to tune the indexer settings as the following example shows:
```json
{
    "classes": [
        {
            "class": "Publication",
            "properties": [],
            "vectorIndexType": "hnsw" // <== the current ANN algorithm
            "vectorIndexConfig": { // <== the vector index settings
                "skip": false,
                "cleanupIntervalSeconds": 300,
                "pq": {"enabled": False,}
                "maxConnections": 64,
                "efConstruction": 128,
                "ef": -1,
                "dynamicEfMin": 100,
                "dynamicEfMax": 500,
                "dynamicEfFactor": 8,
                "vectorCacheMaxObjects": 2000000,
                "flatSearchCutoff": 40000,
                "distance": "cosine"
            }
        },
        { } // <== the Author class
    ]
}
```

## Modular approach
Weaviate's modular approach to vectorizers and LLMs makes for a simple query language.
### Ingesting data
Ingesting data is done with a simple call specifying the class to load data into.
```python
        properties = {
            "answer": d["Answer"],
            "question": d["Question"],
            "category": d["Category"],
        }

        client.batch.add_data_object(
            properties,
            "Question",
        )
```
### Searching for nearest vectors
Querying for semantically comparable vectors is done easily as well
```python
nearText = {"concepts": ["biology"]}

response = (
    client.query
    .get("Question", ["question", "answer", "category"])
    .with_near_text(nearText)
    .with_limit(2)
    .do()
)

print(json.dumps(response, indent=4))
```
### Chaining for Retrieval Augmented Generation
The modular approach of Weaviate allows for simple chaining as follows:
```python
response = (
    client.query
    .get("Question", ["question", "answer", "category"])
    .with_near_text({"concepts": ["biology"]})
    .with_generate(grouped_task="Write a tweet with emojis about these facts.")
    .with_limit(2)
    .do()
)

print(response["data"]["Get"]["Question"][0]["_additional"]["generate"]["groupedResult"])
```

## Colab notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xR6wUizivxkkqHZxEDd1ApYqxhDBWqg3#scrollTo=3k9zbRyQt2gt)

[Back to top](../README.md)