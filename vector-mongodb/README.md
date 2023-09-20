# Vector demo with MongoDB

## Configuration of the cluster
Getting the Atlas DBaaS up an running was very easy:
- Just log in with a Google account
- Create your cluster and choose a CSP/region
- Create the basic access and authorization (the requirement to add an IP whitelisting is cumbersome for quick testing)

### Getting your public IP for a Colab notebook
In order to open up MongoDB Atlas to accept traffic from Google Colab, we need to know the public IP address. It's easy to get that using the following code snippet:
```python
!pip install ipwhois

from ipwhois import IPWhois
from requests import get

ip = get('https://api.ipify.org').text
whois = IPWhois(ip).lookup_rdap(depth=1)
cidr = whois['network']['cidr']
name = whois['network']['name']

print('\n')
print('Provider:  ', name)
print('Public IP: ', ip)
print('CIDRs:     ', cidr)
```

## Configuration of the database and Vector indexes
- Create a database and set a collection
- Creating the index (named LangChainDemo) was straighforward using the JSON API as follows:
```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 1536,
        "similarity": "cosine",
        "type": "knnVector"
      }
    }
  }
}
```
This will create a KNN vector Index.
‼️ I wasn't able to create an ANN index though. KNN improves relevancy over performance a bit so I wanted to test out the quality of the results using ANN. As ANN improves scalability over quality with just a small impact on quality, it seems like the best index type to use overall.

## Langchain integration
Langchain has a nice MongoDB wrapper which makes using MongoDB as a Vector Database pretty simple.
Make sure you have your cluster URL and OpenAI key ready for use.
### Ingesting data
Ingesting data is done with a simple call returning a vector queryable object right away.
```python
docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=index_name
)
```
### Searching for nearest vectors
Querying for semantically comparable vectors is done easily as well
```python
query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)
```

## Colab notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x0tg8TpL4rhuC5hYu0-tlWLXP922DsDI#scrollTo=u1UL38A5TnYN)

[Back to top](../README.md)