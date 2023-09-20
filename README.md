# Collection of Vector Search playgrounds using different databases
This repo contains multiple examples on how to use Vector Search with a selection of databases.
The goal is to learn how to set them up, how to connect and how to semantically query the database. In some instances there's an integration with the popular framework [LangChain](https://python.langchain.com/docs/get_started/introduction).

- [MongoDB](./vector-mongodb/README.md)
- [Neo4J](./vector-neo4j/README.md)
- [PGVector](./vector-pgvector/README.md)
- [Redis](./vector-redis/README.md)

## Conclusion
These tests are merely a `hello-world` example, they are not meant to be extensive. However it is interesting to be pointed to some intersting findings:
### MongoDB
- Easy to set up in the cloud
- Nice UI for index management
- Only KNN searches from which performance will suffer
- Nice integration with LangChain
### Neo4J
- Easy to set up in the cloud
- Supports ANN indexing
- For most Vector use cases, I don't like to complex query language
### PGVector
- Interesting layer over PostgreSQL compatible databases
- It's an add-on
- Only supports up to 2000 dimensions
### Redis
- I used the option to run Redis inside the Colab execution environment which makes for a quick start
- Index creation is pretty simple
- Only KNN searches from which performance will suffer
- Query syntax is pretty straighforward
