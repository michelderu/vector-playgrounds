//Import the Cassandra Driver and Cassandra Vector Store from LangChain
import { CassandraStore } from "langchain/vectorstores/cassandra";

// Import Chat Model
import { ChatOpenAI } from "langchain/chat_models/openai";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PromptTemplate } from "langchain/prompts";
import { RunnableSequence, RunnablePassthrough } from "langchain/schema/runnable";
import { StringOutputParser } from "langchain/schema/output_parser";
import { Document } from "langchain/document";

//Import the OpenAPI Large Language Model (you can import other models here eg. Cohere)
import { OpenAI } from "langchain/llms/openai";

//Load environment variables (populate process.env from .env file)
import * as dotenv from "dotenv";
dotenv.config();

export const run = async () => {

    // Get Chat Model
    const model = new ChatOpenAI({});

    // Get OpenAI embedder
    const embedder = new OpenAIEmbeddings();

    // Get the Vector Store from Astra DB
    //const vectorStore = await CassandraStore.fromExistingIndex(
    //    embedder,
    //    {
    //        cloud: { secureConnectBundle: String(process.env.ASTRA_SCB_PATH) },
    //        credentials: { username: String(process.env.ASTRA_CLIENT_ID), password: String(process.env.ASTRA_SECRET) },
    //        table: "cnn_dailymail",
    //        keyspace: "vector_preview",
    //        dimensions: 1536,
    //        primaryKey: { name:"row_id", type: "text" },
    //        metadataColumns: [ { name:"metadata_s", type: "map<text, text>" } ]
    //    }
    //)

    const vectorStore = await CassandraStore.fromTexts(
        [
            "Tortoise: Labyrinth? Labyrinth? Could it Are we in the notorious Little\
                    Harmonic Labyrinth of the dreaded Majotaur?",
            "Achilles: Yiikes! What is that?",
            "Tortoise: They say-although I person never believed it myself-that an I\
                    Majotaur has created a tiny labyrinth sits in a pit in the middle of\
                    it, waiting innocent victims to get lost in its fears complexity.\
                    Then, when they wander and dazed into the center, he laughs and\
                    laughs at them-so hard, that he laughs them to death!",
            "Achilles: Oh, no!",
            "Tortoise: But it's only a myth. Courage, Achilles.",
        ],
        [{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }, { id: 5 }],
        embedder,
        {
            cloud: { secureConnectBundle: process.env.ASTRA_SCB_PATH as string },
            credentials: { username: process.env.ASTRA_CLIENT_ID as string, password: process.env.ASTRA_CLIENT_SECRET as string },
            table: "goldel_escher_bach",
            keyspace: "vector_preview",
            dimensions: 1536,
            primaryKey: { name: "row_id", type: "text" },
            metadataColumns: [ { name: "metadata", type: "map<text, text>" } ]
        }
    )

    const results = vectorStore.similaritySearch("Who is scared?", 2);
    console.log({ results });

    // Get a Retriever from the Vector Store
    const retriever = vectorStore.asRetriever();

    // Define the prompt based on a template
    const prompt = PromptTemplate.fromTemplate(
`Answer the question based only on the following context: {context}
Question: {question}`
    );

    const serializeDocs = (docs: Document[]) =>
        docs.map((doc) => doc.pageContent).join("\n");

    // Create a Q&A chain
    const chain = RunnableSequence.from([
        {
            context: retriever.pipe(serializeDocs),
            question: new RunnablePassthrough(),
        },
        prompt,
        model,
        new StringOutputParser(),
    ]);

    const result = await chain.invoke("Who is scared?");
    console.log(result);
};

run();
