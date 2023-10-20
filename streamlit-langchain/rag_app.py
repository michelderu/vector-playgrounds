import streamlit as st

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Cassandra
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import ChatMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

st.title('🦜🔗 Enterprise Chat Agent')

# Cache Astra DB session for future runs
@st.cache_resource(show_spinner="Setting up Astra DB connection...")
def load_session():
    # Connect to Astra DB
    cluster = Cluster(cloud={'secure_connect_bundle': st.secrets["ASTRA_SCB_PATH"]}, 
                      auth_provider=PlainTextAuthProvider(st.secrets["ASTRA_CLIENT_ID"], 
                                                          st.secrets["ASTRA_CLIENT_SECRET"]))
    return cluster.connect()
session = load_session()

# Cache OpenAI Embedding for future runs
@st.cache_resource(show_spinner="Getting the OpenAI embedding...")
def load_embedding():
    # Get the OpenAI Embedding
    return OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
embedding = load_embedding()

# Cache Vector Store for future runs
@st.cache_resource(show_spinner="Getting the Vector Store from Astra DB...")
def load_vectorstore():
    # Get the vector store from Astra DB
    return Cassandra(
        embedding=embedding,
        session=session,
        keyspace='vector_preview',
        table_name='romeo'
    )
vectorstore = load_vectorstore()

# Cache OpenAI Chat Model for future runs
@st.cache_resource(show_spinner="Getting the OpenAI Chat Model...")
def load_model():
    # Get the OpenAI Chat Model
    return ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"],
                        streaming=True,
                        verbose=True)
llm = load_model()

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_file):
  file = [uploaded_file.read().decode()]
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size = 1000,
      chunk_overlap  = 10
  )  
  texts = text_splitter.create_documents(file)
  vectorstore.add_documents(texts)

  st.info(f"Loaded {len(texts)} chunks into Astra DB")

# Cache Conversational Chain for future runs
@st.cache_resource(show_spinner="Getting the Conversational Chain...")
def load_qa_chain():

    template = """
        Given the following conversation respond to the best of your ability in a pirate voice and end every sentence with Ay Ay Matey.
        CONTEXT:
        {context}
        
        QUESTION: 
        {question}

        CHAT HISTORY: 
        {chat_history}
        
        ANSWER:
    """

    prompt = PromptTemplate(
        input_variables=["chat_history", "question", "context"], 
        template=template
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False,
        verbose=True,
        memory=ConversationBufferMemory(memory_key="chat_history", 
                                        input_key='question', 
                                        output_key='answer',
                                        return_messages=True),
        combine_docs_chain_kwargs={"prompt": prompt}, 
        chain_type="stuff"
    )
qa_chain = load_qa_chain()

# Include the upload form for new data to be Vectorized
with st.form('upload'):
  uploaded_file = st.file_uploader('Upload an article', type='txt')
  submitted = st.form_submit_button('Submit')
  if submitted:
    vectorize_text(uploaded_file)

# Start with empty messages, stored in session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

# Redraw all messages, both user and agent so far (every time the app reruns)
for message in st.session_state.messages:
    st.chat_message(message.role).markdown(message.content)

# Now get a prompt from a user
if prompt := st.chat_input("What is up?"):
     # Add the prompt to messages, stored in session state
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))

    # Draw the prompt on the page
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the results from Langchain
    with st.chat_message("assistant"):
        # UI placeholder to start filling with agent response
        response_placeholder = st.empty()

        callback = StreamHandler(response_placeholder)
        response = qa_chain.run({'question': prompt}, callbacks=[callback])

        # Write the final answer without the cursor
        response_placeholder.markdown(response)

        # Add the answer to the messages session state
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))