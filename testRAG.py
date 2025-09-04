import getpass
import os


# Load API key
if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


# Load model
from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


# Load documents
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Large_language_model")
document = loader.load()


# Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(document)


# Embedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings()


# Vectoring
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents, embeddings)


# Similatiry search
query = "What are some challenges faced by large language models?"
result = vectorstore.similarity_search(query)
print(result[0].page_content)


# Prompt template
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """
    You are an expert assistant. Answer the following question based only on the provided context:
    
    <context>
    {context}
    </context>
    
    Question: {question}
    """
)


# Build RAG chain
from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(model, prompt)

retriever = vectorstore.as_retriever()
from langchain.chains.retrieval import create_retrieval_chain 

retrieval_chain = create_retrieval_chain(retriever, document_chain)

result = retrieval_chain.invoke({
  "input": "what are some challenges faced by large language models?"
})
print(result['answer'])




