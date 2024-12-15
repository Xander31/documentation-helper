#Scrappe Langchain doc: wget --recursive --no-clobber --html-extension --convert-links --domains python.langchain.com --no-parent https://python.langchain.com/docs/introduction/
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")

def ingest_docs():
    #Load
    loader = ReadTheDocsLoader(path="langchain-docs/api.python.langchain.com/en/latest", encoding="utf-8")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    #Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    #Generating a credible Url
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https://")
        doc.metadata.update({"source": new_url})

    #Embeed and Load
    print(f"Going to add {len(documents)} chunks to Pinecone.")
    PineconeVectorStore.from_documents(
        embedding=embeddings, documents=documents, index_name = os.getenv("INDEX_NAME")
    )
    print("*** Loaded to Pinecone vectorstore ***")

if __name__ == "__main__":
    ingest_docs()