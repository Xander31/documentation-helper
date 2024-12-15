import os

from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()

def run_llm(query:str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, verbose=True)
    docsearch = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query})
    return result

if __name__ == "__main__":
    res = run_llm(query= "What is a LangChain Chain?")
    print(res["answer"])





