import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from collections import Counter


def langchain(question):
    loader = DirectoryLoader('./ElementGPT/data', glob="*.json",
                             loader_cls=lambda path: TextLoader(path, encoding='utf-8'))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    source_lst = []
    for i in range(0, len(texts)):
        source_lst.append(texts[i].metadata['source'])
    element_counts = Counter(source_lst)
    embedding = OpenAIEmbeddings(openai_api_key="openai_api_key")
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding)
    retriever = vectordb.as_retriever(search_kwargs={'k': 2}) # k는 top-k
    docs = retriever.get_relevant_documents(question)
    return docs

if __name__=='__main__':
    question = input().strip()
    docs = langchain(question)