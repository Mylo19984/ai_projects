from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from typing import List
import bs4
import getpass
import os

if not os.environ.get("ANTHROPIC_API_KEY"):
  os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

promt_question = 'What is chain of thought?'

def retriever_similarity(query: str) -> List[Document]:
    docs, scores = zip(*vectorstore.similarity_search_with_score(query))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    return docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# user agent for web_base_loader
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
print(f"Split blog post into {len(all_splits)} sub-documents.")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma.from_documents(documents=all_splits, 
                                    embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

docs_similarity = retriever_similarity(promt_question)
docs_similarity_sorted = sorted(docs_similarity, key=lambda x: x.metadata["score"], reverse=True)
print('sorted similarity overview ---------')
print(docs_similarity_sorted)

llm = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")
prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result = rag_chain.invoke(promt_question)
print('output ---------')
print(result)