from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from typing import List
import bs4
import getpass
import os
from operator import itemgetter

if not os.environ.get("ANTHROPIC_API_KEY"):
  os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

promt_question = 'What is chain of thought?'
url_link = 'https://lilianweng.github.io/posts/2023-06-23-agent/'
parse_elements =  ["post-content", "post-title", "post-header"]
use_question_decomposition = True

def retriever_similarity(query: str) -> List[Document]:
    docs, scores = zip(*vectorstore.similarity_search_with_score(query))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    return docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def load_website(url_link, parse_elements):
    loader = WebBaseLoader(
    web_paths=(url_link,),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_= parse_elements
        )
        ),
    )
    return loader.load()

# Decomposition
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output should be only (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

# user agent for web_base_loader
docs = load_website(url_link, parse_elements)

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

generate_queries_decomposition = (prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))
questions = generate_queries_decomposition.invoke({"question":promt_question})
print(f'questions are: {questions}')

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

template = """Here is the question you need to answer:
\n --- \n {question} \n --- \n
Here is any available background question + answer pairs:
\n --- \n {q_a_pairs} \n --- \n
Here is additional context relevant to the question: 
\n --- \n {context} \n --- \n
Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)

def format_qa_pair(question, answer):
    """Format Q and A pair"""
    
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

q_a_pairs = ""
for q in questions:
    rag_chain_decomp = (
    {"context": itemgetter("question") | retriever, 
     "question": itemgetter("question"),
     "q_a_pairs": itemgetter("q_a_pairs")} 
    | decomposition_prompt
    | llm
    | StrOutputParser())

    answer = rag_chain_decomp.invoke({"question":q,"q_a_pairs":q_a_pairs})
    q_a_pair = format_qa_pair(q,answer)
    q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

print('output decomposition ---------')
print(answer)