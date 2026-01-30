import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 1. Load ALL PDFs from data folder
documents = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("data", file))
        documents.extend(loader.load())

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
docs = text_splitter.split_documents(documents)

# 3. FREE embeddings (CPU)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Vector store
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# 5. Load SMALL local LLM (CPU-friendly)
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256
)

llm = HuggingFacePipeline(pipeline=pipe)

# 6. RAG chain WITH source documents
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 7. Ask questions
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"query": query})

    print("\nAnswer:")
    print(result["result"])

    print("\nSources:")
    for doc in result["source_documents"]:
        source = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", "Unknown page")
        print(f"- {source}, page {page}")
