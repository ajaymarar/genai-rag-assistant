import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --------------------------------------------------
# 1. Load PDFs
# --------------------------------------------------
documents = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("data", file))
        documents.extend(loader.load())

# --------------------------------------------------
# 2. Split text
# --------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120
)
docs = text_splitter.split_documents(documents)

# --------------------------------------------------
# 3. Embeddings
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------------------------------
# 4. Vector store
# --------------------------------------------------
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --------------------------------------------------
# 5. Local LLM
# --------------------------------------------------
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.3
)

llm = HuggingFacePipeline(pipeline=pipe)

# --------------------------------------------------
# 6. Prompt (forces explanation, not Yes/No)
# --------------------------------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an enterprise document analyst.

Using ONLY the information in the context:
- Explain what the document says that is relevant to the question.
- Do NOT answer with just Yes or No.
- If profitability is discussed indirectly, explain how.
- If it is not explicitly stated, say so clearly and explain what is mentioned instead.

Context:
{context}

Question:
{question}

Answer (full explanation, no Yes/No):
"""
)

# --------------------------------------------------
# 7. Chain
# --------------------------------------------------
llm_chain = LLMChain(llm=llm, prompt=prompt)

qa_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context"
)

# --------------------------------------------------
# 8. Public function
# --------------------------------------------------
def answer_question(query: str) -> dict:
    docs = retriever.get_relevant_documents(query)

    # 🔑 THIS IS THE KEY CHANGE
    # Rephrase question to force explanation
    rewritten_question = (
        f"Explain based on the document whether or how the following is discussed: {query}"
    )

    answer = qa_chain.run(
        input_documents=docs,
        question=rewritten_question
    )

    return {
        "result": answer,
        "source_documents": docs
    }
