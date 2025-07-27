import os
import fitz  # PyMuPDF
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load env variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY not found in .env")

# Initialize FastAPI
app = FastAPI(title="Multi-PDF Chatbot API")

# Initialize global variables
vectorstore = None
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o", temperature=1)

# Function to extract text from a PDF
def extract_text_from_pdf(file: UploadFile) -> str:
    pdf = fitz.open(stream=file.file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in pdf])

# Create chunks and build vectorstore
def create_vectorstore_from_text(text: str, source_name: str) -> FAISS:
    doc = Document(page_content=text, metadata={"source": source_name})
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = splitter.split_documents([doc])
    return FAISS.from_documents(chunks, embeddings)

# Update global vectorstore
def update_vectorstore(text: str, source_name: str):
    global vectorstore
    new_vs = create_vectorstore_from_text(text, source_name)
    if vectorstore is None:
        vectorstore = new_vs
    else:
        vectorstore.merge_from(new_vs)

# Setup QA Chain
def get_qa_chain():
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 50}),
        chain_type="stuff"
    )

# API route to upload PDFs only
@app.post("/upload/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    global vectorstore
    vectorstore = None  # Reset for new session
    for file in files:
        print(f"Received: {file.filename}")

    for file in files:
        if not file.filename.endswith(".pdf"):
            return JSONResponse(content={"error": "Only PDF files are supported."}, status_code=400)
        try:
            text = extract_text_from_pdf(file)
            update_vectorstore(text, file.filename)
        except Exception as e:
            return JSONResponse(content={"error": f"Failed to process {file.filename}: {str(e)}"}, status_code=500)

    return {"message": f"Successfully processed {len(files)} PDF(s)."}

# API route to ask a question
@app.post("/query/")
async def ask_query(question: str = Form(...)):
    if vectorstore is None:
        return JSONResponse(content={"error": "❌ No PDFs uploaded. Please upload PDFs first."}, status_code=400)

    qa = get_qa_chain()
    try:
        answer = qa.run(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#NEW COMBINED API route to upload PDFs and ask a question in one call
@app.post("/upload-and-ask/")
async def upload_and_ask(files: List[UploadFile] = File(...), question: str = Form(...)):
    global vectorstore
    vectorstore = None  # Reset for new session

    for file in files:
        if not file.filename.endswith(".pdf"):
            return JSONResponse(
                content={"error": f"Only PDF files are supported. Got: {file.filename}"},
                status_code=400
            )
        try:
            text = extract_text_from_pdf(file)
            update_vectorstore(text, file.filename)
        except Exception as e:
            return JSONResponse(content={"error": f"Failed to process {file.filename}: {str(e)}"}, status_code=500)

    # Now handle the query
    qa = get_qa_chain()
    try:
        answer = qa.run(question)
        return {
            "message": f"Processed {len(files)} file(s) and answered the query.",
            "question": question,
            "answer": answer
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
