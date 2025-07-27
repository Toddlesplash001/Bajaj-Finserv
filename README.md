# Bajaj-Finserv

A **Streamlit-based intelligent chatbot** that allows users to upload and query multiple PDF documents. It leverages **LangChain**, **FAISS**, and **OpenAI's GPT-4o** to provide accurate, context-aware answers based on the uploaded documents. It also includes **objection detection** and **off-topic filtering** to guide responses appropriately.

---

## 🚀 Features

- 📄 Upload and embed multiple PDFs
- 🔍 Extracts and chunks text using `RecursiveCharacterTextSplitter`
- 🤖 Context-aware question answering using `RetrievalQA`
- 📚 Vector-based search powered by **FAISS**
- ⚠️ Objection detection using regex
- 🚫 Off-topic query filtering (e.g., weather, sports, politics)
- 💬 Chat interface with memory of previous interactions
- 📝 All chats are logged to `logs/chat_logs.txt`

---

## 🛠️ Tech Stack

- **Streamlit** – UI framework
- **LangChain** – LLM orchestration
- **FAISS** – Vector similarity search
- **OpenAI GPT-4o** – Language model
- **PyMuPDF (fitz)** – PDF parsing
- **dotenv** – API key handling

---

## 📂 Project Structure

```bash
📁 your_project/
│
├── app.py                 # Main Streamlit app
├── .env                   # OpenAI API key
├── requirements.txt       # All Python dependencies
