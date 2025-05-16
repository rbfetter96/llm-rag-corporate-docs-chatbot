# llm-rag-corporate-docs-chatbot

# 🧠 LLM + RAG Chatbot for Corporate Document Search

This project is a prototype of an intelligent chatbot that leverages **LLMs (Large Language Models)** and **RAG (Retrieval-Augmented Generation)** to search and summarize information from corporate documents such as meeting minutes, bylaws, and other structured text files.

## 📌 Objective

To simulate how Generative AI can be applied in business environments to reduce time spent searching for relevant information in long and repetitive documents, improving decision-making and operational efficiency.

## ⚙️ Tech Stack

- **Python 3.11**
- **LangChain** (for chaining components)
- **ChromaDB** (vector store)
- **HuggingFace Transformers** (`all-mpnet-base-v2` and `all-MiniLM-L6-v2` for embeddings)
- **Unstructured** (document parsing)
- **OpenAI or LLaMA API** (LLM of your choice)



## 📂 Folder Structure
├── chat.py # Runs the RAG chatbot in terminal
├── ingest.py # Script to process, chunk, embed, and store PDFs
├── requirements.txt # All Python dependencies
├── chroma_db/ # Persisted vector DB
└── Files/ # Folder with PDF documents


## 🚀 How It Works

1. **Ingest step** (`ingest.py`)
   - Reads `.pdf` files in the `Files/` folder
   - Extracts text using `UnstructuredPDFLoader`
   - Splits text into chunks
   - Embeds with HuggingFace models
   - Saves to `ChromaDB` locally

2. **Query step** (`chat.py`)
   - Accepts a user question via terminal
   - Retrieves top-k similar chunks using vector similarity
   - Sends the context and question to an LLM (via API)
   - Returns an answer based on retrieved context


## 📝 Prompt Engineering

Prompt design plays a critical role in LLM-based applications. In this project, the system prompt was tailored to ensure that the chatbot responds as a **legal/administrative assistant**, citing content only from the retrieved documents.

## 📄 Dataset

For confidentiality reasons, real company documents were **not used**. Instead, the dataset consists of publicly available **academic articles** on:
- Psychological safety in the workplace
- Feedback culture
- Meaningful work
- Job crafting
- Employee engagement

This ensures no sensitive data is exposed, while still simulating realistic use cases.


## 🎯 Business Relevance

During my professional experience, I often witnessed how external audits and paralegal activities demanded a lot of time to locate documents—even when those documents were well-organized. This project shows how Generative AI with RAG can streamline such workflows and reduce cognitive load for knowledge workers.
