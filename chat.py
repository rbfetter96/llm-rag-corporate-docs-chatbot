# Project Chatbot using LLM & RAG

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Responda à pergunta do usuário usando documentos fornecidos no contexto.
    No contexto estão documentos que devem conter uma resposta.
    Sempre faça referência ao ID do documento (entre colchetes, por exemplo [0], [1]) do documento que foi usado para fazer uma consulta.
    Use quantas citações e documentos forem necessários para responder à pergunta.

    Contexto:
    {context}

    Pergunta:
    {question}
    """
)

PERSIST_DIRECTORY = "chroma_db"

def main():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

    llm = ChatOpenAI(
    openai_api_base="https://integrate.api.nvidia.com/v1",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="meta/llama-3.3-70b-instruct",
    temperature=0.2,
    streaming=True  # mantenha se quiser saída em tempo real
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt}
    )

    while True:
        query = input("\nPergunta (ou 'sair' para encerrar): ")
        if query.lower() in ["sair", "exit", "quit"]:
            break

        result = qa_chain({"query": query})
        print("\nResposta:")
        print(result["result"])

        print("\nDocumentos usados:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"[{i}] {doc.metadata.get('source', 'sem origem identificada')}")

if __name__ == "__main__":
    main()