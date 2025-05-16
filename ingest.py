#Project Chatbot using LLM & RAG
#PDF Ingestion
#This step extracts and preprocesses informations from PDF documents for chunking and vector storage.




import os
from os.path import isfile, isdir, join
from langchain_community.document_loaders import PyMuPDFLoader as PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Used for chunking 
from langchain_community.vectorstores import Chroma   # VectorDB
from langchain_huggingface import HuggingFaceEmbeddings  # Generate vectors (embeddings) from texts
from dotenv import load_dotenv



load_dotenv()


# Function to list all files paths
def ingest_documents(directory):

	# Initialize an empty list to store the file paths
	docs_list = []

	# Iterate over all files and directories in the specified directory
	for file in os.listdir(directory):

		# If it is a file, add to the list
		if isfile(join(directory, file)):
			docs_list.append(join(directory, file))

		# If it's a directory, call the function recursively and add the results to the list
		elif isdir(join(directory, file)):
			docs_list += ingest_documents(join(directory, file))

	# Return list with file paths
	return docs_list



def main_indexing(mypath):
	# Embedding local com modelo leve e eficiente
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Cria ou carrega a base vetorial
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")

    docs_path = ingest_documents(mypath)

	# Iterate over all files in the list
    for file in docs_path:
        if file.lower().endswith(".pdf"):
            print(f"processing {file}")
            try:
                loader = PDFLoader(file)
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                chunks = splitter.split_documents(pages)
                vectorstore.add_documents(chunks)
            except Exception as e:
                print(f"Error processing file: {file} -> {e}")

    print("Ingestion complete")



if __name__ == "__main__":
    import sys
    arguments = sys.argv

    if len(arguments) > 1:
        main_indexing(arguments[1])
    else:
        print("You must provide a folder path containing documents to index.")