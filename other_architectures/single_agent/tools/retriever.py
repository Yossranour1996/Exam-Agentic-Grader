# src/tools/retriever.py
"""Optional course-material retriever used by future grounded agents."""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


KNOWLEDGE_BASE = [
]

def create_retriever():
	if not KNOWLEDGE_BASE:
		raise ValueError("KNOWLEDGE_BASE is empty; configure course PDF paths first.")
	# Create documents
	documents = []
	for path in KNOWLEDGE_BASE:

		# Safety measure
		if not os.path.exists(path):
			raise FileNotFoundError(f"PDF file not found: {path}")

		# This loads the PDF
		try:
			loader = PyPDFLoader(path)
			pages = loader.load()
		except Exception as e:
			print(f"Error loading PDF: {e}")
			raise

		# Chunking Process
		text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=1000,
			chunk_overlap=200
		)

		documents = documents + text_splitter.split_documents(pages)

	# Our Embedding Model - has to also be compatible with the LLM
	embeddings = OpenAIEmbeddings(
		model="text-embedding-3-small",
	)

	persist_directory = os.path.split(os.getcwd())[0]
	collection_name = "Java_Knowledge_Base"
	if not os.path.exists(persist_directory):
		os.makedirs(persist_directory)

	try:
		# Here, we actually create the chroma database using our embeddigns model
		vectorstore = Chroma.from_documents(
			documents=documents,
			embedding=embeddings,
			persist_directory=persist_directory,
			collection_name=collection_name
		)
	except Exception as e:
		print(f"Error setting up ChromaDB: {str(e)}")
		raise

	# Now we create our retriever 
	retriever = vectorstore.as_retriever(
		search_type="similarity",
		search_kwargs={"k": 5} # K is the amount of chunks to return
	)

	return retriever
