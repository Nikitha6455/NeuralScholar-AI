# NeuralScholar AI Agent

## Agent Name

NeuralScholar AI

## Purpose

NeuralScholar AI is an AI-powered PDF Research Assistant that uses Retrieval Augmented Generation (RAG) to answer user questions from uploaded PDF documents.

## Responsibilities

* Accept PDF uploads
* Extract document text
* Split documents into chunks
* Generate embeddings
* Store embeddings in FAISS
* Retrieve relevant chunks
* Generate answers using Gemini
* Display source citations

## Workflow

1. User uploads PDF
2. Text is extracted
3. Text is chunked
4. Embeddings are generated
5. FAISS index is created
6. User asks question
7. Relevant chunks are retrieved
8. Gemini generates answer
9. Sources are displayed

## Technologies

* Streamlit
* LangChain
* FAISS
* Sentence Transformers
* Google Gemini
