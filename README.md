pdFRAG Chatbot

This repository contains a Retrieval-Augmented Generation (RAG) chatbot built using Python and Streamlit. The chatbot allows users to interact with PDF documents by asking questions and receiving summarized responses based on the content of the documents. It leverages advanced natural language processing (NLP) techniques, including text embedding, semantic search, and language models, to provide accurate and context-aware answers.

Key Features:

1. PDF Text Extraction: Extracts text from uploaded PDF documents.

2. Text Chunking: Splits large documents into smaller, manageable chunks for processing.

3. Semantic Search: Uses FAISS for efficient similarity search to retrieve relevant text chunks.

4. Language Model Integration: Generates summaries and responses using Ollama and its language models (e.g., llama2).

5. Interactive Web Interface: Built with Streamlit, providing an easy-to-use chat interface.

6. Metadata Extraction: Extracts document metadata (e.g., title, authors, DOI) for better context.

Use Cases

** Research Assistance: Quickly extract and summarize information from academic papers or reports.

** Document Q&A: Ask questions about the content of PDFs and get instant answers.

** Knowledge Management: Organize and interact with large collections of documents.

Technologies Used

** Python Libraries: streamlit, pypdf, langchain, sentence-transformers, faiss, ollama.

** NLP Models: all-MiniLM-L6-v2 for embeddings, llama2 (or other Ollama models) for summarization.

** Vector Database: FAISS for fast and efficient similarity search.
    
The following provides an overview of the dependencies required for the RAG (Retrieval-Augmented Generation) chatbot script. It explains the purpose of each library, how to install them, and any additional setup steps.

1. Core Dependencies

1.1. streamlit

Purpose: Used to create the web-based user interface for the chatbot.

Installation:

    pip install streamlit

Notes: Streamlit is a lightweight framework for building interactive web apps in Python. It is used here to create the chat interface and handle user inputs.

1.2. pypdf

Purpose: Used to extract text from PDF files.

Installation:

    pip install pypdf

Notes: This library reads PDF files and extracts text content, which is then processed by the chatbot.

1.3. langchain

Purpose: Provides the RecursiveCharacterTextSplitter for splitting text into smaller chunks.

Installation:

    pip install langchain

Notes: The RecursiveCharacterTextSplitter is used to divide the extracted PDF text into manageable chunks for processing and embedding.

1.4. sentence-transformers

Purpose: Used to generate embeddings (vector representations) of text chunks using the all-MiniLM-L6-v2 model.

Installation:

    pip install sentence-transformers

Notes:

 This library is built on top of PyTorch, so PyTorch will be installed automatically as a dependency.

 The all-MiniLM-L6-v2 model is a lightweight and efficient model for generating sentence embeddings.

1.5. faiss

Purpose: A library for efficient similarity search and clustering of dense vectors. Used to build and query the FAISS index for retrieving relevant text chunks.

Installation:

    For CPU version:

        pip install faiss-cpu

    For GPU version (requires CUDA):

        pip install faiss-gpu

Notes:

FAISS is optimized for fast nearest-neighbor search in high-dimensional spaces.

The CPU version is sufficient for most use cases, but the GPU version can provide significant speedups for large datasets.

1.6. ollama

Purpose: Used to interact with the Ollama API for generating summaries using a language model.

Installation:
    
        pip install ollama

Notes:

 Ollama is a tool for running large language models locally.

 You need to download a model (e.g., llama2) using the command:

        ollama pull llama2

Ensure the Ollama server is running while using the chatbot.

1.7. json

Purpose: A built-in Python library for working with JSON data. Used for saving summaries and chat history.

Installation: No installation required (comes with Python).

1.8. re

Purpose: A built-in Python library for regular expressions. Used for extracting metadata like DOI from text.

Installation: No installation required (comes with Python).

2. Optional Dependencies

2.1. numpy

Purpose: Often required by faiss and sentence-transformers for numerical operations.

Installation:

    pip install numpy

Notes: This library is typically installed automatically as a dependency of faiss or sentence-transformers.

2.2. tqdm

Purpose: Useful for displaying progress bars during long-running tasks (e.g., embedding generation).

Installation:

    pip install tqdm

Notes: This is optional and can be added if you want to monitor the progress of tasks like text splitting or embedding generation.

3. External Tools

3.1. Ollama Server

Purpose: The code relies on the Ollama server to generate summaries using a language model.

Setup:

Install Ollama on your system (follow the instructions at Ollama's official website).

Download a model (e.g., llama2) using the command:

        ollama pull llama2

Ensure the Ollama server is running while using the chatbot.

4. Environment Setup

4.1. Create a requirements.txt File

To simplify the installation of all dependencies, create a requirements.txt file with the following content:

        streamlit
        pypdf
        langchain
        sentence-transformers
        faiss-cpu
        ollama
        numpy

4.2. Install Dependencies

Run the following command to install all dependencies:

        pip install -r requirements.txt

5. Additional Notes

5.1. PyTorch

The sentence-transformers library requires PyTorch, which will be installed automatically when you install sentence-transformers.

If you want to avoid installing PyTorch (e.g., due to its large size), you would need to replace sentence-transformers with an alternative embedding library, but this would require significant changes to the code.

5.2. TensorFlow

TensorFlow is not required for this code unless you are using advanced features of FAISS with GPU support.

5.3. PDF Files

The code expects PDF files to be placed in the ./documents/ directory (or the directory specified in the sidebar). Ensure the PDFs are accessible and readable.

6. Running the Chatbot

Ensure all dependencies are installed.

Start the Ollama server (if not already running):
    

        ollama serve

Run the Streamlit app:
    

        streamlit run RAG_Chatbot_StandaloneLLM.py

Open the provided URL in your browser to interact with the chatbot.

7. Troubleshooting

FAISS Installation Issues: If you encounter issues installing faiss, ensure you have the correct version for your system (CPU or GPU). For CPU-only systems, use faiss-cpu.

Ollama Model Not Found: Ensure the model you select in the Streamlit sidebar (e.g., llama2) is downloaded and available in your Ollama environment.

PDF Text Extraction Issues: Verify that the PDF files are not scanned images and contain extractable text.

For questions or feedback, feel free to reach out:

Email: yu@optibayeslab.com
