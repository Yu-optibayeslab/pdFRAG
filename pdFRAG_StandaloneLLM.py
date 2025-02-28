import os
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import ollama
import json
import re

DEFAULT_DATA_DIR = "./documents/"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to extract metadata from PDF text
def extract_metadata(text):
    # Example: Extract title, authors, affiliations, year, and journal from the text
    lines = text.split("\n")
    title = lines[0].strip() if len(lines) > 0 else "Unknown"
    authors = lines[1].strip() if len(lines) > 1 else "Unknown"
    affiliations = lines[2].strip() if len(lines) > 2 else "Unknown"
    year = "Unknown"
    journal = "Unknown"
    doi = "Unknown"

    # Simple logic to extract year, journal, and DOI (can be improved)
    for line in lines:
        if "Journal:" in line:
            journal = line.replace("Journal:", "").strip()
        if "Year:" in line:
            year = line.replace("Year:", "").strip()
        if "DOI:" in line:
            doi = line.replace("DOI:", "").strip()
        # Alternatively, use regex to extract DOI
        doi_match = re.search(r"\b(10\.\d{4,}\/[\S]+)\b", line)
        if doi_match:
            doi = doi_match.group(1)

    return {
        "title": title,
        "authors": authors,
        "affiliations": affiliations,
        "year": year,
        "journal": journal,
        "doi": doi
    }

# Function to split text into chunks
def split_text_into_chunks(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to build FAISS index
def build_faiss_index(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return model, index, chunks

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, model, index, chunks, k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

# Function to summarize chunks using Ollama
def summarize_chunks(chunks, model_name):
    combined_context = "\n".join(chunks)
    response = ollama.generate(
        model=model_name,
        prompt=f"""Summarize the following text. The summary should be structured, well-organized, and concise, providing a clear understanding of the paper's significance.:\n\n Text to be summarized: \n\n {combined_context}"""
    )
    return response["response"]

# Function to save summaries and metadata locally
def save_summary(summary, metadata, output_dir="summaries"):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{metadata['title']}.json")
    with open(filename, "w") as f:
        json.dump({"summary": summary, "metadata": metadata}, f, indent=4)

# Function to save chat history to a JSON file
def save_chat_history(chat_history, output_dir="chat_history"):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "chat_history.json")
    with open(filename, "w") as f:
        json.dump(chat_history, f, indent=4)

# Get available models from Ollama
def get_available_models():
    """Fetches the list of available models from Ollama."""
    try:
        models_info = ollama.list()  # Fetch the model list

        # Safely extract model names
        available_models = [model.model for model in getattr(models_info, 'models', [])]

        if not available_models:
            st.warning("No models found in Ollama.")

        return available_models

    except Exception as e:
        st.error(f"Failed to fetch models from Ollama: {str(e)}")
        return []  # Ensure an empty list is returned on failure

# Run the function and store the models
available_models = get_available_models()

###################################
# Sidebar for LLM model selection #
###################################
st.sidebar.title("pdFRAG")
st.sidebar.write("## LLM model Settings")

# Dropdown to select a model
if available_models:
    model_name = st.sidebar.selectbox(
        "Choose a model",
        options=available_models
    )
else:
    st.sidebar.warning("No models available. Please ensure Ollama is running and models are downloaded.")
    model_name = None

# Sliders for chunk size and overlap
chunk_size = st.sidebar.slider("Chunk Size", 500, 3000, 2000)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 600, 400)

st.sidebar.write("## Documents")
# Directory input
pdf_directory = st.sidebar.text_input("PDF Directory:", DEFAULT_DATA_DIR)

# Check if the directory exists
if pdf_directory:
    if not os.path.isdir(pdf_directory):
        st.sidebar.error(f"Directory not found: {pdf_directory}.")
        
# Initialize session state for document metadata and FAISS components
if "document_metadata" not in st.session_state:
    st.session_state.document_metadata = {}
    
if "faiss_components" not in st.session_state:
    st.session_state.faiss_components = {}
    
if "selected_document" not in st.session_state:
    st.session_state.selected_document = None

# Process PDFs in the directory
if pdf_directory and os.path.isdir(pdf_directory):
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]
    ##############################
    # Sidebar for chunk settings #
    ##############################
    
    # Dropdown to select a document
    selected_document = st.sidebar.selectbox(
        "Select a document",
        options=pdf_files,
        index=None  # No document is selected by default
    )

    # Check if the selected document has changed
    if selected_document != st.session_state.selected_document:
        # Reset chat input and history
        st.session_state.chat_input = ""
        st.session_state.selected_document = selected_document

        # Add a new tab for the selected document if it doesn't exist
        if selected_document and selected_document not in st.session_state.tabs:
            st.session_state.tabs[selected_document] = []  # Initialize chat history for the new tab
            st.session_state.active_tab = selected_document  # Set the new tab as active
    
    if selected_document:
        if selected_document not in st.session_state.document_metadata:
            pdf_path = os.path.join(pdf_directory, selected_document)
            pdf_text = extract_text_from_pdf(pdf_path)

            # Extract metadata
            metadata = extract_metadata(pdf_text)
            st.session_state.document_metadata[selected_document] = metadata

            # Split text into chunks
            chunks = split_text_into_chunks(pdf_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # Build FAISS index
            model, index, chunks = build_faiss_index(chunks)

            # Store FAISS components for this document
            st.session_state.faiss_components[selected_document] = {
                "model": model,
                "index": index,
                "chunks": chunks
            }
        # Display metadata of the selected documents
        metadata = st.session_state.document_metadata[selected_document]
        st.sidebar.write("### Document Details")
        st.sidebar.write(f"**Title:** {metadata['title']}")
        st.sidebar.write(f"**Authors:** {metadata['authors']}")
        st.sidebar.write(f"**Affiliations:** {metadata['affiliations']}")
        st.sidebar.write(f"**Year:** {metadata['year']}")
        st.sidebar.write(f"**Journal:** {metadata['journal']}")
        st.sidebar.write(f"**DOI:** {metadata['doi']}")
    else:
        st.warning("Please select a document to chat with.")  
        
############################
# Top Row: Chat Window #
############################
# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

if "new_query" not in st.session_state:  # Temporary state variable to track input changes
    st.session_state.new_query = ""
    
# Initialize session state for tabs and chat history
if "tabs" not in st.session_state:
    st.session_state.tabs = {}  # Dictionary to store chat history for each article

#with upper row:
with st.container():
    st.subheader("What do you want to ask your PDF?")
    query = st.text_input("Enter your query:", key="chat_input")

    if query and selected_document in st.session_state.faiss_components:
        # Retrieve FAISS components for the selected document
        faiss_components = st.session_state.faiss_components[selected_document]
        model = faiss_components["model"]
        index = faiss_components["index"]
        chunks = faiss_components["chunks"]

        # Add user query to chat history
        st.session_state.tabs[selected_document].append({"role": "User", "content": query})

        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(
            query,
            model,
            index,
            chunks
        )

        # Generate a response using the selected model
        summary = summarize_chunks(relevant_chunks, model_name)

        # Add assistant's response to chat history
        st.session_state.tabs[selected_document].append({
                "role": "Assistant",
                "content": summary,
                "references": [f"Chunk {i+1}" for i, _ in enumerate(relevant_chunks)],
                "chunks": relevant_chunks
            })

        # Display the latest response at the top
        st.write("### Latest Response")
        st.write(f"**Assistant:** {summary}")       
##############################
# Bottom Row: Chat History #
##############################
with st.container():
#with right_column:
    # Apply light gray background to the right column
    st.markdown(
        """
        <style>
        .st-emotion-cache-1v0mbdj {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    #st.header("Chat History")
    st.subheader("Chat History")
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = None  # Track the currently active tab
        
    # Display tabs in the bottom row
    for tab_name in list(st.session_state.tabs.keys()):
        with st.expander(f"{tab_name} (Click to expand)"):
            for message in st.session_state.tabs[tab_name]:
                st.write(f"**{message['role']}:** {message['content']}")

    # Move "Save Chat History" and "Load Chat History" buttons to the bottom
    st.markdown("---")  # Add a horizontal line for separation
    if st.button("Save Chat History"):
        save_chat_history(st.session_state.tabs)
        st.success("Chat history saved successfully!")

