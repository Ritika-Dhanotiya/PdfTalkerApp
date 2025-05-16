import streamlit as st
import PyPDF2
import google.generativeai as genai
import os
from dotenv import load_dotenv
from time import sleep
from tenacity import retry, stop_after_attempt, wait_exponential
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set page config with increased upload size
st.set_page_config(
    page_title="PDF AI Chat",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Modern ChatGPT style UI
st.markdown("""
    <style>
        /* Main background and text colors */
        .stApp {
            background-color: #f5f5f5;
        }
        
        /* Header styling */
        h1 {
            color: #2c3e50 !important;
            font-family: 'Helvetica', sans-serif;
            font-size: 2rem;
            font-weight: 600;
            text-align: center;
            padding: 1rem;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #ffffff;
        }
        
        /* File uploader styling */
        .stFileUploader {
            padding: 1rem;
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Chat message containers */
        [data-testid="stChatMessage"] {
            background-color: #ffffff !important;
            border: none !important;
            margin-bottom: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* User message styling */
        [data-testid="user-message"] {
            background-color: #ffffff !important;
            padding: 1.5rem 1rem;
            color: #2c3e50 !important;
        }
        
        /* Assistant message styling */
        [data-testid="assistant-message"] {
            background-color: #ffffff !important;
            padding: 1.5rem 1rem;
            color: #2c3e50 !important;
        }
        
        /* Chat input styling */
        .stChatInput {
            background-color: #ffffff !important;
            border: 1px solid #e0e0e0 !important;
            color: #2c3e50 !important;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Success message styling */
        .stSuccess {
            background-color: #ffffff !important;
            color: #27ae60 !important;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Info message styling */
        .stInfo {
            background-color: #ffffff !important;
            color: #2980b9 !important;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Remove Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Button styling */
        .stButton > button {
            background-color: #ffffff !important;
            color: #2c3e50 !important;
            border: 1px solid #e0e0e0 !important;
            padding: 0.5rem 1rem !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .stButton > button:hover {
            background-color: #f8f9fa !important;
        }

        /* Text color fixes */
        .st-emotion-cache-1y4p8pa,
        .st-emotion-cache-16idsys,
        .st-emotion-cache-16idsys p,
        .st-emotion-cache-1104ytp {
            color: #2c3e50;
        }
        
        /* Centering for app info cards */
        .feature-card {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .center-icon {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("No API key found. Please set GEMINI_API_KEY in your environment or .env file.")
    st.stop()

genai.configure(api_key=api_key)

@st.cache_data(ttl=24*3600, max_entries=1000)
def extract_text_from_pdf(uploaded_file):
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        total_pages = len(pdf_reader.pages)
        
        # Show progress bar
        progress_bar = st.progress(0)
        
        for page_num in range(total_pages):
            progress = (page_num + 1) / total_pages
            progress_bar.progress(progress)
            
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text() or ""  # Handle None return
            
            # Add page number reference for better context
            text += f"\n\n=== Page {page_num + 1} ===\n{page_text}\n"
            
        progress_bar.empty()
        
        # Clean text - remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def split_text_into_chunks(text, chunk_size=2000, overlap=500):
    """Split text into overlapping chunks for better context preservation"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        # If this is the last chunk and it's too small, merge with previous
        if i + chunk_size >= len(words) and len(words) - i < chunk_size/2 and len(chunks) > 1:
            merged_chunk = chunks[-2] + " " + chunks[-1]
            chunks = chunks[:-2]
            chunks.append(merged_chunk)
            
    return chunks

def find_relevant_chunks(query, text_chunks, top_n=3):
    """Find the most relevant chunks using TF-IDF and cosine similarity"""
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Add query to chunks for vectorization
    all_documents = text_chunks + [query]
    
    try:
        # Generate TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(all_documents)
        
        # Calculate cosine similarity between query and chunks
        query_vector = tfidf_matrix[-1]  # Last vector is the query
        chunk_vectors = tfidf_matrix[:-1]  # All except last are chunks
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, chunk_vectors)[0]
        
        # Get indices of top N chunks
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Return top chunks and their similarity scores
        return [(text_chunks[i], similarities[i]) for i in top_indices]
    except Exception as e:
        st.warning(f"Error finding relevant chunks: {str(e)}. Using default chunks.")
        # Fallback: return first chunks if vectorization fails
        return [(chunk, 0.5) for chunk in text_chunks[:min(top_n, len(text_chunks))]]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_answer(query, text_chunks, chat_history=None):
    """Get answer using semantic search to find relevant chunks and Gemini for generation"""
    if not chat_history:
        chat_history = []
    
    # Find most relevant chunks for the query
    relevant_chunks = find_relevant_chunks(query, text_chunks)
    
    # Prepare context from relevant chunks
    context = "\n\n".join([f"Chunk (relevance {score:.2f}):\n{chunk}" for chunk, score in relevant_chunks])
    
    # Format chat history for context
    history_context = ""
    if chat_history:
        history_context = "Previous conversation:\n" + "\n".join([
            f"{'Q' if i%2==0 else 'A'}: {msg}" 
            for i, msg in enumerate(chat_history[-4:])  # Include last 2 Q&A pairs
        ]) + "\n\n"
    
    # Prepare prompt with better instructions
    prompt = f"""{history_context}Context from document:
{context}

Question: {query}

Instructions:
1. Answer the question based ONLY on the provided context
2. If the context doesn't contain enough information to answer, say "I don't have enough information in this document to answer that question" - don't try to make up an answer
3. If you're unsure, indicate your level of confidence
4. Cite specific parts or pages of the document when possible
5. Keep your answer clear and concise
6. If asked about information outside the document, only answer based on what's in the document

Answer:"""

    try:
        # Use more capable model for complex reasoning
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        
        if response.text:
            answer = response.text.strip()
            # Check if response indicates information wasn't found
            if any(phrase in answer.lower() for phrase in [
                "i don't have enough information", 
                "doesn't provide information",
                "not mentioned in the document",
                "not provided in the context"
            ]):
                return answer
            else:
                return answer
        else:
            return "I couldn't generate an answer from the document. Please try rephrasing your question."
            
        sleep(1)  # Add small delay between API calls
    except Exception as e:
        st.warning(f"Error generating response: {str(e)}")
        return f"I encountered an error while processing your question. Please try again or rephrase your query. Error details: {str(e)}"

# Initialize session state
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []  # Store Q&A pairs for context

# Enhanced header with improved icon
st.title("🤖 PDF Chat Assistant")

# Sidebar for PDF upload and controls
with st.sidebar:
    st.markdown("### 📚 Upload Your PDF")
    uploaded_file = st.file_uploader("", type="pdf")
    
    # Add chunk size and overlap controls
    st.markdown("### ⚙️ Advanced Settings")
    chunk_size = st.slider("Chunk Size (words)", 500, 5000, 2000, 100, 
                         help="Size of text chunks. Larger chunks provide more context but may reduce precision.")
    chunk_overlap = st.slider("Chunk Overlap (words)", 0, 1000, 500, 50, 
                            help="Overlap between chunks. Higher overlap helps maintain context across chunks.")
    
    # Add clear chat button with better icon
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.qa_pairs = []
        st.success("Chat history cleared!")
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
            if st.session_state.pdf_text:
                st.session_state.text_chunks = split_text_into_chunks(
                    st.session_state.pdf_text, 
                    chunk_size=chunk_size, 
                    overlap=chunk_overlap
                )
                st.success("✅ PDF processed successfully!")
                st.info(f"📄 Pages: {len(PyPDF2.PdfReader(uploaded_file).pages)}")
                st.info(f"🧩 Chunks: {len(st.session_state.text_chunks)}")
                
        # Add document details section
        st.markdown("### 📊 Document Details")
        st.markdown(f"**Filename:** {uploaded_file.name}")
        st.markdown(f"**Size:** {round(uploaded_file.size/1024, 2)} KB")
        
        # Add option to view extracted text
        if st.checkbox("View extracted text sample"):
            st.markdown("### 📝 Extracted Text (Sample)")
            sample_text = st.session_state.pdf_text[:500] + "..." if st.session_state.pdf_text else "No text extracted"
            st.text_area("", sample_text, height=150)

# Main chat interface
if not st.session_state.pdf_text:
    # App info section that disappears after PDF upload
    st.markdown("## 🌟 Welcome to PDF Chat Assistant!")
    
    # Feature cards with icons
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="center-icon">📚</div>
            <h3 style="text-align: center;">Upload Any PDF</h3>
            <p>Support for all types of PDF documents, including technical papers, books, and reports.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="center-icon">🔍</div>
            <h3 style="text-align: center;">Smart Search</h3>
            <p>Semantic search finds the most relevant parts of your document for each question.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="center-icon">💬</div>
            <h3 style="text-align: center;">Natural Conversations</h3>
            <p>Ask questions in plain language and get meaningful responses with context awareness.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="center-icon">🚀</div>
            <h3 style="text-align: center;">Powered by Gemini</h3>
            <p>Using Google's powerful Gemini AI to deliver accurate and helpful answers with relevant context.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How to use section
    st.markdown("## 🔰 How to Use")
    st.markdown("""
    1. **Upload your PDF** using the sidebar on the left
    2. **Wait** for processing to complete
    3. **Ask questions** about the content of your PDF
    4. **Get instant answers** powered by AI with relevant context from your document
    5. **Adjust settings** in the sidebar to optimize performance for your specific document
    """)
    
    # Call to action
    st.info("👈 Please upload a PDF document to start chatting")
else:
    # Chat input
    user_input = st.chat_input("Ask about your PDF...")
    
    if user_input:
        # Add user message to chat history display
        st.session_state.chat_history.append(("user", user_input))
        
        # Add question to Q&A pairs for context
        st.session_state.qa_pairs.append(user_input)
        
        with st.spinner("Analyzing document and finding answer..."):
            answer = get_answer(user_input, st.session_state.text_chunks, st.session_state.qa_pairs)
            
            # Add answer to chat history display and Q&A pairs for context
            st.session_state.chat_history.append(("assistant", answer))
            st.session_state.qa_pairs.append(answer)

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
