import streamlit as st
import PyPDF2
import google.generativeai as genai
import os
from dotenv import load_dotenv
from time import sleep
from tenacity import retry, stop_after_attempt, wait_exponential

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
genai.configure(api_key="AIzaSyDietYE2i25i_EyNULT9RiNZlD64PoLZY8")

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
            text += page.extract_text() + "\n"
            
        progress_bar.empty()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def split_text_into_chunks(text, chunk_size=3000):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_answer(query, text_chunks):
    # First try to find answer in PDF
    for chunk in text_chunks:
        prompt = f"Context:\n{chunk}\n\nQuestion: {query}\nAnswer:"
        model = genai.GenerativeModel("gemini-2.0-flash")
        try:
            response = model.generate_content(prompt)
            if response.text and "does not contain the answer" not in response.text.lower():
                return response.text.strip()
            sleep(1)  # Add small delay between API calls
        except Exception as e:
            st.warning(f"Retrying due to error: {str(e)}")
            continue
    
    # If answer not found in PDF, use Gemini for general knowledge
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""Please provide a detailed and accurate answer to this question: {query}

Rules:
1. Be specific and factual
2. If you're not sure about something, say so
3. Provide examples if relevant
4. Keep the answer clear and well-structured"""
        
        response = model.generate_content(prompt)
        return "Sorry, I couldn't find this information in the PDF. However, here's the answer to your question:\n\n" + response.text.strip()
    except Exception as e:
        return f"Error getting answer: {str(e)}"

# Initialize session state
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Enhanced header with improved icon
st.title("🤖 PDF Chat Assistant")

# Sidebar for PDF upload and controls
with st.sidebar:
    st.markdown("### 📚 Upload Your PDF")
    uploaded_file = st.file_uploader("", type="pdf")
    
    # Add clear chat button with better icon
    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
            if st.session_state.pdf_text:
                st.session_state.text_chunks = split_text_into_chunks(st.session_state.pdf_text)
                st.success("✅ PDF processed successfully!")
                st.info(f"📄 Pages: {len(PyPDF2.PdfReader(uploaded_file).pages)}")
                
        # Add document details section
        st.markdown("### 📊 Document Details")
        st.markdown(f"**Filename:** {uploaded_file.name}")
        st.markdown(f"**Size:** {round(uploaded_file.size/1024, 2)} KB")

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
            <p>Advanced algorithms to find precise information within your documents.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="center-icon">💬</div>
            <h3 style="text-align: center;">Natural Conversations</h3>
            <p>Ask questions in plain language and get meaningful responses.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="center-icon">🚀</div>
            <h3 style="text-align: center;">Powered by Gemini</h3>
            <p>Using Google's powerful Gemini AI to deliver accurate and helpful answers.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How to use section
    st.markdown("## 🔰 How to Use")
    st.markdown("""
    1. **Upload your PDF** using the sidebar on the left
    2. **Wait** for processing to complete
    3. **Ask questions** about the content of your PDF
    4. **Get instant answers** powered by AI
    """)
    
    # Call to action
    st.info("👈 Please upload a PDF document to start chatting")
else:
    # Chat input
    user_input = st.chat_input("Ask about your PDF...")
    
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.spinner("Thinking..."):
            answer = get_answer(user_input, st.session_state.text_chunks)
            st.session_state.chat_history.append(("assistant", answer))

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
