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
# ... existing code ...

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
    </style>
""", unsafe_allow_html=True)
# ... rest of the existing code ...
# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
        model = genai.GenerativeModel("gemini-1.5-pro")
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
        model = genai.GenerativeModel("gemini-1.5-pro")
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

# Simple header
st.title("💬 PDF Chat Assistant")

# Sidebar for PDF upload and controls
with st.sidebar:
    st.markdown("### 📁 Upload Your PDF")
    uploaded_file = st.file_uploader("", type="pdf")
    
    # Add clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
            if st.session_state.pdf_text:
                st.session_state.text_chunks = split_text_into_chunks(st.session_state.pdf_text)
                st.success("PDF processed successfully!")
                st.info(f"📄 Pages: {len(PyPDF2.PdfReader(uploaded_file).pages)}")

# Main chat interface
if not st.session_state.pdf_text:
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