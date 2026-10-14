import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from local .env file
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="TN Farmer Subsidies Expert RAG",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- EXPERT UI & STYLING (CSS) ---
st.markdown("""
    <style>
    /* Global Theme & Font Styling */
    .stApp {
        background: linear-gradient(135deg, #f3f8f4 0%, #e8f5e9 100%);
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #43a047 100%);
        padding: 30px 40px;
        border-radius: 16px;
        color: white;
        box-shadow: 0 10px 25px rgba(27, 94, 32, 0.15);
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: white !important;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #c8e6c9;
        margin-top: 5px;
        margin-bottom: 0;
    }

    /* Metric / Highlight Cards */
    .metric-container {
        display: flex;
        gap: 15px;
        margin-bottom: 25px;
    }
    .metric-card {
        background: #ffffff;
        padding: 15px 20px;
        border-radius: 12px;
        border: 1px solid #c8e6c9;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        flex: 1;
        text-align: center;
    }
    .metric-card h4 {
        margin: 0;
        color: #2e7d32;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card p {
        margin: 5px 0 0 0;
        color: #1f2937;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
        padding-top: 20px;
    }

    /* Chat Messages Look & Feel */
    .stChatMessage {
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR COMPONENT ---
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/b/b3/Emblem_of_Tamil_Nadu.svg", 
        width=80
    )
    st.markdown("### **Govt. of Tamil Nadu**")
    st.caption("Department of Agricultural & Farmers Welfare")
    st.markdown("---")
    
    st.markdown("### 🔍 **System Status**")
    st.success("🟢 PMKSY Micro-Irrigation Scheme)")
    st.success("🟢 SOLAR Pumpset Subsidy")
    st.success("🟢 Sub-Mission on Agricultural Mechanisation (SMAM)")
    
    st.markdown("---")
    st.markdown("### 📌 **Quick Help**")
    st.info("Click any sample query below or type your question directly in the chat input to explore scheme rules instantly.")

# --- HERO BANNER ---
st.markdown("""
    <div class="hero-banner">
        <div>
            <h1 class="hero-title">🌾 Tamil Nadu Farmer Subsidies Assistant</h1>
            <p class="hero-subtitle">Formars Scheme</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- QUICK METRICS ROW ---
st.markdown("""
    <div class="metric-container">
        <div class="metric-card">
            <h4>💧 Irrigation Schemes</h4>
            <p>Up to 100% Subsidy</p>
        </div>
        <div class="metric-card">
            <h4>☀️ Solar Energy</h4>
            <p>70% Pumpset Subsidy</p>
        </div>
        <div class="metric-card">
            <h4>🚜 Mechanisation</h4>
            <p>50% Machinery Grant</p>
        </div>
        <div class="metric-card">
            <h4>📱 Application Mode</h4>
            <p>Uzhavan Mobile App</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Configuration for local vector database and PDF
PDF_PATH = "tamil_nadu_farmer_subsidies.pdf"
DB_FAISS_PATH = "vectorstore/faiss_index"

@st.cache_resource
def initialize_vector_db():
    """Loads PDF, splits text, creates OpenAI embeddings, and stores them using local FAISS."""
    if not os.path.exists(PDF_PATH):
        st.error(f"❌ PDF file '{PDF_PATH}' not found in the directory. Please place it there.")
        return None

    # 1. Load the PDF document
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 3. Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. Create and save FAISS vector database locally
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(DB_FAISS_PATH)
    
    return vectorstore

# Initialize Vector Store
with st.spinner("🔄 Initializing local FAISS vector database and indexing documents..."):
    vector_store = initialize_vector_db()

if vector_store:
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Setup LLM using ChatOpenAI
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    except Exception as e:
        st.error("⚠️ Failed to initialize ChatOpenAI. Make sure your `OPENAI_API_KEY` is correctly set in your .env file.")
        st.stop()

    # RAG Prompt Template
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    
    Answer clearly and concisely based on the document provided:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Setup chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Interactive Quick-Click Query Buttons
    st.markdown("#### **💡 Quick Suggested Questions:**")
    col1, col2, col3, col4 = st.columns(4)
    
    selected_preset = None
    with col1:
        if st.button("💧 Micro-Irrigation Subsidy", use_container_width=True):
            selected_preset = "What is the subsidy breakdown for the PMKSY Micro-Irrigation Scheme?"
    with col2:
        if st.button("☀️ Solar Pumpsets", use_container_width=True):
            selected_preset = "What is the subsidy percentage for solar-powered pumpsets?"
    with col3:
        if st.button("🌱 Organic Farming", use_container_width=True):
            selected_preset = "What financial incentives are available for organic farming development?"
    with col4:
        if st.button("📱 How to Apply?", use_container_width=True):
            selected_preset = "How can farmers apply for these benefits digitally?"

    st.markdown("---")

    # Display prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Determine user query from either chat input box or preset button click
    user_query = st.chat_input("💬 Ask a question regarding subsidies, schemes, or eligibility...")
    
    if selected_preset:
        user_query = selected_preset

    if user_query:
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response using RAG chain
        with st.chat_message("assistant"):
            with st.spinner("🔍 Querying FAISS vector index & generating expert answer..."):
                # Retrieve relevant docs
                retrieved_docs = retriever.invoke(user_query)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

                # Build chain & invoke
                rag_chain = (
                    {"context": lambda x: context_text, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                response = rag_chain.invoke(user_query)
                st.markdown(response)
                
                # Add assistant response to state
                st.session_state.messages.append({"role": "assistant", "content": response})