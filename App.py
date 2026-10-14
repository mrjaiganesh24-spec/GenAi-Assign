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

# Load environment variables from a local .env file
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="TN Farmer Subsidies RAG Assistant",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- CUSTOM COLORFUL CSS STYLING ---
st.markdown("""
    <style>
    /* Main Background & Font Styling */
    .stApp {
        background: linear-gradient(to bottom, #f4fbf7, #e8f5e9);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Container Styling */
    .main-header {
        background: linear-gradient(135deg, #2e7d32, #66bb6a);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2);
        margin-bottom: 25px;
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.2rem;
        margin-bottom: 5px;
    }
    .main-header p {
        color: #e8f5e9;
        font-size: 1.1rem;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #c8e6c9;
    }

    /* Custom Cards for Quick Suggestions */
    .suggestion-card {
        background-color: #ffffff;
        padding: 12px 18px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        font-size: 0.95rem;
        color: #2c3e50;
    }

    /* Chat input box styling */
    stTextInput > div > div > input {
        border-radius: 20px !important;
        border: 2px solid #81c784 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR BRANDING & QUICK INFO ---
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/b/b3/Emblem_of_Tamil_Nadu.svg", 
        width=90
    )
    st.markdown("### 🏛️ **Govt. of Tamil Nadu**")
    st.markdown("**Department of Agricultural and Farmers Welfare**")
    st.markdown("---")
    
    st.markdown("### 🚀 **Quick Scheme Ideas:**")
    st.markdown("<div class='suggestion-card'>💧 <b>Micro-Irrigation:</b> What is the subsidy for drip/sprinkler systems?</div>", unsafe_allow_html=True)
    st.markdown("<div class='suggestion-card'>☀️ <b>Solar Pumpsets:</b> How much subsidy is offered for solar pumpsets?</div>", unsafe_allow_html=True)
    st.markdown("<div class='suggestion-card'>🌱 <b>Organic Farming:</b> What are the incentives for organic cultivation?</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("💡 *Powered by LangChain, FAISS & OpenAI*")

# --- CREATIVE MAIN HEADER ---
st.markdown("""
    <div class="main-header">
        <h1>🌾 Tamil Nadu Farmer Subsidies Assistant</h1>
        <p>Your intelligent companion for exploring agricultural welfare & scheme details instantly[cite: 3].</p>
    </div>
""", unsafe_allow_html=True)

# Farmer Banner Image with decorative layout
st.image(
    "https://images.unsplash.com/photo-1595974482597-4f82fb67e4e1?auto=format&fit=crop&w=1200&q=80", 
    caption="Empowering Tamil Nadu Farmers through Digital Governance and Sustainable Agriculture", 
    use_container_width=True
)

st.markdown("<br>", unsafe_allow_html=True)

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

# Initialize Vector Store with a smooth spinner
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

    # Display prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input chat box
    if user_query := st.chat_input("💬 Ask a question about any subsidy or scheme..."):
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response using RAG chain
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching vector database & formulating response..."):
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