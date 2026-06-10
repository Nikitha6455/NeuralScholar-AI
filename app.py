import streamlit as st
import tempfile

from sentence_transformers import SentenceTransformer

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import create_embeddings
from src.vector_store import create_faiss_index
from src.retriever import retrieve_chunks
from src.llm import get_llm


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="NeuralScholar AI",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(
        135deg,
        #020617,
        #081028,
        #11235a
    );
    color:white;
}

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:rgba(0,0,0,0.25);
    color:white !important;
}

/* Title */
.main-title{
    text-align:center;
    font-size:58px;
    font-weight:700;
    color:#60a5fa;
}

.sub-title{
    text-align:center;
    font-size:24px;
    color:#dbeafe;
    margin-bottom:40px;
}

/* Glass Card */
.glass{
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(10px);
    padding:25px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.1);
}

/* Metrics */
[data-testid="metric-container"]{
    background:rgba(255,255,255,0.08);
    border-radius:15px;
    padding:15px;
}

/* Text Input */
.stTextInput input{
    background:rgba(255,255,255,0.08);
    color:white;
}

/* Upload Box */
[data-testid="stFileUploader"]{
    background:rgba(255,255,255,0.08);
    padding:15px;
    border-radius:15px;
}
/* Answer Box */
.answer-box{
    background:rgba(255,255,255,0.08);
    padding:20px;
    border-radius:15px;
    border-left:5px solid #60a5fa;
}

/* Source Box */
.source-box{
    background:rgba(255,255,255,0.05);
    padding:10px;
    border-radius:10px;
    margin-bottom:10px;
}

/* Sidebar text white */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* Sidebar markdown text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] span {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)
# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

with st.sidebar:
    st.title("🧠 NeuralScholar AI")
    st.markdown("### Research • Retrieve • Reason")

    st.markdown("---")

    st.markdown("""
    ✅ PDF Question Answering

    ✅ Source Citations

    ✅ Gemini Powered

    ✅ FAISS Retrieval

    ✅ Research Assistant
    """)

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown(
    '<div class="main-title">🧠 NeuralScholar AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Ask Questions. Get Cited Answers. Instantly.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="glass">
    <h2>📑 AI Research Assistant</h2>
    Upload research papers, reports, technical documents and PDFs.
    Ask questions and receive context-aware answers with citations.
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

llm = get_llm()

# ------------------------------------------------
# PDF UPLOAD
# ------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type=["pdf"]
)

# ------------------------------------------------
# PROCESS PDF
# ------------------------------------------------

if uploaded_file:

    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            pdf_path = temp_file.name

        pages = load_pdf(pdf_path)

        chunks = split_documents(pages)

        embeddings = create_embeddings(chunks)

        index = create_faiss_index(embeddings)

        st.session_state.index = index
        st.session_state.chunks = chunks

    st.success("PDF processed successfully")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Pages", len(pages))

    with col2:
        st.metric("Chunks", len(chunks))

    st.divider()

    # ------------------------------------------------
    # QUESTION
    # ------------------------------------------------

    question = st.text_input(
        "💬 Ask a question about your PDF"
    )

    if question:

        with st.spinner("Analyzing document..."):

            query_embedding = embedding_model.encode(
                question
            )

            results = retrieve_chunks(
                query_embedding,
                st.session_state.index,
                st.session_state.chunks,
                k=5
            )

            context = "\n\n".join(
                [chunk["text"] for chunk in results]
            )

            prompt = f"""
Answer ONLY from the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

            response = llm.invoke(prompt)

        st.markdown("## 🤖 Answer")

        st.markdown(
            f"""
            <div class="answer-box">
            {response.content}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.markdown("## 📚 Source Pages")

        pages_used = sorted(
            set(
                chunk["page"]
                for chunk in results
            )
        )

        for page in pages_used:
            st.markdown(
                f"""
                <div class="source-box">
                📄 Page {page}
                </div>
                """,
                unsafe_allow_html=True
            )