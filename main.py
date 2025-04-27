import os
import streamlit as st
import pickle
import time
#from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

#load_dotenv()

# 🖌 Inject Custom CSS Theme
st.markdown("""
    <style>
    /* Change background color */
    body {
        background-color: black;

    }

    /* Glass Effect Main Content */
    .stApp {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        color: white;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #1c1c1c, #363636);
        color: #ffffff;
    }

    /* Titles */
    h1 {
        color: #9CBD9D;
        text-align: center;
        font-family: 'Trebuchet MS', sans-serif;
        font-size: 3em;
    }

    h2 {
        color: #3399FF;
        font-family: 'Trebuchet MS', sans-serif;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #28a745;
        color: white;
    }

    /* Input Text Box */
    .stTextInput>div>div>input {
        background-color: #333;
        color: #ffffff;

    }

    /* Answer Section */
    .stMarkdown h3 {
        color: #00FFAB;
    }
    </style>
""", unsafe_allow_html=True)

# 🎨 Title and Sidebar Title with color
st.markdown(
    "<h1>Dev: Research ChatBot 📈</h1>",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<h2>News Article URL's</h2>",
    unsafe_allow_html=True
)

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

# Unwanted keywords
unwanted_keywords = [
    "Advertisement", "Remove Ad", "Follow Us On", "moneycontrol",
    "Facebook", "twitter", "instagram", "linkedin", "telegram", "youtube",
    "Set Alert", "Portfolio", "Watchlist", "Message", "live", "bselive", "nselive", "Volume", "Today's L/H"
]

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    for doc in data:
        doc.page_content = "\n".join([
            line for line in doc.page_content.split("\n")
            if all(kw.lower() not in line.lower() for kw in unwanted_keywords) and len(line.strip()) > 20
        ])

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

            retrieved_docs = vectorstore.similarity_search(query, k=1)

            st.markdown("### Answer:")

            if retrieved_docs:
                clean_text = retrieved_docs[0].page_content.strip()

                st.write(clean_text)

                if 'source' in retrieved_docs[0].metadata:
                    st.markdown(f"[Source Link]({retrieved_docs[0].metadata['source']})")
            else:
                st.write("No relevant answer found.")

# Latest Version Of RockyBot, working on the side of maintenance 🚀


# Final 3.0
