import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import requests
import numpy as np
import faiss
import torch
import re

# ========== Setup ==========
st.set_page_config(page_title="University of Memphis Agentic AI Information Retrieval System", layout="wide")

# URLs to scrape
urls = [
    "https://www.memphis.edu/cs/",
    "https://www.memphis.edu/cs/programs/",
    "https://www.memphis.edu/cs/people/",
    "https://www.memphis.edu/cs/research/index.php",
    "https://www.memphis.edu/cs/about/index.php",
    "https://www.memphis.edu/cs/contact/index.php",
    "https://mdotcenter.org/santosh/?utm_source=chatgpt.com"
]

@st.cache_data
def scrape_pages(urls):
    documents = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=' ')
        documents.append(text)
    return documents

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_llama_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load resources
documents = scrape_pages(urls)
embedding_model = load_embedding_model()
llm = load_llama_pipeline()

# Create FAISS index
embeddings = embedding_model.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ========== Query Functions ==========

def retrieve_documents(query, model, index, documents, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [documents[i] for i in indices[0]]

def generate_answer_no_rag(query):
    prompt = f"Question: {query}\nAnswer:"
    response = llm(prompt, max_new_tokens=150, temperature=0.9, do_sample=True)[0]["generated_text"]
    return response

def generate_answer_with_rag(query, retrieved_docs):
    context = " ".join(retrieved_docs)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = llm(prompt, max_new_tokens=150, temperature=0.9, do_sample=True)[0]["generated_text"]
    answers = re.findall(r"Answer:\s*(.*?)(?=\n[A-Z][a-z]+:|\Z)", response, re.DOTALL)
    return " ".join([ans.strip() for ans in answers])

# ========== Streamlit UI ==========

st.title("📘 University of Memphis Agentic AI Information Retrieval System")
query = st.text_input("Enter your question:")

model_choice = st.selectbox("Choose response mode:", ["RAG", "No RAG"])

if st.button("Generate Answer") and query:
    with st.spinner("Thinking..."):
        if model_choice.startswith("RAG"):
            retrieved = retrieve_documents(query, embedding_model, index, documents)
            answer = generate_answer_with_rag(query, retrieved)
        else:
            answer = generate_answer_no_rag(query)

    st.markdown("### 🧠 Answer")
    cleaned_answer = answer.strip()

    # Optionally remove everything before 'Answer:' in case model repeats prompt
    if "Answer:" in cleaned_answer:
        cleaned_answer = cleaned_answer.split("Answer:", 1)[-1].strip()

    st.markdown(f"""
<div style="
    background-color: #1e1e1e;
    color: #f0f0f0;
    padding: 1rem;
    border-radius: 0.5rem;
    font-size: 1rem;
    line-height: 1.6;
    border: 1px solid #333;
    word-wrap: break-word;
    white-space: pre-wrap;">
    {cleaned_answer.replace('\n', '<br>')}
</div>
""", unsafe_allow_html=True)

    # Optional: show retrieved docs for transparency (for RAG)
    if model_choice.startswith("RAG"):
        with st.expander("🔍 Show Retrieved Context"):
            for i, doc in enumerate(retrieved, 1):
                preview = doc.strip().replace("\n", " ")
                preview = " ".join(preview.split())  # Collapse extra whitespace
                st.markdown(f"**Document {i}:**\n\n{preview[:1000]}{'...' if len(preview) > 1000 else ''}")


# Who is the department chair of the Computer Science in the  University of Memphis?
# Who are the current students under Dr Santosh Kumar in the University of Memphis?