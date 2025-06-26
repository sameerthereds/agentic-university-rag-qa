import requests
from bs4 import BeautifulSoup

# List of URLs to scrape
urls = [
    "https://www.memphis.edu/cs/",
    "https://www.memphis.edu/cs/programs/",
    "https://www.memphis.edu/cs/people/",
    "https://www.memphis.edu/cs/research/index.php",
    "https://www.memphis.edu/cs/about/index.php",
    "https://www.memphis.edu/cs/contact/index.php",
    "https://mdotcenter.org/santosh/?utm_source=chatgpt.com"
]

def scrape_pages(urls):
    documents = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove scripts and styles
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator=' ')
        documents.append(text)
    return documents

documents = scrape_pages(urls)


from sentence_transformers import SentenceTransformer
import numpy as np

# Load pre-trained model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2',token=False)

# Generate embeddings
embeddings = embedding_model.encode(documents)


import faiss

# Dimension of embeddings
dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


def retrieve_documents(query, model, index, documents, top_k=5):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    retrieved_docs = [documents[idx] for idx in indices[0]]
    return retrieved_docs


from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# # Load LLaMA model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
# # model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )

def load_llama_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = load_llama_pipeline()
def generate_answer_no_rag(query):
    # context = " ".join(retrieved_docs)
    prompt = f"Question: {query}\nAnswer:"
    # inputs = tokenizer(prompt, return_tensors="pt")
    # outputs = model.generate(**inputs, max_new_tokens=100)
    # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # return answer

    response = llm(prompt, max_new_tokens=60, temperature=0.9, do_sample=True)[0]["generated_text"]
    return response


query = "Who is the department head of the University of Memphis?"

print(generate_answer_no_rag(query))
import re
def generate_answer(query, retrieved_docs):
    context = " ".join(retrieved_docs)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    # inputs = tokenizer(prompt, return_tensors="pt")
    # outputs = model.generate(**inputs, max_new_tokens=100)
    # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # return answer
    response = llm(prompt, max_new_tokens=60, temperature=0.9, do_sample=True)[0]["generated_text"]
    answers = re.findall(r"Answer:\s*(.*?)(?=\n[A-Z][a-z]+:|\Z)", response, re.DOTALL)

    # Clean and return all answers
    return " ".join([ans.strip() for ans in answers])



retrieved_docs = retrieve_documents(query, model, index, documents)
# print(retrieved_docs)
# generate_answer_no_rag(query, retrieved_docs=retrieved_docs)
print(generate_answer(query,retrieved_docs=retrieved_docs))

