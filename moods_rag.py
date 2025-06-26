# moods_rag.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from llama_model import load_llama_pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_experimental.agents import create_pandas_dataframe_agent

# === Setup ===
st.set_page_config(page_title="Stress RAG App", layout="wide")

@st.cache_resource
def get_llm():
    pipe = load_llama_pipeline()
    return HuggingFacePipeline(pipeline=pipe)

@st.cache_data
def load_data():
    df = pd.read_pickle("/home/sneupane/Side Project/RAG/final_df_stressed_with_category.pickle")
    df['starttime'] = pd.to_datetime(df['starttime'])
    df.dropna(subset=['mod_stressor'], inplace=True)
    return df

llm = get_llm()
data = load_data()

# === Agent Setup ===
agent = create_pandas_dataframe_agent(
    llm,
    data,
    verbose=True,
    handle_parsing_errors=True,
    agent_type="openai-tools",  # or "zero-shot-react-description" depending on your setup
    allow_dangerous_code=True
)

# === UI ===
st.title("🧠 Stressor Data Explorer with AI Agent")
st.write("Ask anything about the stressor dataset or generate visual insights using natural language.")

query = st.text_area("Ask your question about the data:")

if st.button("Submit Query") and query:
    with st.spinner("🤖 Thinking..."):
        try:
            answer = agent.run(query)
            st.success("Answer:")
            st.write(answer)
        except Exception as e:
            st.error("⚠️ An error occurred while executing the generated code:")
            st.exception(e)