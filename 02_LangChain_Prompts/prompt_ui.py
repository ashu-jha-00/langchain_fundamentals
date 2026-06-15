from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate , load_prompt


load_dotenv() 
llm = HuggingFaceEndpoint (
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

st.header("Chat with LLM")

paper_input = st.selectbox("Choose your book:", ["The Psychology of Money", "Rich Dad Poor Dad", "The Intelligent Investor"])

style_input = st.selectbox("Choose your style:", ["Formal", "Informal", "Humorous"])

length_input = st.selectbox("Choose response length:", ["Short (1-2 sentences)", "Medium (3-5 sentences)", "Long (6+ sentences)"])

template = load_prompt("template.json")

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })
    st.write(result.content)