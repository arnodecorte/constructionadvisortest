import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load .env file

load_dotenv()

#UI prompt to enter API key

st.sidebar.title("OpenAI API Key required")
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

if not api_key:
    st.warning("App cannot function without API key")
    st.stop()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key
st.title("Construction Advisor")
st.markdown("Stel vragen over de BBL")

#load building regulations

with open("bbl_full_text.txt", "r", encoding="utf-8") as f:
    regulation_text = f.read()

#split into chunks so the AI can handle it

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_text(regulation_text)

#Create embeddings from chunks and store them as vectors

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

#user input
question = st.text_input("Ask a question about Dutch building regulations:")
if question:
    result=qa_chain(question)

    st.markdown("### Answer:")
    st.write(result["result"])

    st.markdown("### Source used:")
    for doc in result["source_documents"]:
        st.write(doc.page_content[:300])
