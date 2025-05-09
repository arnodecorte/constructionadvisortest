import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from supabase import create_client, Client
import datetime
import os

# Load .env file

load_dotenv()
    
# Load Supabase credentials from environment variables
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]

# Initiate Supabase Client
supabase: Client = create_client(supabase_url, supabase_key)

# Feedback Submission Function
def submit_feedback(question, answer, source_chunks, rating, comment=""):
    timestamp = datetime.datetime.now().isoformat()
    data = {
        "question": question,
        "answer": answer,
        "source_chunks": source_chunks,
        "rating": rating,
        "comment": comment,
        "timestamp": timestamp
    }

    st.write("📤 Submitting this to Supabase:")
    st.json(data)  # Add this line temporarily for debugging

    try:
        response = supabase.table("ZJAC - feedback").insert(data).execute()
        return response
    except Exception as e:
        st.error(f"An error occurred while submitting feedback: {e}")
        return None

#Load the OpenAI API key from the environment

st.sidebar.title("OpenAI API Key required")
api_key = st.secrets["OPENAI_API_KEY"]
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
    
    #Feedback Section
    st.markdown("### Was this answer helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes"):
            submit_feedback(question, result["result"], "\n---\n".join([doc.page_content for doc in result["source_documents"]]), True)
            st.success("Thank you for your feedback!")
    with col2:
        if st.button("👎 No"):
            feedback_comment = st.text_input("What was wrong with the answer?")
            if feedback_comment:
                submit_feedback(question, result["result"], "\n---\n".join([doc.page_content for doc in result["source_documents"]]), False, feedback_comment)
                st.warning("Feedback submitted for training data, thanks")
