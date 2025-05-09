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

#Load the OpenAI API key from the environment

st.sidebar.title("OpenAI API Key required")
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key

if not api_key:
    st.warning("App cannot function without API key")
    st.stop()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key
st.title("ZJAC - BBL AI Assistant")
st.markdown("Deze applicatie helpt je bij het vinden van informatie over de Nederlandse bouwvoorschriften (Bouwbesluit). Stel een vraag en ontvang een antwoord gebaseerd op de regelgeving.")

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

# Check for Debug Mode
debug_mode = st.secrets.get("DEBUG_MODE", False)
if debug_mode:
    st.warning("DEBUG MODE IS AAN")

# User input
question = st.text_input("Stel een vraag over de Nederlandse bouwvoorschriften:")
if question:
    # Provide dummy response for debugging purposes

    if debug_mode:
        result = {
            "result": "Dit is een voorbeeldantwoord voor debugdoeleinden.",
            "source_documents": [
                {"page_content": "Dit is een voorbeeldbron."}
            ]
        }
    else:
        # actual GPT-3.5 call
        result = qa_chain(question)

    st.markdown("### Antwoord:")
    st.write(result["result"])
    
    # Feedback Section
    st.markdown("### Was dit antwoord nuttig?")
    col1, col2 = st.columns(2)
    
    # Use session state to track the feedback button state
    if "no_feedback" not in st.session_state:
        st.session_state.no_feedback = False
    
    with col1:
        if st.button("👍 Ja"):
            submit_feedback(
                question,
                result["result"],
                "\n---\n".join([doc.page_content for doc in result["source_documents"]]),
                True
            )
            st.success("Bedankt voor je feedback!")
    with col2:
        if st.button("👎 Nee"):
            st.session_state.no_feedback = True  # This will trigger the feedback comment input
            
    if st.session_state.no_feedback:
        st.markdown('### Feedback:')
        feedback_comment = st.text_area("Wat was er mis met het antwoord?", height=150)
        if st.button("Verzend feedback"):
            if feedback_comment.strip(): # Ensure comment is not empty
                submit_feedback(
                    question,
                    result["result"],
                    "\n---\n".join([doc.page_content for doc in result["source_documents"]]),
                    False,
                    feedback_comment
                )
            st.success("Feedback ingediend voor trainingsdata, bedankt!")
            st.session_state.no_feedback = False  # Reset the feedback state after submission
        else:
            st.warning("Vul alstublieft een feedbackcommentaar in voordat u verzendt.")

    st.markdown("### Gebruikte bron:")
    for doc in result["source_documents"]:
        if isinstance(doc, dict):
            st.write(doc["page_content"][:300]) # Process as a dictionary if debug mode is on
        else:
            st.write(doc.page_content[:300]) # Process as a LangChain document if not in debug mode
