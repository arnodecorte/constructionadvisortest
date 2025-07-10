import streamlit as st
from dotenv import load_dotenv
import re

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
    try:
        response = supabase.table("feedback").insert(data).execute()
        # If insert is successful, response.data will be a list with the inserted row
        if response.data and isinstance(response.data, list):
            st.write("✅ Feedback opgeslagen in Supabase.")
        else:
            st.error(f"❌ Fout bij opslaan in Supabase: {response.data}")
    except Exception as e:
        st.error(f"❌ Fout bij opslaan in Supabase: {e}")

#Load the OpenAI API key from the environment

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

# Create a slider for the temperature control
with st.expander("Advanced controls"):
    temperature = st.slider(
        "Creativiteit van het antwoord (temperature)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.05,
        help="Lager = preciezer, hoger = creatiever"
    )

# Use the selected temperature when creating the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Check for Debug Mode
debug_mode = st.secrets.get("DEBUG_MODE", False)
if debug_mode:
    st.warning("DEBUG MODE IS AAN")

# Initialize session state for question and result
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = ""

# User input
question = st.text_input("Stel een vraag over de Nederlandse bouwvoorschriften:")

col1, col2 = st.columns(3,1)
with col2:
    submit = st.button("Submit")

if submit and question:
    # Only call the LLM if the question is new or changed
    if question != st.session_state.last_question:
        st.session_state.last_question = question
        if debug_mode:
            st.session_state.last_result = {
                "result": "Dit is een voorbeeldantwoord voor debugdoeleinden.",
                "source_documents": [
                    {"page_content": "Dit is een voorbeeldbron."}
                ]
            }
        else:
            st.session_state.last_result = qa_chain(question)
else:
    # If no new question submitted, do not call LLM again
    pass

result = st.session_state.last_result

if result:
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

# Function to extract article or section numbers and create hyperlinks
def create_hyperlinked_source(source_text):
    # Match article numbers (e.g., Artikel 2.24)
    article_match = re.search(r'Artikel\s(\d+\.\d+)', source_text)
    if article_match:
        article_id = f"artikel-{article_match.group(1)}"
        return f'<a href="#{article_id}" target="iframe">{source_text}</a>'
    
    # Match section numbers (e.g., § 2.3.2)
    section_match = re.search(r'§\s(\d+\.\d+\.\d+)', source_text)
    if section_match:
        section_id = f"section-{section_match.group(1)}"
        return f'<a href="#{section_id}" target="iframe">{source_text}</a>'
    
    # If no match, return the plain source text
    return source_text

# Display AI-generated sources with hyperlinks
st.markdown("### Gebruikte bron:")
for doc in result["source_documents"]:
    if isinstance(doc, dict):
        source_text = doc["page_content"][:300]  # Process as a dictionary if debug mode is on
    else:
        source_text = doc.page_content[:300]  # Process as a LangChain document if not in debug mode
    
    # Create a hyperlink for the source text
    hyperlinked_source = create_hyperlinked_source(source_text)
    st.markdown(hyperlinked_source, unsafe_allow_html=True)
    
# Display BBL Html as a source
st.markdown("### Volledige Bouwbesluit:")
with open("bbl_full_text.html", "r", encoding="utf-8") as html_file:
    html_content = html_file.read()
    st.components.v1.html(html_content, height=2000, width=None, scrolling=True)