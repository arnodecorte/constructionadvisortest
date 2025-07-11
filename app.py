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

# Welcome Tab

if "welcome_acknowledged" not in st.session_state:
    st.session_state.welcome_acknowledged = False

def show_welcome_popup():
    """Display the welcome tab with instructions and acknowledgments."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.container():
            
            st.markdown("""
            **Belangrijke informatie over het gebruik van deze applicatie (English + OK button below):**
            
            Dit is een experimentele applicatie, dus antwoorden kunnen onjuist zijn. Het programma zelf kan ook bugs bevatten, dus meld ze als je ze vindt.

            Elke vraag kost (een beetje) geld, omdat we de API en tokens van OpenAI gebruiken om vragen te beantwoorden. Het is geen groot bedrag, maar zorg ervoor dat je geen vragen spamt, zodat we binnen het maandelijkse budget blijven.

            Je feedback is belangrijk. Als ZJAC een goed antwoord geeft, geweldig! Als ZJAC onjuist is en je het antwoord toevallig zelf weet, helpt het enorm om feedback te geven voor verdere verbetering.

            Veel plezier en als je vragen of suggesties hebt, of als je ook mee wilt doen, neem dan contact met mij (Arno) op.
                        
            ---------
                        
            This is an experimental application so answers may be incorrect. The program itself may also contain bugs, so please report them if you find any.

            Each question costs (a little bit) of money as we are using OpenAI's API and tokens to answer questions. It is not a large amount, but please be mindful not to spam questions so that we remain within the monthly budget.

            Your feedback is important. If ZJAC provides a good answer, great! If ZJAC is incorrect, and you happen to know the answer yourself, it helps alot to submit feedback for fine-tuning.

            Have fun and if you have any questions, suggestions or if you want to also take part, please reach out to me (Arno)
            """)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Centered OK button
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn2:
                if st.button("OK", type="primary", use_container_width=True):
                    st.session_state.welcome_acknowledged = True
                    st.rerun()

def main_app():
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
            response = supabase.table("ZJAC - feedback").insert(data).execute()
            # If insert is successful, response.data will be a list with the inserted row
            if response.data and isinstance(response.data, list):
                st.write("✅ Feedback opgeslagen in Supabase.")
            else:
                st.error(f"❌ Fout bij opslaan in Supabase: {response.data}")
        except Exception as e:
            st.error(f"❌ Fout bij opslaan in Supabase: {e}")

    # Load the OpenAI API key from the environment
    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key

    if not api_key:
        st.warning("App cannot function without API key")
        st.stop()

    # Set OpenAI API key
    os.environ["OPENAI_API_KEY"] = api_key
    st.title("ZJAC - BBL AI Assistant")
    st.markdown("Deze applicatie helpt je bij het vinden van informatie over de Nederlandse bouwvoorschriften (Bouwbesluit). Stel een vraag en ontvang een antwoord gebaseerd op de regelgeving.")

    # Load building regulations
    with open("bbl_full_text.txt", "r", encoding="utf-8") as f:
        regulation_text = f.read()

    # Split into chunks so the AI can handle it
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(regulation_text)

    # Create embeddings from chunks and store them as vectors
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Advanced controls
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

    # User input and LLM call in a form
    with st.form("ask_form"):
        question = st.text_input("Stel een vraag over de Nederlandse bouwvoorschriften:")
        submit_question = st.form_submit_button("Vraag")

    if submit_question:
        if debug_mode:
            result = {
                "result": "Dit is een voorbeeldantwoord voor debugdoeleinden.",
                "source_documents": [
                    {"page_content": "Dit is een voorbeeldbron."}
                ]
            }
        else:
            result = qa_chain(question)
        st.session_state["last_result"] = result
        st.session_state["last_question"] = question
        st.session_state.no_feedback = False  # Reset feedback state on new question

    # Display the last result and feedback UI if available
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        question = st.session_state["last_question"]

        st.markdown("### Antwoord:")
        st.write(result["result"])

        st.markdown("### Was dit antwoord nuttig?")
        col1, col2 = st.columns(2)

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
                if feedback_comment.strip():  # Ensure comment is not empty
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

        # Function to display source chunk as plain text
        def display_source_chunk(source_text):
            return source_text[:300]

        # Display AI-generated sources as plain text
        st.markdown("### Gebruikte bron:")
        for doc in result["source_documents"]:
            if isinstance(doc, dict):
                source_text = doc["page_content"]
            else:
                source_text = doc.page_content
            plain_source = display_source_chunk(source_text)
            st.markdown(plain_source)

        # Display BBL Html as a source
        st.markdown("### Volledige Bouwbesluit:")
        with open("bbl_full_text.html", "r", encoding="utf-8") as html_file:
            html_content = html_file.read()
            st.components.v1.html(html_content, height=2000, width=None, scrolling=True)

# Control flow: Show popup or main app
if not st.session_state.welcome_acknowledged:
    show_welcome_popup()
else:
    main_app()