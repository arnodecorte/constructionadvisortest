import streamlit as st
from dotenv import load_dotenv
import re
import time

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

# JavaScript for navigation
js_code = """
<script>
function scrollToElement(elementId) {
    // Get the iframe element
    const iframe = document.getElementById('bbl-iframe');
    
    // Make sure the iframe exists and is loaded
    if (iframe && iframe.contentDocument) {
        // Find the element inside the iframe
        const targetElement = iframe.contentDocument.getElementById(elementId);
        
        // If the element exists, scroll to it
        if (targetElement) {
            // Scroll the element into view
            targetElement.scrollIntoView({behavior: 'smooth', block: 'start'});
            
            // Highlight the element temporarily
            const originalBg = targetElement.style.backgroundColor;
            const originalOutline = targetElement.style.outline;
            
            targetElement.style.backgroundColor = 'rgba(255, 255, 0, 0.3)';
            targetElement.style.outline = '2px solid orange';
            
            setTimeout(() => {
                targetElement.style.backgroundColor = originalBg;
                targetElement.style.outline = originalOutline;
            }, 3000); // Remove highlight after 3 seconds
        } else {
            console.error('Element with ID ' + elementId + ' not found in the iframe');
            alert('Kon het element "' + elementId + '" niet vinden in het document.');
        }
    } else {
        console.error('Iframe not loaded yet or not found');
        alert('Het document is nog niet volledig geladen. Probeer het opnieuw.');
    }
    
    // Prevent the default link behavior
    return false;
}
</script>
"""

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

# Function to extract article or section numbers and create hyperlinks
def create_hyperlinked_source(source_text):
    """
    Creates hyperlinks from source text that point to specific sections in the BBL document.
    
    Args:
        source_text (str): The source text containing article or section references
        
    Returns:
        str: HTML with hyperlinks to the referenced sections
    """
    # Match article numbers (e.g., Artikel 2.24)
    article_match = re.search(r'Artikel\s+(\d+\.\d+)', source_text)
    if article_match:
        article_num = article_match.group(1)
        article_id = f"artikel-{article_num}"
        # Create a hyperlink that targets the iframe and points to the article ID
        linked_text = source_text.replace(
            f"Artikel {article_num}", 
            f'<a href="#" onclick="return scrollToElement(\'{article_id}\')">Artikel {article_num}</a>'
        )
        return linked_text
    
    # Match section numbers (e.g., § 2.3.2)
    section_match = re.search(r'§\s+(\d+\.\d+\.\d+)', source_text)
    if section_match:
        section_num = section_match.group(1)
        section_id = f"section-{section_num}"
        # Create a hyperlink that targets the iframe and points to the section ID
        linked_text = source_text.replace(
            f"§ {section_num}", 
            f'<a href="#" onclick="return scrollToElement(\'{section_id}\')">§ {section_num}</a>'
        )
        return linked_text
    
    # Additional patterns to match
    # For example, match references like "Hoofdstuk 2"
    chapter_match = re.search(r'Hoofdstuk\s+(\d+)', source_text)
    if chapter_match:
        chapter_num = chapter_match.group(1)
        chapter_id = f"hoofdstuk-{chapter_num}"
        linked_text = source_text.replace(
            f"Hoofdstuk {chapter_num}", 
            f'<a href="#" onclick="return scrollToElement(\'{chapter_id}\')">Hoofdstuk {chapter_num}</a>'
        )
        return linked_text
    
    # If no match, return the plain source text
    return source_text

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

# Add an info message about the clickable sources feature
st.info("💡 Tip: Klik op artikelnummers in de bronvermeldingen om direct naar dat gedeelte in het Bouwbesluit te navigeren.")

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
@st.cache_resource
def load_vector_store():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

vectorstore = load_vector_store()
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
    # Show a spinner while processing
    with st.spinner('Bezig met zoeken in de regelgeving...'):
        # Provide dummy response for debugging purposes
        if debug_mode:
            result = {
                "result": "Dit is een voorbeeldantwoord voor debugdoeleinden.",
                "source_documents": [
                    {"page_content": "Dit is een voorbeeldbron met Artikel 2.24 en § 2.3.2 als test."}
                ]
            }
            # Simulate processing time
            time.sleep(1)
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
                "\n---\n".join([doc.page_content for doc in result["source_documents"]] if not debug_mode else 
                              [doc["page_content"] for doc in result["source_documents"]]),
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
                    "\n---\n".join([doc.page_content for doc in result["source_documents"]] if not debug_mode else 
                                  [doc["page_content"] for doc in result["source_documents"]]),
                    False,
                    feedback_comment
                )
                st.success("Feedback ingediend voor trainingsdata, bedankt!")
                st.session_state.no_feedback = False  # Reset the feedback state after submission
            else:
                st.warning("Vul alstublieft een feedbackcommentaar in voordat u verzendt.")

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
    
    # Add the JavaScript after displaying sources
    st.components.v1.html(js_code, height=0)
    
    # Display BBL Html as a source
    st.markdown("### Volledige Bouwbesluit:")
    with open("bbl_full_text.html", "r", encoding="utf-8") as html_file:
        html_content = html_file.read()
        
        # Create the iframe with ID for targeting by JavaScript
        iframe_html = f"""
        <div style="width:100%; height:2000px;">
            <iframe id="bbl-iframe" srcdoc='{html_content}' width="100%" height="100%" 
            style="border:none; background-color:white; overflow:auto;"></iframe>
        </div>
        """
        
        st.components.v1.html(iframe_html, height=2000, scrolling=True)