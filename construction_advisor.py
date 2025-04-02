from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# read .env for the API key
load_dotenv()

#load our sample text
with open("bouwbesluit_sample.txt","r", encoding="utf-8") as f:
    regulation_text = f.read()

#split the text in to chunks (approx 500 characters with overlap so that it is more manageable when being looked up
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_text(regulation_text)

# Display the number of chunks
print(f"Split into {len(chunks)} chunks.")

# create embeddings from chunks
embeddings = OpenAIEmbeddings()

# create a FAISS vector store (in-memory)
vectorstore = FAISS.from_texts(chunks, embeddings)

# create an LLM instance
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create a retriever (searches relevant chunks)
retriever = vectorstore.as_retriever()

# Create a RetrievalQA chain (search + answer)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Ask a sample question
query = "What is the minimum sound insulation requirement for apartments?"
response = qa_chain(query)

print("\n Answer:")
print(response["result"])

print("\n Source used:")
for doc in response["source_documents"]:
    print(doc.page_content[:200])