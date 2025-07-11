 # ZJAC - BBL AI Assistant
 
-This Streamlit application helps users search and understand the Dutch building regulations ("Bouwbesluit"). The app loads the full text of the regulations, embeds it with OpenAI embeddings and FAISS, and allows you to ask questions in natural language. Feedback on the answers can be saved to Supabase for further improvement.
+This repository contains a simple Streamlit application that helps users search and understand the Dutch building regulations ("Bouwbesluit"). It loads the full text of the regulations, creates embeddings with OpenAI and FAISS, and lets you query them in natural language. Feedback on each answer can be stored in Supabase for later analysis and model improvement.

 ## Features
-- Loads the full Bouwbesluit text from `bbl_full_text.txt` and presents the HTML version from `bbl_full_text.html`.
-- Uses LangChain with OpenAI models to create embeddings and answer queries.
+- Embeds the full Bouwbesluit text from `bbl_full_text.txt` and shows the accompanying HTML document.
+- Answers natural language questions using LangChain and OpenAI models.
+- Displays the source text used to generate each answer.
 - Stores feedback (question, answer, rating and optional comment) in a Supabase table.
 - Adjustable response creativity via the temperature slider.
-- Optionally run in debug mode to return a static example answer.
+- Optionally run in debug mode to return a static example answer without calling OpenAI.

 ## Files
 - `app.py` – main Streamlit application.
 - `bbl_full_text.txt` – plain text of the Dutch building regulations used to build the vector store.
 - `bbl_full_text.html` – HTML version of the regulations displayed within the app.
 - `bouwbesluit_sample.txt` – short example excerpt of the regulations.
 - `requirements.txt` – Python dependencies.

## Future optimisations
+- **Dedicated pre-prompt** – prime the language model with a statement such as "You are a BBL advisor and nothing else" so responses always follow the style and scope of the building regulations.
+- **Conversational mode** – move from single question/answer pairs to a conversation where both the AI and the user can ask follow-up questions.
+- **Fine-tuning** – once enough feedback has been collected, train a custom model to further improve accuracy and relevance.
