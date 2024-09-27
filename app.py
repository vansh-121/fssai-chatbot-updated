import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
import PyPDF2
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import tiktoken
import faiss
import numpy as np
from gtts import gTTS  # Google Text-to-Speech
import tempfile
import speech_recognition as sr  # Speech Recognition
import io
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configure the Gemini Pro model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Setup Google Drive API credentials
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'D:/fssai testing/dotted-chariot-436805-d7-4634e58b6c3c.json'

# Function to load Gemini Pro model and get responses
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def download_pdfs_from_gdrive_folder(folder_id):
    """Download all PDFs from the specified Google Drive folder."""
    credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)
    
    query = f"'{folder_id}' in parents and mimeType='application/pdf'"
    response = drive_service.files().list(q=query).execute()

    file_ids = [file['id'] for file in response['files']]
    
    pdf_texts = []
    for file_id in file_ids:
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()

        # Extract text from downloaded PDF
        fh.seek(0)
        reader = PyPDF2.PdfReader(fh)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        pdf_texts.append(text)
    
    return pdf_texts

def split_text_into_chunks(text, max_token_length=1000):
    """Split text into smaller chunks to ensure context accuracy."""
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_token_length):
        chunk_tokens = tokens[i:i + max_token_length]
        chunks.append(encoder.decode(chunk_tokens))
    
    return chunks

# Hypothetical method to generate embeddings
# Hypothetical method to generate embeddings using the Gemini model


# Load a pre-trained model for embedding generation
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings_for_chunks(chunks):
    """Generate embeddings using Hugging Face model."""
    embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    return embeddings



def build_semantic_index(embeddings):
    """Build a semantic index using FAISS."""
    embedding_dim = len(embeddings[0])  # Get the dimensionality of embeddings
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index

    # Convert embeddings list to numpy array
    embeddings_array = np.array(embeddings).astype(np.float32)
    index.add(embeddings_array)
    
    return index

# Example: Save index to disk
def save_index(index, path='semantic_index.faiss'):
    """Save the FAISS index to a file."""
    faiss.write_index(index, path)


def search_with_query(query, index, all_chunks):
    """Perform semantic search using query embeddings."""
    # Use SentenceTransformer to generate the query embedding
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    query_embedding_np = np.array(query_embedding).astype(np.float32)  # Ensure it's a numpy array

    # FAISS expects a 2D array, so we reshape the query_embedding if necessary
    if len(query_embedding_np.shape) == 1:
        query_embedding_np = np.expand_dims(query_embedding_np, axis=0)

    # Search the FAISS index for the most relevant chunk
    distances, indices = index.search(query_embedding_np, k=5)  # Get top 5 results
    relevant_chunks = [all_chunks[i] for i in indices[0]]
    
    return relevant_chunks



def get_gemini_response_based_on_chunks(question, relevant_chunks):
    """Get a response from Gemini API based on the retrieved chunks."""
    context = "\n\n".join(relevant_chunks)
    response = chat.send_message(context + "\n" + question, stream=True)
    return ''.join([chunk.text for chunk in response])

def text_to_speech(text):
    """Converts text to speech and returns the path to the audio file."""
    # Show "Generating Audio..." message
    st.info("Generating Audio...")
    
    # Convert text to speech
    tts = gTTS(text=text, lang='en')
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
    tts.save(temp_file_path)
    
    # Update the status to "Audio generated!"
    st.success("Audio generated!")
    
    return temp_file_path

def recognize_speech():
    """Captures voice input from the user and returns it as text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak into the microphone.")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError:
            st.error("Error with the Speech Recognition service.")

# Initialize Streamlit app
st.set_page_config(page_title="AI Chatbot")
st.title("FSSAI : AI Chatbot")

# Initialize session state for FAISS index and chunks if they don't exist
if 'index' not in st.session_state:
    st.session_state['index'] = None

if 'all_chunks' not in st.session_state:
    st.session_state['all_chunks'] = []

# Google Drive folder upload and processing
st.sidebar.header("Upload Database link here")
folder_id = st.sidebar.text_input("Google Drive Folder ID")
if st.sidebar.button("Load Database"):
    if folder_id:
        pdf_texts = download_pdfs_from_gdrive_folder(folder_id)
        st.session_state['pdf_context'] = ' '.join(pdf_texts)
        st.sidebar.success("Database loaded successfully!")

        # Process PDFs into chunks and embeddings
        all_chunks = []
        for pdf_text in pdf_texts:
            all_chunks.extend(split_text_into_chunks(pdf_text))

        st.session_state['all_chunks'] = all_chunks  # Save chunks to session state
        
        embeddings = generate_embeddings_for_chunks(all_chunks)
        index = build_semantic_index(embeddings)
        save_index(index)

        st.session_state['index'] = index  # Save index to session state
        st.sidebar.success("Text processing completed!")
    else:
        st.sidebar.error("Please enter a valid folder ID.")

# Voice input button
st.header("Voice Input")
if st.button("Speak Your Question"):
    input_text = recognize_speech()
    if input_text and st.session_state['pdf_context']:
        if st.session_state['index'] is not None:
            relevant_chunks = search_with_query(input_text, st.session_state['index'], st.session_state['all_chunks'])
            response = get_gemini_response_based_on_chunks(input_text, relevant_chunks)

            # Display "Response" and the text response
            st.subheader("Response")
            st.write(response)

            # Generate and display the audio after showing the response text
            audio_path = text_to_speech(response)
            st.session_state['latest_audio_path'] = audio_path  # Update latest audio path

            # Display the audio player
            audio_bytes = open(st.session_state['latest_audio_path'], "rb").read()
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.write("Please load PDFs first.")
    else:
        st.write("Please load PDFs first or check your input.")

# Text input for fallback or preference
st.header("Text Input")
input_text = st.text_input("Type Your Question:")
submit = st.button("Ask the question")

if submit and input_text:
    if st.session_state['pdf_context']:
        if st.session_state['index'] is not None:
            relevant_chunks = search_with_query(input_text, st.session_state['index'], st.session_state['all_chunks'])
            response = get_gemini_response_based_on_chunks(input_text, relevant_chunks)

            # Display "Response" and the text response
            st.subheader("Response")
            st.write(response)

            # Generate and display the audio after showing the response text
            audio_path = text_to_speech(response)
            st.session_state['latest_audio_path'] = audio_path  # Update latest audio path

            # Display the audio player
            audio_bytes = open(st.session_state['latest_audio_path'], "rb").read()
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.write("Please load Database first.")
    else:
        st.write("Please load Database first.")
