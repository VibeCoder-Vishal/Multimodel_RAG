import os
import requests
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pymongo import MongoClient
import warnings
import json
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import time
import pyaudio
import wave
import numpy as np
import threading
from faster_whisper import WhisperModel

# Page config
st.set_page_config(
    page_title="E-Commerce Application System",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .ai-response {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin: 20px 0;
        animation: fadeIn 0.5s;
    }
    
    /* Updated image styling for better fit */
    .stImage > img {
        max-width: 100%;
        max-height: 450px; /* Limit height */
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        padding: 10px;
        border-radius: 8px;
        margin: 0 auto;
        display: block;
    }
    
    /* Apply styling to the image container */
    .stImage {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        overflow: hidden; /* Prevent image overflow */
        display: flex;
        justify-content: center;
        align-items: center;
        height: auto; /* Auto height */
    }
    
    /* Caption styling */
    .image-caption {
        text-align: center;
        padding: 8px 5px;
        font-size: 0.9rem;
        color: #666;
        background-color: #f9f9f9;
        border-radius: 0 0 8px 8px;
        margin-top: 5px;
    }
    
    .stImage:hover {
        transform: scale(1.02);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    
    /* Audio recording button styling */
    .recording-button {
        background-color: #DC2626 !important;
    }
    
    /* Adjust the caption container */
    .css-1kyxreq {
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
warnings.filterwarnings("ignore")

# App title
st.markdown("<h1 class='main-header'>E-Commerce Application System</h1>", unsafe_allow_html=True)

# Audio Transcription Class
class LiveAudioTranscriber:
    def __init__(self, model_size="medium", device="auto"):
        # Audio recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.recording = False
        self.frames = []
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Initialize Whisper model
        self.model = WhisperModel(model_size, device=device)
        
        # Temporary file path
        self.temp_file = "temp_recording.wav"
    
    def start_recording(self):
        self.recording = True
        self.frames = []
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record)
        self.record_thread.start()
    
    def _record(self):
        while self.recording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)
    
    def stop_recording(self):
        if not self.recording:
            return "Not currently recording."
        
        self.recording = False
        self.record_thread.join()
        
        # Close and terminate audio stream
        self.stream.stop_stream()
        self.stream.close()
        
        # Save recording to temporary file
        self._save_audio()
        
        # Transcribe the audio
        result = self._transcribe()
        
        # Clean up temporary file
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        
        return result
    
    def _save_audio(self):
        wf = wave.open(self.temp_file, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
    
    def _transcribe(self):
        segments, info = self.model.transcribe(self.temp_file, beam_size=5)
        
        # Collect transcription text
        text = " ".join([segment.text for segment in segments])
        return text
    
    def close(self):
        """Clean up resources"""
        self.audio.terminate()

# Initialization function
@st.cache_resource
def initialize_databases():
    # MongoDB connection
    client = MongoClient('mongodb://localhost:27017/')
    db = client['E_Commerce']
    mongo_collection = db['samples']
    
    # Create directory to save images
    dataset_folder = "./dataset/mongodb_images"
    os.makedirs(dataset_folder, exist_ok=True)
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./data/e_commerce.db")
    image_loader = ImageLoader()
    embedding_function = OpenCLIPEmbeddingFunction()
    chroma_collection = chroma_client.get_or_create_collection(
        "e_commerce_collection",
        embedding_function=embedding_function,
        data_loader=image_loader,
    )
    
    return mongo_collection, chroma_collection, dataset_folder

# Initialize connections
mongo_collection, chroma_collection, dataset_folder = initialize_databases()

# Initialize audio transcriber in session state if not already there
if 'transcriber' not in st.session_state:
    st.session_state.transcriber = LiveAudioTranscriber(model_size="medium")
    st.session_state.recording = False

# Functions for data processing
def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            st.error(f"Failed to download image, status code: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"Error downloading image: {e}")
        return False

def save_images_from_mongodb():
    # Path to track which images we've already downloaded
    download_tracker = os.path.join(dataset_folder, "downloaded_ids.json")
    
    # Load previously downloaded image IDs
    downloaded_ids = set()
    if os.path.exists(download_tracker):
        try:
            with open(download_tracker, 'r') as f:
                downloaded_ids = set(json.load(f))
        except Exception as e:
            st.error(f"Error loading download tracker: {e}")
    
    # Query documents from MongoDB
    documents = mongo_collection.find().limit(200)
    
    count = 0
    new_count = 0
    skipped_count = 0
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Keep track of IDs for this run
    current_ids = set()
    total_docs = mongo_collection.count_documents({})
    
    for i, doc in enumerate(documents):
        progress_bar.progress((i + 1) / min(200, total_docs))
        status_text.text(f"Processing document {i+1}/{min(200, total_docs)}")
        
        doc_id = str(doc.get('_id', ''))
        current_ids.add(doc_id)
        
        # Skip if already downloaded
        if doc_id in downloaded_ids:
            skipped_count += 1
            continue
            
        if 'image' in doc and doc['image']:
            image_url = doc['image']
            # Extract filename from ObjectId or create a sequential one
            filename = f"{doc_id}_{doc.get('title', '')}.jpg"
            # Clean filename of any invalid characters
            filename = ''.join(c for c in filename if c.isalnum() or c in ['_', '-', '.']).strip()
            
            save_path = os.path.join(dataset_folder, filename)
            
            status_text.text(f"Downloading image {i+1}: {image_url}")
            success = download_image(image_url, save_path)
            
            if success:
                count += 1
                new_count += 1
                downloaded_ids.add(doc_id)
    
    # Update our tracker file with all downloaded IDs
    with open(download_tracker, 'w') as f:
        json.dump(list(downloaded_ids), f)
    
    progress_bar.empty()
    status_text.empty()
    
    return new_count, skipped_count, len(downloaded_ids)

def update_chromadb():
    # Track which files are already in ChromaDB
    chromadb_tracker = os.path.join(dataset_folder, "chromadb_ids.json")
    chromadb_ids = set()
    
    # Load previously added ChromaDB IDs
    if os.path.exists(chromadb_tracker):
        try:
            with open(chromadb_tracker, 'r') as f:
                chromadb_ids = set(json.load(f))
        except Exception as e:
            st.error(f"Error loading ChromaDB tracker: {e}")
    
    # Get current count in collection
    existing_count = chroma_collection.count()
    
    # Prepare new images to add
    new_ids = []
    new_uris = []
    added_ids = []
    
    for filename in sorted(os.listdir(dataset_folder)):
        if filename.endswith((".jpg", ".png")) and not filename.startswith("."):
            # Extract the MongoDB ID from the filename
            mongo_id = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
            
            # Skip if already in ChromaDB
            if mongo_id in chromadb_ids:
                continue
                
            file_path = os.path.join(dataset_folder, filename)
            new_ids.append(mongo_id)
            new_uris.append(file_path)
            added_ids.append(mongo_id)
    
    # Only add if there are new images
    if new_ids:
        with st.spinner(f"Adding {len(new_ids)} new images to ChromaDB"):
            chroma_collection.add(ids=new_ids, uris=new_uris)
        
        # Update our tracker with newly added IDs
        chromadb_ids.update(added_ids)
        with open(chromadb_tracker, 'w') as f:
            json.dump(list(chromadb_ids), f)
        
        return len(new_ids), chroma_collection.count(), len(chromadb_ids)
    else:
        return 0, existing_count, len(chromadb_ids)

def query_db(query, results=2):
    with st.spinner("Searching for relevant images..."):
        results = chroma_collection.query(
            query_texts=[query], n_results=results, include=["uris", "distances"]
        )
    return results

def format_prompt_inputs(data, user_query):
    with st.spinner("Processing images for AI analysis..."):
        inputs = {}
        
        inputs["user_query"] = user_query
        
        image_path_1 = data["uris"][0][0]
        image_path_2 = data["uris"][0][1]
        
        with open(image_path_1, "rb") as image_file:
            image_data_1 = image_file.read()
        inputs["image_data_1"] = base64.b64encode(image_data_1).decode("utf-8")
        
        with open(image_path_2, "rb") as image_file:
            image_data_2 = image_file.read()
        inputs["image_data_2"] = base64.b64encode(image_data_2).decode("utf-8")
        
    return inputs, image_path_1, image_path_2

def get_ai_response(user_query, prompt_inputs):
    # Initialize the vision model
    vision_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    parser = StrOutputParser()
    
    # Create messages directly
    messages = [
        {
            "role": "system",
            "content": "You are a talented Philatelist you have been assigned to create a collection of stamps for a specific buyers. Answer the user's question using the given image context with direct references to the parts of images provided. Maintain a more conversational tone, don't make too many lists. Use markdown formatting for highlights, emphasis, and structure."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": f"What are some good ideas for choosing the stamps {prompt_inputs['user_query']}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{prompt_inputs['image_data_1']}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{prompt_inputs['image_data_2']}"
                    }
                }
            ]
        }
    ]
    
    # AI processing with animated spinner
    with st.spinner("Generating AI response..."):
        response = vision_model.invoke(messages)
        parsed_response = parser.invoke(response)
    
    return parsed_response

# Function to resize and normalize images for better display
def preprocess_image_for_display(image_path):
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Calculate aspect ratio
        aspect_ratio = img.width / img.height
        
        # Determine new dimensions while preserving aspect ratio
        max_width = 600  # Maximum width for display
        max_height = 400  # Maximum height for display
        
        if aspect_ratio > 1:  # Wider than tall
            new_width = min(img.width, max_width)
            new_height = int(new_width / aspect_ratio)
        else:  # Taller than wide
            new_height = min(img.height, max_height)
            new_width = int(new_height * aspect_ratio)
        
        # Resize the image if needed
        if img.width > max_width or img.height > max_height:
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        return img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return Image.open(image_path)  # Return original if processing fails

# Function to start recording
def start_recording():
    st.session_state.recording = True
    st.session_state.transcriber.start_recording()
    st.rerun()

# Function to stop recording
def stop_recording():
    st.session_state.recording = False
    transcribed_text = st.session_state.transcriber.stop_recording()
    # Set the transcribed text as the query
    st.session_state.query_input = transcribed_text
    st.rerun()

# App workflow
with st.sidebar:
    st.markdown("<h2 class='sub-header'>System Status</h2>", unsafe_allow_html=True)
    
    # Data refresh button
    if st.button("Refresh Data"):
        with st.spinner("Processing images from MongoDB..."):
            new_count, skipped_count, total_count = save_images_from_mongodb()
            st.success(f"Downloaded {new_count} new images")
            st.info(f"Skipped {skipped_count} existing images")
            st.info(f"Total images in dataset: {total_count}")
        
        with st.spinner("Updating ChromaDB..."):
            new_added, total_chroma, total_tracked = update_chromadb()
            if new_added > 0:
                st.success(f"Added {new_added} new images to ChromaDB")
            else:
                st.info("No new images to add to ChromaDB")
            st.info(f"Total images in ChromaDB: {total_chroma}")
    
    # Show system information
    st.markdown("### System Information")
    st.markdown("- Application: Streamlit + Python")
    st.markdown("- Database: MongoDB + ChromaDB")
    st.markdown("- GEN AI Model: Gemini")
    st.markdown("- Image Embedding: OpenCLIP")
    st.markdown("- Audio: OpenAI Whisper")

# Add text input with previous transcription if available
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

query = st.text_input("What kind of stamps are you interested in?", value=st.session_state.query_input, key="query_input")

# Create a layout for the input options
col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

with col2:
    # Create a button to start/stop recording
    if not st.session_state.recording:
        st.button("🎤 Voice Search", on_click=start_recording, use_container_width=True)
    else:
        st.button("⏹️ Stop Recording", on_click=stop_recording, use_container_width=True, 
                 help="Stop recording and transcribe")

with col3:
    search_button = st.button("🔍 Search", use_container_width=True)

# Display recording status
if st.session_state.recording:
    st.info("🎙️ Recording... Speak clearly and then click 'Stop Recording' when finished.")

# Process user query
if search_button and query:
    # Query processing
    results = query_db(query)
    
    if len(results["uris"][0]) >= 2:
        prompt_inputs, image_path_1, image_path_2 = format_prompt_inputs(results, query)
        
        # Generate and display AI response with typing animation effect
        response = get_ai_response(query, prompt_inputs)
        
        # Create a placeholder for the typing animation
        response_container = st.empty()
        
        # Simulate typing animation
        displayed_text = ""
        for i in range(len(response) + 1):
            displayed_text = response[:i]
            response_container.markdown(f"<div class='ai-response'>{displayed_text}</div>", unsafe_allow_html=True)
            time.sleep(0.01)  # Adjust speed of typing animation
        
        # Display related images
        st.markdown("<h2 class='sub-header'>Relevant Stamps</h2>", unsafe_allow_html=True)
        
        # Create two columns for the images with a bit of spacing
        col1, col2 = st.columns(2)
        
        with col1:
            # Container for better styling
            with st.container():
                # Process image for display
                img1 = preprocess_image_for_display(image_path_1)
                # Display the image with caption
                st.image(img1, caption="Stamp Sample 1", use_container_width=True)
            
        with col2:
            # Container for better styling
            with st.container():
                # Process image for display
                img2 = preprocess_image_for_display(image_path_2)
                # Display the image with caption
                st.image(img2, caption="Stamp Sample 2", use_container_width=True)
    else:
        st.error("Not enough relevant images found. Please try a different query.")
elif search_button:
    st.warning("Please enter a query or use voice search to find stamps.")

# Handle app close
def on_close():
    if 'transcriber' in st.session_state:
        st.session_state.transcriber.close()

# Register the on_close function to be called when the app is closed
try:
    # This is a workaround as Streamlit doesn't have a built-in on_close event
    st.cache_resource.clear()
except:
    pass

# Footer
st.markdown("---")
st.markdown("© 2025 E-Commerce Application System - Powered by AI")