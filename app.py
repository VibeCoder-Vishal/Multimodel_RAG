import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
from pymongo import MongoClient
import warnings
import json
from dotenv import load_dotenv

# Load environment variables from the .env file (if present)
load_dotenv()

# Access environment variables as if they came from the actual environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

warnings.filterwarnings("ignore")
# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Update with your connection string if needed
db = client['E_Commerce']
collection = db['samples']
# Create directory to save images
dataset_folder = "./dataset/mongodb_images"
os.makedirs(dataset_folder, exist_ok=True)
# Path to track which images we've already downloaded
download_tracker = os.path.join(dataset_folder, "downloaded_ids.json")
# Load previously downloaded image IDs
downloaded_ids = set()
if os.path.exists(download_tracker):
    try:
        with open(download_tracker, 'r') as f:
            downloaded_ids = set(json.load(f))
        print(f"Found {len(downloaded_ids)} previously downloaded images")
    except Exception as e:
        print(f"Error loading download tracker: {e}")
def show_image_from_uri(uri):
    img = Image.open(uri)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download image, status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False
def save_images_from_mongodb(collection, dataset_folder, limit=200):
    # Query documents from MongoDB
    documents = collection.find().limit(limit)
    
    count = 0
    new_count = 0
    skipped_count = 0
    
    # Keep track of IDs for this run
    current_ids = set()
    
    for i, doc in enumerate(documents):
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
            
            print(f"Downloading image {i+1}: {image_url}")
            success = download_image(image_url, save_path)
            
            if success:
                count += 1
                new_count += 1
                downloaded_ids.add(doc_id)
    
    # Update our tracker file with all downloaded IDs
    with open(download_tracker, 'w') as f:
        json.dump(list(downloaded_ids), f)
    
    print(f"Downloaded {new_count} new images to {dataset_folder}")
    print(f"Skipped {skipped_count} already downloaded images")
    print(f"Total images in dataset: {len(downloaded_ids)}")
# Download images from MongoDB
save_images_from_mongodb(collection, dataset_folder)

#vector db
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
#cromadb object
chroma_client = chromadb.PersistentClient(path="./data/e_commerce.db")
#instantiate image loader
image_loader = ImageLoader()
#MM embedding function
embedding_function = OpenCLIPEmbeddingFunction()
#collection
collection = chroma_client.get_or_create_collection(
    "e_commerce_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

# Track which files are already in ChromaDB
chromadb_tracker = os.path.join(dataset_folder, "chromadb_ids.json")
chromadb_ids = set()

# Load previously added ChromaDB IDs
if os.path.exists(chromadb_tracker):
    try:
        with open(chromadb_tracker, 'r') as f:
            chromadb_ids = set(json.load(f))
        print(f"Found {len(chromadb_ids)} images already in ChromaDB")
    except Exception as e:
        print(f"Error loading ChromaDB tracker: {e}")

# Get current count in collection
existing_count = collection.count()
print(f"Current ChromaDB collection has {existing_count} items")

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
    print(f"Adding {len(new_ids)} new images to ChromaDB")
    collection.add(ids=new_ids, uris=new_uris)
    
    # Update our tracker with newly added IDs
    chromadb_ids.update(added_ids)
    with open(chromadb_tracker, 'w') as f:
        json.dump(list(chromadb_ids), f)
else:
    print("No new images to add to ChromaDB")

# print(f"ChromaDB collection now has {collection.count()} items")
# print(f"Total images tracked in ChromaDB: {len(chromadb_ids)}")

#query vectorDB
def query_db(query, results=5):
    print(f"Querying the database: {query}")
    results = collection.query(
        query_texts=[query], n_results=results, include=["uris", "distances"]
    )
    return results

def print_results(results):
    for idx, uri in enumerate(results["uris"][0]):
        print(f"ID: {results['ids'][0][idx]}")
        print(f"Distances: {results['distances'][0][idx]}")
        print(f"Path: {uri}")
        show_image_from_uri(uri)
        print("\n")

# query = "i like to buy sports related stamps"
# results = query_db(query)
# print_results(results)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
import base64
import os

# Make sure you have the Google API key in your environment variables
vision_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")  # Make sure this is set in your .env file
)

parser = StrOutputParser()

def format_prompt_inputs(data, user_query):
    print("Formatting prompt inputs...")
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
    
    print("Prompt inputs formatted.....")
    return inputs

query = input("Enter your query:\n")
# Retrieval & Generation
results = query_db(query, results=2)
prompt_inputs = format_prompt_inputs(results, query)

# Create messages directly instead of using a template
messages = [
    {
        "role": "system",
        "content": "You are a talented Philatelist you have been assigned to create a collection of stamps for a specific buyers.Answer the user's question using the given image context with direct references to the parts of images provided.Maintain a more conversational tone, don't make too many list.Use markdown formatting for highlights, emphasis, and structure."
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

# Invoke the model with messages directly
response = vision_model.invoke(messages)
parsed_response = parser.invoke(response)

print("\n --------\n")
print(parsed_response)
show_image_from_uri(results["uris"][0][0])
show_image_from_uri(results["uris"][0][1])