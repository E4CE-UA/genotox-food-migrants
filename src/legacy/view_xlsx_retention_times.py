import chainlit as cl
import chromadb
import re
import os

# ChromaDB location: set CAS_CHROMA_PATH, or run from a directory holding ./chroma_db
chroma_path = os.environ.get("CAS_CHROMA_PATH", "chroma_db")
client = chromadb.PersistentClient(path=chroma_path)

# Get collection
collection = client.get_collection("nist_retention_data")

def clean_compound_name(name):
    """Thoroughly clean up the compound name"""
    if not name:
        return 'Unknown Compound'
    
    # Remove common encoding artifacts
    name = name.replace('*x0028*', '(')
    name = name.replace('_x0029_', ')')
    name = name.replace('_', ' ')
    
    # Remove any remaining HTML or XML-like encodings
    name = re.sub(r'[*_]x[0-9a-fA-F]+', '', name)
    
    # Remove extra spaces
    name = ' '.join(name.split())
    
    return name.strip()

@cl.on_chat_start
async def start():
    """Grab and display the first retention time"""
    try:
        # Get the first document
        results = collection.get(limit=1, include=["metadatas", "documents"])
        
        if not results['documents']:
            await cl.Message(content="No retention times found.").send()
            return
        
        # Extract metadata and document
        metadata = results['metadatas'][0]
        document = results['documents'][0]
        
        # Clean up compound name
        compound_name = clean_compound_name(metadata.get('substance_name', 'Unknown Compound'))
        cas_number = metadata.get('cas', 'N/A')
        
        # Prepare HTML display
        html_content = f"""
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ font-family: 'Arial', sans-serif; line-height: 1.6; }}
            .compound-info {{ 
                background-color: #f4f4f4; 
                border-left: 5px solid #007bff; 
                padding: 15px; 
                margin-bottom: 20px; 
            }}
            table {{ 
                width: 100%; 
                margin-bottom: 20px; 
            }}
            th, td {{ 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: left; 
            }}
            th {{ 
                background-color: #007bff; 
                color: white; 
            }}
            .retention-times table {{ 
                box-shadow: 0 2px 3px rgba(0,0,0,0.1); 
            }}
        </style>
        <div class="container-fluid p-4">
            <div class="compound-info">
                <h2 class="mb-3">Compound Details</h2>
                <p class="mb-2"><strong>Name:</strong> {compound_name}</p>
                <p class="mb-0"><strong>CAS Number:</strong> {cas_number}</p>
            </div>
            
            <div class="retention-times">
                <h3 class="mb-3">Retention Times</h3>
                {document}
            </div>
        </div>
        """
        
        await cl.Message(content=html_content).send()
        
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()

@cl.on_message
async def handle_message(message: cl.Message):
    """Simple pass-through"""
    await cl.Message(content="Retention time already displayed. Type 'list' to see more options.").send()

if __name__ == "__main__":
    pass  # Chainlit handles running the application