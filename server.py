from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import uuid
from datetime import datetime
import os
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Print configuration for debugging
print("Azure OpenAI Configuration:")
print(f"Endpoint: {AZURE_OPENAI_ENDPOINT}")
print(f"API Key: {'Set' if AZURE_OPENAI_KEY else 'Not Set'}")
print(f"Deployment: {AZURE_OPENAI_DEPLOYMENT}")
print(f"API Version: {AZURE_OPENAI_API_VERSION}")

app = FastAPI(title="Chat API with Azure GPT-4o")

# Define data models
class Message(BaseModel):
    role: Literal["user", "ai", "system"]
    content: str
    timestamp: datetime = None
    
    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    messages: List[Message]

class ChatResponse(BaseModel):
    id: str
    messages: List[Message]
    created_at: datetime

# In-memory storage for chats
chat_history = {}

# Definition of a message:
# A message represents a single unit of communication in the chat.
# - 'role' indicates who sent the message ('user', 'system', or 'ai')
# - 'content' contains the actual text of the message
# - 'timestamp' records when the message was sent
# Messages are exchanged between the user and AI to form a conversation.

async def get_gpt4o_response(messages):
    """
    Get a response from Azure GPT-4o for the given messages.
    """
    # Check if environment variables are properly loaded
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
        error_message = "Azure OpenAI credentials not configured correctly."
        print(f"ERROR: {error_message}")
        print("Make sure your .env file exists and contains the required variables.")
        print("Required variables: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY")
        print("Optional variables: AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION")
        raise HTTPException(
            status_code=500, 
            detail=error_message
        )
    
    # Convert our messages to OpenAI format
    openai_messages = []
    for msg in messages:
        role = "assistant" if msg.role == "ai" else msg.role
        openai_messages.append({"role": role, "content": msg.content})
    
    # Prepare the request
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_KEY
    }
    payload = {
        "messages": openai_messages,
        "max_tokens": 800
    }
    
    # Make the API call
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Azure OpenAI API error: {response.text}"
            )
        
        result = response.json()
        return result["choices"][0]["message"]["content"]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint to post a conversation between a user and AI.
    Expects a list of messages with roles ('user' or 'ai').
    Automatically generates AI responses using Azure GPT-4o.
    """
    # Generate a unique ID for this chat
    chat_id = str(uuid.uuid4())
    
    # Add timestamps to messages if not provided
    processed_messages = []
    for message in request.messages:
        if not message.timestamp:
            message.timestamp = datetime.now()
        processed_messages.append(message)
    
    # Get AI response for the latest message if it's from a user
    if processed_messages and processed_messages[-1].role == "user":
        ai_response_text = await get_gpt4o_response(processed_messages)
        
        # Add AI response to messages
        ai_message = Message(
            role="ai",
            content=ai_response_text,
            timestamp=datetime.now()
        )
        processed_messages.append(ai_message)
    
    # Store the chat
    chat_response = ChatResponse(
        id=chat_id,
        messages=processed_messages,
        created_at=datetime.now()
    )
    
    chat_history[chat_id] = chat_response
    
    return chat_response

@app.get("/chat/{chat_id}", response_model=ChatResponse)
async def get_chat(chat_id: str):
    """
    Retrieve a specific chat by its ID.
    """
    if chat_id not in chat_history:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    return chat_history[chat_id]

@app.get("/chats", response_model=List[ChatResponse])
async def get_all_chats():
    """
    Retrieve all stored chats.
    """
    return list(chat_history.values())

class ChatTestRequest(BaseModel):
    message: str

class ChatTestResponse(BaseModel):
    message: str
    response: str

@app.post("/chat-test", response_model=ChatTestResponse)
async def chat_test(request: ChatTestRequest):
    """
    Simple endpoint to test the chat functionality.
    Send a single message and get a direct response from Azure GPT-4o.
    """
    # Create a simple message array with just this one message
    test_message = Message(
        role="user",
        content=request.message,
        timestamp=datetime.now()
    )
    
    # Get AI response
    ai_response = await get_gpt4o_response([test_message])
    
    return ChatTestResponse(
        message=request.message,
        response=ai_response
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)