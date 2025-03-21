from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from typing import List, Literal
import uuid
from datetime import datetime
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path


# Custom configuration loading
def load_config_from_env():
    """Load configuration from .env file with robust error handling"""
    # Try multiple possible .env file locations
    possible_paths = [
        Path.cwd() / ".env",  # Current working directory
        Path(__file__).parent / ".env",  # Same directory as this file
        Path(__file__).parent.parent / ".env",  # Parent directory
    ]

    env_file = None
    for path in possible_paths:
        if path.exists():
            env_file = path
            break

    if env_file:
        print(f"Found .env file at: {env_file}")
        try:
            load_dotenv(dotenv_path=env_file, override=True)
        except Exception as e:
            print(f"Error loading .env with dotenv: {e}")
            # Manual loading as fallback
            try:
                with open(env_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            try:
                                key, value = line.split("=", 1)
                                os.environ[key.strip()] = value.strip()
                            except ValueError:
                                print(f"Skipping invalid line in .env: {line}")
            except Exception as e2:
                print(f"Error with manual .env loading: {e2}")
    else:
        print("No .env file found in any expected location.")
        print(f"Searched in: {[str(p) for p in possible_paths]}")
        print("Will rely on system environment variables.")


# Load environment variables
load_config_from_env()

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

# Create an Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
)

app = FastAPI(
    title="Chat API with Azure GPT-4o",
    description="A RESTful API for chatting with Azure OpenAI GPT-4o model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


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

# System message/context to be included in all conversations
with open(Path.cwd() / "prompts" / "system_prompt.txt", "rb") as f:
    SYSTEM_CONTEXT = f.read().decode("utf-8")

with open(Path.cwd() / "prompts" / "abfallentsorgung.txt", "rb") as f:
    SYSTEM_CONTEXT += f.read().decode("utf-8")

# Definition of a message:
# A message represents a single unit of communication in the chat.
# - 'role' indicates who sent the message ('user', 'system', or 'ai')
# - 'content' contains the actual text of the message
# - 'timestamp' records when the message was sent
# Messages are exchanged between the user and AI to form a conversation.


async def get_gpt4o_response(messages):
    """
    Get a response from Azure GPT-4o for the given messages using the updated OpenAI package.
    """
    # Check if environment variables are properly loaded
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
        error_message = "Azure OpenAI credentials not configured correctly."
        print(f"ERROR: {error_message}")
        print("Make sure your .env file exists and contains the required variables.")
        print("Required variables: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY")
        print("Optional variables: AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION")
        raise HTTPException(status_code=500, detail=error_message)

    # Convert our messages to OpenAI format
    openai_messages = [{"role": "system", "content": SYSTEM_CONTEXT}]

    for msg in messages:
        role = "assistant" if msg.role == "ai" else msg.role
        openai_messages.append({"role": role, "content": msg.content})

    try:
        # Make the OpenAI API call with the updated client
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT, messages=openai_messages, max_tokens=800
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail=f"Azure OpenAI API error: {str(e)}")


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
            role="ai", content=ai_response_text, timestamp=datetime.now()
        )
        processed_messages.append(ai_message)

    # Store the chat
    chat_response = ChatResponse(
        id=chat_id, messages=processed_messages, created_at=datetime.now()
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
    include_context: bool = False


class ChatTestResponse(BaseModel):
    message: str
    response: str
    system_context: str = None


@app.post("/chat-test", response_model=ChatTestResponse)
async def chat_test(request: ChatTestRequest):
    """
    Simple endpoint to test the chat functionality.
    Send a single message and get a direct response from Azure GPT-4o.
    Optional: include_context=true to see the system context in the response.
    """
    # Create a simple message array with just this one message
    test_message = Message(
        role="user", content=request.message, timestamp=datetime.now()
    )

    # Get AI response
    ai_response = await get_gpt4o_response([test_message])

    response = ChatTestResponse(message=request.message, response=ai_response)

    # Optionally include system context in the response
    if request.include_context:
        response.system_context = SYSTEM_CONTEXT

    return response


class UpdateContextRequest(BaseModel):
    context: str


class UpdateContextResponse(BaseModel):
    previous_context: str
    new_context: str
    updated: bool


@app.post("/update-context", response_model=UpdateContextResponse)
async def update_context(request: UpdateContextRequest):
    """
    Update the system context/knowledge used by the chatbot.
    """
    global SYSTEM_CONTEXT
    previous_context = SYSTEM_CONTEXT
    SYSTEM_CONTEXT = request.context

    return UpdateContextResponse(
        previous_context=previous_context, new_context=SYSTEM_CONTEXT, updated=True
    )


@app.get("/context")
async def get_context():
    """
    Get the current system context/knowledge used by the chatbot.
    """
    return {"context": SYSTEM_CONTEXT}


@app.get("/hello")
async def hello_world():
    """
    Simple hello world endpoint to test the API is working.
    """
    return {"message": "Hello, World!"}


@app.get("/swagger", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    Custom Swagger UI endpoint.
    """
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Chat API - Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3005)
