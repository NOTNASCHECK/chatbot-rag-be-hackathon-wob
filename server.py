from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import List, Literal, Dict, Any, Optional
import uuid
from datetime import datetime
import os
from openai import OpenAI  # Changed from AzureOpenAI to OpenAI
from dotenv import load_dotenv
from pathlib import Path
import json


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

# OpenAI configuration - Changed from Azure-specific to standard OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Default model name

# Print configuration for debugging
print("OpenAI Configuration:")
print(f"API Key: {'Set' if OPENAI_API_KEY else 'Not Set'}")
print(f"Model: {OPENAI_MODEL}")

# Create an OpenAI client - Changed from AzureOpenAI to OpenAI
client = OpenAI(
    api_key=OPENAI_API_KEY,
)

# Create FastAPI app with custom configuration
app = FastAPI(
    title="Chat API with OpenAI GPT-4o and Function Calling",  # Removed Azure from title
    description="A RESTful API for chatting with OpenAI GPT-4o model, with function calling capabilities",  # Removed Azure
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


# Define simplified message model
class Message(BaseModel):
    role: Literal["user", "ai", "ai_function", "function_call"]
    content: Any  # Can be a string or JSON object depending on role
    timestamp: Optional[datetime] = None

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

# Define available functions
available_functions = [
    {
        "type": "function",
        "function": {
            "name": "abfallentsorgung",
            "description": "Informationen zur Abfallentsorgung in der Stadt",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aufenthaltserlaubnis",
            "description": "Informationen zu Aufenthaltserlaubnis und Aufenthaltstiteln",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verlust",
            "description": "Informationen bei Verlust von Personalausweis oder Reisepass",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# File path for prompt files
PROMPTS_DIR = Path("prompts")


# Function implementations
async def abfallentsorgung():
    """Provide information about waste disposal"""
    try:
        file_path = PROMPTS_DIR / "abfallentsorgung.json"
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return {"content": content}
    except Exception as e:
        return {"error": f"Error loading waste disposal information: {str(e)}"}


async def aufenthaltserlaubnis():
    """Provide information about residence permits"""
    try:
        file_path = PROMPTS_DIR / "aufenthaltserlaubnis.json"
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return {"content": content}
    except Exception as e:
        return {"error": f"Error loading residence permit information: {str(e)}"}


async def verlust():
    """Provide information about lost ID or passport"""
    try:
        file_path = PROMPTS_DIR / "verlust.json"
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return {"content": content}
    except Exception as e:
        return {"error": f"Error loading lost ID/passport information: {str(e)}"}


# Function dispatcher
function_map = {
    "abfallentsorgung": abfallentsorgung,
    "aufenthaltserlaubnis": aufenthaltserlaubnis,
    "verlust": verlust,
}


async def execute_function(function_name, arguments=None):
    """Execute a function by name with the given arguments"""
    if function_name not in function_map:
        raise ValueError(f"Unknown function: {function_name}")

    function = function_map[function_name]
    # For these specific functions, we don't need any arguments
    if function_name in ["abfallentsorgung", "aufenthaltserlaubnis", "verlust"]:
        return await function()
    else:
        return await function(**(arguments or {}))


# System message/context to be included in all conversations
with open(Path.cwd() / "prompts" / "system_prompt.txt", "rb") as f:
    SYSTEM_CONTEXT = f.read().decode("utf-8")


async def get_gpt4o_response(messages):
    """
    Get a response from OpenAI GPT-4o for the given messages, with function calling.
    """
    # Check if environment variables are properly loaded
    if not OPENAI_API_KEY:
        error_message = "OpenAI API key not configured correctly."
        print(f"ERROR: {error_message}")
        print("Make sure your .env file exists and contains the required variables.")
        print("Required variables: OPENAI_API_KEY")
        print("Optional variables: OPENAI_MODEL")
        raise HTTPException(status_code=500, detail=error_message)

    # Always include system instruction
    system_message = {
        "role": "system",
        "content": SYSTEM_CONTEXT,
    }

    # Convert our messages to OpenAI format
    # Start with the system message and add the conversation messages
    openai_messages = [system_message]
    
    for msg in messages:
        if msg.role == "user":
            openai_messages.append({"role": "user", "content": msg.content})
        elif msg.role == "ai":
            openai_messages.append({"role": "assistant", "content": msg.content})
        elif msg.role == "ai_function":
            # Function call from assistant
            function_call_data = msg.content.copy()
            if "arguments" in function_call_data and isinstance(
                function_call_data["arguments"], dict
            ):
                function_call_data["arguments"] = json.dumps(
                    function_call_data["arguments"]
                )

            openai_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": function_call_data,
                }
            )
        elif msg.role == "function_call":
            # Function response
            openai_messages.append(
                {
                    "role": "function",
                    "name": msg.content.get("name"),
                    "content": json.dumps(msg.content.get("content")),
                }
            )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,  # Changed from AZURE_OPENAI_DEPLOYMENT to OPENAI_MODEL
            messages=openai_messages,
            max_tokens=8000,
            tools=available_functions,
            tool_choice="auto",  # Let the model decide when to call functions
            temperature=0.2,  # Lower temperature for more focused, deterministic responses
            top_p=0.9,  # Slightly reduced top_p to focus on more likely tokens
            frequency_penalty=0.0,  # Default, can be adjusted to reduce repetition
            presence_penalty=0.0,  # Default, can be adjusted to encourage topic diversity
            seed=42,  # Set a consistent seed for reproducibility
            # Add metadata for tracing and evaluation
            user="system-api",  # Identifies system-initiated requests
            stop=None,  # Optional sequences where the API will stop generating further tokens
        )

        ai_message = response.choices[0].message

        # Check if the model wants to call a function
        if ai_message.tool_calls:
            # The model wants to call a function
            function_call = ai_message.tool_calls[0]
            function_name = function_call.function.name
            function_args = json.loads(function_call.function.arguments)

            # Return the function call information as ai_function role
            return {
                "role": "ai_function",
                "content": {
                    "id": function_call.id,
                    "name": function_name,
                    "arguments": function_args,
                    "text": ai_message.content,  # This might be None or contain explanatory text
                }
            }
        else:
            # Regular text response with ai role
            return {"role": "ai", "content": ai_message.content}
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint to post a conversation between a user and AI.
    Supports function calling and multi-turn conversations.
    """
    # Generate a unique ID for this chat
    chat_id = str(uuid.uuid4())

    # Add timestamps to messages if not provided
    processed_messages = []
    for message in request.messages:
        if not message.timestamp:
            message.timestamp = datetime.now()
        processed_messages.append(message)

    # Return an error if there are no messages or if the last message is not from a user
    if not processed_messages or processed_messages[-1].role != "user":
        raise HTTPException(
            status_code=400, detail="The conversation must end with a user message"
        )

    # Initial AI response
    ai_response = await get_gpt4o_response(processed_messages)

    # Create AI message based on response type
    if ai_response["role"] == "ai":
        # Simple text response
        ai_message = Message(
            role="ai",
            content=ai_response["content"],
            timestamp=datetime.now(),
        )
        processed_messages.append(ai_message)
    elif ai_response["role"] == "ai_function":
        # The model wants to call a function
        function_data = ai_response["content"]

        # Add the function call message
        function_call_message = Message(
            role="ai_function",
            content=function_data,
            timestamp=datetime.now(),
        )
        processed_messages.append(function_call_message)

        # Execute the function
        try:
            function_result = await execute_function(
                function_data["name"], function_data["arguments"]
            )

            # Add the function response message
            function_response_message = Message(
                role="function_call",
                content={
                    "name": function_data["name"],
                    "content": function_result,
                },
                timestamp=datetime.now(),
            )
            processed_messages.append(function_response_message)

            # Get a follow-up response from the AI after the function call
            follow_up_response = await get_gpt4o_response(processed_messages)

            # Add the follow-up response
            if follow_up_response["role"] == "ai":
                follow_up_message = Message(
                    role="ai",
                    content=follow_up_response["content"],
                    timestamp=datetime.now(),
                )
                processed_messages.append(follow_up_message)
            # Note: We're not handling nested function calls here for simplicity

        except Exception as e:
            # Handle function execution errors
            error_message = Message(
                role="function_call",
                content={
                    "name": function_data["name"],
                    "content": {"error": str(e)},
                },
                timestamp=datetime.now(),
            )
            processed_messages.append(error_message)

            # Get a follow-up response to handle the error
            error_follow_up = await get_gpt4o_response(processed_messages)
            error_response = Message(
                role="ai",
                content=error_follow_up["content"],
                timestamp=datetime.now(),
            )
            processed_messages.append(error_response)
            
    # Filter messages to include only user and ai messages for frontend
    frontend_messages = [msg for msg in processed_messages if msg.role in ["user", "ai"]]

    # Store the full chat history internally
    chat_history[chat_id] = ChatResponse(
        id=chat_id, messages=processed_messages, created_at=datetime.now()
    )

    # Return only user-AI conversation
    return ChatResponse(
        id=chat_id, messages=frontend_messages, created_at=datetime.now()
    )


@app.get("/chat/{chat_id}", response_model=ChatResponse)
async def get_chat(chat_id: str):
    """
    Retrieve a specific chat by its ID.
    Only returns user and ai messages to the frontend.
    """
    if chat_id not in chat_history:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    chat = chat_history[chat_id]
    
    # Filter to only include user and ai messages
    frontend_messages = [msg for msg in chat.messages if msg.role in ["user", "ai"]]
    
    return ChatResponse(
        id=chat.id, 
        messages=frontend_messages, 
        created_at=chat.created_at
    )


@app.get("/chats", response_model=List[ChatResponse])
async def get_all_chats():
    """
    Retrieve all stored chats.
    Only returns user and ai messages to the frontend.
    """
    frontend_chats = []
    
    for chat in chat_history.values():
        frontend_messages = [msg for msg in chat.messages if msg.role in ["user", "ai"]]
        frontend_chats.append(
            ChatResponse(
                id=chat.id,
                messages=frontend_messages,
                created_at=chat.created_at
            )
        )
    
    return frontend_chats


@app.get("/hello")
async def hello_world():
    """
    Simple hello world endpoint to test the API is working.
    """
    return {"message": "Hello, World!"}


@app.get("/functions")
async def get_available_functions():
    """
    Get a list of available functions that the AI can call.
    """
    return {"functions": available_functions}


@app.post("/chat-with-functions", response_model=ChatResponse)
async def chat_with_functions(request: ChatRequest):
    """
    Explicit endpoint for chatting with function calling capabilities.
    Same as /chat but makes it clear functions are available.
    """
    return await chat(request)


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


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """
    Returns the OpenAPI schema.
    """
    return get_openapi(
        title="Chat API with OpenAI GPT-4o and Function Calling",  # Removed Azure from title
        version="1.0.0",
        description="A RESTful API for chatting with OpenAI GPT-4o model, with function calling capabilities",  # Removed Azure
        routes=app.routes,
    )


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Run the FastAPI server with optional HTTPS support")
    parser.add_argument("--secure", action="store_true", help="Enable HTTPS with SSL certificates")
    args = parser.parse_args()
    
    # Certificate paths from your Let's Encrypt setup
    ssl_keyfile = "/etc/letsencrypt/live/invade.phat-invaders.com/privkey.pem"
    ssl_certfile = "/etc/letsencrypt/live/invade.phat-invaders.com/cert.pem"
    ssl_ca_certs = "/etc/letsencrypt/live/invade.phat-invaders.com/chain.pem"
    
    # Run with or without SSL based on the --secure flag
    if args.secure:
        print("Starting server with HTTPS enabled")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=3005,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            ssl_ca_certs=ssl_ca_certs
        )
    else:
        print("Starting server in HTTP mode (no SSL)")
        uvicorn.run(app, host="0.0.0.0", port=3005)