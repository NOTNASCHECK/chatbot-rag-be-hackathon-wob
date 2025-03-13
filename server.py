from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import List, Literal, Dict, Any, Optional, Union
import uuid
from datetime import datetime
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from pathlib import Path
import json
import random


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

# Create FastAPI app with custom configuration
app = FastAPI(
    title="Chat API with Azure GPT-4o and Function Calling",
    description="A RESTful API for chatting with Azure OpenAI GPT-4o model, with function calling capabilities",
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
class MessageContent(BaseModel):
    type: Literal["text", "function_call", "function_response"] = "text"
    text: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    function_response: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    role: Literal["user", "ai", "system", "function"]
    content: Union[str, MessageContent]
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
    Get a response from Azure GPT-4o for the given messages, with function calling.
    """
    # Check if environment variables are properly loaded
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
        error_message = "Azure OpenAI credentials not configured correctly."
        print(f"ERROR: {error_message}")
        print("Make sure your .env file exists and contains the required variables.")
        print("Required variables: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY")
        print("Optional variables: AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION")
        raise HTTPException(status_code=500, detail=error_message)

    # Always include system instruction to respond in HTML
    system_message = {
        "role": "system",
        "content": SYSTEM_CONTEXT,
    }

    # Convert our messages to OpenAI format
    # Start with the system message and add the conversation messages
    openai_messages = [system_message]
    for msg in messages:
        if isinstance(msg.content, str):
            role = "assistant" if msg.role == "ai" else msg.role
            openai_messages.append({"role": role, "content": msg.content})
        else:
            # Handle structured message content
            if msg.role == "ai" and msg.content.type == "function_call":
                # Function call from assistant
                # The API expects arguments as a string, not an object
                function_call_data = msg.content.function_call.copy()
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
            elif msg.role == "function" and msg.content.type == "function_response":
                # Function response
                openai_messages.append(
                    {
                        "role": "function",
                        "name": msg.content.function_response.get("name"),
                        "content": json.dumps(
                            msg.content.function_response.get("content")
                        ),
                    }
                )
            else:
                # Regular message with text
                role = "assistant" if msg.role == "ai" else msg.role
                openai_messages.append(
                    {"role": role, "content": msg.content.text or ""}
                )

    try:
        # Make the OpenAI API call with function calling
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=openai_messages,
            max_tokens=800,
            tools=available_functions,
            tool_choice="auto",  # Let the model decide when to call functions
        )

        ai_message = response.choices[0].message

        # Check if the model wants to call a function
        if ai_message.tool_calls:
            # The model wants to call a function
            function_call = ai_message.tool_calls[0]
            function_name = function_call.function.name
            function_args = json.loads(function_call.function.arguments)

            # Return the function call information
            return {
                "type": "function_call",
                "function_call": {
                    "id": function_call.id,
                    "name": function_name,
                    "arguments": function_args,
                },
                "text": ai_message.content,  # This might be None or contain explanatory text
            }
        else:
            # Regular text response
            return {"type": "text", "text": ai_message.content}
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise HTTPException(status_code=500, detail=f"Azure OpenAI API error: {str(e)}")


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
    if ai_response["type"] == "text":
        # Simple text response
        ai_message = Message(
            role="ai",
            content=MessageContent(type="text", text=ai_response["text"]),
            timestamp=datetime.now(),
        )
        processed_messages.append(ai_message)
    elif ai_response["type"] == "function_call":
        # The model wants to call a function
        function_call = ai_response["function_call"]

        # Add the function call message
        function_call_message = Message(
            role="ai",
            content=MessageContent(
                type="function_call",
                function_call=function_call,
                text=ai_response.get("text"),
            ),
            timestamp=datetime.now(),
        )
        processed_messages.append(function_call_message)

        # Execute the function
        try:
            function_result = await execute_function(
                function_call["name"], function_call["arguments"]
            )

            # Add the function response message
            function_response_message = Message(
                role="function",
                content=MessageContent(
                    type="function_response",
                    function_response={
                        "name": function_call["name"],
                        "content": function_result,
                    },
                ),
                timestamp=datetime.now(),
            )
            processed_messages.append(function_response_message)

            # Get a follow-up response from the AI after the function call
            follow_up_response = await get_gpt4o_response(processed_messages)

            # Add the follow-up response
            if follow_up_response["type"] == "text":
                follow_up_message = Message(
                    role="ai",
                    content=MessageContent(
                        type="text", text=follow_up_response["text"]
                    ),
                    timestamp=datetime.now(),
                )
                processed_messages.append(follow_up_message)
            # Note: We're not handling nested function calls here for simplicity
            # In a production system, you might want to support multiple levels of function calls

        except Exception as e:
            # Handle function execution errors
            error_message = Message(
                role="function",
                content=MessageContent(
                    type="function_response",
                    function_response={
                        "name": function_call["name"],
                        "content": {"error": str(e)},
                    },
                ),
                timestamp=datetime.now(),
            )
            processed_messages.append(error_message)

            # Get a follow-up response to handle the error
            error_follow_up = await get_gpt4o_response(processed_messages)
            error_response = Message(
                role="ai",
                content=MessageContent(type="text", text=error_follow_up["text"]),
                timestamp=datetime.now(),
            )
            processed_messages.append(error_response)

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
        title="Chat API with Azure GPT-4o and Function Calling",
        version="1.0.0",
        description="A RESTful API for chatting with Azure OpenAI GPT-4o model, with function calling capabilities",
        routes=app.routes,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3005)
