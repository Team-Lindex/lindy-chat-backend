from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import uuid
import json
import os
from io import BytesIO
from PIL import Image

from chat_api import ChatChain, pil_image_to_bytes, bytes_to_pil_image
from google import genai

app = FastAPI(title="AI Chat API", description="FastAPI server for AI chat with image support", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for chat sessions
chat_sessions: Dict[str, ChatChain] = {}

# Pydantic models for request/response
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    system_prompt: Optional[str] = "You are Lindy a personal style assistant from Lindex."
    enable_image_generation: Optional[bool] = False
    generate_image: Optional[bool] = None

class ChatWithImagesRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    images: List[str]  # Base64 encoded images
    system_prompt: Optional[str] = "You are Lindy a personal style assistant from Lindex."
    enable_image_generation: Optional[bool] = False
    generate_image: Optional[bool] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    generated_images: Optional[List[str]] = None  # Base64 encoded images
    message_count: int

class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    system_prompt: str
    enable_image_generation: bool

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Configuration - Use environment variables
API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyAfMPu8d-uzzXQ-a2xaKxNMr_K9_oKDIVo')
MODEL = os.getenv('MODEL_NAME', 'gemini-2.5-flash-image-preview')

def get_genai_client():
    """Get Google GenAI client"""
    return genai.Client(api_key=API_KEY)

def base64_to_pil(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

def pil_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def create_or_get_session(session_id: Optional[str], system_prompt: str, enable_image_generation: bool) -> tuple[str, ChatChain]:
    """Create new session or get existing one"""
    if session_id and session_id in chat_sessions:
        return session_id, chat_sessions[session_id]
    
    # Create new session
    new_session_id = session_id or str(uuid.uuid4())
    client = get_genai_client()
    chat_chain = ChatChain(
        model=MODEL,
        client=client,
        system_prompt=system_prompt,
        enable_image_generation=enable_image_generation
    )
    chat_sessions[new_session_id] = chat_chain
    return new_session_id, chat_chain

@app.get("/")
async def root():
    return {"message": "AI Chat API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a text message to the AI"""
    try:
        session_id, chat_chain = create_or_get_session(
            request.session_id,
            request.system_prompt,
            request.enable_image_generation
        )
        
        # Send message
        response_text = chat_chain.chat(
            message=request.message,
            generate_image=request.generate_image
        )
        
        # Get generated images if any
        generated_images_b64 = None
        last_generated = chat_chain.get_last_generated_images()
        if last_generated:
            generated_images_b64 = [
                pil_to_base64(img) for img in last_generated
            ]
        
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            generated_images=generated_images_b64,
            message_count=len(chat_chain.get_messages())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/with-images", response_model=ChatResponse)
async def chat_with_images(request: ChatWithImagesRequest):
    """Send a message with images to the AI"""
    try:
        session_id, chat_chain = create_or_get_session(
            request.session_id,
            request.system_prompt,
            request.enable_image_generation
        )
        
        # Convert base64 images to PIL Images
        pil_images = []
        for img_b64 in request.images:
            try:
                pil_img = base64_to_pil(img_b64)
                pil_images.append(pil_img)
            except Exception as img_error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image data: {str(img_error)}"
                )
        
        # Send message with images
        response_text = chat_chain.chat(
            message=request.message,
            images=pil_images,
            generate_image=request.generate_image
        )
        
        # Get generated images if any
        generated_images_b64 = None
        last_generated = chat_chain.get_last_generated_images()
        if last_generated:
            generated_images_b64 = [
                pil_to_base64(img) for img in last_generated
            ]
        
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            generated_images=generated_images_b64,
            message_count=len(chat_chain.get_messages())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/upload", response_model=ChatResponse)
async def chat_with_file_upload(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    system_prompt: str = Form("You are Lindy a personal style assistant from Lindex."),
    enable_image_generation: bool = Form(False),
    generate_image: Optional[bool] = Form(None),
    files: List[UploadFile] = File(...)
):
    """Send a message with uploaded image files to the AI"""
    try:
        session_id, chat_chain = create_or_get_session(
            session_id,
            system_prompt,
            enable_image_generation
        )
        
        # Process uploaded files
        pil_images = []
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not an image"
                )
            
            try:
                contents = await file.read()
                pil_img = Image.open(BytesIO(contents))
                pil_images.append(pil_img)
            except Exception as img_error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not process image {file.filename}: {str(img_error)}"
                )
        
        # Send message with images
        response_text = chat_chain.chat(
            message=message,
            images=pil_images,
            generate_image=generate_image
        )
        
        # Get generated images if any
        generated_images_b64 = None
        last_generated = chat_chain.get_last_generated_images()
        if last_generated:
            generated_images_b64 = [
                pil_to_base64(img) for img in last_generated
            ]
        
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            generated_images=generated_images_b64,
            message_count=len(chat_chain.get_messages())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get information about a chat session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_chain = chat_sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        message_count=len(chat_chain.get_messages()),
        system_prompt=chat_chain.system_prompt,
        enable_image_generation=chat_chain.enable_image_generation
    )

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get all messages from a chat session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_chain = chat_sessions[session_id]
    messages = chat_chain.get_messages()
    
    # Convert images in messages to base64 for JSON serialization
    serializable_messages = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            # Handle user messages with images
            content = []
            for part in msg["content"]:
                if isinstance(part, dict) and part.get("type") == "image":
                    if "image_bytes" in part:
                        # Convert bytes to base64
                        img_b64 = base64.b64encode(part["image_bytes"]).decode('utf-8')
                        content.append({
                            "type": "image",
                            "image_data": img_b64,
                            "mime_type": part.get("mime_type", "image/jpeg")
                        })
                    else:
                        content.append(part)
                else:
                    content.append(part)
            serializable_messages.append({
                "role": msg["role"],
                "content": content
            })
        elif isinstance(msg.get("content"), dict):
            # Handle assistant messages with generated images
            content = msg["content"].copy()
            if "generated_images" in content:
                converted_images = []
                for gen_img in content["generated_images"]:
                    if "image_bytes" in gen_img:
                        # Convert bytes to base64
                        img_b64 = base64.b64encode(gen_img["image_bytes"]).decode('utf-8')
                        converted_images.append({
                            "image_data": img_b64,
                            "mime_type": gen_img.get("mime_type", "image/jpeg")
                        })
                    else:
                        converted_images.append(gen_img)
                content["generated_images"] = converted_images
            serializable_messages.append({
                "role": msg["role"],
                "content": content
            })
        else:
            # Handle simple text messages
            serializable_messages.append(msg)
    
    return {"messages": serializable_messages}

@app.get("/sessions")
async def list_sessions():
    """List all active chat sessions"""
    sessions = []
    for session_id, chat_chain in chat_sessions.items():
        sessions.append({
            "session_id": session_id,
            "message_count": len(chat_chain.get_messages()),
            "system_prompt": chat_chain.system_prompt,
            "enable_image_generation": chat_chain.enable_image_generation
        })
    return {"sessions": sessions}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del chat_sessions[session_id]
    return {"message": f"Session {session_id} deleted"}

@app.get("/sessions/{session_id}/images/{image_index}")
async def get_generated_image(session_id: str, image_index: int):
    """Get a specific generated image from a session"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    chat_chain = chat_sessions[session_id]
    all_generated = chat_chain.get_all_generated_images()
    
    if not all_generated:
        raise HTTPException(status_code=404, detail="No generated images found")
    
    # Flatten all generated images
    all_images = []
    for _, images in all_generated:
        all_images.extend(images)
    
    if image_index >= len(all_images):
        raise HTTPException(status_code=404, detail="Image index out of range")
    
    # Convert PIL image to bytes for streaming
    img = all_images[image_index]
    img_bytes, mime_type = pil_image_to_bytes(img, format="PNG")
    
    return StreamingResponse(
        BytesIO(img_bytes),
        media_type=mime_type,
        headers={"Content-Disposition": f"inline; filename=generated_image_{image_index}.png"}
    )

@app.post("/sessions/{session_id}/save-images")
async def save_session_images(session_id: str, base_path: str = "generated_image", format: str = "PNG"):
    """Save all generated images from a session to files"""
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        chat_chain = chat_sessions[session_id]
        saved_paths = chat_chain.save_last_generated_images(base_path, format)
        return {"saved_files": saved_paths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    uvicorn.run(app, host=host, port=port)