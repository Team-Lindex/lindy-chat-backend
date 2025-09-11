# AI Chat Backend API

A FastAPI-based backend service for AI chat with image support using Google Gemini models.

## Features

- 🤖 AI chat with Google Gemini models
- 🖼️ Image upload and processing (PIL/Pillow)
- 🎨 Image generation capabilities
- 📝 Session management
- 🔄 Multiple chat endpoints (text, images, file upload)
- 🐳 Docker containerization
- 🚀 Production-ready deployment

## Quick Start

### Using Docker (Recommended)

1. Clone the repository
2. Copy environment file:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` with your Google API key
4. Run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set environment variables:
   ```bash
   export GOOGLE_API_KEY=your_api_key_here
   ```
3. Run the server:
   ```bash
   python main.py
   ```

## API Endpoints

### Core Chat Endpoints

- `POST /chat` - Send text message
- `POST /chat/with-images` - Send message with base64 images
- `POST /chat/upload` - Send message with file uploads

### Session Management

- `GET /sessions` - List all sessions
- `GET /sessions/{session_id}` - Get session info
- `GET /sessions/{session_id}/messages` - Get session messages
- `DELETE /sessions/{session_id}` - Delete session

### Image Endpoints

- `GET /sessions/{session_id}/images/{image_index}` - Get generated image
- `POST /sessions/{session_id}/save-images` - Save generated images

### Health Check

- `GET /health` - Health check endpoint
- `GET /` - API info

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | Required | Your Google AI API key |
| `MODEL_NAME` | `gemini-2.5-flash-image-preview` | Model to use |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

## Docker Deployment

### Build Image
```bash
docker build -t chat-backend .
```

### Run Container
```bash
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key chat-backend
```

### Docker Compose
```bash
docker-compose up -d
```

## API Usage Examples

### Simple Chat
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, how are you?",
    "system_prompt": "You are a helpful assistant."
  }'
```

### Chat with Images (Base64)
```bash
curl -X POST "http://localhost:8000/chat/with-images" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What do you see in this image?",
    "images": ["base64_image_data_here"]
  }'
```

### File Upload
```bash
curl -X POST "http://localhost:8000/chat/upload" \
  -F "message=Describe this image" \
  -F "files=@image.jpg"
```

## Production Deployment

1. Set up environment variables securely
2. Use a reverse proxy (nginx/traefik)
3. Enable HTTPS
4. Set up monitoring and logging
5. Configure resource limits

## Security Notes

- API key is required and should be kept secure
- CORS is enabled for all origins (modify for production)
- Non-root user in Docker container
- Health checks enabled

## Development

The API uses:
- FastAPI for the web framework
- Google GenAI for AI capabilities
- PIL/Pillow for image processing
- Uvicorn as ASGI server

## License

This project is licensed under the MIT License.