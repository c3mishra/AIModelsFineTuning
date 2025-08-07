# 🌐 API Server Guide

## Quick Start

```bash
# Start the API server
python scripts/api_server.py

# Visit the interactive docs
http://localhost:8000/docs
```

## API Endpoints

### 🤖 Chat with AI
```bash
POST /chat
{
  "user_id": "user_001",
  "message": "What should I eat for dinner?"
}
```

### 🍽️ Get Recommendations
```bash
GET /recommendations/{user_id}?top_k=5
```

### 👥 List Users
```bash
GET /users
```

### 🥘 List Foods
```bash
GET /foods
```

## Usage Examples

### Python Client
```python
import requests

# Chat with the AI
response = requests.post("http://localhost:8000/chat", json={
    "user_id": "user_001", 
    "message": "I want something spicy"
})
print(response.json())

# Get recommendations
recommendations = requests.get("http://localhost:8000/recommendations/user_001?top_k=3")
print(recommendations.json())
```

### cURL
```bash
# Chat endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "message": "I want healthy food"}'

# Recommendations endpoint  
curl "http://localhost:8000/recommendations/user_001?top_k=5"
```

## Environment Variables

```bash
export API_HOST=0.0.0.0        # API host
export API_PORT=8000           # API port  
export DEBUG=false             # Debug mode
export MODEL_PATH=/workspace/models  # Model storage path
```

## Deployment

### Local Development
```bash
python scripts/api_server.py
```

### Production (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker scripts.api_server:app
```

### Docker
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r configs/requirements.txt
CMD ["python", "scripts/api_server.py"]
```

## Response Formats

### Chat Response
```json
{
  "response": "I'd recommend the Spicy Thai Curry - it matches your preferences perfectly!",
  "recommendations": [
    {
      "food_name": "Spicy Thai Curry",
      "category": "Thai", 
      "preference_score": 0.89,
      "like_probability": 0.85
    }
  ],
  "user_id": "user_001"
}
```

### Recommendations Response
```json
{
  "user_id": "user_001",
  "recommendations": [
    {
      "food_name": "Spicy Thai Curry",
      "category": "Thai",
      "ingredients": "coconut milk, chilies, vegetables, tofu",
      "preference_score": 0.89,
      "like_probability": 0.85,
      "is_vegetarian": true,
      "is_spicy": true
    }
  ]
}
```

## Error Handling

The API returns standard HTTP status codes:
- `200` - Success
- `400` - Bad Request (invalid user_id or parameters)
- `404` - Not Found (user not found)
- `500` - Internal Server Error

Error responses include details:
```json
{
  "error": "User not found",
  "detail": "User 'invalid_user' does not exist",
  "status_code": 404
}
```
