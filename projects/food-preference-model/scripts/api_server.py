"""
FastAPI server for the TinyLlama Food Preference Chatbot
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import os
from contextlib import asynccontextmanager

# Import our chatbot components
try:
    from tinyllama_food_chatbot import FoodPreferenceChatbot, create_demo_chatbot
except ImportError:
    # Fallback for demo environment
    FoodPreferenceChatbot = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global chatbot instance
chatbot = None

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Chatbot's response")
    user_id: Optional[str] = Field(None, description="Current user ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    query_type: Optional[str] = Field(None, description="Type of query processed")
    confidence: Optional[float] = Field(None, description="Response confidence if applicable")

class UserProfile(BaseModel):
    user_id: str
    age_group: str
    dietary_preference: str
    spice_tolerance: str
    favorite_cuisines: List[str]
    description: str

class FoodRecommendation(BaseModel):
    food_id: str
    name: str
    category: str
    description: str
    confidence: float
    like_probability: float

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    top_k: int = Field(5, description="Number of recommendations", ge=1, le=20)
    min_confidence: float = Field(0.5, description="Minimum confidence threshold", ge=0.0, le=1.0)

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[FoodRecommendation]
    avoid_foods: List[FoodRecommendation]

class PredictionRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    food_id: str = Field(..., description="Food ID")

class PredictionResponse(BaseModel):
    user_id: str
    food_id: str
    predicted_preference: str
    confidence: float
    probabilities: Dict[str, float]
    explanation: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup chatbot"""
    global chatbot
    
    # Startup
    try:
        model_package_path = os.getenv("MODEL_PACKAGE_PATH", "food_preference_model_package.pkl")
        llm_model = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        if FoodPreferenceChatbot:
            chatbot = FoodPreferenceChatbot(
                model_package_path=model_package_path,
                llm_model=llm_model,
                max_length=512,
                temperature=0.7
            )
            logger.info("Chatbot initialized successfully")
        else:
            logger.warning("Chatbot not available - running in demo mode")
            
    except Exception as e:
        logger.error(f"Failed to initialize chatbot: {e}")
        chatbot = None
    
    yield
    
    # Cleanup
    if chatbot:
        logger.info("Shutting down chatbot")

# Create FastAPI app
app = FastAPI(
    title="Food Preference Chatbot API",
    description="AI-powered food recommendation chatbot using TinyLlama and Two-Tower model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage (in production, use Redis or database)
sessions: Dict[str, Dict[str, Any]] = {}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Food Preference Chatbot API",
        "status": "healthy",
        "chatbot_available": chatbot is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot is not None,
        "available_users": len(chatbot.get_available_users()) if chatbot else 0,
        "available_foods": len(chatbot.get_available_foods()) if chatbot else 0
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not available")
    
    try:
        # Get or create session
        session_id = request.session_id or f"session_{len(sessions)}"
        if session_id not in sessions:
            sessions[session_id] = {"conversation_history": []}
        
        # Process chat request
        response = chatbot.chat(
            user_message=request.message,
            user_id=request.user_id
        )
        
        # Store conversation in session
        sessions[session_id]["conversation_history"].extend([
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response}
        ])
        
        # Determine query type (simplified)
        query_type = chatbot._classify_query(request.message) if hasattr(chatbot, '_classify_query') else None
        
        return ChatResponse(
            response=response,
            user_id=chatbot.current_user_id,
            session_id=session_id,
            query_type=query_type
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/users", response_model=List[str])
async def get_users():
    """Get available user IDs"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not available")
    
    return chatbot.get_available_users()

@app.get("/users/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str):
    """Get user profile information"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not available")
    
    try:
        user_info = chatbot.preference_predictor.get_user_info(user_id)
        return UserProfile(**user_info)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"User not found: {user_id}")

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get food recommendations for a user"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not available")
    
    try:
        recommendations = chatbot.preference_predictor.get_user_recommendations(
            user_id=request.user_id,
            top_k=request.top_k,
            min_confidence=request.min_confidence
        )
        
        avoid_foods = chatbot.preference_predictor.get_foods_to_avoid(
            user_id=request.user_id,
            top_k=min(3, request.top_k)
        )
        
        # Convert to response format
        rec_list = [
            FoodRecommendation(
                food_id=rec['food_id'],
                name=rec['name'],
                category=rec['category'],
                description=rec['description'],
                confidence=rec['confidence'],
                like_probability=rec['like_probability']
            )
            for rec in recommendations
        ]
        
        avoid_list = [
            FoodRecommendation(
                food_id=avoid['food_id'],
                name=avoid['name'],
                category=avoid['category'],
                description=avoid['description'],
                confidence=avoid['confidence'],
                like_probability=avoid.get('dislike_probability', 0.0)
            )
            for avoid in avoid_foods
        ]
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=rec_list,
            avoid_foods=avoid_list
        )
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_preference(request: PredictionRequest):
    """Predict user preference for a specific food"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not available")
    
    try:
        prediction = chatbot.preference_predictor.predict_preference(
            user_id=request.user_id,
            food_id=request.food_id
        )
        
        explanation = chatbot.preference_predictor.explain_prediction(
            user_id=request.user_id,
            food_id=request.food_id
        )
        
        return PredictionResponse(
            user_id=request.user_id,
            food_id=request.food_id,
            predicted_preference=prediction['predicted_preference'],
            confidence=prediction['confidence'],
            probabilities=prediction['probabilities'],
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/foods")
async def get_foods():
    """Get available food items"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not available")
    
    return chatbot.get_available_foods()

@app.get("/foods/search")
async def search_foods(query: str, limit: int = 5):
    """Search for food items"""
    if not chatbot:
        raise HTTPException(status_code=503, detail="Chatbot not available")
    
    try:
        results = chatbot.preference_predictor.search_foods(query, limit=limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id].get("conversation_history", [])

# Development server runner
if __name__ == "__main__":
    import uvicorn
    
    # Set environment variables for demo
    os.environ.setdefault("MODEL_PACKAGE_PATH", "demo_food_preference_model.pkl")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
