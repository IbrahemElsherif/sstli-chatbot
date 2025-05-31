from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

web_router = APIRouter(
    prefix="",
    tags=["web"],
)

# Get the base directory (src folder)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

@web_router.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main management page"""
    file_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("<h1>Welcome to Mini-RAG</h1><p>Web interface not found.</p>")

@web_router.get("/index.html", response_class=HTMLResponse)
async def read_index():
    """Serve the management page"""
    file_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("<h1>Management Page Not Found</h1>")

@web_router.get("/chatbot.html", response_class=HTMLResponse)
async def read_chatbot():
    """Serve the chatbot page"""
    file_path = os.path.join(WEB_DIR, "chatbot.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("<h1>Chatbot Page Not Found</h1>")

@web_router.get("/management.js")
async def read_management_js():
    """Serve the management JavaScript file"""
    file_path = os.path.join(WEB_DIR, "management.js")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/javascript")
    return HTMLResponse("// Management JS not found", media_type="text/plain")

@web_router.get("/chatbot.js")
async def read_chatbot_js():
    """Serve the chatbot JavaScript file"""
    file_path = os.path.join(WEB_DIR, "chatbot.js")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/javascript")
    return HTMLResponse("// Chatbot JS not found", media_type="text/plain")

@web_router.get("/api/v1/welcome")
async def welcome():
    """Welcome endpoint for testing"""
    return {
        "message": "Welcome to Mini-RAG API!",
        "version": "1.0",
        "endpoints": {
            "management": "/",
            "chatbot": "/chatbot.html",
            "upload": "/api/v1/data/upload/{project_id}",
            "process": "/api/v1/data/process/{project_id}",
            "index_push": "/api/v1/nlp/index/push/{project_id}",
            "index_info": "/api/v1/nlp/index/info/{project_id}",
            "index_search": "/api/v1/nlp/index/search/{project_id}",
            "rag_answer": "/api/v1/nlp/index/answer/{project_id}"
        }
    }

@web_router.get("/test_chatbot.html", response_class=HTMLResponse)
async def read_test_chatbot():
    """Serve the test chatbot widget page"""
    file_path = os.path.join(WEB_DIR, "test_chatbot.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse("<h1>Test Chatbot Page Not Found</h1>") 