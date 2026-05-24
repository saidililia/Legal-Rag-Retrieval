from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")


# Request model
class ChatRequest(BaseModel):
    message: str


# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# Chat endpoint
@app.post("/chat")
async def chat(req: ChatRequest):

    user_message = req.message

    # Replace this later with OpenAI / AI model call
    response = f"AI says: {user_message}"

    return {
        "reply": response
    }