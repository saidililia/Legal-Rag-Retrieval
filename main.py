from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ai_response import stream_ai_response

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/chat")
async def chat(req: ChatRequest):

    user_message = req.message

    def generate(): # this function will yield chunks of the AI response as they are generated

        for chunk in stream_ai_response(user_message):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )