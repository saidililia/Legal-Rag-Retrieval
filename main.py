from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ai import ask_ai
from evaluator import run_ragas_eval

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 🔥 MAIN CHAT ENDPOINT
@app.post("/chat")
async def chat(req: ChatRequest):

    user_message = req.message

    # 1. Generate answer
    ai_answer = ask_ai(user_message)

    # 2. OPTIONAL: RAGAS evaluation (mock context for now)
    # Later replace with real retrieved docs (RAG system)
    try:
        eval_result = run_ragas_eval(
            question=user_message,
            answer=ai_answer,
            context="Contract law provides damages for breach of contract.",
            ground_truth="Damages are the primary remedy for breach of contract."
        )

        print("\n📊 RAGAS SCORE:\n", eval_result)

    except Exception as e:
        print("Ragas skipped:", e)

    return {
        "reply": ai_answer
    }