from fastapi import FastAPI, UploadFile, File, Form
from agent import run_agent_task, get_llm  
import os
import time

app = FastAPI()

@app.on_event("startup")
def startup_event():
    try:
        llm = get_llm()
        app.state.llm = llm
    except Exception as e:
        print("Warning: LLM init failed at startup:", e)

@app.post("/ask")
async def ask(query: str = Form(...), screenshot: UploadFile | None = File(None)):
    screenshot_path = None
    if screenshot:
        save_path = f"screenshots/upload_{int(time.time())}_{screenshot.filename}"
        with open(save_path, "wb") as f:
            f.write(await screenshot.read())
        screenshot_path = save_path

    llm = getattr(app.state, "llm", None)
    result = run_agent_task(query, use_screenshot=False, screenshot_path=screenshot_path, llm=llm)
    return result
