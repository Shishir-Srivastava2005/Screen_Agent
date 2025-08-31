import os
import base64
import time
import subprocess
from typing import Optional, Dict, Any, Union

import pyautogui
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

_default_llm = None  


def get_llm(api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
    """
    Lazily initialize and return the ChatGoogleGenerativeAI instance.
    Passing an api_key overrides the environment value.
    """
    global _default_llm
    if _default_llm is not None:
        return _default_llm

    gemini_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is not set (pass api_key or set in environment).")

    _default_llm = ChatGoogleGenerativeAI(model=model, google_api_key=gemini_key)
    return _default_llm

def take_screenshot(filename: str = "latest_capture.png", delay: float = 3.0) -> Optional[str]:
    """
    Capture the current screen and save to screenshots/<filename>.
    Returns the path on success, or None on failure.
    """
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    try:
        if delay and delay > 0:
            time.sleep(delay)
        screenshot = pyautogui.screenshot()
        screenshot.save(filepath)
        return filepath
    except Exception as e:
        # do not raise here — return None so caller can handle gracefully
        print(f"[agent.take_screenshot] failed: {e}")
        return None


def encode_image_from_path(image_path: str) -> str:
    """Return base64 string for an image file path."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_from_bytes(image_bytes: bytes) -> str:
    """Return base64 string from raw image bytes."""
    return base64.b64encode(image_bytes).decode("utf-8")


# --- LLM wrapper functions ---
def get_conversational_response(
    query: str,
    screenshot_path: Optional[str] = None,
    screenshot_bytes: Optional[bytes] = None,
    llm: Optional[Any] = None,
) -> str:
    """
    Ask the LLM for a conversational answer. Either screenshot_path or screenshot_bytes may be provided.
    If llm is None, module get_llm() will be used (which reads GEMINI_API_KEY).
    Returns the textual content of the LLM response.
    """
    if llm is None:
        llm = get_llm()

    prompt_text = f"""Act as 'Nexus', a proactive and helpful Windows desktop assistant. Your goal is to provide a clear, direct, and conversational answer to the user's request.

User Request: {query}
Nexus Response:"""

    if screenshot_bytes:
        b64 = encode_image_from_bytes(screenshot_bytes)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        )
    elif screenshot_path:
        b64 = encode_image_from_path(screenshot_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        )
    else:
        message = HumanMessage(content=prompt_text)

    response = llm.invoke([message])
    return response.content


def get_structured_commands(
    query: str,
    screenshot_path: Optional[str] = None,
    screenshot_bytes: Optional[bytes] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Ask the LLM to extract structured (JSON) commands. Returns a dict like:
      {"commands": [...], "explanation": "..."}
    If parsing fails, returns {"commands": [], "explanation": "..."}.
    """
    if llm is None:
        llm = get_llm()

    response_schemas = [
        ResponseSchema(
            name="commands",
            description=(
                "A list of executable commands. These can be full .exe files (like 'notepad.exe'), "
                "short system commands (like 'osk' or 'ipconfig'), or commands with arguments "
                "(like 'shutdown /r')."
            ),
        ),
        ResponseSchema(
            name="explanation",
            description="A brief, one-sentence explanation for why these commands are relevant."
        ),
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)

    prompt_template = """You are a command extraction expert for Windows. Your goal is to analyze the user's request and identify any relevant commands that could help them.
Do your best to find a command if one is applicable.
Your output MUST be ONLY the JSON structure described below.
If no command is relevant, return an empty list for the 'commands' field.

{format_instructions}

User Request: {query}
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    if screenshot_bytes:
        vision_prompt = prompt.format(query=query)
        b64 = encode_image_from_bytes(screenshot_bytes)
        message = HumanMessage(
            content=[
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        )
        json_output = llm.invoke([message])
    elif screenshot_path:
        vision_prompt = prompt.format(query=query)
        b64 = encode_image_from_path(screenshot_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": vision_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        )
        json_output = llm.invoke([message])
    else:
        # text-only path: we can compose prompt -> llm chain
        chain = prompt | llm
        json_output = chain.invoke({"query": query})

    try:
        parsed = parser.parse(json_output.content)
        # ensure shape is JSON-friendly (list/dict)
        if not isinstance(parsed, dict):
            return {"commands": [], "explanation": "Parser returned unexpected type."}
        return parsed
    except Exception as e:
        print(f"[agent.get_structured_commands] parse failed: {e}")
        return {"commands": [], "explanation": "Failed to parse commands."}


# --- Orchestrator (primary importable function) ---
def run_agent_task(
    query: str,
    use_screenshot: bool = False,
    screenshot_path: Optional[str] = None,
    screenshot_bytes: Optional[bytes] = None,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:

    if llm is None:
        llm = get_llm()

    final_screenshot_path = None

    if screenshot_bytes:
        fname = f"upload_{int(time.time())}.png"
        final_screenshot_path = os.path.join(SCREENSHOT_DIR, fname)
        with open(final_screenshot_path, "wb") as f:
            f.write(screenshot_bytes)
    elif screenshot_path:
        final_screenshot_path = screenshot_path
    elif use_screenshot:
        final_screenshot_path = take_screenshot()
        if not final_screenshot_path:
            return {
                "conversational_response": "Sorry, I failed to capture the screen. Please try again.",
                "structured_commands": {},
                "screenshot_path": None,
            }

    conversational_answer = get_conversational_response(
        query,
        screenshot_path=final_screenshot_path,
        screenshot_bytes=None, 
        llm=llm,
    )

    structured_data = get_structured_commands(
        query,
        screenshot_path=final_screenshot_path,
        screenshot_bytes=None,
        llm=llm,
    )

    return {
        "conversational_response": conversational_answer,
        "structured_commands": structured_data,
        "screenshot_path": final_screenshot_path,
    }


# Exported names for easy imports
__all__ = [
    "get_llm",
    "take_screenshot",
    "get_conversational_response",
    "get_structured_commands",
    "run_agent_task",
]

# Minimal CLI for testing when run directly (won't run on import)
if __name__ == "__main__":
    print("AI Agent CLI — quick test mode")
    q = input("Type a quick query: ").strip()
    res = run_agent_task(q, use_screenshot=False)
    print("--- Conversational ---")
    print(res["conversational_response"])
    print("--- Structured ---")
    print(res["structured_commands"])
