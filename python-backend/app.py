# from fastapi import FastAPI
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from pathlib import Path

# # Load env
# env_path = Path(__file__).parent / ".env"
# if env_path.exists():
#     load_dotenv(env_path)
#     print("📄 Loaded .env")

# # Import your helper
# from core.react_agent.holistic_ai_bedrock import HolisticAIBedrockChat, get_chat_model


# app = FastAPI()


# class ChatRequest(BaseModel):
#     message: str

# def extract_text(response):
#     """
#     Extracts plain text from various LLM response formats:
#     - Anthropic messages
#     - Bedrock Claude
#     - LangChain AIMessage
#     - OpenAI-compatible content blocks
#     """

#     # Case 1 — LangChain AIMessage / HumanMessage
#     # These have .content directly as a string
#     if hasattr(response, "content") and isinstance(response.content, str):
#         return response.content

#     # Case 2 — Anthropic Content Blocks (list)
#     if hasattr(response, "content") and isinstance(response.content, list):
#         parts = response.content
#         texts = []
#         for p in parts:
#             if hasattr(p, "text"):
#                 texts.append(p.text)
#         if texts:
#             return "\n".join(texts)

#     # Case 3 — Anthropic single content block
#     if hasattr(response, "content") and hasattr(response.content, "text"):
#         return response.content.text

#     # Fallback: stringify
#     return str(response)

# @app.post("/chat")
# async def chat(req: ChatRequest):
#     # Load your model once per request
#     llm = get_chat_model("claude-3-5-sonnet")

#     # Call model
#     raw_response = llm.invoke(req.message)

#     # Extract readable text
#     clean_text = extract_text(raw_response)

#     print("\n🧹 RAW RESPONSE:")
#     print(raw_response)

#     return {"response": clean_text}

from fastapi import FastAPI
from pydantic import BaseModel
from model_router import route_tasks


app = FastAPI()

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    user_input = req.message

    # MAIN LOGIC
    result = route_tasks(user_input)

    return {"response": result}
