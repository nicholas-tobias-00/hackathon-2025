from typing import List, Dict, Any
# from model import get_tasks   # your black-box classifier
# from googleapiclient.discovery import build

from dotenv import load_dotenv
from pathlib import Path
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

# Load env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print("📄 Loaded .env")

# Import your helper
from core.react_agent.holistic_ai_bedrock import HolisticAIBedrockChat, get_chat_model

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply ⁠ a ⁠ and ⁠ b ⁠.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds ⁠ a ⁠ and ⁠ b ⁠.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide ⁠ a ⁠ and ⁠ b ⁠.

    Args:
        a: First int
        b: Second int
    """
    return a / b


def extract_text(response):
    if hasattr(response, "content") and isinstance(response.content, str):
        return response.content

    if hasattr(response, "content") and isinstance(response.content, list):
        texts = []
        for p in response.content:
            if hasattr(p, "text"):
                texts.append(p.text)
        return "\n".join(texts)

    if hasattr(response, "content") and hasattr(response.content, "text"):
        return response.content.text

    return str(response)

# ---------------------------
# HELPERS TO CALL MODELS
# ---------------------------

def run_small_llm(query: str) -> str:
    llm = get_chat_model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    response = llm.invoke(query)
    return extract_text(response)


def run_large_llm(query: str) -> str:
    llm = get_chat_model("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    response = llm.invoke(query)
    return extract_text(response)


# def run_google_search(query: str) -> str:
#     service = build("customsearch", "v1",
#                     developerKey="AIzaSyDEc5q53ghqRCgvX_8jNi5ykpCL9BJvAZE")

#     results = service.cse().list(
#         q=query, cx="YOUR_CSE_ID"
#     ).execute()

#     # Format a simple output
#     formatted = "\n".join([
#         item["snippet"] for item in results.get("items", [])[:3]
#     ])

#     return formatted or "No search results found."


def run_math_agent(query: str) -> str:
    """Run small LLM with tools (ReAct agent)."""
    llm = get_chat_model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    math_agent = create_react_agent(llm, [add, multiply, divide])
    response = math_agent.invoke({"messages": [HumanMessage(content=query)]})
    return extract_final_answer(response)


# ---------------------------
# ROUTER FUNCTION
# ---------------------------
from transformers import pipeline
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List


MODEL_PATH = "./model"
SYSTEM_MESSAGE_FOR_CLASSIFIER = (
        "You are an expert at task decomposition (break the questions / tasks). Your job is to take a complex, "
        "multi-part user query and break it down into a list of "
        "sub-tasks. "
        "Do not add any commentary. Only output the list of tasks ['task 1', 'task 2' etc]. No any extra information from the "
    )

class TaskList(BaseModel):
        """Structured output for a list of tasks."""
        tasks: List[str] = Field(description="List of tasks to perform")

# Initialize classifier and LLM once (reuse across recursive calls)
_classifier = None
_llm_structured = None


def _init_models():
    """Initialize classifier and LLM once for performance"""
    global _classifier, _llm_structured
    if _classifier is None:
        print(f"Loading model from {MODEL_PATH}...")
        _classifier = pipeline(
            "text-classification",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH
        )
        print("Model loaded successfully!")
    
    if _llm_structured is None:
        llm = get_chat_model("claude-3-5-sonnet")
        _llm_structured = llm.with_structured_output(TaskList)
    
    return _classifier, _llm_structured

def _decompose_task_recursive(task: str, classifier, llm_structured, depth: int = 0) -> list:
    """
    Recursively decompose a task if it's a multi-task query.
    
    Args:
        task: The task to classify and potentially decompose
        classifier: The text classification model
        llm_structured: The structured LLM for task decomposition
        depth: Current recursion depth (for logging)
    
    Returns:
        List of task dictionaries with classification and query
    """
    indent = "  " * depth
    
    # Classify the task
    result = classifier(task)
    # classifier returns a list like [{'label':..., 'score':...}]
    result_list = result.copy() if isinstance(result, list) else [result]
    
    print(f"{indent}Classification: {result_list[0]['label']} (score: {result_list[0]['score']:.4f})")
    
    tasks_out = []
    # If it's a multi-task query, decompose it further
    if result_list[0]['label'] == 'multi_task_query':
        print(f"{indent}→ Multi-task detected, decomposing...")
        messages = [
            SystemMessage(content=SYSTEM_MESSAGE_FOR_CLASSIFIER),
            HumanMessage(task)
        ]
        response = llm_structured.invoke(messages)
        sub_tasks = response.tasks
        
        # Recursively process each sub-task and collect their classified dicts
        for sub_task in sub_tasks:
            print(f"{indent}├─ Sub-task: '{sub_task}'")
            sub_results = _decompose_task_recursive(
                sub_task, 
                classifier, 
                llm_structured, 
                depth + 1
            )
            # sub_results is a list of dicts each already containing 'label','score','query'
            tasks_out.extend(sub_results)
    else:
        # Single task - attach query to classifier result and return it
        result_list[0]['query'] = task
        tasks_out.append(result_list[0])
    
    return tasks_out

def get_tasks(query: str) -> list:
    """
    Decompose a user query into a list of tasks with classifications.
    
    Handles nested multi-task queries recursively.
    
    Args:
        query: User query. Examples:
            - 'What is 5 + 5 and also 10 plus 10?'
            - 'What is capital of Mongolia?'
            - 'Calculate 5+5, tell me a joke, and what is 10*10?'
    
    Returns:
        List of task dictionaries with structure:
            [
                {'label': 'run_calculation_tool', 'score': 0.97, 'query': 'Calculate 5 + 5'},
                {'label': 'run_calculation_tool', 'score': 0.97, 'query': 'Calculate 10 + 10'}
            ]
    """
    print(f"\n{'='*70}")
    print(f"Processing query: '{query}'")
    print(f"{'='*70}")
    
    # Initialize models
    classifier, llm_structured = _init_models()
    
    # Classify the initial query (for logging)
    print(f"\nInitial classification:")
    result_of_initial_classification = classifier(query)
    print(f"Result: {result_of_initial_classification}")
    
    # Decompose recursively - this call will return only leaf tasks with 'query' attached
    print(f"\nTask decomposition:")
    combined_tasks = _decompose_task_recursive(
        query, 
        classifier, 
        llm_structured, 
        depth=0
    )
    
    print(f"\n{'─'*70}")
    print(f"Final task list ({len(combined_tasks)} tasks):")
    for i, task in enumerate(combined_tasks, 1):
        print(f"  {i}. [{task['label']}] {task.get('query', 'N/A')}")
    print(f"{'='*70}\n")
    
    return combined_tasks


def extract_final_answer(result_dict):
    """
    Extract final AI answer from a LangChain agent trace:
    - find last AIMessage with non-empty content
    - ignore ToolMessage and tool-calling AIMessages
    """

    messages = result_dict.get("messages", [])

    final_answer = None

    for msg in messages[::-1]:   # iterate BACKWARDS
        # Skip tool results
        if msg.__class__.__name__ == "ToolMessage":
            continue

        # Skip AI messages that are tool-calls (they have tool_calls list)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            continue

        # FIND the final AIMessage with actual content
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            final_answer = msg.content.strip()
            break

    return final_answer or "No answer found."


def route_tasks(query: str) -> str:
    """
    1. Use get_tasks(query) to predict the required actions.
    2. Execute each task in order.
    3. Combine final results into one answer.
    """
    tasks = get_tasks(query) # THIS IS USUKS AND SARTHAKS FUNCTION
    # tasks = [
    #     {"query": "Hi there!", "tool_label": "simple_chitchat"},
    #     {"query": "Solve for x in 3x = 21", "tool_label": "run_calculation_tool"}]
    results = []

    for task in tasks:
        label = task.get("label")
        task_query = task.get("query", query)  # fallback if query missing

        if label == "simple_chitchat":
            result = run_small_llm(task_query)
            print("*******************************************************************")
            print("We are in simple chitchat")
            print("*******************************************************************")

        elif label == "google_search":
            result = run_small_llm(task_query)
            print("*******************************************************************")
            print("We are in google search")
            print("*******************************************************************")

        elif label == "run_calculation_tool":
            result = run_math_agent(task_query)
            print("*******************************************************************")
            print("We are in calculation tool")
            print("*******************************************************************")

        elif label == "run_large_model_agent":
            result = run_large_llm(task_query)
            print("*******************************************************************")
            print("We are in large model agent")
            print("*******************************************************************")

        else:
            result = f"Unknown task: {label}"
        print("result:", result)

        results.append(result)

    # Combine all results (multi-task)
    return "\n\n".join(results)
