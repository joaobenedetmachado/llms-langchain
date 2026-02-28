# Baseado na documentação LangChain
# pip install -qU "langchain[anthropic]" langchain-anthropic python-dotenv

import json
import os
import re
import tomllib
from dotenv import load_dotenv, find_dotenv
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
import html2text

_ = load_dotenv(find_dotenv())  

from langchain.agents import create_agent
from langchain_core.tools import tool

name = "Joao Victor"


CONVERSATION_FILE = os.path.join(os.path.dirname(__file__), "conversation.txt")
FINANCE_DATA_FILE = os.path.join(os.path.dirname(__file__), "finance_data.txt")


@tool
def exec_code(code: str) -> str:
    """Give the agent a way to execute Python code. The code is logged to conversation.txt."""
    with open(CONVERSATION_FILE, "a", encoding="utf-8") as f:
        f.write(f"[exec_code]\n{code}\n\n")
    exec(code)
    return "Code executed and saved to conversation.txt"

@tool
def write_finance_data(value: float, description: str, date: str, type: str):
    """Write the finance data to the file conversation.txt"""
    with open(FINANCE_DATA_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{description} : {value} : {date} : {type}\n")
    return "Data written to file"

@tool
def read_finance_data():
    """Read the finance data from the file conversation.txt"""
    with open(FINANCE_DATA_FILE, 'r', encoding='utf-8') as f:
        return f.read()

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for the query. Returns title, link and snippet for each result."""
    try:
        results = list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        return f"Search error: {e}"
    if not results:
        return "No results found."
    lines = []
    for r in results:
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"- {title}\n  {href}\n  {body}")
    return "\n\n".join(lines)

@tool
def scrape_website(url: str) -> str:
    """Scrape a website and return the title, description and content."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Error fetching URL: {e}"
    soup = BeautifulSoup(response.text, "html.parser")
    title_tag = soup.find("title")
    title = (title_tag.get_text(strip=True) if title_tag else "") or "(no title)"
    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = (meta_desc.get("content", "").strip() if meta_desc else "") or "(no description)"
    content = soup.get_text(separator="\n", strip=True)
    return f"Title: {title}\nDescription: {description}\n\nContent:\n{content[:8000]}"


@tool
def kill_chat():
    """kill the chat"""
    exit()

@tool
def get_name() -> str:
    """Returns the name of the person. Use when the user asks for the name or who they are."""
    return name

@tool
def get_history() -> str:
    """Returns the last messages between the agent and the user in the file conversation.txt"""
    with open(CONVERSATION_FILE, 'r', encoding='utf-8') as f:
        return f.read()

REASONING_SYSTEM_PROMPT = """You are a helpful assistant. You can write finance data to the file finance_data.txt and read finance data from the file finance_data.txt.
When asked to reason: first think step by step (your reasoning), then when you have the final answer give it clearly. You may receive your own previous reasoning and be asked to continue."""

REASONING_JSON_INSTRUCTION = """Respond with ONLY a valid JSON object, nothing else. Format:
{"text": "your reasoning or final answer here", "status": "thinking" or "finished"}
- Use status "thinking" when you are still reasoning and need another turn.
- Use status "finished" when you have the final answer for the user."""

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_history, kill_chat, exec_code, write_finance_data, read_finance_data, search_web, scrape_website],
    system_prompt=REASONING_SYSTEM_PROMPT,
)

MAX_REASONING_STEPS = 15


def _save_and_print(text: str, last_content: str, result):
    with open(CONVERSATION_FILE, "a", encoding="utf-8") as f:
        f.write(f"{text} : {last_content}\n")
    print(last_content)


def _parse_reasoning_json(content: str) -> dict | None:
    """Extrai JSON da resposta do modelo (pode vir com markdown ou texto extra)."""
    if not content or not content.strip():
        return None
    content = content.strip()
    # Tenta extrair bloco ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
    if match:
        content = match.group(1)
    # Ou procura por { ... } no texto
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        content = match.group(0)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def polish_tool(full_context: list[str]) -> str:
    """Refina todo o contexto de raciocínio numa resposta final clara para o utilizador."""
    context_str = "\n\n".join(full_context)
    prompt = f"""Refine the following reasoning into a clear, concise final answer for the user. Write only the final answer, no meta-commentary.

Reasoning context:
{context_str}

Final answer:"""
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if content:
            return content
    return context_str


def reasoning_recursive(
    user_message: str,
    context_so_far: list[str] | None = None,
    step: int = 0,
    max_steps: int = MAX_REASONING_STEPS,
):
    """
    O modelo devolve JSON com {text, status}. Se status == "thinking", chama-se a si mesma
    juntando o texto ao contexto. Quando status == "finished", envia todo o contexto para
    polish_tool e devolve a resposta refinada.
    """
    if context_so_far is None:
        context_so_far = []
    if step >= max_steps:
        return polish_tool(context_so_far)

    if not context_so_far:
        prompt = f"{user_message}\n\n{REASONING_JSON_INSTRUCTION}"
    else:
        prev = "\n\n".join(context_so_far)
        prompt = f"""Previous reasoning:

{prev}

Continue reasoning. If you have the final answer, set status to "finished".

{REASONING_JSON_INSTRUCTION}"""

    print("Thinking...")
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result.get("messages", [])
    last_content = ""
    if messages:
        last_content = getattr(messages[-1], "content", "") or ""

    data = _parse_reasoning_json(last_content)
    if not data or "status" not in data:
        # trata como resposta final
        context_so_far = context_so_far + [last_content] if last_content else context_so_far
        return polish_tool(context_so_far) if context_so_far else last_content

    text = data.get("text", "")
    status = data.get("status", "finished").lower()
    context_so_far = context_so_far + [text] if text else context_so_far

    if status == "thinking":
        return reasoning_recursive(user_message, context_so_far, step + 1, max_steps)
    # status == "finished" ou outro
    return polish_tool(context_so_far)


def create_message(text: str, reasoning_mode: bool = False):
    if reasoning_mode:
        final = reasoning_recursive(text)
        _save_and_print(text, final, None)
        return
    result = agent.invoke({"messages": [{"role": "user", "content": text}]})
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None) or str(result)
        if content:
            _save_and_print(text, content, result)
            return
    with open(CONVERSATION_FILE, "a", encoding="utf-8") as f:
        f.write(f"{text} : {result}\n")
    print(result)

while True:
    message = input("> ")
    if message.lower() == "exit":
        break
    if message == "" or message is None:
        continue
    if message.lower() == "help":
        print("Help:")
        print("exit: exit the program")
        print("help: show this help")
        print(f"r <message>  : reasoning mode (JSON thinking/finished, max {MAX_REASONING_STEPS} steps, then polish)")
        print("write_finance_data / read_finance_data / get_history / kill_chat / get_name")
        continue
    if message.lower().startswith("r "):
        create_message(message[2:].strip(), reasoning_mode=True)
    else:
        create_message(message)
