# Baseado na documentação LangChain
# pip install -qU "langchain[anthropic]" langchain-anthropic python-dotenv

import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  

from langchain.agents import create_agent
from langchain_core.tools import tool

name = "Joao Victor"


@tool
def get_name() -> str:
    """Returns the name of the person. Use when the user asks for the name or who they are."""
    return name


agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_name],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the name?"}]
})

messages = result.get("messages", [])
if messages:
    last = messages[-1]
    if hasattr(last, "content") and last.content:
        print(last.content)
    else:
        print(result)
