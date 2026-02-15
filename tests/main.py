# Baseado na documentação LangChain
# pip install -qU "langchain[anthropic]" langchain-anthropic python-dotenv

import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  

from langchain.agents import create_agent
from langchain_core.tools import tool

name = "Joao Victor"

db = {}


@tool
def get_name() -> str:
    """Returns the name of the person. Use when the user asks for the name or who they are."""
    return name

@tool
def get_history() -> str:
    """Returns the last messages between the agent and the user"""
    return db

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_history],
    system_prompt="You are a helpful assistant.",
)


def create_message(text):
    result = agent.invoke({
    "messages": [{"role": "user", "content": f"{text}"}]
    })

    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        if hasattr(last, "content") and last.content:
            db[len(db.items())]= [f'"{text}" : "{last.content}"']

            print(last.content)
        else:
            db[len(db.items())]= [f'"{text}" : "{result}"']
            print(result)






while True:
    message = input("> ")
    create_message(message)

    