from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents.structured_output import ProviderStrategy
from typing import List
from pydantic import BaseModel, Field
import os

load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent."""
    title: str = Field(description="The title of the source")
    url: str = Field(description="The url of the source")


class AgentResponse(BaseModel):
    """Schema for agent's response with answer and sources"""
    answer: str = Field(description="The agent answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of source used to generate an answer")


# @tool
# def search(query: str) -> str:
#     """
#     Search for information on the web.

#     Args:
#         query: The search query

#     Returns:
#         The search result
#     """

#     print("Using TavilySearch tool.... ")
#     print(f"Search result for {query}")
#     result = TavilySearch().invoke(query)
#     return str(result["results"])


# llm = ChatOllama(model="qwen3:4b")
llm = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    extra_body={
        "models": [
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "google/gemma-4-31b-it:free",
            "recursive/flint-v1.0:free",
        ]
    },
    max_tokens=4096,
    temperature=0,
)

# because TavilySearch() is already a tools, then we can immediat use it
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course: create agent!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(content="You MUST call the tavily_search tool first to get real data. "
                     "Search for AI engineer jobs that mention LangChain, located in Jakarta Indonesia, restricted to linkedin.com "
                     "and in the answer field, summarize each posting with title, company, and key requirements. "
                     "summarize each posting briefly (1-2 lines each): title, company, key requirements. "
                     "Keep the total answer under 150 words. "
                     "Do not set start_date or end_date in the search. "
                     "List exactly 3 postings. For each: title, company, one-line requirements.")
            ]
        }
    )
   
    print(result)


if __name__ == "__main__":
    main()
