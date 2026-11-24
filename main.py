from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()
# tavily = TavilyClient()


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


llm = ChatOllama(model="llama3.2")

#because TavilySearch() is already a tools, then we can immediat use it
# tools = [search]
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course: create agent!")
    # search("Tokyo")
    result = agent.invoke(
        {"messages": HumanMessage(content="Search for 3 job posting for an AI engineer using langchain in Jakarta Indonesia area on linkedin and list of their detail.")}
    )
    # print(result)
    print(result["messages"])


if __name__ == "__main__":
    main()
