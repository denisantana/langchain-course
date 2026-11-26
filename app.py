import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent."""
    title: str = Field(description="The title of the source")
    url: str = Field(description="The url of the source")


class JobPosting(BaseModel):
    """Schema for a single job posting."""
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(default="", description="Job location, e.g. 'Central Jakarta'")
    requirements: str = Field(default="", description="Key requirements, one line")
    url: str = Field(default="", description="URL of the job posting")


class AgentResponse(BaseModel):
    """Schema for agent's response with answer, jobs and sources"""
    answer: str = Field(description="One-sentence overview of the search results")
    jobs: List[JobPosting] = Field(default_factory=list, description="List of job postings found")
    sources: List[Source] = Field(default_factory=list, description="List of source used to generate an answer")


class SearchRequest(BaseModel):
    query: str
    site: str = "linkedin.com"
    count: int = 3


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

app = FastAPI(title="Job Search Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/api/search")
def search(req: SearchRequest):
    tavily_kwargs = {"max_results": min(max(req.count, 1) * 2, 20)}
    if req.site:
        tavily_kwargs["include_domains"] = [req.site]
    tools = [TavilySearch(**tavily_kwargs)]
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

    site_clause = f"restricted to {req.site}" if req.site else "on job listing websites"
    prompt = (
        "You MUST call the tavily_search tool first to get real data. "
        f"Search for {req.query}, {site_clause}. "
        f"Fill the jobs field with exactly {req.count} postings. For each: title, company, "
        "location, key requirements in one line, and the posting URL. "
        "Put a one-sentence overview in the answer field. "
        "Do not set start_date or end_date in the search. "
        "Never invent job postings, companies, or dates."
    )
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
    return result["structured_response"]
