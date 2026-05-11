import os 
from dotenv import load_dotenv
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

load_dotenv()  # Load environment variables from .env file

class FlowState(TypedDict, total=False):
    query: str
    search_results: Any
    answer: str


# Node 1: Web search (Tavily)
def search_web(state: FlowState) -> FlowState:
    query = state["query"]
    tavily_tool = TavilySearchResults(max_results=5)
    results = tavily_tool.invoke(query)
    return {"search_results": results}


# Node 2: Send query + search results to GROQ model
def ask_groq(state: FlowState) -> FlowState:
    query = state["query"]
    results = state.get("search_results", [])

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"],
    )

    prompt = f"""
You are given a user query and web search results.
Use the results to produce a concise, accurate answer.

User Query:
{query}

Web Search Results:
{results}
"""

    response = llm.invoke(prompt)
    return {"answer": response.content}


def build_graph():
    graph = StateGraph(FlowState)

    graph.add_node("search_web", search_web)
    graph.add_node("ask_groq", ask_groq)

    graph.add_edge(START, "search_web")
    graph.add_edge("search_web", "ask_groq")
    graph.add_edge("ask_groq", END)

    return graph.compile()


if __name__ == "__main__":
    # Required environment variables:
    #   TAVILY_API_KEY
    #   GROQ_API_KEY

    app = build_graph()
    user_query = "Latest updates on open-source AI agents"

    result = app.invoke({"query": user_query})
    print("\n=== FINAL ANSWER ===")
    print(result["answer"])