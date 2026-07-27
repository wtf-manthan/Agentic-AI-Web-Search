from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
import os
import re

load_dotenv(find_dotenv())

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")
os.environ["USER_AGENT"] = os.getenv("USER_AGENT", "StateGraphRAGAgent/1.0")


class WebSearchAgent:
    """
    Wraps all search tools and exposes the LLM model.
    Falls back through Tavily → DuckDuckGo → Wikipedia until one succeeds.
    """

    def __init__(self):
        self.model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.fast_model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        self._tavily    = TavilySearch(max_results=3)
        self._ddg       = DuckDuckGoSearchRun(max_results=3)
        self._wikipedia = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3)
        )

    # ------------------------------------------------------------------
    # Individual tool runners
    # ------------------------------------------------------------------

    def _run_tavily(self, query: str):
        try:
            result = self._tavily.invoke(query)
            print(f"[Tavily] ✅ Success for: '{query}'")
            return {"success": True, "response": result, "search_engine_used": "Tavily"}
        except Exception as e:
            print(f"[Tavily] ❌ Failed: {e}")
            return {"success": False, "response": None, "search_engine_used": "Tavily"}

    def _run_ddg(self, query: str):
        try:
            result = self._ddg.run(query)
            print(f"[DuckDuckGo] ✅ Success for: '{query}'")
            return {"success": True, "response": result, "search_engine_used": "DuckDuckGo"}
        except Exception as e:
            print(f"[DuckDuckGo] ❌ Failed: {e}")
            return {"success": False, "response": None, "search_engine_used": "DuckDuckGo"}

    def _run_wikipedia(self, query: str):
        try:
            result = self._wikipedia.run(query)
            print(f"[Wikipedia] ✅ Success for: '{query}'")
            return {"success": True, "response": result, "search_engine_used": "Wikipedia"}
        except Exception as e:
            print(f"[Wikipedia] ❌ Failed: {e}")
            return {"success": False, "response": None, "search_engine_used": "Wikipedia"}

    # ------------------------------------------------------------------
    # Query sanitizer
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_query(query: str) -> str:
        """
        Sanitize a raw query before sending it to any search engine.

        Removes characters that confuse Tavily's search parser
        (commas, semicolons, stray quotes, newlines) and collapses
        any runs of whitespace into a single space.
        """
        # Replace newlines / tabs with a space
        query = query.replace("\n", " ").replace("\t", " ")
        # Strip commas and semicolons (confirmed Tavily parser breakers)
        query = re.sub(r"[,;]", " ", query)
        # Remove stray quotation marks
        query = re.sub(r'["\u2018\u2019\u201c\u201d]', "", query)
        # Collapse multiple spaces into one and strip leading/trailing
        query = re.sub(r"\s+", " ", query).strip()
        return query

    # ------------------------------------------------------------------
    # Fallback chain: Tavily → DuckDuckGo → Wikipedia
    # ------------------------------------------------------------------

    def search_with_fallback(self, query: str) -> dict:
        clean = self._clean_query(query)
        if clean != query:
            print(f"[Search] Query sanitized: '{clean}'")
        for runner in (self._run_tavily, self._run_ddg, self._run_wikipedia):
            payload = runner(clean)
            if payload["success"] and payload["response"]:
                return payload
        return {
            "success": False,
            "response": None,
            "search_engine_used": "None"
        }


# Singleton — imported by nodes.py
web_search_agent = WebSearchAgent()