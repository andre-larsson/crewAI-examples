from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5, safesearch="off")
search=DuckDuckGoSearchResults(api_wrapper=search_wrapper)


class SearchQuery(BaseModel):
    """Query input for the tool."""
    query: str = Field(..., description="Description of the argument.")

class SearchTool(BaseTool):
    name: str = "Search"
    api_wrapper: DuckDuckGoSearchAPIWrapper = search_wrapper
    search: DuckDuckGoSearchResults = search
    description: str = "Search the internet to get updated information on current events and trends."
    args_schema: Type[BaseModel] = SearchQuery

    def __init__(self):
        super().__init__()
        self.api_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        self.search = DuckDuckGoSearchResults(api_wrapper=self.api_wrapper)

    # def _run(self, query: str) -> str:
    #     # Your tool's logic here
    #     return "Tool's result"

    def _run(self, query: str) -> str:
        """Execute the search query and return results"""
        try:
            return self.search.invoke(query)
        except Exception as e:
            raise RuntimeError(f"Error performing search: {str(e)}") from e