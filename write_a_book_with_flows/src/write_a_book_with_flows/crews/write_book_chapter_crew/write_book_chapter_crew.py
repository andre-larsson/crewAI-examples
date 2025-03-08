from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import BaseTool
# from crewai_tools import SerperDevTool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI
from write_a_book_with_flows.types import Chapter
from pydantic import Field

search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)


class SearchTool(BaseTool):
    name: str = "Search"
    description: str = "Search the internet to get updated information on current events and trends."
    search: DuckDuckGoSearchResults = Field(default_factory=DuckDuckGoSearchResults(api_wrapper=search_wrapper))

    def _run(self, query: str) -> str:
        """Execute the search query and return results"""
        try:
            return self.search.invoke(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"


@CrewBase
class WriteBookChapterCrew:
    """Write Book Chapter Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    # llm = ChatOpenAI(model="gpt-4o")
    llm = ChatOpenAI(model="ollama/phi4-mini", base_url="http://localhost:11434/slv1")

    @agent
    def researcher(self) -> Agent:
        # search_tool = SerperDevTool()
        search_tool = SearchTool()
        return Agent(
            config=self.agents_config["researcher"],
            tools=[search_tool],
            llm=self.llm,
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"],
            llm=self.llm,
        )

    @task
    def research_chapter(self) -> Task:
        return Task(
            config=self.tasks_config["research_chapter"],
        )

    @task
    def write_chapter(self) -> Task:
        return Task(config=self.tasks_config["write_chapter"], output_pydantic=Chapter)

    @crew
    def crew(self) -> Crew:
        """Creates the Write Book Chapter Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
