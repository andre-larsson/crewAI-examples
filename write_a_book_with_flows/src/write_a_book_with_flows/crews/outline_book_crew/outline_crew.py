from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from write_a_book_with_flows.llm import outline_llm, outline_researcher_llm
# from crewai_tools import SerperDevTool
from write_a_book_with_flows.types import BookOutline
from write_a_book_with_flows.tools import SearchTool


@CrewBase
class OutlineCrew:
    """Book Outline Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    # llm = ChatOpenAI(model="chatgpt-4o-latest")
    # llm = ChatOpenAI(model="gpt-4o")
    llm = outline_llm
    tool_llm = outline_researcher_llm

    @agent
    def researcher(self) -> Agent:
        search_tool = SearchTool()
        return Agent(
            config=self.agents_config["researcher"],
            tools=[search_tool],
            llm=self.tool_llm,
            verbose=True,
        )

    @agent
    def outliner(self) -> Agent:
        return Agent(
            config=self.agents_config["outliner"],
            llm=self.llm,
            verbose=True,
        )

    @task
    def research_topic(self) -> Task:
        return Task(
            config=self.tasks_config["research_topic"],
        )

    @task
    def generate_outline(self) -> Task:
        return Task(
            config=self.tasks_config["generate_outline"], output_pydantic=BookOutline
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Book Outline Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
