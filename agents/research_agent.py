from typing import Optional, Type

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """
You are a research agent.
Your task is to search for information on the given topic and provide facts and feedback.

Iterate until until the research is satisfactory and complete.
Don't pass it on to the research_manager until it is complete.

Return all facts gathered. Include other useful facts if you find any.
Strategize, then execute

Return the long-form research in full, including citations.
"""


search = TavilySearchAPIWrapper()
talivy_tool = TavilySearchResults(search=search)


class ResearchAgentInput(BaseModel):
    topic: str = Field(description="Topic of research.")
    feedback: Optional[str] = Field(
        description="Feedback on the partner, if any. If none, leave empty."
    )


class ResearchAgent(BaseTool):
    name = "research_agent"
    description = "Agent(ResearchAgent) - The Agent that does the research. Always use this when you need to search for facts."
    args_schema: Type[BaseModel] = ResearchAgentInput
    tools = [talivy_tool]

    def _run(
        self,
        topic: str,
        feedback: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "Topic: {topic}\n\nFeedback: {feedback}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
        agent = create_openai_functions_agent(llm, self.tools, prompt)
        runnable = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        return runnable.invoke({"topic": topic, "feedback": feedback})["output"]
