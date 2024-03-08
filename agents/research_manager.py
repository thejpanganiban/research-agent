from enum import Enum
from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.output_parser import StrOutputParser
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT = """
You are a research manager.
Your tasks are to strategize, review the output of the research_agent, and decide on the next steps.

If not output is given, then you'll need to strategize.
If the output is satisfactory and complete, return the output.
Otherwise, ask the research_agent to continue their research.
"""


class ResearchManagerTask(str, Enum):
    STRATEGIZE = "Strategize"
    REVIEW = "Review"
    DECIDE = "Decide"


class ResearchManagerInput(BaseModel):
    task: ResearchManagerTask = Field(description="Task to be done.")
    topic: str = Field(description="The topic of research.")
    output: Optional[str] = Field(
        description="Full output provided by the research_agent."
    )


class ResearchManagerAgent(BaseTool):
    name = "research_manager"
    description = "Agent(ResearchManager) - Strategizes, reviews the output of the research_agent, and decides on the next steps."
    args_schema: Type[BaseModel] = ResearchManagerInput

    def _run(
        self,
        task: str,
        topic: str,
        output: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Task: {task}\n\nOutput: {output}"),
            ]
        )
        prompt = prompt.partial(output=output)
        llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
        output_parser = StrOutputParser()
        runnable = prompt | llm | output_parser
        return runnable.invoke({"task": task, "topic": topic})
