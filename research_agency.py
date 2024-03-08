from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from agents.research_agent import ResearchAgent
from agents.research_manager import ResearchManagerAgent

SYSTEM_PROMPT = """
You are Jared, a research director.
Your task is to complete the research on a topic until it is satisfactory and complete.

Return the full long-form research, including citations.

Always use the agents.
Iterate as many times as needed (up to 15 times).
"""


tools = [
    ResearchManagerAgent(),
    ResearchAgent(),
]
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="metadata", optional=True),
    ]
)
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
runnable = AgentExecutor(agent=agent, tools=tools, verbose=True)

__all__ = ["runnable"]


if __name__ == "__main__":
    output = runnable.invoke(
        {"input": "Engineering Management Approaches (for Software Engineers)"}
    )
    print(output["output"])
