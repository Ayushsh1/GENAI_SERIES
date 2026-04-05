from dotenv import load_dotenv  
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm = ChatGroq(model="openai/gpt-oss-120b")
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model=llm,
    tools=[search.run],
    checkpointer=MemorySaver(),
    system_prompt="You are a agent who can search the web using google search"
)

quit_commands = {"exit", "quit", "bye", "goodbye", "see you later", "stop", "q"}

while True:
    query = input("User:")
    if query.strip().lower() in quit_commands:
        print("Agent: Goodbye!")
        break
    res = agent.invoke(
        {"messages":[{"role":"user","content":query}]},
        {"configurable":{"thread_id":"google_agent_thread"}}
        )
    print("Agent:" , res['messages'][-1].content)
