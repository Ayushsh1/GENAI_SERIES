from dotenv import load_dotenv  
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent

llm = ChatGroq(model="openai/gpt-oss-120b")
search = GoogleSerperAPIWrapper()

agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="You are a agent who can search the web using google search"
)

while True:
    query = input("User:")
    if query.lower() == "exit":
        print("Agent: Goodbye!")
        break
    res = agent.invoke({"messages":[{"role":"user","content":query}]})
    print("Agent:" , res['messages'][-1].content)
