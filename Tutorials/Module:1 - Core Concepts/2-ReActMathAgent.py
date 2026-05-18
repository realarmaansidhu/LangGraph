import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate  
from langchain_google_genai import ChatGoogleGenerativeAI as Gemini
from langchain_community.agent_toolkits.load_tools import load_tools

load_dotenv()  # take environment variables from .env file
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = Gemini(model="gemini-2.5-flash", temperature=0.0)

tools = load_tools(["llm-math"], llm=llm)

llm = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can reason and use tools to answer questions."),
    ("placeholder", "{messages}")
])

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt,
    )
    
result = agent.invoke({"messages": [{"role": "user", "content": "What's the result of 12345 * 6789?"}]})
print(result['messages'][-1].content)

graph = agent.get_graph()
graph.print_ascii()