# LangGraph code to create a ReAct agent that uses a language model to reason and act, with a weaather API tool.
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate  
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI as Gemini
load_dotenv(".env")  # Load environment variables from .env file
GEMINI_API_TOKEN = os.getenv("GOOGLE_API_KEY")

def get_weather(city: str) -> str:
    '''Get the current weather for a given city.'''
    return f"The weather in {city} is sunny with a high of 75°F."

llm = Gemini(
    model="gemini-2.5-flash",
    temperature=0.0
).bind_tools([get_weather])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can reason and use tools to answer questions."),
    ("placeholder", "{messages}")
])

agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt=prompt,
    ) 

result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Toronto?"}]})

print(result['messages'][-1].content)