import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI as Gemini
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = Gemini(
    model="gemini-2.5-flash",
    temperature=0.0
)

# --- INITIALIZE THE TOOL DIRECTLY ---
# 1. Create the API wrapper
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
# 2. Create the tool instance
tools = [
    WikipediaQueryRun(api_wrapper=api_wrapper)
]

# Bind the tools to the LLM
llm = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can reason and use tools to answer questions."),
    ("placeholder", "{messages}")
])

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt
)

# Store the result of the agent invocation
result = agent.invoke({"messages": [{"role": "user", "content": "Who is Mark Carney, give a short summary of his career"}]})

# Print the final message from the 'result' object
print(result['messages'][-1].content)