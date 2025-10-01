import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI as Gemini
from langchain.tools import tool

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = Gemini(
    model="gemini-2.5-flash",
    temperature=0.0
)

@tool
def financial_report(company_name: str, revenue: float, expense: float) -> str:
    """Generates a financial report summary for a given company based on its revenue and expenses."""
    profit = revenue - expense
    profit_margin = (profit / revenue) * 100 if revenue != 0 else 0
    report = (
        f"Financial Report for {company_name}:\n"
        f"Total Revenue: ${revenue:,.2f}\n"
        f"Total Expenses: ${expense:,.2f}\n"
        f"Net Profit: ${profit:,.2f}\n"
        f"Profit Margin: {profit_margin:.2f}%\n"
    )
    return report
    
tools = [financial_report]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant, that is helping a user in Financial Reporting. When he shares the data, use the tools to generate the report."),
    ("placeholder", "{messages}")
])

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt
)

results = agent.invoke({"messages": [{"role": "user", "content": "Generate a financial report for 'Tech Innovators Inc.' with a revenue of $5,000,000 and expenses of $3,500,000."}]})
print(results["messages"][-1].content)