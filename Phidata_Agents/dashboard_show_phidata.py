import openai
from dotenv import load_dotenv
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
import os
import phi
from phi.playground import Playground,serve_playground_app
from phi.model.groq import Groq



# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PHIDATA_API_KEY = os.getenv("PHIDATA_API_KEY")



# web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for latest information",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)


# finance data agent
finance_data_agent = Agent(
    name="Finance Data Agent",
    role="Collect financial data from Yahoo Finance",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[
        YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True,company_news=True),
        ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

app = Playground(agents=[web_search_agent,finance_data_agent]).get_app()


if __name__ == "__main__":
    serve_playground_app("dashboard_show_phidata:app",reload=True)