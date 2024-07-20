import os
import streamlit as st
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# Environment setup using Streamlit secrets
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
os.environ["MODEL_ENDPOINT"] = st.secrets["MODEL_ENDPOINT"]

# Custom AI model configuration using ChatOpenAI and OpenRouter
custom_model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url=os.environ["MODEL_ENDPOINT"]
)

# Tool for internet search
class InternetSearchTool(SerperDevTool):
    def __init__(self):
        super().__init__()

    def run(self, query):
        results = self.search(query, num_results=3)
        return results

# Create agents with custom AI model
keyword_extractor = Agent(
    role="Keyword Extractor",
    goal="Extract 10 most relevant keywords based on the research title",
    model=custom_model,
    verbose=True,
    memory=True,
    backstory="Expert in keyword extraction using natural language processing techniques."
)

internet_researcher = Agent(
    role="Internet Researcher",
    goal="Perform internet search using extracted keywords and extract top 3 relevant links for each keyword",
    model=custom_model,
    verbose=True,
    memory=True,
    backstory="Skilled in performing comprehensive internet searches and filtering relevant information.",
    tools=[InternetSearchTool()]
)

content_summarizer = Agent(
    role="Content Summarizer",
    goal="Summarize the content of each link in relation to the user-provided research title",
    model=custom_model,
    verbose=True,
    memory=True,
    backstory="Proficient in summarizing web content effectively and efficiently."
)

literature_reviewer = Agent(
    role="Literature Reviewer",
    goal="Perform a comprehensive literature review and write the literature review section",
    model=custom_model,
    verbose=True,
    memory=True,
    backstory="Expert in academic writing and literature review."
)

introduction_writer = Agent(
    role="Introduction Writer",
    goal="Write the introduction section for the medical research proposal",
    model=custom_model,
    verbose=True,
    memory=True,
    backstory="Experienced in writing engaging and informative introduction sections for research proposals."
)

methodology_writer = Agent(
    role="Methodology Writer",
    goal="Write the methodology section based on the literature review and introduction",
    model=custom_model,
    verbose=True,
    memory=True,
    backstory="Skilled in writing detailed and clear methodology sections for research proposals."
)

statistical_methods_writer = Agent(
    role="Statistical Methods Writer",
    goal="Write the statistical methods section, performing all necessary calculations and explanations",
    model=custom_model,
    verbose=True,
    memory=True,
    backstory="Proficient in statistical analysis and writing detailed statistical methods."
)

# Create tasks
extract_keywords_task = Task(
    description="Extract the 10 most relevant keywords based on the research title provided by the user.",
    expected_output="A list of 10 relevant keywords for performing a literature search.",
    agent=keyword_extractor
)

internet_search_task = Task(
    description="Perform an internet search using the extracted keywords, extract the top 3 most relevant links for each keyword, and ensure at least 30 unique links.",
    expected_output="A list of at least 30 unique links relevant to the research title.",
    agent=internet_researcher
)

summarize_content_task = Task(
    description="Summarize the content of each link in relation to the user-provided research title.",
    expected_output="A list of summarized contents for each link.",
    agent=content_summarizer
)

write_literature_review_task = Task(
    description="Perform a comprehensive literature review based on the user-provided research title and write the literature review section, including a numbered list of references.",
    expected_output="A complete literature review section with a numbered list of references.",
    agent=literature_reviewer
)

write_introduction_task = Task(
    description="Write the introduction section for the medical research proposal based on the research title and literature review, including a numbered list of references.",
    expected_output="A complete introduction section with a numbered list of references.",
    agent=introduction_writer
)

write_methodology_task = Task(
    description="Write the methodology section based on the literature review and introduction, including a numbered list of references.",
    expected_output="A complete methodology section with a numbered list of references.",
    agent=methodology_writer
)

write_statistical_methods_task = Task(
    description="Write the statistical methods section, performing all necessary calculations step-by-step and explaining each step in detail, including a numbered list of references.",
    expected_output="A complete statistical methods section with a numbered list of references.",
    agent=statistical_methods_writer
)

# Create crew
crew = Crew(
    agents=[
        keyword_extractor,
        internet_researcher,
        content_summarizer,
        literature_reviewer,
        introduction_writer,
        methodology_writer,
        statistical_methods_writer
    ],
    tasks=[
        extract_keywords_task,
        internet_search_task,
        summarize_content_task,
        write_literature_review_task,
        write_introduction_task,
        write_methodology_task,
        write_statistical_methods_task
    ],
    process=Process.sequential
)

def kickoff_crew(research_title):
    return crew.kickoff(inputs={'research_title': research_title})

# Streamlit App
st.title("Medical Research Proposal Generator")

research_title = st.text_input("Enter your research title:")

if st.button("Generate Proposal"):
    if research_title:
        result = kickoff_crew(research_title)
        st.write(result)
    else:
        st.error("Please enter a research title")
