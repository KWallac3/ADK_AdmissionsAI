from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools import google_search
from pydantic import BaseModel, conlist

# --- Global Configuration ---
GEMINI_MODEL = "gemini-2.0-flash-001"

# --- Schemas ---
class UniversityInfo(BaseModel):
    college_name: str
    college_type: str
    campus_setting: str
    campus_size: str
    graduation_rate: str
    acceptance_rate: str
    website: str
    sat_range: str
    act_range: str
    regular_application_due: str
    tuition: str
    other_fees: str
    student_body: str
    race_ethnicity: str

class SearchResults(BaseModel):
    urls: conlist(str, min_length=1, max_length=5)

# --- Search Agent ---
search_agent = LlmAgent(
    name="search_agent",
    model=GEMINI_MODEL,
    instruction="""
    You are a research assistant for student-athletes.
    Given a college or university name, use the google_search tool to find 1–3 official .edu web pages
    related to admissions, student body, and general university info.
    Prioritize links to:
    - Admissions
    - Fast Facts
    - Common Data Set
    - About pages

    Output just the top 1–3 relevant URLs as a list.
    """,
    description="Finds relevant official .edu URLs for college information.",
    tools=[google_search],
    output_key="search_results",
    #output_schema=SearchResults
)

# --- University Info Extractor ---
university_info_extractor = LlmAgent(
    name="university_info_extractor",
    model=GEMINI_MODEL,
    instruction="""
    You are given 1–3 URLs pointing to college websites. Visit those URLs, read the content, and extract
    relevant structured data. Use 'N/A' for missing fields.

    Return data in the format defined by the UniversityInfo schema.
    """,
    description="Extracts structured university data from page content.",
    tools=[google_search],  # Temporarily reuse google_search to simulate browsing for now
    output_key="university_info",
    #output_schema=UniversityInfo
)

# --- Root Agent: Executes Workflow ---
root_agent = SequentialAgent(
    name="university_info_pipeline",
    description="Searches for and extracts university information.",
    sub_agents=[search_agent, university_info_extractor]
    
)