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

class SportsInfo(BaseModel): # added
    ncaa_division: str
    conference: str
    facilities: str
    coaching_staff: str
    team_roster: str
    team_page_url: str


class SearchResults(BaseModel):
    urls: conlist(str, min_length=1, max_length=5)

class SportsSearchResults(BaseModel):
    urls: conlist(str, min_length=1, max_length=5) # added

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

#-------Missing Info Resolver --------
missing_info_resolver = LlmAgent(
    name="missing_info_resolver",
    model=GEMINI_MODEL,
    instruction="""
    You are a follow-up researcher. You're given a partially complete UniversityInfo object.
    Identify which fields are missing (marked 'N/A') and perform targeted searches using the google_search tool.

    For example:
    - If 'sat_range' is missing, search: "SAT range [college name] site:.edu"
    - If 'campus_size' is missing, search: "[college name] campus size site:.edu"
    
    Your goal is to gather useful snippets or URLs to help fill in the gaps.
    """,
    description="Finds data for missing university fields via follow-up search.",
    tools=[google_search],
    output_key="followup_snippets"
)


# --- Sport Info Extractor ---
sports_info_extractor = LlmAgent(
    name="sports_info_extractor",
    model=GEMINI_MODEL,
    instruction="""
    You are given links or content about a college's athletics program.
    Extract information for the following fields:
    - NCAA division
    - Conference
    - Facilities
    - Coaching staff
    - Team roster
    - Team page URL

    Use 'N/A' for any unknown or missing field.
    """,
    description="Extracts structured athletics program data.",
    tools=[google_search],  # reuse google_search to simulate browsing
    output_key="sports_info"
)
sports_info_filler = LlmAgent(
    name="sports_info_filler",
    model=GEMINI_MODEL,
    instruction="""
    You are a fact-checking agent that improves sports information.
    Given sports data with 'N/A' values and URLs related to a specific college team,
    issue targeted follow-up searches using google_search to fill in the missing fields.

    For example:
    - "UCLA womens track and field NCAA division"
    - "Yale mens soccer coaching staff"
    - "University of Florida athletics facilities"

    Update only the missing fields. If nothing useful is found, leave the value as 'N/A'.
    Return a complete and cleaned SportsInfo schema.
    """,
    description="Fills in missing sports data using targeted queries.",
    tools=[google_search],
    output_key="completed_sports_info"
)


# --- Report Writer Agent ---
report_writer_agent = LlmAgent(
    name="report_writer_agent",
    model=GEMINI_MODEL,
   instruction="""
   You are an assistant writing reports for student-athletes.
    You will be given two inputs:
    - UniversityInfo: structured information about the college
    - SportsInfo: structured info about the team (e.g., women's track and field)

    Write a readable summary that combines both academic and athletics details.
    Mention:
    - College name, setting, size, graduation rate, tuition, student body, etc.
    - Team division, conference, coaching staff, facilities, and page URL.
    Clearly indicate if any data is missing (i.e., 'N/A').
    
    """,
    description="Final step: converts content into structured university info.",
    tools=[],  
    output_key="final_report",
    output_schema=None,
)

# --- Root Agent: Executes Workflow ---
root_agent = SequentialAgent(
    name="college_info_pipeline",
    description="Gathers both academic and athletics info.",
    sub_agents=[
        search_agent,
        university_info_extractor,
        missing_info_resolver,
        sports_info_extractor,
        sports_info_filler,              
        report_writer_agent
    ]
)
