from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools import google_search
from pydantic import BaseModel, conlist, Field
from typing import Optional

# --- Global Configuration ---
GEMINI_MODEL = "gemini-2.0-flash-001"

# --- Schemas ---
class UniversityInfo(BaseModel):
    """Schema for comprehensive institutional and admissions data."""
    college_name: str = Field(description="Full legal name of the college")
    college_type: str = Field(description="Institution type (e.g., 'Private', 'Public')")
    campus_setting: str = Field(description="General campus environment (e.g., 'Urban', 'Rural', 'Suburban')")
    campus_size: str = Field(description="Physical campus size and facilities overview (e.g., 'large, sprawling', 'compact urban')")
    graduation_rate: str = Field(description="Current graduation rate percentage (e.g., '85%')")
    acceptance_rate: str = Field(description="Most recent acceptance rate percentage (e.g., '15%')")
    website: str = Field(description="Official institution website URL")
    sat_range: str = Field(description="Typical SAT score range for admitted students (e.g., '1300-1500')")
    act_range: str = Field(description="Typical ACT score range for admitted students (e.g., '28-34')")
    regular_application_due: str = Field(description="Regular decision application deadline (e.g., 'January 1', 'Rolling')")
    tuition: str = Field(description="Detailed annual cost breakdown, ideally with in-state/out-of-state distinctions (e.g., 'in-state: $10,000, out-of-state: $25,000')")
    other_fees: str = Field(description="Comprehensive annual fee structure including room, board, books, and other mandatory fees (e.g., 'room: $8,000, board: $5,000, books: '1,200, fees: $500')")
    student_body: str = Field(description="Total undergraduate enrollment (e.g., '5,000 students')")
    race_ethnicity: str = Field(description="Detailed demographic breakdown by race and ethnicity (e.g., '40% White, 30% Asian, 20% Hispanic, 10% Black')")

class SportsInfo(BaseModel): 
    """Schema for detailed athletic program information."""
    ncaa_division: str = Field(description="NCAA division classification (e.g., 'NCAA Division I', 'NCAA Division III')")
    conference: str = Field(description="Athletic conference affiliation (e.g., 'Big East Conference', 'Ivy League')")
    facilities: str = Field(description="Descriptions and amenities of athletic facilities (e.g., 'Olympic-size pool, modern gymnasium, track stadium with seating for 5,000')")
    coaching_staff: str = Field(description="Current coaching staff information including names and titles (e.g., 'Head Coach: John Doe, Assistant Coach: Jane Smith')")
    team_roster: str = Field(description="Current team composition and size (e.g., '25 athletes, 10 freshmen, 8 sophomores, 7 juniors', 'Current roster of 30 players')")
    team_page_url: str = Field(description="Official team webpage URL (e.g., 'https://athletics.college.edu/sports/team-name')")
    #recruiting_standards: str = Field(description="recruiting standard or from https://runcruit.com/")

# Note: SearchResults schemas removed since we can't use output_schema with tools

# --- Search Agent ---
search_agent = LlmAgent(
    name="search_agent",
    model=GEMINI_MODEL,
    instruction="""
    You are a research assistant for student-athletes.
    Given a college or university name, use the google_search tool to find 1–3 official .edu web pages
    related to admissions, student body, and general university info.
    
    Prioritize links to:
    - Admissions pages with statistics
    - Fast Facts or at-a-glance pages
    - Common Data Set pages
    - About/overview pages
    - Cost and financial aid pages

    Search with specific queries like:
    - "[college name] admissions statistics site:edu"
    - "[college name] fast facts site:edu"
    - "[college name] common data set site:edu"

    Return the top 3-5 most relevant URLs as a simple list.
    """,
    description="Finds relevant official .edu URLs for college information.",
    tools=[google_search],
    output_key="search_results"
    # No output_schema when using tools
)

# --- University Info Extractor ---
university_info_extractor = LlmAgent(
    name="university_info_extractor",
    model=GEMINI_MODEL,
    instruction="""
    You are given URLs pointing to college websites. Use google_search to visit those URLs, 
    read the content, and extract relevant structured data.

    Extract the following information and format it as JSON matching the UniversityInfo schema:
    - college_name: Full official name
    - college_type: Public/Private
    - campus_setting: Urban/Rural/Suburban
    - campus_size: Description of campus size
    - graduation_rate: Percentage
    - acceptance_rate: Percentage
    - website: Official URL
    - sat_range: Score range (e.g., "1300-1500")
    - act_range: Score range (e.g., "28-34")
    - regular_application_due: Deadline
    - tuition: Cost breakdown
    - other_fees: Additional fees
    - student_body: Enrollment number
    - race_ethnicity: Demographic breakdown

    Use 'N/A' for any missing fields. Be as specific and accurate as possible.
    """,
    description="Extracts structured university data from page content.",
    tools=[google_search],
    output_key="university_info"
    # No output_schema when using tools
)

# --- Missing Info Resolver ---
missing_info_resolver = LlmAgent(
    name="missing_info_resolver",
    model=GEMINI_MODEL,
    instruction="""
    You are a follow-up researcher. You're given university information that may have 'N/A' values.
    Identify which fields are missing and perform targeted searches using the google_search tool.

    For missing fields, use specific search queries:
    - If 'sat_range' is missing: "[college name] SAT range admission requirements site:edu"
    - If 'campus_size' is missing: "[college name] campus size acres facilities site:edu"
    - If 'tuition' is missing: "[college name] tuition costs 2024-2025 site:edu"
    - If 'acceptance_rate' is missing: "[college name] acceptance rate admission statistics site:edu"
    
    Return updated information with any newly found data, keeping 'N/A' only for truly unavailable information.
    Focus on the most important missing fields first.
    """,
    description="Fills in missing university information through targeted searches.",
    tools=[google_search],
    output_key="followup_university_info"
)

# --- Sports Info Extractor ---
sports_info_extractor = LlmAgent(
    name="sports_info_extractor",
    model=GEMINI_MODEL,
    instruction="""
    You need to search for and extract athletics information for a specific college and sport.
    Use the google_search tool to find official athletics pages.

    Search for athletics information using queries like:
    - "[college name] athletics [sport] site:edu"
    - "[college name] [sport] roster coaching staff site:edu" 
    - "[college name] athletics facilities NCAA division site:edu"

    Extract and return information as JSON matching the SportsInfo schema:
    - ncaa_division: Division level
    - conference: Athletic conference
    - facilities: Athletic facilities description
    - coaching_staff: Coach names and titles
    - team_roster: Team size and composition
    - team_page_url: Official team page URL
    

    Use 'N/A' for any unknown or missing fields.
    """,
    description="Searches for and extracts athletics program data.",
    tools=[google_search],
    output_key="sports_info"
)

# --- Sports Info Filler ---
sports_info_filler = LlmAgent(
    name="sports_info_filler",
    model=GEMINI_MODEL,
    instruction="""
    You are given sports information that may have 'N/A' values.
    Use google_search to find missing information with targeted queries.

    For missing sports fields, search for:
    - NCAA division: "[college name] athletics NCAA division conference"
    - Coaching staff: "[college name] [sport] coaching staff directory"
    - Facilities: "[college name] athletics facilities tour"
    - Team roster: "[college name] [sport] roster team size"

    Update the sports information with any newly found data.
    Return complete information with 'N/A' only for truly unavailable data.
    """,
    description="Fills in missing sports data using targeted searches.",
    tools=[google_search],
    output_key="completed_sports_info"
)

# --- Report Writer Agent (No tools, can use output_schema) ---
report_writer_agent = LlmAgent(
    name="report_writer_agent",
    model=GEMINI_MODEL,
    instruction="""
    You are an admissions consultant writing a comprehensive report for student-athletes.
    You will receive:
    - University information (academic data)
    - Sports information (athletics data)
    - Any follow-up information found

    Write a comprehensive, readable report that includes:

    **ACADEMIC PROFILE:**
    - College name, type, location, and setting
    - Acceptance rate and test score ranges
    - Graduation rate and student body size
    - Tuition costs and fees
    - Student demographics

    **ATHLETIC PROGRAM:**
    - NCAA division and conference
    - Athletic facilities quality
    - Coaching staff details
    - Team roster information
    - Official team page

    **ASSESSMENT & RECOMMENDATIONS:**
    - Overall fit assessment for student-athletes
    - Specific action items and next steps
    - Contact information if available

    **DATA QUALITY NOTES:**
    - Clearly indicate any missing information (marked as 'N/A')
    - Note limitations in available data
    - Suggest additional research needed

    Present information clearly and professionally. Do not invent data - be transparent about gaps.
    """,
    description="Creates comprehensive student-athlete college report.",
    tools=[],  # No tools
    output_key="final_report"
    # Could add output_schema here since no tools, but keeping flexible for now
)

# --- Root Agent: Sequential Workflow ---
root_agent = SequentialAgent(
    name="college_info_pipeline",
    description="Comprehensive college research pipeline for student-athletes - gathers both academic and athletics information.",
    sub_agents=[
        search_agent,                # Find official college URLs
        university_info_extractor,   # Extract academic data
        missing_info_resolver,       # Fill gaps in university info
        sports_info_extractor,       # Search for and extract sports data
        sports_info_filler,          # Fill gaps in sports info
        report_writer_agent          # Generate final comprehensive report
    ]
)
