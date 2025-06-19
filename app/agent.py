import asyncio
from typing import List, Optional

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools import google_search
from pydantic import BaseModel, Field

# --- Student Profile Collection ---
# This profile is built directly into the code for simplicity.
# CHANGE these values to simulate different students.
HARDCODED_STUDENT_PROFILE = {
    "full_name": "Alex Morgan",
    "grade_level": "11th", # Junior year
    "primary_sport": "Track and Field",
    "events": ["55m","100m", "200m", "Long Jump"],  # Student's specific events
    "personal_records": {
        "55M": "7.6",
        "100m": "12.45",
        "200m": "25.80", 
        "Long_Jump": "5.20m"
    },
    "gender": "Female",
    "unweighted_gpa": 3.75,
    "gpa_scale": 4.0,
    "sat_score": 1380, # Example SAT score
    "act_score": "N/A", # Example ACT score, N/A if not taken
    "high_school": "My High School",
    # Note: target_college will be provided by user input during runtime
}

# Helper function to convert dict to formatted string for LLM instructions
def profile_to_string(profile: dict) -> str:
    """Converts a student profile dictionary into a readable string format for the AI."""
    s = []
    for k, v in profile.items():
        s.append(f"- {k.replace('_', ' ').title()}: {v}")
    return "\n".join(s)

# Format the hardcoded profile for agent instructions
PROFILE_FOR_AGENT = profile_to_string(HARDCODED_STUDENT_PROFILE)
print("PROFILE_FOR_AGENT content:")
print(repr(PROFILE_FOR_AGENT))

# --- Global Configuration ---
GEMINI_MODEL = "gemini-2.0-flash-001"

# --- Schemas ---
# Original UniversityInfo schema 
class UniversityInfo(BaseModel):
    college_name: str = Field(description="Full legal name of the college")
    college_type: str = Field(description="Institution type (e.g., 'Private', 'Public')")
    campus_setting: str = Field(description="General campus environment (e.g., 'Urban', 'Rural', 'Suburban')")
    campus_size: str = Field(description="Physical campus size and facilities overview")
    graduation_rate: str = Field(description="Current graduation rate percentage (e.g., '85%')")
    acceptance_rate: str = Field(description="Most recent acceptance rate percentage (e.g., '15%')")
    website: str = Field(description="Official institution website URL")
    sat_range: str = Field(description="Typical SAT score range for admitted students")
    act_range: str = Field(description="Typical ACT score range for admitted students")
    regular_application_due: str = Field(description="Regular decision application deadline")
    tuition: str = Field(description="Detailed annual cost breakdown")
    other_fees: str = Field(description="Comprehensive annual fee structure")
    student_body: str = Field(description="Total undergraduate enrollment")
    race_ethnicity: str = Field(description="Demographic breakdown by race and ethnicity")

# NEW: Pydantic model for individual current performers 
class CurrentPerformer(BaseModel):
    name: str = Field(description="Name of the student-athlete")
    mark: str = Field(description="Their best mark or time for the event")
    ranking: Optional[str] = Field(description="Their ranking, if available (e.g., '1st in conference', 'Top 25 NCAA')", default="N/A")
    class_year: Optional[str] = Field(description="The athlete's academic class (e.g., 'Freshman', 'Sophomore', 'JR-3')", default="N/A") # NEW FIELD
    event_category: str = Field(description="The category of the event (e.g., 'sprint', 'distance', 'long jump')")
    sport: str = Field(description="The specific sport (e.g., 'Track and Field', 'Swimming')")

# NEW: Pydantic model for a single conference championship result 
class ChampionshipResult(BaseModel):
    event_name: str = Field(description="Name of the event (e.g., 'Men's 100m Dash', 'Women's 200 Yard Freestyle')")
    gender: str = Field(description="Gender for the event ('Men' or 'Women')")
    finalists: List[dict] = Field(description="List of dictionaries, each containing 'name', 'school_team', and 'mark_time' for finalists.")
    
# Updated SportsInfo schema to include new sports performance data 
class SportsInfo(BaseModel):
    ncaa_division: str = Field(description="NCAA division classification")
    conference: str = Field(description="Athletic conference affiliation")
    facilities: str = Field(description="Descriptions and amenities of athletic facilities")
    coaching_staff: str = Field(description="Current coaching staff information")
    team_roster: str = Field(description="Current team composition and size")
    team_page_url: str = Field(description="Official team webpage URL")
    current_performers: List[CurrentPerformer] = Field(description="List of best current performers with their names, marks, and rankings for the 2024-2025 season in specified event categories.")
    championship_results: List[ChampionshipResult] = Field(description="List of most recent conference championship final results, including event names, gender, finalists, and their marks/times.")
    recruiting_standards: str = Field(description="Published recruiting standards or typical performance levels for recruited athletes")
    walk_on_standards: str = Field(description="Walk-on tryout standards or minimum performance levels")
    conference_qualifying_marks: str = Field(description="Conference championship qualifying standards for relevant events")
    team_depth_analysis: str = Field(description="Analysis of roster depth and competition for spots in relevant events")



# --- Agents ---

# 1. University Info Search Agent 
search_agent = LlmAgent(
    name="search_agent",
    model=GEMINI_MODEL,
    instruction="""
    You are a research assistant for student-athletes.
    Your task is to find official .edu webpages related to the student's target college, including admissions stats, cost, fast facts, common data set, and about pages.

Use google_search tool to find 1–3 relevant URLs with queries such as:
- "[college name] admissions statistics site:.edu"
- "[college name] fast facts site:.edu"
- "[college name] common data set site:.edu"
- "[college name] tuition site:.edu"

Return the top 3–5 most relevant URLs as a simple list.
""",
    description="Finds relevant official .edu URLs for college information.",
    tools=[google_search],
    output_key="university_search_results"
)

# 2. University Info Extraction Agent
university_info_extractor = LlmAgent(
    name="university_info_extractor",
    model=GEMINI_MODEL,
    instruction=f"""
    === STUDENT PROFILE ===
    {PROFILE_FOR_AGENT}
    === END STUDENT PROFILE ===

    You are a specialized sports data researcher focusing on recruitment fit analysis.

    CRITICAL TASK: You must collect specific performance data to assess whether this student-athlete could compete at this college level.

    For Track and Field, search for and extract:
    1. **Current team performance standards:**
    - What times/marks are the current athletes posting in the student's events?
    - What are the team's recruiting standards or walk-on standards?
    - Conference qualifying marks for the student's events

    2. **Recent recruiting classes:**
    - What caliber of athletes has this program recruited recently?
    - What were their high school PRs when recruited?

    3. **Team depth and opportunities:**
    - How many athletes compete in the student's events?
    - Are there scholarship spots typically available?
    - What's the competition level for roster spots?

    Use queries like:
    - "[college name] track field recruiting standards 2024-2025"
    - "[college name] track field roster [student's events] times marks"
    - "[conference name] qualifying standards track field"
    - "[college name] track field walk-on tryout standards"

    IMPORTANT: Include specific performance benchmarks in your extracted data so we can compare against the student's abilities.
 

    You are given URLs pointing to official college websites. Visit those URLs and extract relevant data matching the UniversityInfo schema below.
    
    IMPORTANT: After extracting the standard data, you MUST also analyze and comment on fit:
    - Compare the student's GPA to typical admitted students
    - Compare the student's test scores to the college's ranges
    - Note any information relevant to their sport
    - Include these comparisons in the appropriate fields (don't create new fields, but include analysis within existing fields)


    Return data in JSON format strictly adhering to the UniversityInfo schema. Use 'N/A' for any missing fields.

UniversityInfo fields:
- college_name, college_type, campus_setting, campus_size, graduation_rate, acceptance_rate,
  website, sat_range, act_range, regular_application_due, tuition, other_fees, student_body, race_ethnicity
""",
    description="Extracts structured university data from page content.",
    tools=[google_search], 
    output_key="university_info"
)

# 3. Sports Info Search Agent 
sports_info_extractor = LlmAgent(
    name="sports_info_extractor",
    model=GEMINI_MODEL,
    instruction=f"""
    === STUDENT PROFILE ===
    {PROFILE_FOR_AGENT}
    === END STUDENT PROFILE ===

    You MUST consider this student's specific characteristics when extracting data:
    - Focus on admission requirements relevant to their GPA ({HARDCODED_STUDENT_PROFILE['unweighted_gpa']}) and test scores
    - Prioritize cost information since financial considerations are important
    - Look for information about {HARDCODED_STUDENT_PROFILE['primary_sport']} programs if available

    You are a specialized sports data researcher. Your task is to find official athletics information for a given college and sport,
    specifically focusing on current season top performers and recent conference championship results.
    Student Profile:
    {PROFILE_FOR_AGENT}

    Search for current  {HARDCODED_STUDENT_PROFILE['gender']} {HARDCODED_STUDENT_PROFILE['primary_sport']} roster and times for events like sprints, distance, throws, jumps - whatever the student's focus area is

    When searching, prioritize authoritative and up-to-date platforms.
    Dynamically select your search strategy based on the sport:

    If the sport is 'Track and Field':
    - Prioritize TFRRS (Track & Field Results Reporting System) and official conference athletics websites.
    - Use queries like:
        - "TFRRS [college name] track and field current season results"
        - "NCAA Track and Field Statistics and Records official website"
        - "[conference name] Track and Field Championship results 2024-2025 [gender]"
        - "[college name] track and field top performers 2024-2025 [event categories]"
        - "official website [conference name] athletics results"

    If the sport is 'Swimming':
    - Prioritize USA Swimming's SWIMS database and official conference athletics websites.
    - Use queries like:
        - "SWIMS 3.0 [college name] swimming current season"
        - "NCAA Swimming and Diving Statistics and Records official website"
        - "[conference name] Swimming and Diving Championship results 2024-2025 [gender]"
        - "[college name] swimming top performers 2024-2025 [event categories]"
        - "official website [conference name] swimming and diving results"

    Extract the following information for the 2024-2025 season:
    1.  **Top current performers:** For the specified college and within its conference, in event categories like 'sprint', 'distance', 'long jump' (adjust based on user input). Include their names, marks, rankings, and academic class.
    2.  **Most recent conference championship final results:** For the specified sport, broken down by gender and event. Include event names, a list of all finalists (including their names and school/team), and their marks/times.

    Return the extracted data in JSON format matching the SportsInfo schema below.
    Use 'N/A' for any missing fields if information cannot be found.
    Be precise in extracting names, marks/times, rankings, and academic class as they appear on official sources.

    SportsInfo fields:
    - ncaa_division, conference, facilities, coaching_staff, team_roster, team_page_url, current_performers, championship_results
    """,
    description="Searches for and extracts athletics program data, including current performers and championship results.",
    tools=[google_search],
    output_key="sports_info_extracted"
)

# 4. Missing Info Resolver Agent (single for university and sports) 
missing_info_resolver = LlmAgent(
    name="missing_info_resolver",
    model=GEMINI_MODEL,
    instruction="""
    You are a meticulous data consolidator. You are given partial JSON data for university and sports information, some fields marked 'N/A'.

Your tasks:
-Review the provided 'university_info_extracted' and 'sports_info_extracted' objects
- Identify all fields that are currently 'N/A' or clearly missing values.
- For each missing field, generate highly targeted google_search queries to fill missing fields, focusing on official .edu sources where possible.
- Examples:
  - "[college_name] tuition fees site:.edu"
  - "[college_name] {HARDCODED_STUDENT_PROFILE['primary_sport]} coaching staff site:.edu"
- Attempt to fill in as many missing fields as possible; keep 'N/A' only if data cannot be found.
- **Crucially, Rcombine the data from both university and sports information into a single JSON 
object that conforms to the combined fields of both UniversityInfo and SportsInfo schemas.** Ensure all fields from both schemas are present in the final output.
Return the updated, combined data as a single JSON object.
""",
    description="Fills missing university and sports info fields by targeted searches.",
    tools=[google_search],
    output_key="completed_info"
)

# 5. Report Writer Agent 
report_writer_agent = LlmAgent(
    name="report_writer_agent",
    model=GEMINI_MODEL,
    instruction=f"""
    === STUDENT PROFILE ===
    {PROFILE_FOR_AGENT}
    === END STUDENT PROFILE ===

CRITICAL: You must provide a detailed athletic fit assessment that includes:

1. **Performance Gap Analysis:**
   - Compare the student's current times/marks to the team's current performers
   - Compare to conference qualifying standards
   - Assess how much improvement would be needed to contribute

2. **Recruitment Viability:**
   - Based on team recruiting standards, assess recruitment likelihood
   - Evaluate walk-on opportunities if applicable
   - Consider the student's improvement trajectory and potential

3. **Competitive Opportunity:**
   - Analyze roster depth in the student's events
   - Assess realistic timeline for contributing to the team
   - Consider scholarship availability

4. **Development Potential:**
   - Factor in the student's current level vs. what's typical for recruits
   - Consider coaching program's track record of athlete development

Make this assessment specific and realistic - don't just say "great program" but actually analyze whether this student could realistically compete there.
    
    You are a professional college advisor specializing in student-athlete recruitment.
    Write a comprehensive, professional, and empathetic report for the student-athlete based on the university and athletics data provided.
    
    The report should include the following sections:
    - Academic profile summary
    - Athletics program summary
    - Fit assessment and recommendations
    - Note any missing data clearly as 'N/A'

    **Important Content Rules:*
    -Do not fabricate data.
    -Maintain Professional advisory tone.
    -Do not make subjective judgements or provide personal opions on demographics. Simple state the facts.
""",
    description="Creates comprehensive student-athlete college report.",
    tools=[],
    output_key="final_report"
)

# --- Root Sequential Agent  ---
root_agent = SequentialAgent(
    name="college_info_pipeline",
    description="Comprehensive college research pipeline for student-athletes.",
    sub_agents=[
        search_agent,
        university_info_extractor,
        sports_info_extractor,
        missing_info_resolver,  # Single gap-filler for both university & sports
        report_writer_agent
    ]
)

 