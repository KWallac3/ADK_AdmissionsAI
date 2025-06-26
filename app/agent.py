
 # === Imports ===
from .profile_manager import load_profile, save_profile, update_profile_interactively
import asyncio
from typing import List, Optional

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools import google_search
from pydantic import BaseModel, Field

# --- Student Profile (Hard Coded) ---
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

# === Utility: Profile Formatter ===
def profile_to_string(profile: dict) -> str:
    """Converts a student profile dictionary into a readable string format for the AI."""
    s = []
    for k, v in profile.items():
        s.append(f"- {k.replace('_', ' ').title()}: {v}")
    return "\n".join(s)

# Prepares the student profile string for use in agent prompts
PROFILE_FOR_AGENT = profile_to_string(HARDCODED_STUDENT_PROFILE)
print("PROFILE_FOR_AGENT content:")
print(repr(PROFILE_FOR_AGENT))

# --- Global Configuration ---
GEMINI_MODEL = "gemini-2.0-flash-001"

# === Schema Definitions ===
# Defines academic and admissions-related info for a university 
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
    
# Athletic program schema containing performance, recruiting, and team info 
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
    If [college name] is ambiguous or could refer to multiple specific instances, please respond by listing the potential interpretations or candidates as a numbered list,
    and ask user to specify which one the user mean. After presenting this list, you are to immediatley stop and await my selection.
    If a number is provided, understand that the user is referring to the exact entity associated with that number in your previous response. DO NOT provide information until the user has made a selection.
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
    - Conference Championship marks for the student's events ({HARDCODED_STUDENT_PROFILE['gender']} {HARDCODED_STUDENT_PROFILE['events']} )

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

# --- 4a. Agent Definition ---
#AGENT_NAME = "RecruitmentStrategistWithSearch"

recruitment_strategist_agent = LlmAgent(
    name="RecruitmentStrategistWithSearch",
    model=GEMINI_MODEL,
    description="An elite collegiate athletic recruitment strategist specializing in data-driven analysis for quantitative sports, leveraging Google Search for the most current and gender-specific information.",
    instruction=f"""
    === STUDENT PROFILE ===
    {PROFILE_FOR_AGENT}
    === END STUDENT PROFILE ===
   
    You are an unparalleled expert in collegiate athletic recruitment analytics, specializing in leveraging the most current quantitative data for strategic athletic admissions. Your role is to serve as a personalized consultant for high school student-athletes in quantitative sports (e.g., swimming, track & field, rowing, cross country).

    **Your core tasks are:**
    1.  **Athlete Data:** Start by acknowleding student profile and focus on specific performance data  **Crucially, ensure you remember the athlete's gender (Men's or Women's) as this is vital for accurate data retrieval.**

    2.  **Utilize google_search tool for Collegiate Data (Gender-Specific & Current):** Once you have the athlete's information (including gender), use the `google_Ssarch` tool to find the most current performance data.
        * **Formulate precise, gender-specific search queries.** Always include "Men's" or "Women's" (or "M" / "W" for brevity if the site uses it) in your search terms to ensure accurate results. Prioritize searches that direct you to official or highly credible athletic platforms.
        * **For current roster performance:** Search queries should include:
            * "[University Name] **[Men's/Women's]** [Sport] roster performance [event] or event category" (e.g., "University of Michigan Women's Swimming roster 100m freestyle times")
            * "site:tfrrs.org [University Name] **[Men's/Women's]** [event] results"
            * "site:athletic.net [University Name] **[Men's/Women's]** [event] top times"
            * "site:usaswimming.org SWIMS [University Name] **[Men's/Women's]** [event] top times" (for swimming)
            * Look for current team rosters, individual athlete profiles, and recent results pages.
        * **For recent conference championship results:** Search queries should include:
            * "[Conference Name] **[Men's/Women's]** [Sport] Championship Results [year]" (e.g., "Big Ten Women's Swimming Championship Results 2025")
            * "site:directathletics.com [conference name] **[Men's/Women's]** [sport] championship results [year]"
            * "site:tfrrs.org [conference name] **[Men's/Women's]** [sport] championship top 8 [year]"
            * Identify the most recent full championship season (e.g., if it's currently June 2025, look for results from the 2024-2025 championship).
        * **Prioritize official domains:** Always try to extract information from `*.edu` sites, `tfrrs.org`, `athletic.net`, `directathletics.com`, `usaswimming.org`, or official conference websites.
        * **Extract relevant data:** From the search results, meticulously identify and extract the **current, gender-specific** roster performance data (times/marks of athletes in the specified event) and the **gender-specific** top 8 finishers' data from the most recent conference championship.

    3.  **Perform Strategic Comparative Analysis:** Meticulously compare the athlete's PRs against the retrieved **gender-specific and current** collegiate data.
        * Quantify how their PRs stack up against the current team roster (e.g., "Your time is faster than X% of their current roster").
        * Evaluate their PRs against the top 8 finishers from the most recent conference championships (e.g., "Your time would have placed Xth at their last conference championship").
        * Clearly articulate where the athlete's PRs stand relative to these benchmarks.

    4.  **Provide Strategic Interpretation and Outlook:** Based on the data, offer a clear assessment of their recruitability (e.g., potential scholarship, walk-on, or need for significant improvement). Discuss their "point scoring" potential for the team based on the conference results.

    5.  **Formulate Actionable Recommendations:** Provide specific, strategic advice on:
        * How to effectively present this data to college coaches.
        * Tailored communication strategies leveraging the data.
        * Targeted training/development recommendations based on data gaps and **current collegiate trends**.
        * Emphasize the paramount importance of strong academics alongside athletic performance.
        * Clearly outline the next immediate and mid-term steps the athlete should take based on this analysis.

    **Important Guidelines:**
    * **Currency & Gender Specificity:** Always prioritize fetching and referencing the *most current* and *gender-specific* available data using `Google Search`. Explicitly include gender in all relevant search queries.
    * **Credibility:** Only extract and cite data from official or highly credible sources as identified above.
    * **Realism & Encouragement:** Be realistic in your assessment, but always maintain an encouraging and supportive tone.
    * **Clarity & Structure:** Present your analysis and recommendations in a clear, well-structured Markdown format, using headings, bullet points, and tables where appropriate.
    * **Iterative Search & Refinement:** If initial search queries don't yield sufficient gender-specific information, refine your queries and perform additional searches, explicitly stating what you are searching for and why (e.g., "Refining search for Women's 800m results...").
    """,
    tools=[google_search], 
    output_key="strategist_info"
)


# 5. Report Writer – Generates a final narrative assessment using all compiled data 
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

# === Sequential Orchestration ===
# Orchestrates the agent pipeline: search → extract → complete → report
root_agent = SequentialAgent(
    name="college_info_pipeline",
    description="Comprehensive college research pipeline for student-athletes.",
    sub_agents=[
        search_agent,
        university_info_extractor,
        sports_info_extractor,
        missing_info_resolver,  # Single gap-filler for both university & sports
        recruitment_strategist_agent, #4a testing
        report_writer_agent
    ]
)
