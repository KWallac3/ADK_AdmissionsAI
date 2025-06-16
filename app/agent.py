# agent.py
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools import google_search
from pydantic import BaseModel, Field
from typing import Optional
import asyncio

# --- Student Profile Collection (Option 1) ---
def collect_student_profile_mvp():
    print("\n--- Student-Athlete Pre-Screening & Analysis App ---")
    print("Hello Student-Athlete! Let's get started with your preliminary college analysis.")
    print("Please provide the following information:")

    student_profile = {}

    def get_mandatory_string_input(prompt_text):
        while True:
            value = input(prompt_text).strip()
            if value:
                return value
            else:
                print("This field cannot be empty. Please try again.")

    student_profile["full_name"] = get_mandatory_string_input("1. What is your full name? ")
    student_profile["grade_level"] = get_mandatory_string_input("2. What grade level are you currently in? (e.g., 10th, Junior) ")
    student_profile["primary_sport"] = "Track and Field"
    print(f"3. Primary Sport: {student_profile['primary_sport']} (Autofilled)")

    while True:
        gender_input = input("4. What is your gender? (Please enter 'Male' or 'Female') ").strip().lower()
        if gender_input in ["male", "female"]:
            student_profile["gender"] = gender_input.capitalize()
            break
        else:
            print("Invalid input. Please enter 'Male' or 'Female'.")

    while True:
        try:
            gpa_input = float(input("5. What is your Unweighted GPA? (e.g., 3.8) "))
            if 0.0 <= gpa_input <= 5.0:
                student_profile["unweighted_gpa"] = gpa_input
                break
            else:
                print("GPA must be between 0.0 and 5.0. Please re-enter.")
        except ValueError:
            print("Invalid input. GPA must be a number (e.g., 3.5).")

    while True:
        try:
            gpa_scale_input = float(input("6. On what scale is your Unweighted GPA? (e.g., 4.0, 5.0) "))
            if gpa_scale_input > 0:
                student_profile["gpa_scale"] = gpa_scale_input
                break
            else:
                print("GPA scale must be a positive number. Please re-enter.")
        except ValueError:
            print("Invalid input. GPA scale must be a number (e.g., 4.0).")

    sat_choice = input("7. Do you have an SAT score? (Enter 'yes' or 'no') ").strip().lower()
    if sat_choice == 'yes':
        while True:
            sat_score_str = input("   Please enter your SAT score (e.g., 1450): ").strip()
            if sat_score_str.isdigit():
                sat_score = int(sat_score_str)
                if 400 <= sat_score <= 1600:
                    student_profile["sat_score"] = sat_score
                    break
                else:
                    print("Invalid SAT score. Please enter a number between 400 and 1600.")
            else:
                print("Invalid input. SAT score must be a number.")
    else:
        student_profile["sat_score"] = "N/A"

    act_choice = input("8. Do you have an ACT score? (Enter 'yes' or 'no') ").strip().lower()
    if act_choice == 'yes':
        while True:
            act_score_str = input("   Please enter your ACT score (e.g., 30): ").strip()
            if act_score_str.isdigit():
                act_score = int(act_score_str)
                if 1 <= act_score <= 36:
                    student_profile["act_score"] = act_score
                    break
                else:
                    print("Invalid ACT score. Please enter a number between 1 and 36.")
            else:
                print("Invalid input. ACT score must be a number.")
    else:
        student_profile["act_score"] = "N/A"

    student_profile["high_school"] = get_mandatory_string_input("9. What is the name of your High School? ")
    student_profile["target_college"] = get_mandatory_string_input("10. What college/uni are you interested in analyzing today? ")

    print("\n--- Profile collection complete! ---")
    return student_profile

# --- Global Configuration ---
GEMINI_MODEL = "gemini-2.0-flash-001"

# --- Schemas ---
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

class SportsInfo(BaseModel):
    ncaa_division: str = Field(description="NCAA division classification")
    conference: str = Field(description="Athletic conference affiliation")
    facilities: str = Field(description="Descriptions and amenities of athletic facilities")
    coaching_staff: str = Field(description="Current coaching staff information")
    team_roster: str = Field(description="Current team composition and size")
    team_page_url: str = Field(description="Official team webpage URL")

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
    output_key="search_results"
)

# 2. University Info Extraction Agent
university_info_extractor = LlmAgent(
    name="university_info_extractor",
    model=GEMINI_MODEL,
    instruction="""
You are given URLs pointing to official college websites. Visit those URLs and extract relevant data matching the UniversityInfo schema below.

Return data in JSON format. Use 'N/A' for any missing fields.

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
    instruction="""
Search for official athletics info about the student's target college and sport.

Use google_search to find athletics pages, team rosters, coaching staff info, and facilities.

Return data in JSON matching the SportsInfo schema below, using 'N/A' for missing fields.

SportsInfo fields:
- ncaa_division, conference, facilities, coaching_staff, team_roster, team_page_url
""",
    description="Searches for and extracts athletics program data.",
    tools=[google_search],
    output_key="sports_info"
)

# 4. Missing Info Resolver Agent (single for university and sports)
missing_info_resolver = LlmAgent(
    name="missing_info_resolver",
    model=GEMINI_MODEL,
    instruction="""
You are given partial JSON data for university and sports information, some fields marked 'N/A'.

Your tasks:
- Identify missing fields with 'N/A'.
- Generate targeted google_search queries to fill missing fields, focusing on official .edu sources.
- Examples:
  - "[college_name] tuition fees site:.edu"
  - "[college_name] [sport] coaching staff site:.edu"
- Fill in as many fields as possible; keep 'N/A' only if data cannot be found.
- Return updated data as JSON combining both UniversityInfo and SportsInfo schemas.
""",
    description="Fills missing university and sports info fields by targeted searches.",
    tools=[google_search],
    output_key="completed_info"
)

# 5. Report Writer Agent
report_writer_agent = LlmAgent(
    name="report_writer_agent",
    model=GEMINI_MODEL,
    instruction="""
Write a comprehensive report for a student-athlete based on the university and athletics data provided.

Include:
- Academic profile summary
- Athletics program summary
- Fit assessment and recommendations
- Note any missing data clearly as 'N/A'

Do not fabricate data.
""",
    description="Creates comprehensive student-athlete college report.",
    tools=[],
    output_key="final_report"
)

# --- Root Sequential Agent ---
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

if __name__ == "__main__":
    
    import asyncio

    # Collect student profile interactively
    student_profile = collect_student_profile_mvp()

    # Inject into session context
    session_context = {"student_profile": student_profile}

    print("\n=== Running agent pipeline ===\n")

    # Run the root agent with the session context
    result = asyncio.run(root_agent.run(session_context))

    print("\n=== FINAL REPORT ===\n")
    print(result.get("final_report", "No report generated."))
