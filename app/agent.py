# === Imports ===
import asyncio
from typing import List, Optional, Dict, Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools import google_search
from pydantic import BaseModel, Field

# === Utility: Profile Formatter ===
def profile_to_string(profile: dict) -> str:
    """Converts a student profile dictionary into a readable string format for the AI."""
    s = []
    for k, v in profile.items():
        s.append(f"- {k.replace('_', ' ').title()}: {v}")
    return "\n".join(s)

# --- Function to get student profile (CLI for basic input) ---
def get_student_profile() -> Dict[str, Any]:
    """
    Retrieves the student profile by prompting the user for basic information via CLI.
    Other details will use default values.
    """
    print("--- Please Enter Student Profile Information ---")

    full_name = input("Full Name (e.g., Alex Morgan): ") or "Alex Morgan"
    grade_level = input("Grade Level (e.g., 11th): ") or "11th"
    primary_sport = input("Primary Sport (e.g., Track and Field): ") or "Track and Field"
    gender = input("Gender (e.g., Female/Male/Non-binary): ") or "Female"

    unweighted_gpa_str = input("Unweighted GPA (e.g., 3.75): ")
    try:
        unweighted_gpa = float(unweighted_gpa_str) if unweighted_gpa_str else 3.75
    except ValueError:
        print("Invalid GPA format. Using default 3.75.")
        unweighted_gpa = 3.75

    sat_score_str = input("SAT Score (e.g., 1380): ")
    try:
        sat_score = int(sat_score_str) if sat_score_str else 1380
    except ValueError:
        print("Invalid SAT score format. Using default 1380.")
        sat_score = 1380

    # Default values for other fields to keep CLI simple for now
    profile_data = {
        "full_name": full_name,
        "grade_level": grade_level,
        "primary_sport": primary_sport,
        "events": ["55m","100m", "200m", "Long Jump"],  # Default
        "personal_records": { # Default
            "55M": "7.6",
            "100m": "12.45",
            "200m": "25.80",
            "Long_Jump": "5.20m"
        },
        "gender": gender,
        "unweighted_gpa": unweighted_gpa,
        "gpa_scale": 4.0, # Default
        "sat_score": sat_score,
        "act_score": "N/A", # Default
        "high_school": "My High School", # Default
    }
    print("-------------------------------------------------")
    return profile_data

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



# --- Agent Definitions within a Function ---

def create_agent_pipeline(student_profile_data: Dict[str, Any]) -> SequentialAgent:
    """
    Creates and configures the sequential agent pipeline with dynamic student profile information.
    """
    profile_for_agent_string = profile_to_string(student_profile_data)

    # 1. University Info Search Agent (No profile needed for this one)
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
    university_info_extractor_instruction = f"""
    === STUDENT PROFILE ===
    {profile_for_agent_string}
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
    """
    university_info_extractor = LlmAgent(
        name="university_info_extractor",
        model=GEMINI_MODEL,
        instruction=university_info_extractor_instruction,
        description="Extracts structured university data from page content.",
        tools=[google_search],
        output_key="university_info"
    )

    # 3. Sports Info Search Agent
    sports_info_extractor_instruction = f"""
    === STUDENT PROFILE ===
    {profile_for_agent_string}
    === END STUDENT PROFILE ===

    You MUST consider this student's specific characteristics when extracting data:
    - Focus on admission requirements relevant to their GPA ({student_profile_data['unweighted_gpa']}) and test scores
    - Prioritize cost information since financial considerations are important
    - Look for information about {student_profile_data['primary_sport']} programs if available

    You are a specialized sports data researcher. Your task is to find official athletics information for a given college and sport,
    specifically focusing on current season top performers and recent conference championship results.

    Search for current {student_profile_data['gender']} {student_profile_data['primary_sport']} roster and times for events like sprints, distance, throws, jumps - whatever the student's focus area is

    When searching, prioritize authoritative and up-to-date platforms.
    Dynamically select your search strategy based on the sport:

    If the sport is 'Track and Field':
    - Prioritize TFRRS (Track & Field Results Reporting System) and official conference athletics websites.
    - Use queries like:
        - "TFRRS [college name] track and field current season results"
        - "NCAA Track and Field Statistics and Records official website"
        - "[conference name] Track and Field Championship results 2024-2025 {student_profile_data['gender']}"
        - "[college name] track and field top performers 2024-2025 [event categories]"
        - "official website [conference name] athletics results"

    If the sport is 'Swimming':
    - Prioritize USA Swimming's SWIMS database and official conference athletics websites.
    - Use queries like:
        - "SWIMS 3.0 [college name] swimming current season"
        - "NCAA Swimming and Diving Statistics and Records official website"
        - "[conference name] Swimming and Diving Championship results 2024-2025 {student_profile_data['gender']}"
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
    """
    sports_info_extractor = LlmAgent(
        name="sports_info_extractor",
        model=GEMINI_MODEL,
        instruction=sports_info_extractor_instruction,
        description="Searches for and extracts athletics program data, including current performers and championship results.",
        tools=[google_search],
        output_key="sports_info_extracted"
    )

# === Combined Data Schema ===
class ComprehensiveCollegeReportData(BaseModel):
    university_details: UniversityInfo = Field(description="Comprehensive information about the university's academics, admissions, and general facts.")
    athletics_details: SportsInfo = Field(description="Detailed information about the university's athletic programs, team performance, and recruiting.")
    target_college_name: str = Field(description="The name of the college being researched.")


    # 4. Missing Info Resolver Agent (single for university and sports)
    missing_info_resolver_instruction = f"""
    You are a meticulous data consolidator. You are given:
    1. Partially populated JSON data for 'university_info' (conforming to UniversityInfo schema).
    2. Partially populated JSON data for 'sports_info' (conforming to SportsInfo schema).
    3. The 'target_college_name'.

    Your tasks are as follows:

    1.  **Comprehensive Review:**
        *   Thoroughly examine the provided `university_info` (conforming to UniversityInfo schema) and `sports_info` (conforming to SportsInfo schema) objects.
        *   Identify EVERY field, including nested fields within lists or sub-objects, that is currently marked 'N/A', is an empty string/list, or seems to be a placeholder indicating missing information.

    2.  **Targeted Information Retrieval:**
        *   For each identified missing piece of information, formulate highly specific `google_search` queries.
        *   **Prioritize official sources:** For `university_details`, always prefer `.edu` domains of the `target_college_name`. For `athletics_details`, prioritize official athletic department websites, conference websites, and reputable sports data providers (e.g., TFRRS for Track & Field, SWIMS for Swimming).
        *   **Iterative Searching:** If a first query doesn't yield the result, try variations. For example, if "[target_college_name] SAT range site:.edu" fails, try "[target_college_name] admissions statistics site:.edu" and look for test score data.
        *   **Contextual Search for Complex Data:**
            *   For lists like `current_performers` or `championship_results`, if the list is empty or missing, search for the overall data (e.g., "TFRRS [target_college_name] {student_profile_data['gender']} {student_profile_data['primary_sport']} roster 2024-2025").
            *   If specific items within these lists are missing (e.g., a performer's `class_year`), try to find that specific detail if the rest of the item is present.
        *   **Example Queries (adapt these using the actual `target_college_name` and `student_profile_data` from the context):**
            *   Missing `tuition` in `university_info`: "[target_college_name] undergraduate tuition and fees site:.edu"
            *   Missing `coaching_staff` in `sports_info`: "[target_college_name] {student_profile_data['primary_sport']} coaching staff site:.edu athletics"
            *   Missing `current_performers` in `sports_info` for Track & Field: "TFRRS [target_college_name] {student_profile_data['gender']} {student_profile_data['primary_sport']} top performers 2024-2025"
            *   Missing `acceptance_rate` in `university_info`: "[target_college_name] common data set admissions site:.edu" OR "[target_college_name] admissions facts site:.edu"

    3.  **Data Integration and Validation:**
        *   Update the `university_info` and `sports_info` objects with the information you find.
        *   **Persistence of 'N/A':** Only retain 'N/A' (or leave a field empty if appropriate for its type, e.g., an empty list) if, after diligent and varied search attempts, you confirm the information is genuinely unavailable, not published, or not applicable (e.g., ACT range for a test-blind school). Do not use 'N/A' if you simply couldn't find it on the first try.

    4.  **Structured Output Generation:**
        *   After attempting to fill all missing information, you MUST structure your final output as a single JSON object conforming to the `ComprehensiveCollegeReportData` schema.
        *   This means the final JSON must have three top-level keys:
            *   `university_details`: Containing the updated UniversityInfo data.
            *   `athletics_details`: Containing the updated SportsInfo data.
            *   `target_college_name`: The original target college name you were given.

    Example of the final output structure:
    {{
        "university_details": {{ ... all UniversityInfo fields ... }},
        "athletics_details": {{ ... all SportsInfo fields ... }},
        "target_college_name": "Actual College Name"
    }}

    Return this single, structured JSON object.
    """
    missing_info_resolver = LlmAgent(
        name="missing_info_resolver",
        model=GEMINI_MODEL,
        instruction=missing_info_resolver_instruction,
        description="Fills missing university and sports info fields by targeted searches.",
        tools=[google_search],
        output_key="completed_info"
    )

    # 5. Report Writer – Generates a final narrative assessment using all compiled data
    report_writer_agent_instruction = f"""
    === STUDENT PROFILE ===
    {profile_for_agent_string}
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
    Write a comprehensive, professional, and empathetic report for the student-athlete.
    You will receive a JSON object conforming to the `ComprehensiveCollegeReportData` schema. This object contains:
    1. `university_details`: Academic and general information about the college.
    2. `athletics_details`: Specifics about the college's athletic programs, team performance, and recruiting for the student's sport.
    3. `target_college_name`: The name of the college.

    The report should use this data to cover the following sections:
    - **College Overview:** Briefly introduce the {{{{ target_college_name }}}}.
    - **Academic Profile Summary:** Based on `university_details`.
    - **Athletics Program Summary:** Based on `athletics_details` for {{{{ student_profile_data['primary_sport'] }}}}.
    - **Fit Assessment and Recommendations:** This is the most crucial part. Integrate insights from both academic and athletic data.
    - Note any missing data (fields marked 'N/A' within `university_details` or `athletics_details`) clearly.

    **Important Content Rules ( tái khẳng định ):**
    - Do not fabricate data.
    - Maintain Professional advisory tone.
    - Do not make subjective judgements or provide personal opions on demographics. Simple state the facts.
    """
    report_writer_agent = LlmAgent(
        name="report_writer_agent",
        model=GEMINI_MODEL,
        instruction=report_writer_agent_instruction,
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
            missing_info_resolver,
            report_writer_agent
        ]
    )
    return root_agent

# === Main Execution Block (Example) ===
async def main():
    # 1. Get student profile
    student_profile = get_student_profile()
    print("--- Student Profile Loaded ---")
    print(profile_to_string(student_profile))
    print("-----------------------------")

    # 2. Create the agent pipeline with the student's profile
    agent_pipeline = create_agent_pipeline(student_profile)

    # 3. Prepare inputs for the pipeline
    target_college_name = input("Enter the target college name (e.g., Stanford University): ") or "Stanford University"
    initial_input = {"target_college": target_college_name, "student_profile": student_profile}
    
    print(f"\n--- Starting Agent Pipeline for {target_college_name} ---")
    # 4. Run the agent pipeline (example invocation)
    #    Actual execution might vary based on how google-adk expects inputs and runs.
    #    This is a placeholder for how you might run it.
    # result = await agent_pipeline.arun(initial_input)
    # print("\n--- Pipeline Finished ---")
    # print("Final Report (or result):")
    # print(result)
    print("--- Agent Pipeline Created (Execution not yet implemented in this refactor) ---")


if __name__ == "__main__":
    # Note: To run this if it were complete, you'd likely use:
    # asyncio.run(main())
    # For now, just demonstrating the setup
    print("Agent script setup for dynamic profiles. Run main() to initialize.")
    # Example of how main *would* be called:
    # try:
    #     asyncio.run(main())
    # except KeyboardInterrupt:
    #     print("Process interrupted by user.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    pass # Placeholder for actual execution call