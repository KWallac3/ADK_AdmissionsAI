import json
import os
from typing import Dict, Any

PROFILE_FILE = "student_profile.json"

def get_default_profile() -> Dict[str, Any]:
    """
    Returns a default student profile structure.
    """
    return {
        "full_name": "Alice Morgan",
        "grade_level": "11th",
        "primary_sport": "Track and Field",
        "events": ["55m", "100m", "200m", "Long Jump"],
        "personal_records": {
            "55M": "7.6",
            "100m": "12.45",
            "200m": "25.80",
            "Long_Jump": "5.20m"
        },
        "gender": "Female",
        "unweighted_gpa": 3.95,
        "gpa_scale": 4.0,
        "sat_score": 1280,
        "act_score": "N/A",
        "high_school": "My High School",
    }

def save_profile(profile_data: Dict[str, Any]) -> None:
    """
    Saves the provided profile data to the PROFILE_FILE as JSON.
    """
    try:
        with open(PROFILE_FILE, 'w') as f:
            json.dump(profile_data, f, indent=4)
        print(f"Profile saved to {PROFILE_FILE}")
    except IOError as e:
        print(f"Error saving profile to {PROFILE_FILE}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving profile: {e}")

def load_profile() -> Dict[str, Any]:
    """
    Loads the profile from PROFILE_FILE.
    If the file doesn't exist or is invalid, it saves and returns a default profile.
    """
    if not os.path.exists(PROFILE_FILE):
        print(f"{PROFILE_FILE} not found. Creating with default profile.")
        default_profile = get_default_profile()
        save_profile(default_profile)
        return default_profile
    
    try:
        with open(PROFILE_FILE, 'r') as f:
            profile_data = json.load(f)
            # Basic validation: check if it's a dictionary (further schema validation could be added)
            if not isinstance(profile_data, dict):
                raise ValueError("Profile data is not a valid dictionary.")
            print(f"Profile loaded from {PROFILE_FILE}")
            return profile_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {PROFILE_FILE}: {e}. Using default profile.")
        default_profile = get_default_profile()
        save_profile(default_profile) # Overwrite corrupted file
        return default_profile
    except ValueError as e:
        print(f"Error validating profile data from {PROFILE_FILE}: {e}. Using default profile.")
        default_profile = get_default_profile()
        save_profile(default_profile) # Overwrite invalid file
        return default_profile
    except IOError as e:
        print(f"Error loading profile from {PROFILE_FILE}: {e}. Using default profile.")
        default_profile = get_default_profile()
        # Don't save here as it might be a read permission issue, not a corrupt file issue
        return default_profile
    except Exception as e:
        print(f"An unexpected error occurred while loading profile: {e}. Using default profile.")
        default_profile = get_default_profile()
        return default_profile

def update_profile_interactively(current_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interactively prompts the user to update fields of the student profile.
    Skips complex fields (lists/dicts) for interactive update in this version.
    """
    print("\n--- Interactive Profile Update ---")
    print("Press Enter to keep the current value for any field.")
    
    updated_profile = current_profile.copy() # Work on a copy

    # Fields to update interactively (simple top-level fields)
    # Add more fields here as needed for interactive update
    fields_to_update = [
        ("full_name", str),
        ("grade_level", str),
        ("primary_sport", str),
        ("gender", str),
        ("unweighted_gpa", float),
        ("sat_score", int),
        ("high_school", str)
    ]

    for key, field_type in fields_to_update:
        current_value = updated_profile.get(key, "")
        prompt_message = f"{key.replace('_', ' ').title()} (current: {current_value}): "
        
        user_input = input(prompt_message)
        
        if user_input: # If user provided input
            try:
                if field_type == float:
                    new_value = float(user_input)
                elif field_type == int:
                    new_value = int(user_input)
                else: # str
                    new_value = str(user_input)
                updated_profile[key] = new_value
                print(f"Updated {key} to: {new_value}")
            except ValueError:
                print(f"Invalid input type for {key}. Expected {field_type.__name__}. Keeping current value: {current_value}")
        else:
            print(f"{key} kept as: {current_value}")

    # Display complex fields but don't offer detailed interactive update for them in this version
    print("\nComplex fields (view only in this update mode):")
    for key, value in updated_profile.items():
        if isinstance(value, (dict, list)) and key not in dict(fields_to_update):
            print(f"- {key.replace('_', ' ').title()}: {value}")
            # Future enhancement: Allow updating these, e.g. by prompting for JSON input or specific sub-fields.
            
    print("--- Profile update session finished ---")
    return updated_profile

