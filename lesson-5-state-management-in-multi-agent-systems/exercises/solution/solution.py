from typing import Dict, List, Any
import os
import dotenv
import time
from enum import Enum

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)

dotenv.load_dotenv(dotenv_path="../.env")
openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)

# Define seasons enum for type safety
class Season(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    ALL_YEAR = "all_year"

# Fruit database with seasonal availability
fruit_data = {
    "lulo": {
        "description": "A tart, citrusy fruit, often used in juice.", 
        "origin": "Colombia",
        "seasons": [Season.SUMMER, Season.FALL]
    },
    "mango": {
        "description": "Sweet and juicy, a tropical favorite.", 
        "origin": "Colombia",
        "seasons": [Season.SUMMER]
    },
    "granadilla": {
        "description": "A sweet and seedy fruit with a hard shell.", 
        "origin": "Colombia",
        "seasons": [Season.SPRING, Season.SUMMER]
    },
    "chontaduro": {
        "description": "A starchy fruit, often eaten with salt. Rich in nutrients.", 
        "origin": "Colombia",
        "seasons": [Season.ALL_YEAR]
    },
    "feijoa": {
        "description": "A small green fruit with a unique, aromatic flavor.", 
        "origin": "Colombia",
        "seasons": [Season.FALL, Season.WINTER]
    },
    "guava": {
        "description": "A tropical fruit with sweet flesh and edible seeds.", 
        "origin": "Colombia",
        "seasons": [Season.SPRING, Season.FALL]
    },
    "pitahaya": {
        "description": "Dragon fruit with white flesh and black seeds.", 
        "origin": "Colombia",
        "seasons": [Season.SPRING, Season.SUMMER]
    },
    "uchuva": {
        "description": "Small orange berries with a sweet-tart flavor.", 
        "origin": "Colombia",
        "seasons": [Season.WINTER]
    }
}

# Global state storage with enhanced structure
user_states = {}

# Current season - This would be determined by actual date in a real system
current_season = Season.SUMMER

@tool
def get_fruit_description(fruit_name: str) -> str:
    """Retrieves the description of a fruit from the fruit database.
    
    Args:
        fruit_name: The name of the fruit to retrieve information about.
        
    Returns:
        The description of the fruit or an error message if not found.
    """
    if fruit_name in fruit_data:
        return fruit_data[fruit_name]['description']
    return "Fruit not found in database."

@tool
def add_fruit_preference(user_id: str, fruit_name: str) -> str:
    """Adds a fruit to the user's preferences.
    
    Args:
        user_id: The ID of the user.
        fruit_name: The name of the fruit to add to preferences.
        
    Returns:
        A message confirming the addition or indicating it's already in preferences.
    """
    # Initialize user state if it doesn't exist
    if user_id not in user_states:
        user_states[user_id] = {
            "preferences": [],
            "seasonal_preference": True  # Default to seasonal shopping
        }
    user_state = user_states[user_id]
    
    # Ensure preferences list exists
    if "preferences" not in user_state:
        user_state["preferences"] = []
    
    if fruit_name not in user_state["preferences"]:
        user_state["preferences"].append(fruit_name)
        user_states[user_id] = user_state
        return f"Added {fruit_name} to your preferences."
    else:
        return f"{fruit_name} is already in your preferences."

@tool
def get_user_preferences(user_id: str) -> List[str]:
    """Retrieves the user's fruit preferences.
    
    Args:
        user_id: The ID of the user whose preferences to retrieve.
        
    Returns:
        A list of the user's preferred fruits.
    """
    # Initialize user state if it doesn't exist
    if user_id not in user_states:
        user_states[user_id] = {
            "preferences": [],
            "seasonal_preference": True
        }
    user_state = user_states[user_id]
    
    # Ensure preferences list exists
    if "preferences" not in user_state:
        user_state["preferences"] = []
        
    return user_state["preferences"]

@tool
def save_user_state(user_id: str) -> str:
    """Saves the current state for a user.
    
    Args:
        user_id: The ID of the user whose state to save.
        
    Returns:
        A confirmation message.
    """
    # This would connect to a real database in a production environment
    # For this demo, the state is already in memory in the user_states dict
    return f"User state for {user_id} saved successfully."

@tool
def get_current_season() -> str:
    """Gets the current season.
    
    Returns:
        The current season (spring, summer, fall, winter).
    """
    return current_season

@tool
def set_current_season(season: str) -> str:
    """Sets the current season (for demo purposes).
    
    Args:
        season: The season to set (spring, summer, fall, winter).
        
    Returns:
        A confirmation message.
    """
    global current_season
    if season.lower() in [s.value for s in Season]:
        current_season = season.lower()
        return f"Current season set to {season}."
    else:
        return f"Invalid season: {season}. Valid options are spring, summer, fall, winter, or all_year."

@tool
def get_seasonal_fruits(season: str = None) -> List[str]:
    """Gets a list of fruits that are in season.
    
    Args:
        season: The season to check for. If not provided, uses the current season.
        
    Returns:
        A list of fruit names that are in season.
    """
    check_season = season.lower() if season else current_season
    
    if check_season not in [s.value for s in Season]:
        return f"Invalid season: {check_season}. Valid options are spring, summer, fall, winter, or all_year."
    
    seasonal_fruits = []
    for fruit_name, fruit_info in fruit_data.items():
        if check_season in [s.lower() for s in fruit_info["seasons"]] or Season.ALL_YEAR.value in [s.lower() for s in fruit_info["seasons"]]:
            seasonal_fruits.append(fruit_name)
    
    return seasonal_fruits

@tool
def set_user_seasonal_preference(user_id: str, seasonal_only: bool) -> str:
    """Sets whether a user prefers to see only seasonal fruits.
    
    Args:
        user_id: The ID of the user.
        seasonal_only: Whether to show only seasonal fruits (True) or all fruits (False).
        
    Returns:
        A confirmation message.
    """
    # Initialize user state if it doesn't exist
    if user_id not in user_states:
        user_states[user_id] = {
            "preferences": [],
            "seasonal_preference": seasonal_only
        }
    else:
        user_states[user_id]["seasonal_preference"] = seasonal_only
    
    if seasonal_only:
        return f"User {user_id} now prefers to see only seasonal fruits."
    else:
        return f"User {user_id} now prefers to see all available fruits."

@tool
def get_user_seasonal_preference(user_id: str) -> bool:
    """Gets whether a user prefers to see only seasonal fruits.
    
    Args:
        user_id: The ID of the user.
        
    Returns:
        True if the user prefers only seasonal fruits, False otherwise.
    """
    # Initialize user state if it doesn't exist
    if user_id not in user_states:
        user_states[user_id] = {
            "preferences": [],
            "seasonal_preference": True  # Default to seasonal shopping
        }
    
    return user_states[user_id].get("seasonal_preference", True)

@tool
def get_fruit_seasons(fruit_name: str) -> List[str]:
    """Gets the seasons when a fruit is available.
    
    Args:
        fruit_name: The name of the fruit to check.
        
    Returns:
        A list of seasons when the fruit is available.
    """
    if fruit_name in fruit_data:
        return fruit_data[fruit_name]["seasons"]
    return f"Fruit '{fruit_name}' not found in the database."

@tool
def is_fruit_in_season(fruit_name: str, season: str = None) -> bool:
    """Checks if a fruit is in season.
    
    Args:
        fruit_name: The name of the fruit to check.
        season: The season to check for. If not provided, uses the current season.
        
    Returns:
        True if the fruit is in season, False otherwise.
    """
    check_season = season.lower() if season else current_season
    
    if fruit_name not in fruit_data:
        return False
    
    # All-year fruits are always in season
    if Season.ALL_YEAR.value in [s.lower() for s in fruit_data[fruit_name]["seasons"]]:
        return True
    
    return check_season in [s.lower() for s in fruit_data[fruit_name]["seasons"]]

class SeasonalFruitAdvisorAgent(ToolCallingAgent):
    """
    Agent for providing information about Colombian fruits,
    remembering user preferences, and handling seasonal availability.
    """
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                get_fruit_description,
                add_fruit_preference,
                get_user_preferences,
                save_user_state,
                get_current_season,
                set_current_season,
                get_seasonal_fruits,
                set_user_seasonal_preference,
                get_user_seasonal_preference,
                get_fruit_seasons,
                is_fruit_in_season
            ],
            model=model,
            name="seasonal_fruit_advisor_agent",
            description="""
            You are a helpful assistant specializing in Colombian fruits.
            You help users learn about various Colombian fruits, remember their preferences,
            and provide information about seasonal availability.
            
            Use the tools available to you to retrieve information and manage user preferences.
            Be enthusiastic and informative about Colombian fruits!
            
            Remember that some fruits are only available in certain seasons:
            - Spring: March, April, May
            - Summer: June, July, August
            - Fall: September, October, November
            - Winter: December, January, February
            - Some fruits are available all year round
            """,
        )

class SeasonalOrchestratorAgent(ToolCallingAgent):
    """
    Orchestrates the seasonal fruit advisor system, managing user sessions
    and delegating to the fruit advisor agent.
    """
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[],
            model=model,
            name="seasonal_orchestrator_agent",
            description="""
            You are an orchestrator agent that manages the Colombian fruit advisory system.
            Your role is to coordinate interactions between users and the fruit advisor agent,
            ensuring that user state is properly managed and preserved across sessions.
            """,
        )
        self.fruit_advisor = SeasonalFruitAdvisorAgent(model)

    def process_user_message(self, user_id: str, message: str) -> str:
        """
        Process a user message and return a response.
        
        Args:
            user_id: The ID of the user.
            message: The user's message.
            
        Returns:
            A response from the fruit advisor agent.
        """
        # First, we need to load any existing preferences and settings
        current_preferences = get_user_preferences(user_id)
        seasonal_preference = get_user_seasonal_preference(user_id)
        current_season_value = get_current_season()
        seasonal_fruits = get_seasonal_fruits()
        
        # Construct a prompt for the fruit advisor agent
        prompt = f"""
        User ID: {user_id}
        Current preferences: {', '.join(current_preferences) if current_preferences else 'None yet'}
        Current season: {current_season_value}
        User prefers seasonal fruits only: {seasonal_preference}
        Currently in-season fruits: {', '.join(seasonal_fruits)}
        
        The user says: "{message}"
        
        If the user is asking about a fruit, use get_fruit_description to provide information about it.
        Also check if the fruit is in season using is_fruit_in_season and inform the user.
        
        If the user expresses interest in a fruit, use add_fruit_preference to save it.
        
        If the user wants to know what fruits are in season, use get_seasonal_fruits.
        
        If the user wants to change seasons (for demo purposes), use set_current_season.
        
        If the user wants to change their seasonal preference, use set_user_seasonal_preference.
        
        If the user wants to know their preferences, use get_user_preferences.
        
        Remember to save the user's state with save_user_state before ending the conversation.
        
        Respond in a friendly, informative way in both English and a bit of Spanish.
        If a fruit is not in season, suggest alternatives that are currently in season.
        """
        
        return self.fruit_advisor.run(prompt)

def run_seasonal_demo():
    """
    Runs the seasonal fruit advisor demo with simulated user interactions.
    """
    print("🍎 Colombian Fruit Market with Seasonal Availability 🍍")
    print("="*70)
    
    orchestrator = SeasonalOrchestratorAgent(model)
    user_id = "user123"
    
    print("\n--- First Session (Summer) ---")
    print(f"Current Season: {get_current_season()}")
    print(f"Fruits in season: {', '.join(get_seasonal_fruits())}")
    
    # Simulated user messages for summer
    messages = [
        "Hi! What fruits are in season right now?",
        "Tell me about lulo. Is it in season?",
        "I'd like to try lulo. Add it to my preferences.",
        "What about uchuva? Is that available now?",
        "I prefer to see all fruits, not just seasonal ones.",
        "What are my preferences so far?",
        "Thank you! I'll come back later."
    ]
    
    # Process each message
    for i, message in enumerate(messages):
        print(f"\nUser: {message}")
        response = orchestrator.process_user_message(user_id, message)
        print(f"Agent: {response}")
        time.sleep(0.5)  # Brief pause for readability
    
    # Simulate system restart/new session
    print("\n" + "="*70)
    print("--- Simulating a System Restart and Season Change ---")
    print("The system is restarting and will reload user state...")
    set_current_season(Season.WINTER.value)
    print(f"Season has changed to: {get_current_season()}")
    print(f"Fruits in season: {', '.join(get_seasonal_fruits())}")
    print("="*70)
    
    # Create a new orchestrator (simulating a system restart)
    new_orchestrator = SeasonalOrchestratorAgent(model)
    
    # Retrieve the user's preferences to show state persistence
    current_preferences = get_user_preferences(user_id)
    seasonal_preference = get_user_seasonal_preference(user_id)
    print(f"\nAfter system restart, {user_id}'s preferences: {current_preferences}")
    print(f"Seasonal preference setting: {seasonal_preference}")
    
    # Continue the conversation in the new session
    print("\n--- Continuing in New Session (Winter) ---")
    
    # New set of messages for winter
    new_messages = [
        "Hello again! What fruits are available in this season?",
        "Is lulo available now?",
        "What fruits would you recommend that are in season?",
        "Add uchuva to my preferences, please.",
        "I changed my mind, I want to only see seasonal fruits now.",
        "What are all my preferences so far?",
        "Thank you for your help!"
    ]
    
    # Process each new message
    for i, message in enumerate(new_messages):
        print(f"\nUser: {message}")
        response = new_orchestrator.process_user_message(user_id, message)
        print(f"Agent: {response}")
        time.sleep(0.5)  # Brief pause for readability
    
    # Final state check
    final_preferences = get_user_preferences(user_id)
    final_seasonal_pref = get_user_seasonal_preference(user_id)
    print("\n" + "="*70)
    print(f"Final state: {user_id}'s fruit preferences are: {final_preferences}")
    print(f"Final seasonal preference setting: {final_seasonal_pref}")
    print("="*70)
    print("\nDemo complete! This demonstrates seasonal availability state management across sessions.")

if __name__ == "__main__":
    run_seasonal_demo()
