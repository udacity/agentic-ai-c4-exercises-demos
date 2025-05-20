from typing import Dict, List, Any
import os
import dotenv
import time

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

# Fruit database
fruit_data = {
    "lulo": {"description": "A tart, citrusy fruit, often used in juice.", "origin": "Colombia"},
    "mango": {"description": "Sweet and juicy, a tropical favorite.", "origin": "Colombia"},
    "granadilla": {"description": "A sweet and seedy fruit with a hard shell.", "origin": "Colombia"},
    "chontaduro": {"description": "A starchy fruit, often eaten with salt. Rich in nutrients.", "origin": "Colombia"}
}

# Global state storage
user_states = {}

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
    user_state = user_states.get(user_id, {"preferences": []})
    
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
    user_state = user_states.get(user_id, {"preferences": []})
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

class FruitAdvisorAgent(ToolCallingAgent):
    """
    Agent for providing information about Colombian fruits and 
    remembering user preferences.
    """
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                get_fruit_description,
                add_fruit_preference,
                get_user_preferences,
                save_user_state
            ],
            model=model,
            name="fruit_advisor_agent",
            description="""
            You are a helpful assistant specializing in Colombian fruits.
            You help users learn about various Colombian fruits and remember their preferences.
            Use the tools available to you to retrieve information and manage user preferences.
            Be enthusiastic and informative about Colombian fruits!
            """,
        )

class OrchestratorAgent(ToolCallingAgent):
    """
    Orchestrates the entire fruit advisor system, managing user sessions
    and delegating to the fruit advisor agent.
    """
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[],
            model=model,
            name="orchestrator_agent",
            description="""
            You are an orchestrator agent that manages the Colombian fruit advisory system.
            Your role is to coordinate interactions between users and the fruit advisor agent,
            ensuring that user state is properly managed and preserved across sessions.
            """,
        )
        self.fruit_advisor = FruitAdvisorAgent(model)

    def process_user_message(self, user_id: str, message: str) -> str:
        """
        Process a user message and return a response.
        
        Args:
            user_id: The ID of the user.
            message: The user's message.
            
        Returns:
            A response from the fruit advisor agent.
        """
        # First, we need to load any existing preferences
        current_preferences = get_user_preferences(user_id)
        
        # Construct a prompt for the fruit advisor agent
        prompt = f"""
        User ID: {user_id}
        Current preferences: {', '.join(current_preferences) if current_preferences else 'None yet'}
        
        The user says: "{message}"
        
        If the user is asking about a fruit, use get_fruit_description to provide information.
        If the user expresses interest in a fruit, use add_fruit_preference to save it.
        If the user wants to know their preferences, use get_user_preferences.
        Remember to save the user's state with save_user_state before ending the conversation.
        
        Respond in a friendly, informative way in both English and a bit of Spanish.
        """
        
        return self.fruit_advisor.run(prompt)

def run_demo():
    """
    Runs the fruit advisor demo with simulated user interactions.
    """
    print("🍎 Colombian Fruit Market State Management Demo 🍍")
    print("="*60)
    
    orchestrator = OrchestratorAgent(model)
    user_id = "user123"
    
    print("\n--- First Session ---")
    
    # Simulated user messages
    messages = [
        "Hi! What kind of fruits do you have?",
        "Tell me about lulo.",
        "That sounds interesting! I'd like to try lulo.",
        "What about mango?", 
        "I love mangoes! Add that to my preferences too.",
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
    print("\n" + "="*60)
    print("--- Simulating a System Restart ---")
    print("The system is restarting and will reload user state...")
    print("="*60)
    
    # Create a new orchestrator (simulating a system restart)
    new_orchestrator = OrchestratorAgent(model)
    
    # Retrieve the user's preferences to show state persistence
    current_preferences = get_user_preferences(user_id)
    print(f"\nAfter system restart, {user_id}'s preferences: {current_preferences}")
    
    # Continue the conversation in the new session
    print("\n--- Continuing in New Session ---")
    
    # New set of messages
    new_messages = [
        "Hello again! I'm back to learn more about Colombian fruits.",
        "Can you remind me what fruits I liked?",
        "Tell me about granadilla.",
        "I'd like to try that too! Add granadilla to my list.",
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
    print("\n" + "="*60)
    print(f"Final state: {user_id}'s fruit preferences are: {final_preferences}")
    print("="*60)
    print("\nDemo complete! This demonstrates state persistence across agent sessions.")

if __name__ == "__main__":
    run_demo()
