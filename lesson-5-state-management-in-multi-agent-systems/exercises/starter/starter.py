"""
EXERCISE: Transaction History State Management

In this exercise, you'll extend the Colombian Fruit Market system by implementing
a purchase history tracking feature. This represents a practical extension of state
management that's common in real e-commerce systems.

You'll need to:
1. Add a purchase_fruit tool to record transactions
2. Track purchase dates and quantities
3. Add a view_purchase_history tool
4. Implement purchase summary functionality
5. Make the FruitAdvisorAgent aware of purchase history

This exercise demonstrates how state management can be extended to handle
complex transaction data while maintaining user context across sessions.
"""

from typing import Dict, List, Any
import os
import dotenv
import time
from datetime import datetime  # Added for timestamp functionality

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
    "lulo": {"description": "A tart, citrusy fruit, often used in juice.", "origin": "Colombia", "price": 2.50},
    "mango": {"description": "Sweet and juicy, a tropical favorite.", "origin": "Colombia", "price": 1.75},
    "granadilla": {"description": "A sweet and seedy fruit with a hard shell.", "origin": "Colombia", "price": 3.00},
    "chontaduro": {"description": "A starchy fruit, often eaten with salt. Rich in nutrients.", "origin": "Colombia", "price": 2.25}
}

# Enhanced global state storage - now includes purchase history
user_states = {}

# Existing tools from the demo
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
    user_state = user_states.get(user_id, {"preferences": [], "purchases": []})
    
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
    user_state = user_states.get(user_id, {"preferences": [], "purchases": []})
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

# TODO: Implement this new tool to record a fruit purchase
@tool
def purchase_fruit(user_id: str, fruit_name: str, quantity: int) -> str:
    """Records a fruit purchase in the user's purchase history.
    
    Args:
        user_id: The ID of the user making the purchase.
        fruit_name: The name of the fruit being purchased.
        quantity: The quantity of fruit being purchased.
        
    Returns:
        A confirmation message with purchase details.
    """
    # TODO: Create a purchase record and add it to the user's state
    # Be sure to include timestamp, fruit name, quantity, price, and total cost
    # Make sure to initialize the purchases list if it doesn't exist
    pass

# TODO: Implement this new tool to retrieve purchase history
@tool
def get_purchase_history(user_id: str) -> List[Dict]:
    """Retrieves the purchase history for a user.
    
    Args:
        user_id: The ID of the user whose purchase history to retrieve.
        
    Returns:
        A list of the user's past purchases.
    """
    # TODO: Return the user's purchase history
    # Make sure to handle the case where there is no purchase history
    pass

# TODO: Implement this new tool to provide a summary of purchases
@tool
def get_purchase_summary(user_id: str) -> Dict:
    """Calculates a summary of the user's purchase history.
    
    Args:
        user_id: The ID of the user whose purchase summary to calculate.
        
    Returns:
        A dictionary containing the total spent, number of transactions, 
        and most purchased fruit.
    """
    # TODO: Calculate a summary of the user's purchase history
    # The summary should include:
    # - Total amount spent
    # - Total number of transactions
    # - Most frequently purchased fruit
    # - Total number of fruits purchased
    pass

class EnhancedFruitAdvisorAgent(ToolCallingAgent):
    """
    Enhanced agent for providing information about Colombian fruits,
    remembering user preferences, and tracking purchase history.
    """
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                get_fruit_description,
                add_fruit_preference,
                get_user_preferences,
                save_user_state,
                purchase_fruit,  # New tool
                get_purchase_history,  # New tool
                get_purchase_summary,  # New tool
            ],
            model=model,
            name="enhanced_fruit_advisor_agent",
            description="""
            You are a helpful assistant specializing in Colombian fruits.
            You help users learn about various Colombian fruits, remember their preferences,
            and now you can also process fruit purchases and track purchase history.
            
            Use the tools available to you to:
            - Retrieve information about fruits
            - Manage user preferences
            - Process purchases
            - Provide purchase history and summaries
            
            Be enthusiastic and informative about Colombian fruits!
            """,
        )

class EnhancedOrchestratorAgent(ToolCallingAgent):
    """
    Orchestrates the enhanced fruit advisor system with purchase tracking.
    """
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[],
            model=model,
            name="enhanced_orchestrator_agent",
            description="""
            You are an orchestrator agent that manages the Colombian fruit advisory system.
            Your role is to coordinate interactions between users and the fruit advisor agent,
            ensuring that user state (preferences and purchase history) is properly 
            managed and preserved across sessions.
            """,
        )
        self.fruit_advisor = EnhancedFruitAdvisorAgent(model)

    def process_user_message(self, user_id: str, message: str) -> str:
        """
        Process a user message and return a response.
        
        Args:
            user_id: The ID of the user.
            message: The user's message.
            
        Returns:
            A response from the fruit advisor agent.
        """
        # Load any existing preferences
        current_preferences = get_user_preferences(user_id)
        
        # TODO: Load any existing purchase history
        # You'll need to use your get_purchase_history tool
        
        # Construct a prompt for the fruit advisor agent
        prompt = f"""
        User ID: {user_id}
        Current preferences: {', '.join(current_preferences) if current_preferences else 'None yet'}
        
        The user says: "{message}"
        
        If the user is asking about a fruit, use get_fruit_description to provide information.
        If the user expresses interest in a fruit, use add_fruit_preference to save it.
        If the user wants to know their preferences, use get_user_preferences.
        
        If the user wants to buy a fruit, use purchase_fruit to record the transaction.
        If the user asks about their purchase history, use get_purchase_history.
        If the user wants a summary of their purchases, use get_purchase_summary.
        
        Remember to save the user's state with save_user_state before ending the conversation.
        
        Respond in a friendly, informative way in both English and a bit of Spanish.
        """
        
        return self.fruit_advisor.run(prompt)

def run_enhanced_demo():
    """
    Runs the enhanced fruit advisor demo with purchase tracking.
    """
    print("🍎 Enhanced Colombian Fruit Market with Purchase Tracking 🍍")
    print("="*70)
    
    orchestrator = EnhancedOrchestratorAgent(model)
    user_id = "user123"
    
    print("\n--- First Session ---")
    
    # Simulated user messages
    messages = [
        "Hi! What kind of fruits do you have?",
        "Tell me about lulo.",
        "I'd like to buy 3 lulos please.",
        "I also want to try mango. Can I buy 2 mangos?",
        "What's my purchase history so far?",
        "Can you give me a summary of my purchases?",
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
    print("--- Simulating a System Restart ---")
    print("The system is restarting and will reload user state...")
    print("="*70)
    
    # Create a new orchestrator (simulating a system restart)
    new_orchestrator = EnhancedOrchestratorAgent(model)
    
    # Continue the conversation in the new session
    print("\n--- Continuing in New Session ---")
    
    # New set of messages
    new_messages = [
        "Hello again! I'd like to see my purchase history.",
        "Great! I'd like to buy 4 granadillas.",
        "Can you give me an updated summary of all my purchases?",
        "Thank you for your help!"
    ]
    
    # Process each new message
    for i, message in enumerate(new_messages):
        print(f"\nUser: {message}")
        response = new_orchestrator.process_user_message(user_id, message)
        print(f"Agent: {response}")
        time.sleep(0.5)  # Brief pause for readability
    
    # Final state check
    # TODO: Add code to display final purchase history and summary
    
    print("\n" + "="*70)
    print("Demo complete! This demonstrates state persistence and transaction tracking across sessions.")

if __name__ == "__main__":
    run_enhanced_demo()