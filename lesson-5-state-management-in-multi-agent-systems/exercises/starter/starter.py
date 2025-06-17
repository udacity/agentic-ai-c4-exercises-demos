"""
EXERCISE: Multi-Agent State Management with Purchase Tracking

In this exercise, you'll extend the Colombian Fruit Market system by implementing
purchase tracking functionality while maintaining the multi-agent orchestration patterns.

You'll need to:
1. Implement the purchase_fruit tool to record transactions
2. Implement get_purchase_summary tool for analytics
3. Add the purchase summary tool to the PurchaseAgent
4. Add 'summary' action support to the orchestrator's handle_purchase tool

This demonstrates extending multi-agent systems with new capabilities while
maintaining proper orchestration patterns.
"""

from typing import Dict, List, Any
import os
import dotenv
import time
from datetime import datetime
from collections import Counter

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
    return f"User state for {user_id} saved successfully."

# TODO: Implement this tool to record fruit purchases
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
    # TODO: Implement purchase recording with timestamps and pricing
    pass

@tool
def get_purchase_history(user_id: str) -> List[Dict]:
    """Retrieves the purchase history for a user.
    
    Args:
        user_id: The ID of the user whose purchase history to retrieve.
        
    Returns:
        A list of the user's past purchases.
    """
    if user_id not in user_states:
        user_states[user_id] = {"preferences": [], "purchases": []}
    elif "purchases" not in user_states[user_id]:
        user_states[user_id]["purchases"] = []
    
    return user_states[user_id]["purchases"]

# TODO: Implement this tool for purchase analytics
@tool
def get_purchase_summary(user_id: str) -> Dict:
    """Calculates a summary of the user's purchase history.
    
    Args:
        user_id: The ID of the user whose purchase summary to calculate.
        
    Returns:
        A dictionary containing the total spent, number of transactions, 
        and most purchased fruit.
    """
    # TODO: Implement purchase summary calculation
    pass

# Specialized agents with real responsibilities

class FruitInfoAgent(ToolCallingAgent):
    """Agent specialized in providing fruit information."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[get_fruit_description],
            model=model,
            name="fruit_info_agent",
            description="Provides detailed information about Colombian fruits.",
        )

class PreferenceAgent(ToolCallingAgent):
    """Agent specialized in managing user preferences."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[add_fruit_preference, get_user_preferences, save_user_state],
            model=model,
            name="preference_agent",
            description="Manages user fruit preferences and saves user state.",
        )

class PurchaseAgent(ToolCallingAgent):
    """Agent specialized in handling purchases and purchase history."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[purchase_fruit, get_purchase_history],  # TODO: Add get_purchase_summary
            model=model,
            name="purchase_agent",
            description="Handles fruit purchases, purchase history, and purchase summaries.",
        )

class Orchestrator(ToolCallingAgent):
    """Orchestrator that coordinates workflow between specialized agents."""
    
    def __init__(self, model: OpenAIServerModel):
        self.model = model
        
        # Initialize specialized agents
        self.fruit_info = FruitInfoAgent(model)
        self.preferences = PreferenceAgent(model)
        self.purchases = PurchaseAgent(model)

        @tool
        def get_fruit_info(fruit_name: str) -> str:
            """Get information about a specific fruit.
            
            Args:
                fruit_name: Name of the fruit to get information about
                
            Returns:
                Detailed fruit information
            """
            return self.fruit_info.run(f"Tell me about {fruit_name}. Use get_fruit_description to provide detailed information.")

        @tool
        def manage_preferences(user_id: str, action: str, fruit_name: str = None) -> str:
            """Manage user fruit preferences.
            
            Args:
                user_id: ID of the user
                action: Either 'add', 'get', or 'save'
                fruit_name: Name of fruit (required for 'add' action)
                
            Returns:
                Result of preference management operation
            """
            if action == "add" and fruit_name:
                return self.preferences.run(f"Add {fruit_name} to preferences for user {user_id}")
            elif action == "get":
                return self.preferences.run(f"Get current preferences for user {user_id}")
            elif action == "save":
                return self.preferences.run(f"Save state for user {user_id}")
            return "Invalid preference action"

        @tool
        def handle_purchase(user_id: str, action: str, fruit_name: str = None, quantity: int = None) -> str:
            """Handle purchase operations including buying fruits and viewing history.
            
            Args:
                user_id: ID of the user
                action: Either 'buy', 'history', or 'summary'
                fruit_name: Name of fruit (required for 'buy')
                quantity: Quantity to purchase (required for 'buy')
                
            Returns:
                Result of purchase operation
            """
            if action == "buy" and fruit_name and quantity:
                return self.purchases.run(f"Record purchase for user {user_id}: {quantity} {fruit_name}")
            elif action == "history":
                return self.purchases.run(f"Get purchase history for user {user_id}")
            # TODO: Add support for action == "summary"
            return "Invalid purchase action"

        super().__init__(
            tools=[get_fruit_info, manage_preferences, handle_purchase],
            model=model,
            name="orchestrator",
            description="""
            Orchestrates the Colombian fruit market system by coordinating between
            specialized agents for fruit information, preferences, and purchases.
            
            Workflow:
            1. For fruit questions: Route to fruit info agent
            2. For preference management: Route to preference agent  
            3. For purchases: Route to purchase agent, then update preferences if new fruit
            4. For purchase history/summaries: Route to purchase agent
            5. Always save state after preference or purchase changes
            """,
        )

    def process_user_message(self, user_id: str, message: str) -> str:
        """
        Process user message through coordinated agent workflow.
        """
        # Load current user context
        current_preferences = get_user_preferences(user_id)
        purchase_history = get_purchase_history(user_id)
        
        context = f"""
        User ID: {user_id}
        Current preferences: {current_preferences}
        Purchase history: {len(purchase_history)} previous transactions
        
        User message: "{message}"
        
        Coordinate the appropriate agents to handle this request:
        - Use get_fruit_info for fruit information requests
        - Use manage_preferences for adding/viewing preferences  
        - Use handle_purchase for buying fruits, viewing purchase history, or getting purchase summaries
        - Always save user state after preference or purchase changes
        
        Follow this workflow:
        1. If asking about fruit info → get fruit info
        2. If expressing interest → get info first, then add to preferences
        3. If wanting to buy → handle purchase, then add to preferences if new fruit
        4. If asking for purchase history → handle_purchase with action 'history'
        5. If asking for purchase summary → handle_purchase with action 'summary'
        6. Always save state after changes
        """
        
        return self.run(context)

def run_demo():
    """
    Runs the fruit advisor demo with purchase tracking and real orchestration.
    """
    print("🍎 Colombian Fruit Market with Purchase Tracking 🍍")
    print("="*70)
    
    orchestrator = Orchestrator(model)
    user_id = "miercoles"
    
    print("\n--- First Session ---")
    
    messages = [
        "Hi! What kind of fruits do you have?",
        "Tell me about lulo.",
        "I'd like to buy 3 lulos please.",
        "I also want to try mango. Can I buy 2 mangos?",
        "What's my purchase history so far?",
        "Can you give me a summary of my purchases?",
        "Thank you! I'll come back later."
    ]
    
    for message in messages:
        print(f"\nUser: {message}")
        response = orchestrator.process_user_message(user_id, message)
        print(f"Agent: {response}")
        time.sleep(0.5)
    
    print("\n--- Continuing in New Session ---")
    
    new_messages = [
        "Hello again! I'd like to see my purchase history.",
        "Great! I'd like to buy 4 granadillas.",
        "Can you give me an updated summary of all my purchases?",
        "Thank you for your help!"
    ]
    
    for message in new_messages:
        print(f"\nUser: {message}")
        response = orchestrator.process_user_message(user_id, message)
        print(f"Agent: {response}")
        time.sleep(0.5)
    
    # Final state check - display purchase history and summary
    purchase_history = get_purchase_history(user_id)
    # TODO: Uncomment when get_purchase_summary is implemented
    # purchase_summary = get_purchase_summary(user_id)
    
    print("\n" + "="*70)
    print("Final Purchase History:")
    for i, purchase in enumerate(purchase_history):
        print(f"  {i+1}. [Purchase details will show when purchase_fruit is implemented]")
    
    print("\nPurchase Summary:")
    print("  [Summary will show when get_purchase_summary is implemented]")
    
    print("\n" + "="*70)
    print("Demo complete! This demonstrates state persistence and transaction tracking across sessions.")

if __name__ == "__main__":
    run_demo()