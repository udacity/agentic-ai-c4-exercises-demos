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
    "lulo": {"description": "A tart, citrusy fruit, often used in juice.", "origin": "Colombia", "price": 2.50},
    "mango": {"description": "Sweet and juicy, a tropical favorite.", "origin": "Colombia", "price": 1.75},
    "granadilla": {"description": "A sweet and seedy fruit with a hard shell.", "origin": "Colombia", "price": 3.00},
    "chontaduro": {"description": "A starchy fruit, often eaten with salt. Rich in nutrients.", "origin": "Colombia", "price": 2.25}
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

@tool
def record_purchase(user_id: str, fruit_name: str, quantity: int) -> str:
    """Records a fruit purchase.
    
    Args:
        user_id: The ID of the user making the purchase.
        fruit_name: The name of the fruit being purchased.
        quantity: The quantity being purchased.
        
    Returns:
        A confirmation message with purchase details.
    """
    if fruit_name not in fruit_data:
        return f"Sorry, {fruit_name} is not available."
    
    price = fruit_data[fruit_name]["price"]
    total = price * quantity
    
    user_state = user_states.get(user_id, {"preferences": [], "purchases": []})
    if "purchases" not in user_state:
        user_state["purchases"] = []
    
    purchase = {
        "fruit": fruit_name,
        "quantity": quantity,
        "total_cost": total,
        "timestamp": time.time()
    }
    
    user_state["purchases"].append(purchase)
    user_states[user_id] = user_state
    
    return f"Purchase recorded: {quantity} {fruit_name}(s) for ${total:.2f}"

@tool
def get_purchase_history(user_id: str) -> List[Dict]:
    """Gets user's purchase history.
    
    Args:
        user_id: The ID of the user.
        
    Returns:
        List of purchase records.
    """
    user_state = user_states.get(user_id, {"preferences": [], "purchases": []})
    return user_state.get("purchases", [])

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
    """Agent specialized in handling purchases."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[record_purchase, get_purchase_history],
            model=model,
            name="purchase_agent",
            description="Handles fruit purchases and purchase history.",
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
            """Handle purchase operations.
            
            Args:
                user_id: ID of the user
                action: Either 'buy' or 'history'
                fruit_name: Name of fruit (required for 'buy')
                quantity: Quantity to purchase (required for 'buy')
                
            Returns:
                Result of purchase operation
            """
            if action == "buy" and fruit_name and quantity:
                return self.purchases.run(f"Record purchase for user {user_id}: {quantity} {fruit_name}")
            elif action == "history":
                return self.purchases.run(f"Get purchase history for user {user_id}")
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
            3. For purchases: Route to purchase agent, then update preferences
            4. Always save state after preference/purchase changes
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
        Purchase history: {len(purchase_history)} previous purchases
        
        User message: "{message}"
        
        Coordinate the appropriate agents to handle this request:
        - Use get_fruit_info for fruit information requests
        - Use manage_preferences for adding/viewing preferences  
        - Use handle_purchase for buying fruits or viewing purchase history
        - Always save user state after preference or purchase changes
        
        Follow this workflow:
        1. If asking about fruit info → get fruit info
        2. If expressing interest → get info first, then add to preferences
        3. If wanting to buy → handle purchase, then add to preferences if new
        4. Always save state after changes
        """
        
        return self.run(context)

def run_demo():
    """
    Runs the fruit advisor demo with real orchestration.
    """
    print("🍎 Colombian Fruit Market - True Orchestration Demo 🍍")
    print("="*60)
    
    orchestrator = Orchestrator(model)
    user_id = "putzie"
    
    print("\n--- Session 1: Demonstrating Agent Coordination ---")
    
    messages = [
        "Hi! What can you tell me about lulo?",
        "That sounds great! I'd like to try it - add it to my preferences.",
        "What about mango? I love tropical fruits.",
        "Perfect! Add mango to my preferences too.",
        "I want to buy 3 lulos please.",
        "Can I also buy 2 mangoes?",
        "What are my current preferences?",
        "Show me my purchase history.",
    ]
    
    for message in messages:
        print(f"\nUser: {message}")
        response = orchestrator.process_user_message(user_id, message)
        print(f"System: {response}")
        time.sleep(0.5)
    
    print("\n--- Session 2: State Persistence Test ---")
    
    # Create new orchestrator to test state persistence
    new_orchestrator = Orchestrator(model)
    
    persistence_messages = [
        "Hi again! What are my preferences?",
        "What's my purchase history?",
        "I'd like to try granadilla - tell me about it first.",
        "Sounds good! Add it to my preferences and I'll buy 1.",
    ]
    
    for message in persistence_messages:
        print(f"\nUser: {message}")
        response = new_orchestrator.process_user_message(user_id, message)
        print(f"System: {response}")
        time.sleep(0.5)
    
    # Final state
    final_preferences = get_user_preferences(user_id)
    final_purchases = get_purchase_history(user_id)
    
    print("\n" + "="*60)
    print("FINAL STATE:")
    print(f"Preferences: {final_preferences}")
    print(f"Purchases: {len(final_purchases)} total")
    for purchase in final_purchases:
        print(f"  - {purchase['quantity']} {purchase['fruit']} for ${purchase['total_cost']:.2f}")
    print("="*60)
    print("\nDemo complete! Shows real orchestration with agent coordination.")

if __name__ == "__main__":
    run_demo()
