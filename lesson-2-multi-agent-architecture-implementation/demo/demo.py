import os
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from typing import Dict, Any, List
import json

load_dotenv()

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
    api_base="https://openai.vocareum.com/v1",
)

# Global state for simplicity
DISTRIBUTION_HISTORY = {}
@tool
def check_history(penguin_name: str) -> Dict[str, Any]:
    """
    Check the recent resource distribution history for a specific penguin.

    This function retrieves and calculates the recent food distribution 
    and tool ownership status for a given penguin.

    Args:
        penguin_name (str): The unique name or identifier of the penguin 
                            whose resource distribution history will be 
                            retrieved and analyzed.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'recent_food' (int): Total food received in the last 3 rounds
            - 'has_tool' (bool): Whether the penguin has possessed a tool 
                                 in recent history
    """
    history = DISTRIBUTION_HISTORY.get(penguin_name, [])
    recent_food = sum(h["food"] for h in history[-3:]) if history else 0
    has_tool = any(h["has_tool"] for h in history) if history else False
    return {"recent_food": recent_food, "has_tool": has_tool}

@tool
def record_distribution(penguin_name: str, food: int, has_tool: bool) -> str:
    """
    Record the distribution of resources to a specific penguin.

    This function logs the resources (food and tool) given to a penguin 
    in the global distribution history.

    Args:
        penguin_name: The unique name or identifier of the penguin 
                            receiving resources.
        food: The number of food units being given to the penguin. 
                    This represents the quantity of food distributed in 
                    this specific transaction. Must be an integer between 
                    0 and 5.
        has_tool: A flag indicating whether a tool is being distributed. 
                     True means a tool is given, False means no tool is given.

    Returns:
        str: A confirmation message describing the resource distribution.
    """
    if penguin_name not in DISTRIBUTION_HISTORY:
        DISTRIBUTION_HISTORY[penguin_name] = []
    DISTRIBUTION_HISTORY[penguin_name].append({"food": food, "has_tool": has_tool})
    return f"Recorded: {penguin_name} got {food} food and {'a' if has_tool else 'no'} tool"
class ScientistAgent(ToolCallingAgent):
    def __init__(self, initial_food_supply: int = 20, refresh_interval: int = 5) -> None:
        super().__init__(
            tools=[check_history, record_distribution],
            model=model,
            name="scientist",
            description="A scientist responding to penguin actions"
        )
        self.initial_food_supply = initial_food_supply
        self.food_supply = initial_food_supply
        self.tool_available = True
        self.refresh_interval = refresh_interval
        self.turn_counter = 0

    def refresh_resources(self):
        """Periodically refresh the scientist's food supply."""
        self.food_supply = self.initial_food_supply
        self.tool_available = True
        print(f"\n🔄 Scientist Resources Refreshed!")
        print(f"Food Supply Reset to: {self.food_supply}")
        print(f"Tool Availability Reset to: {self.tool_available}")

    def respond_to_action(self, penguin: 'PenguinAgent', penguin_action: Dict[str, Any]) -> None:
        """Respond to a penguin's action with enhanced diagnostics."""
        self.turn_counter += 1
        
        # Check for resource refresh
        if self.turn_counter % self.refresh_interval == 0:
            self.refresh_resources()

        # Pre-action diagnostics
        print(f"\n--- Turn {self.turn_counter}: Scientist Responds to {penguin.name} ---")
        print(f"Penguin Action: {penguin_action}")
        print(f"Penguin State:")
        print(f"  - Food: {penguin.food}")
        print(f"  - Has Tool: {penguin.has_tool}")
        
        history = check_history(penguin.name)
        print(f"Penguin History:")
        print(f"  - Recent Food: {history['recent_food']}")
        print(f"  - Has Had Tool: {history['has_tool']}")
        
        print(f"\nScientist Resources:")
        print(f"  - Food Supply: {self.food_supply}")
        print(f"  - Tool Available: {self.tool_available}")

        response = self.run(
            f"""Penguin {penguin.name} took action: {penguin_action}
            Penguin's current state:
            - Food: {penguin.food}
            - Has Tool: {penguin.has_tool}
            
            Recent History: {history['recent_food']} recent food, {'has' if history['has_tool'] else 'no'} tool.
            Available Scientist Resources: {self.food_supply} food, Tool: {self.tool_available}
            
            Respond with JSON: {{"give_food": <0-5>, "give_tool": <bool>}}"""
        )
        
        try:
            # Check if response is already a dictionary
            if isinstance(response, dict):
                decision = response
            else:
                # If it's a string, try to parse the JSON
                decision = json.loads(response.split("final_answer:")[-1].strip())
            
            food = min(int(decision.get('give_food', 0)), self.food_supply)
            tool = decision.get('give_tool', False) and self.tool_available
            
            # Post-decision diagnostics
            print(f"\nScientist's Decision:")
            print(f"  - Food to Give: {food}")
            print(f"  - Tool to Give: {tool}")

            if food > 0:
                self.food_supply -= food
                penguin.food += food
            if tool:
                penguin.has_tool = True
                self.tool_available = False
                
            record_distribution(penguin.name, food, tool)
            
            # Post-action state
            print(f"\nPost-Action State:")
            print(f"Scientist Resources:")
            print(f"  - Remaining Food Supply: {self.food_supply}")
            print(f"  - Tool Available: {self.tool_available}")
            print(f"Penguin {penguin.name}:")
            print(f"  - Food: {penguin.food}")
            print(f"  - Has Tool: {penguin.has_tool}")
            
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error processing scientist's response: {e}")

class PenguinAgent(ToolCallingAgent):
    def __init__(self, name: str) -> None:
        super().__init__(tools=[], model=model, name=name)
        self.name = name
        self.food = 0
        self.has_tool = False

    def take_action(self) -> Dict[str, Any]:
        """Penguin decides on an action each round."""
        history = check_history(self.name)
        
        response = self.run(
            f"""You are Penguin {self.name}. You have no persistent resources from previous rounds.
            
            Decide on an action:
            1. Request food for this round
            2. Request tool for this round
            
            You cannot hoard resources between rounds. 
            Each round is a fresh start.
            
            Respond with JSON: {{"action": <action_string>, "details": <additional_details>}}"""
        )
        
        try:
            action = json.loads(response.split("final_answer:")[-1].strip())
            return action
        except json.JSONDecodeError:
            print(f"Error processing {self.name}'s action")
            return {"action": "request_food", "details": "default safe action"}

def run_simulation():
    scientist = ScientistAgent(initial_food_supply=20, refresh_interval=5)
    penguins = [PenguinAgent(f"Penguin {i}") for i in range(4)]
    
    print("\nStarting Simulation...")
    for round in range(3):
        print(f"\n{'='*50}")
        print(f"ROUND {round + 1}")
        print(f"{'='*50}")
        
        # Penguins take actions
        penguin_actions = {}
        for penguin in penguins:
            action = penguin.take_action()
            penguin_actions[penguin.name] = action
            print(f"{penguin.name} Action: {action}")
        
        # Scientist responds to actions
        for penguin in penguins:
            scientist.respond_to_action(penguin, penguin_actions[penguin.name])
    
    print(f"\nFinal State:")
    print(f"Remaining: {scientist.food_supply} food, {'🔨' if scientist.tool_available else ''}")
    
    for penguin in penguins:
        history = check_history(penguin.name)
        print(f"{penguin.name} - Total Food: {penguin.food}, Has Tool: {history['has_tool']}")
        
if __name__ == "__main__":
    run_simulation()