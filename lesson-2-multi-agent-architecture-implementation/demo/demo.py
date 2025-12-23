import os
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from typing import Dict, Any, List, Optional
import json
import random

load_dotenv()

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
    api_base="https://openai.vocareum.com/v1",
)

DISTRIBUTION_HISTORY: Dict[str, List[Dict[str, Any]]] = {}

@tool
def check_history(penguin_name: str) -> Dict[str, Any]:
    """
    Check the recent resource distribution history for a specific penguin.

    Args:
        penguin_name (str): The unique name or identifier of the penguin.

    Returns:
        Dict[str, Any]: A dictionary containing recent_food (int) and has_tool (bool) status.
    """
    history = DISTRIBUTION_HISTORY.get(penguin_name, [])
    recent_food = sum(h["food"] for h in history[-3:]) if history else 0
    has_tool = any(h["has_tool"] for h in history) if history else False
    return {"recent_food": recent_food, "has_tool": has_tool}

@tool
def record_distribution(penguin_name: str, food: int, has_tool: bool) -> str:
    """
    Record the distribution of resources to a specific penguin.

    Args:
        penguin_name (str): The unique name or identifier of the penguin.
        food (int): The number of food units being given (0-5).
        has_tool (bool): Whether a tool is being distributed.

    Returns:
        str: A confirmation message.
    """
    if penguin_name not in DISTRIBUTION_HISTORY:
        DISTRIBUTION_HISTORY[penguin_name] = []
    DISTRIBUTION_HISTORY[penguin_name].append({"food": food, "has_tool": has_tool})
    return f"Recorded: {penguin_name} got {food} food and {'a tool' if has_tool else 'no tool'}."

class PenguinAgent(ToolCallingAgent):
    def __init__(self, name: str, initial_food: int = 0, initial_has_tool: bool = False) -> None:
        super().__init__(tools=[], model=model, name=name, description=f"A penguin named {name}")
        self.name = name
        self.food = initial_food
        self.has_toy = initial_has_tool

    def take_action(self) -> Dict[str, Any]:
        self.memory.steps = []
        print(f"\n--- {self.name}'s Turn to Act (Demo Version) ---")
        print(f"Current State for {self.name}: Food: {self.food}, Has Toy: {self.has_toy}")
        
        prompt = f"""
        You are Penguin {self.name}.
        Your current (temporary for this round) state: Food: {self.food}, Has Toy: {self.has_toy}.
        
        Decide what to request from the scientist for this round:
        1. Request some food.
        2. Request a tool.
        
        State your request naturally.
        """
        
        final_llm_text_response = self.run(prompt)
        
        action_result: Dict[str, Any] = {}
        text_to_parse = final_llm_text_response

        if self.memory.steps:
            last_step = self.memory.steps[-1]
            if hasattr(last_step, 'tool_calls') and last_step.tool_calls and \
               hasattr(last_step.tool_calls[0], 'name') and last_step.tool_calls[0].name == 'final_answer' and \
               hasattr(last_step.tool_calls[0], 'arguments'):
                text_to_parse = str(last_step.tool_calls[0].arguments.get('answer', final_llm_text_response))

        response_str_lower = text_to_parse.lower()
        if "tool" in response_str_lower:
            action_result = {"action_type": "request_tool", "details": text_to_parse}
        else:
            action_result = {"action_type": "request_food", "details": text_to_parse}
        
        print(f"  {self.name} (Request): Type: {action_result.get('action_type')}, Details: '{action_result.get('details')}'")
        return action_result

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
        self.toy_available = True
        self.refresh_interval = refresh_interval
        self.turn_counter = 0

    def refresh_resources(self):
        self.food_supply = self.initial_food_supply
        self.toy_available = True
        print(f"\n🔄 Scientist Resources Refreshed!")
        print(f"Food Supply Reset to: {self.food_supply}")
        print(f"Toy Availability Reset to: {self.toy_available}")

    def respond_to_action(self, penguin: 'PenguinAgent', penguin_action_details: Dict[str, Any]) -> None:
        self.turn_counter += 1
        if self.turn_counter > 0 and (self.turn_counter -1) % self.refresh_interval == 0 and self.turn_counter != 1:
            self.refresh_resources()

        print(f"\n--- Turn {self.turn_counter}: Scientist Responds to {penguin.name} ---")
        print(f"Penguin Action/Request: {penguin_action_details}")
        print(f"Penguin State (before scientist action): Food: {penguin.food}, Has Toy: {penguin.has_toy}")

        current_penguin_history = check_history(penguin.name)
        print(f"Penguin History (direct check for {penguin.name}): Recent Food Recorded: {current_penguin_history['recent_food']}, Has Been Given Tool (in history): {current_penguin_history['has_tool']}")
        print(f"\nScientist Resources (before decision): Food Supply: {self.food_supply}, Toy Available: {self.toy_available}")

        self.memory.steps = []
        prompt = f"""
        You are the Scientist. Respond to penguin {penguin.name}'s request.
        Penguin {penguin.name} requested: {json.dumps(penguin_action_details)}
        Penguin's current state (for this round): Food: {penguin.food}, Has Toy: {penguin.has_toy}.
        Recorded history for {penguin.name} (past distributions): {json.dumps(current_penguin_history)}.
        Your available resources: Food Supply: {self.food_supply}, Toy Available: {self.toy_available}.

        Based on the request and available resources, decide:
        - How much food to give (integer 0-5).
        - Whether to give a tool (true or false).
        You MUST use the 'record_distribution' tool ONCE to log your decision.
        Example: record_distribution(penguin_name="{penguin.name}", food=3, has_tool=True)
        After the tool call, your final text response should be simple, like "Distribution decision logged."
        """
        
        final_llm_text_response = self.run(prompt)

        llm_intended_food = 0
        llm_intended_tool = False
        record_distribution_tool_called_by_llm = False
        confirmation_msg_from_tool_execution = "Tool 'record_distribution' output not found."
        
        for step in self.memory.steps:
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tc in step.tool_calls:
                    if tc.name == 'record_distribution':
                        llm_intended_food = tc.arguments.get('food', 0)
                        llm_intended_tool = tc.arguments.get('has_tool', False)
                        record_distribution_tool_called_by_llm = True
                        if hasattr(step, 'observations') and step.observations is not None:
                            confirmation_msg_from_tool_execution = str(step.observations)
                        else:
                            confirmation_msg_from_tool_execution = f"Observation missing on step with record_distribution call (ID: {tc.id})"
                        break 
            if record_distribution_tool_called_by_llm:
                break 
        
        if record_distribution_tool_called_by_llm:
            print(f"\nScientist LLM intended (via 'record_distribution' args): Food: {llm_intended_food}, Tool: {llm_intended_tool}")
            print(f"  Confirmation from tool execution (observation): {confirmation_msg_from_tool_execution}")
        else:
            print(f"\nWARN: Scientist LLM did NOT use 'record_distribution' tool as instructed. LLM's final text response was: '{final_llm_text_response}'")

        actual_food_given = 0
        actual_tool_given = False

        if record_distribution_tool_called_by_llm:
            food_to_attempt = min(max(0, int(llm_intended_food)), 5) 
            if food_to_attempt > 0:
                if self.food_supply >= food_to_attempt:
                    actual_food_given = food_to_attempt
                else: 
                    actual_food_given = self.food_supply 
                if actual_food_given > 0:
                    self.food_supply -= actual_food_given
                    self.food_supply = max(0, self.food_supply)
                    penguin.food += actual_food_given
                if actual_food_given < food_to_attempt : 
                    print(f"  INFO: Scientist intended {food_to_attempt} food, but only {actual_food_given} was available/given.")
            
            if llm_intended_tool:
                if self.toy_available:
                    actual_tool_given = True
                    penguin.has_toy = True 
                    self.toy_available = False 
                else:
                    print(f"  INFO: Scientist LLM intended to give a toy to {penguin.name}, but no toy was available.")
        
        print(f"\nScientist's Actual Distribution (applied after constraints): Food Actually Given to {penguin.name}: {actual_food_given}, Toy Actually Given to {penguin.name}: {actual_tool_given}")
        print(f"\nPost-Action State (End of Scientist's turn for {penguin.name}):")
        print(f"Scientist Resources: Remaining Food Supply: {self.food_supply}, Toy Available: {self.toy_available}")
        print(f"Penguin {penguin.name}: Food: {penguin.food}, Has Toy: {penguin.has_toy}")

def run_simulation(num_rounds=3, num_penguins=2):
    penguin_names = ['miercoles', 'putzie'] 
    
    scientist = ScientistAgent(initial_food_supply=20, refresh_interval=3)
    
    penguins = []
    for i in range(num_penguins):
        name_base = penguin_names[i % len(penguin_names)]
        name_suffix = str(i // len(penguin_names) + 1) if num_penguins > len(penguin_names) and i >= len(penguin_names) else ""
        if num_penguins <= len(penguin_names) : # if only 1 or 2 penguins, don't add suffix to first ones
             name_suffix = str(i + 1) # Use 1-based indexing for names like miercoles1, putzie2
        else: # For more penguins than names, use base + number
            name_suffix = str(i // len(penguin_names) +1) if i >= len(penguin_names) else str(i+1)
            if i < len(penguin_names): # ensure miercoles1, putzie2
                 name = f"{name_base}{str(i+1)}"
            else: # miercoles2, putzie3 etc for more
                 name = f"{name_base}{str(i // len(penguin_names) + 1)}"


        # Simplified naming for consistency with common patterns if num_penguins matches len(penguin_names)
        if num_penguins == 2 and i < 2 :
            name = f"{penguin_names[i]}{i+1}"
        elif num_penguins == 1:
            name = f"{penguin_names[0]}1"
        else: # Fallback for other cases, can be refined
            name = f"{name_base}{i+1}"


        penguins.append(PenguinAgent(name, initial_food=0, initial_has_tool=False)) 
    
    print(f"\n🐧🐟🛠️ Starting DEMO Simulation! ({num_rounds} rounds, {num_penguins} penguins) 🛠️🐟🐧")
    
    for r_num in range(num_rounds):
        print(f"\n{'='*50}\nROUND {r_num + 1} of {num_rounds}\n{'='*50}")
        
        for p in penguins:
            p.food = 0
            p.has_toy = False
            # print(f"Resetting {p.name} for new round: Food: {p.food}, Toy: {p.has_toy}")

        all_penguin_actions: Dict[str, Dict[str,Any]] = {}
        for penguin in penguins:
            action_details = penguin.take_action()
            all_penguin_actions[penguin.name] = action_details
        
        print(f"\n--- Scientist's Interaction Phase (Round {r_num + 1}) ---")
        for penguin in penguins:
            penguin_action_details = all_penguin_actions[penguin.name]
            scientist.respond_to_action(penguin, penguin_action_details)
    
    print(f"\n{'='*50}\n🏁 FINAL DEMO SIMULATION STATE 🏁\n{'='*50}")
    print(f"Scientist Final Resources: Food: {scientist.food_supply}, Toy Available: {'🔨' if scientist.toy_available else 'No Toy'}")
    print(f"Total turns scientist responded: {scientist.turn_counter}")
    
    print("\nPenguins' Final States (reflects last round's distribution):")
    for penguin in penguins:
        print(f"  {penguin.name}: Food: {penguin.food}, Has Tool: {penguin.has_toy}")
        history = check_history(penguin.name)
        print(f"    Overall History (recorded by scientist): Recent Food: {history['recent_food']}, Had Tool Ever: {history['has_tool']}")
        
if __name__ == "__main__":
    run_simulation()