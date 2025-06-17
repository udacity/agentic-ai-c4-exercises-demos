import os
from dotenv import load_dotenv
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from typing import Dict, Any, List, Optional # Added Optional
import json
import random

load_dotenv()

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
    api_base="https://openai.vocareum.com/v1",
)

# Global state for simplicity
DISTRIBUTION_HISTORY: Dict[str, List[Dict[str, Any]]] = {} # Type hint for clarity

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
    # print(f"TOOL EXEC [record_distribution]: For {penguin_name}, food {food}, tool {has_tool}.") # For debugging
    return f"Recorded: {penguin_name} got {food} food and {'a tool' if has_tool else 'no tool'}."

# The tool is added by the student
# --- EXAMPLE TOOL (Student can change) ---
@tool
def find_food(penguin_name: str, method: str) -> int:
    pass


class ScientistAgent(ToolCallingAgent):
    def __init__(self, initial_food_supply: int = 20, refresh_interval: int = 5) -> None:
        super().__init__(
            tools=[check_history, record_distribution], # Student might add check_history if they want scientist to use it
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

    def respond_to_action(self, penguin: 'PenguinAgent', penguin_action_details: Dict[str, Any]) -> None:
        """Respond to a penguin's action."""
        self.turn_counter += 1
        # Corrected refresh logic: should refresh at the START of the turn if condition met
        if self.turn_counter > 0 and (self.turn_counter -1) % self.refresh_interval == 0 and self.turn_counter != 1 : # Avoid refresh on very first turn unless interval is 1
            self.refresh_resources()

        print(f"\n--- Turn {self.turn_counter}: Scientist Responds to {penguin.name} ---")
        print(f"Penguin Action/Request: {penguin_action_details}") # Changed from penguin_action to penguin_action_details
        print(f"Penguin State (before scientist action):")
        print(f"  - Food: {penguin.food}")
        print(f"  - Has Tool: {penguin.has_tool}")

        # Student might want the scientist to use check_history tool
        # For now, we'll call it directly as part of the scientist's internal logic.
        # If they want the LLM to decide to use it, they'd add it to the scientist's tools
        # and prompt the LLM accordingly.
        current_penguin_history = check_history(penguin.name) 
        print(f"Penguin History (direct check for {penguin.name}):")
        print(f"  - Recent Food Recorded: {current_penguin_history['recent_food']}")
        print(f"  - Has Been Given Tool (in history): {current_penguin_history['has_tool']}")

        print(f"\nScientist Resources (before decision):")
        print(f"  - Food Supply: {self.food_supply}")
        print(f"  - Tool Available: {self.tool_available}")

        # Prepare prompt for the scientist LLM
        self.memory.steps = [] # Clear memory for this specific interaction
        prompt = f"""
        You are the Scientist. Your task is to respond to penguin {penguin.name}.
        Penguin {penguin.name}'s recent action or request was: {json.dumps(penguin_action_details)}
        Penguin's current actual state: Food: {penguin.food}, Has Tool: {penguin.has_tool}.
        
        Historical distribution for {penguin.name} (what you previously recorded): 
        Recent Food in last 3 distributions: {current_penguin_history['recent_food']}, Has had tool before: {current_penguin_history['has_tool']}.

        Your available resources:
        - Food supply: {self.food_supply} units.
        - Tool available: {self.tool_available}.

        INSTRUCTIONS:
        1. Analyze the penguin's situation (action/request, current state, history) and your available resources.
        2. Decide how many food units to give {penguin.name}. This must be an integer between 0 and 5 (inclusive).
        3. Decide whether to give {penguin.name} a tool. This must be true or false.
        4. You MUST use the 'record_distribution' tool ONCE to log your decision.
           Provide 'penguin_name' (which is "{penguin.name}"), your chosen 'food' amount, and 'has_tool' status as arguments.
           Example tool call: record_distribution(penguin_name="{penguin.name}", food=3, has_tool=True)
        5. Your final response after the tool call should be simple, like "Decision logged."
        """
        
        final_llm_text_response = self.run(prompt) # LLM makes tool call within this

        # ---- START: Robust parsing of record_distribution tool call and output ----
        llm_intended_food = 0
        llm_intended_tool = False
        record_distribution_tool_called_by_llm = False
        confirmation_msg_from_tool_execution = "Tool 'record_distribution' output not found."
        
        for step in self.memory.steps: # step is an ActionStep (or TaskStep)
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tc in step.tool_calls: # tc is a ToolCall object
                    if tc.name == 'record_distribution':
                        llm_intended_food = tc.arguments.get('food', 0)
                        llm_intended_tool = tc.arguments.get('has_tool', False)
                        record_distribution_tool_called_by_llm = True
                        # The observation for this tool call is on THIS SAME STEP
                        if hasattr(step, 'observations') and step.observations is not None:
                            confirmation_msg_from_tool_execution = str(step.observations)
                        else:
                            confirmation_msg_from_tool_execution = f"Observation missing on step with record_distribution call (ID: {tc.id})"
                        break 
            if record_distribution_tool_called_by_llm:
                break 
        # ---- END: Robust parsing ----

        if record_distribution_tool_called_by_llm:
            print(f"\nScientist LLM intended (via 'record_distribution' args):")
            print(f"  - Food for {penguin.name}: {llm_intended_food}")
            print(f"  - Tool for {penguin.name}: {llm_intended_tool}")
            print(f"  - Confirmation from tool execution (observation): {confirmation_msg_from_tool_execution}")
        else:
            # This case should ideally not happen if the prompt is followed.
            # Students might need to debug their prompts or this logic if it occurs.
            print(f"\nWARN: Scientist LLM did NOT use 'record_distribution' tool as instructed.")
            print(f"  LLM's final text response was: '{final_llm_text_response}'")
            print(f"  No resources will be distributed by the scientist this turn due to this.")


        # Actual distribution based on LLM's intent (if tool was called) and scientist's resources
        actual_food_given = 0
        actual_tool_given = False

        if record_distribution_tool_called_by_llm: # Only proceed if LLM made a decision via the tool
            # Sanitize LLM's decision for food amount (0-5)
            food_to_attempt = min(max(0, int(llm_intended_food)), 5) 
            
            if food_to_attempt > 0:
                if self.food_supply >= food_to_attempt:
                    actual_food_given = food_to_attempt
                else: # Not enough supply, give what's available
                    actual_food_given = self.food_supply 
                
                if actual_food_given > 0: # Only update if some food is actually given
                    self.food_supply -= actual_food_given
                    self.food_supply = max(0, self.food_supply) # Ensure not negative
                    penguin.food += actual_food_given
                
                if actual_food_given < food_to_attempt : 
                    print(f"  INFO: Scientist intended {food_to_attempt} food, but only {actual_food_given} was available/given.")
            
            if llm_intended_tool: # If LLM decided to give a tool
                if self.tool_available:
                    actual_tool_given = True
                    penguin.has_tool = True 
                    self.tool_available = False # Tool is now used up
                else:
                    # LLM wanted to give a tool, but none was available.
                    print(f"  INFO: Scientist LLM intended to give a tool to {penguin.name}, but no tool was available.")
        
        # The record_distribution tool was already called by the LLM to log its *intent*.
        # The actual distribution just happened above. The history reflects the LLM's decision.

        print(f"\nScientist's Actual Distribution (applied after constraints):")
        print(f"  - Food Actually Given to {penguin.name}: {actual_food_given}")
        print(f"  - Tool Actually Given to {penguin.name}: {actual_tool_given}")

        print(f"\nPost-Action State (End of Scientist's turn for {penguin.name}):")
        print(f"Scientist Resources:")
        print(f"  - Remaining Food Supply: {self.food_supply}")
        print(f"  - Tool Available: {self.tool_available}")
        print(f"Penguin {penguin.name}:")
        print(f"  - Food: {penguin.food}")
        print(f"  - Has Tool: {penguin.has_tool}")


class PenguinAgent(ToolCallingAgent):
    def __init__(self, name: str, initial_food: int = 0, initial_tool: bool = False) -> None: # Allow initial state
        # Student adds the find_food tool here as per the original starter
        super().__init__(tools=[find_food], model=model, name=name, description=f"A penguin named {name}")
        self.name = name
        self.food = initial_food
        self.has_tool = initial_tool

    def take_action(self) -> Dict[str, Any]:
        """Penguin decides on an action each round."""
        self.memory.steps = [] # Clear memory for this specific interaction
        
        print(f"\n--- {self.name}'s Turn to Act ---")
        print(f"Current State for {self.name}: Food: {self.food}, Has Tool: {self.has_tool}")

        # --- STUDENT TASK: Construct a good prompt for the penguin ---
        # This prompt should guide the LLM to:
        # 1. Use the 'find_food' tool if it wants to find food for itself.
        #    It needs to specify 'penguin_name' (self.name) and 'method' ('fishing' or 'foraging').
        # 2. If it wants to request food or a tool from the scientist, it should just output text
        #    (e.g., "I need food," or "I'd like a tool.").
        # 3. The LLM's final text after using a tool should be simple (e.g., "Okay, I tried fishing.")
        prompt = f"""
        
                """
        # --- END STUDENT TASK ---
        
        final_llm_text_response = self.run(prompt) # LLM will populate self.memory.steps

        # ---- START: Robust parsing of tool call and output ----
        action_result: Dict[str, Any] = {}
        find_food_tool_called_by_llm = False
        find_food_method_argument: Optional[str] = None
        food_value_from_tool_output = 0
        tool_output_successfully_processed = False
        
        for step in self.memory.steps: # step is an ActionStep (or TaskStep)
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tc in step.tool_calls: # tc is a ToolCall object
                    if tc.name == 'find_food':
                        find_food_method_argument = tc.arguments.get('method', 'unknown')
                        find_food_tool_called_by_llm = True
                        # The observation for this tool call is on THIS SAME STEP
                        if hasattr(step, 'observations') and step.observations is not None:
                            try:
                                food_value_from_tool_output = int(str(step.observations)) # Observations is the direct output
                                # IMPORTANT: Update penguin's food supply if tool was successful
                                self.food += food_value_from_tool_output 
                                tool_output_successfully_processed = True
                            except (ValueError, TypeError) as e:
                                print(f"  WARN [{self.name}]: 'find_food' output conversion error. Observation: '{step.observations}', Type: {type(step.observations)}, Error: {e}")
                        else:
                            # This case means the tool was called, but smolagents didn't record an observation on that step.
                            # This would be unusual if the tool function executed and returned a value.
                            print(f"  WARN [{self.name}]: 'find_food' tool called (ID {tc.id}), but 'observations' attribute missing or None on the ActionStep.")
                        break 
            if find_food_tool_called_by_llm: # If we identified the call, no need to check further steps for this specific tool call
                break 
        # ---- END: Robust parsing ----
        
        if find_food_tool_called_by_llm:
            if tool_output_successfully_processed:
                action_result = {
                    "action_type": "find_food", 
                    "method": find_food_method_argument, 
                    "food_found": food_value_from_tool_output, # Inform scientist what was found
                    "message": f"{self.name} used 'find_food' (method: {find_food_method_argument}), found {food_value_from_tool_output}. New food total: {self.food}."
                }
                print(f"  {self.name} (Self-Action Result): {action_result['message']}")
            else: 
                 # Tool was called by LLM but processing its output failed (e.g., observation wasn't an int)
                 action_result = {
                     "action_type": "error_processing_find_food_output", 
                     "message": f"{self.name} attempted 'find_food', but its output could not be processed. LLM's final text: '{final_llm_text_response}'"
                 }
                 print(f"  WARN [{self.name}]: {action_result['message']}")
        else: 
            # No find_food tool call was made by the LLM.
            # Interpret its final text response as a request to the scientist.
            # Student needs to parse 'final_llm_text_response' to determine intent.
            

            # This logic determines if the penguin is requesting food or a tool, or doing nothing.
            # It should populate 'action_result'.
            # Example parsing (can be improved):
            text_to_parse = final_llm_text_response
            # Check if the LLM used the implicit 'final_answer' tool that smolagents might add
            # This is robust for when the LLM wraps its text in final_answer.
            if self.memory.steps:
                last_step = self.memory.steps[-1]
                if hasattr(last_step, 'tool_calls') and last_step.tool_calls and \
                   hasattr(last_step.tool_calls[0], 'name') and last_step.tool_calls[0].name == 'final_answer' and \
                   hasattr(last_step.tool_calls[0], 'arguments'): # Check arguments exist
                    text_to_parse = str(last_step.tool_calls[0].arguments.get('answer', final_llm_text_response))

            response_lower = text_to_parse.lower()
            if "tool" in response_lower and ("request" in response_lower or "need" in response_lower or "want" in response_lower or "like" in response_lower):
                action_result = {"action_type": "request_tool", "message": text_to_parse}
            elif "food" in response_lower and ("request" in response_lower or "need" in response_lower or "want" in response_lower or "hungry" in response_lower or "like" in response_lower):
                action_result = {"action_type": "request_food", "message": text_to_parse}
            else: 
                # If the text is ambiguous or doesn't clearly state a request for food/tool after find_food wasn't called
                action_result = {"action_type": "request_food", "message": f"Ambiguous statement or no specific request: '{text_to_parse}'. Defaulting to request_food."}
            print(f"  {self.name} (Text-Based Action Result): Type: {action_result.get('action_type')}, Msg: '{action_result.get('message')}'")
            
        return action_result

def run_simulation(num_rounds=3, num_penguins=2): # Default to 2 penguins for simpler testing
    # Initialize Scientist
    scientist = ScientistAgent(initial_food_supply=20, refresh_interval=3) # Smaller interval for testing refresh

    # Initialize Penguins
    penguins = [PenguinAgent(f"Pingu_{i+1}", initial_food=random.randint(0,3)) for i in range(num_penguins)]
    # penguins[0].has_tool = True # Example: give one penguin a tool initially

    print(f"\n🐧🐟🛠️ Starting Penguin Colony Simulation! ({num_rounds} rounds, {num_penguins} penguins) 🛠️🐟🐧")

    for round_num in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num + 1} of {num_rounds}")
        print(f"{'='*60}")

        current_penguin_actions: Dict[str, Dict[str,Any]] = {} # Store actions for this round

        # Penguins take actions
        for penguin in penguins:
            action_details = penguin.take_action()
            current_penguin_actions[penguin.name] = action_details
            # The print for the penguin's self-action is now inside take_action()

        # Scientist responds to penguins that made requests or had errors
        print(f"\n--- Scientist's Interaction Phase (Round {round_num + 1}) ---")
        for penguin in penguins:
            penguin_action_details = current_penguin_actions[penguin.name]
            
            # Scientist should respond if the penguin:
            # - explicitly requested food ("request_food")
            # - explicitly requested a tool ("request_tool")
            # - had an error trying to find food for itself ("error_processing_find_food_output")
            if penguin_action_details.get("action_type") != "find_food":
                scientist.respond_to_action(penguin, penguin_action_details)
            else:
                # Penguin successfully found its own food
                print(f"\n--- Scientist notes: {penguin.name} successfully found its own food this turn. No direct interaction needed. ---")
                # We still log the scientist's "non-action" for turn counting if desired, or just skip
                # For simplicity, we can just let the scientist's turn counter advance naturally when respond_to_action is called.

    # Final State Output
    print(f"\n{'='*60}\n🏁 FINAL SIMULATION STATE (After {num_rounds} Rounds) 🏁\n{'='*60}")
    print(f"Scientist Final Resources: Food: {scientist.food_supply}, Tool Available: {scientist.tool_available}")
    print(f"Total turns scientist responded to requests: {scientist.turn_counter}")

    print("\nPenguins' Final States:")
    for penguin in penguins:
        print(f"  {penguin.name}: Food: {penguin.food}, Has Tool: {penguin.has_tool}")
        
        final_recorded_history = DISTRIBUTION_HISTORY.get(penguin.name, [])
        if final_recorded_history:
            total_food_decided_for_penguin = sum(h['food'] for h in final_recorded_history)
            times_tool_decided_for_penguin = sum(1 for h in final_recorded_history if h['has_tool'])
            print(f"    (Scientist's Records for {penguin.name}: {len(final_recorded_history)} decisions logged. "
                  f"Total intended food: {total_food_decided_for_penguin}, "
                  f"Times intended tool: {times_tool_decided_for_penguin})")
        else:
            print(f"    (Scientist's Records for {penguin.name}: No distribution decisions logged by scientist)")

if __name__ == "__main__":
    run_simulation()