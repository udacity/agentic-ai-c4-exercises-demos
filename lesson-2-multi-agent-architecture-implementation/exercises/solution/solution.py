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
    Checks recent resource distribution history for a penguin.

    Args:
        penguin_name (str): The unique name or identifier of the penguin.

    Returns:
        Dict[str, Any]: A dictionary containing recent_food_recorded (int)
                        and has_been_recorded_with_tool (bool).
    """
    history = DISTRIBUTION_HISTORY.get(penguin_name, [])
    recent_food = sum(h["food"] for h in history[-3:]) if history else 0
    has_toy_in_history = any(h["has_toy"] for h in history) if history else False
    return {"penguin_name": penguin_name, "recent_food_recorded": recent_food, "has_been_recorded_with_tool": has_toy_in_history}

@tool
def record_distribution(penguin_name: str, food: int, has_toy: bool) -> str:
    """
    Records the scientist's decision to distribute resources.

    Args:
        penguin_name (str): The unique name of the penguin.
        food (int): The number of food units decided (0-5).
        has_toy (bool): Whether a toy is decided to be given.

    Returns:
        str: A confirmation message.
    """
    if penguin_name not in DISTRIBUTION_HISTORY:
        DISTRIBUTION_HISTORY[penguin_name] = []
    DISTRIBUTION_HISTORY[penguin_name].append({"food": food, "has_toy": has_toy})
    return f"Decision Recorded: {penguin_name} gets {food} food and {'a toy' if has_toy else 'no toy'}."

@tool
def find_food(method: str) -> int:
  """
  Penguin finds food. Returns amount found (int).

  Args:
      method (str): Method ("fishing" or "foraging").

  Returns:
      int: The amount of food found.
  """
  food_found = 0
  if method.lower() == "foraging":
    food_found = random.randint(0, 3)
  elif method.lower() == "fishing":
    food_found = random.randint(2, 7)
  return food_found

class ScientistAgent(ToolCallingAgent):
    def __init__(self, initial_food_supply: int = 20, refresh_interval: int = 5) -> None:
        super().__init__(tools=[check_history, record_distribution], model=model, name="Scientist")
        self.initial_food_supply = initial_food_supply
        self.food_supply = initial_food_supply
        self.toy_available = True
        self.refresh_interval = refresh_interval
        self.turn_counter = 0

    def refresh_resources(self):
        self.food_supply = self.initial_food_supply
        self.tool_available = True
        print(f"\n🔄 Scientist Resources Refreshed! Food: {self.food_supply}, Tool: {self.tool_available}")

    def respond_to_action(self, penguin: 'PenguinAgent', action_details: Dict[str, Any]) -> None:
        self.turn_counter += 1
        if self.turn_counter > 0 and self.turn_counter % self.refresh_interval == 0:
            self.refresh_resources()

        print(f"\n--- Turn {self.turn_counter}: Scientist Responds to {penguin.name} ---")
        print(f"Penguin Action: {action_details}")
        print(f"Penguin State (before): Food: {penguin.food}, Toy: {penguin.has_toy}")
        print(f"Scientist Resources (before): Food: {self.food_supply}, Tool: {self.tool_available}")

        self.memory.steps = []
        prompt = f"""
        You are the Scientist. Respond to penguin {penguin.name}.
        Penguin's last action/request: {json.dumps(action_details)}
        Penguin's current state: Food: {penguin.food}, Has Toy: {penguin.has_toy}.
        Your resources: Food Supply: {self.food_supply}, Tool Available: {self.tool_available}.

        You can use 'check_history' for {penguin.name} if needed.
        Then, decide food (0-5) and tool (true/false) for {penguin.name}.
        MUST use 'record_distribution' tool ONCE for your final decision.
        Example: record_distribution(penguin_name="{penguin.name}", food=3, has_toy=True)
        After 'record_distribution' call, just state "Decision logged."
        """
        final_llm_text_output = self.run(prompt)
        
        # DEBUG: Inspect memory steps
        # print(f"DEBUG Scientist MEMORY for {self.name} after run:")
        # for i, step in enumerate(self.memory.steps):
        #     print(f"  Step {i}: {step}")
        #     if hasattr(step, 'tool_calls') and step.tool_calls: print(f"    Tool Calls: {step.tool_calls}")
        #     if hasattr(step, 'observations'): print(f"    Observations: '{step.observations}' (type: {type(step.observations)})")
        #     if hasattr(step, 'content'): print(f"    Content: '{step.content}' (type: {type(step.content)})")


        intended_food_from_llm = 0
        intended_toy_from_llm = False
        record_distribution_tool_called = False
        confirmation_message_from_tool = "Tool 'record_distribution' output not found."
        
        for step in self.memory.steps: # step is an ActionStep (or TaskStep)
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tc in step.tool_calls:
                    if tc.name == 'record_distribution':
                        intended_food_from_llm = tc.arguments.get('food', 0)
                        intended_toy_from_llm = tc.arguments.get('has_toy', False)
                        record_distribution_tool_called = True
                        # The observation for this tool call is on THIS SAME STEP
                        if hasattr(step, 'observations') and step.observations is not None:
                            confirmation_message_from_tool = str(step.observations)
                        else:
                            confirmation_message_from_tool = f"Observation missing on step with record_distribution call (ID: {tc.id})"
                        break 
            if record_distribution_tool_called:
                break 
        
        if record_distribution_tool_called:
            print(f"\nScientist LLM intended (via 'record_distribution' args): Food: {intended_food_from_llm}, Toy: {intended_toy_from_llm}")
            print(f"  Confirmation from tool execution (observation): {confirmation_message_from_tool}")
        else:
            print(f"\nWARN: Scientist LLM did NOT use 'record_distribution'. Final LLM text: '{final_llm_text_output}'")

        actual_food_given = 0
        actual_tool_given = False
        if record_distribution_tool_called:
            food_to_give = min(max(0, int(intended_food_from_llm)), 5)
            if food_to_give > 0:
                if self.food_supply >= food_to_give:
                    actual_food_given = food_to_give
                else:
                    actual_food_given = self.food_supply
                self.food_supply -= actual_food_given
                self.food_supply = max(0, self.food_supply)
                penguin.food += actual_food_given
                if actual_food_given < food_to_give and food_to_give > 0:
                     print(f"  INFO: Intended {food_to_give}, but only {actual_food_given} was available/given.")
            
            if intended_toy_from_llm:
                if self.toy_available:
                    actual_toy_given = True
                    penguin.has_toy = True
                    self.toy_available = False
                else:
                    print("  INFO: Scientist LLM intended to give a toy, but no toy was available.")
        
        print(f"\nScientist Actual Distribution: Food: {actual_food_given}, Toy: {actual_toy_given}")
        print(f"Scientist Resources (after): Food: {self.food_supply}, Toy: {self.toy_available}")
        print(f"Penguin State (after): Food: {penguin.food}, Toy: {penguin.has_toy}")

class PenguinAgent(ToolCallingAgent):
    def __init__(self, name: str) -> None:
        super().__init__(tools=[find_food], model=model, name=name)
        self.name = name
        self.food = random.randint(0, 2)
        self.has_toy = False
    def take_action(self) -> Dict[str, Any]:
        self.memory.steps = []
        print(f"\n--- {self.name}'s Turn ---")
        print(f"State: Food: {self.food}, Toy: {self.has_toy}")

        prompt = f"""
        You are {self.name}. Your current state is: Food: {self.food}, Has Toy: {self.has_toy}.
        Choose ONE action:
        1. FIND FOOD: Use the 'find_food' tool. You MUST provide 'penguin_name' (which is "{self.name}") and 'method' ("fishing" or "foraging").
           After the tool is used by the system, your final response should be simple, like "Okay, I went searching."
        2. REQUEST FOOD: If you want to ask the scientist for food, respond with a short text message (e.g., "I am hungry and need food."). Do NOT use any tool.
        3. REQUEST TOY: If you want to ask the scientist for a TOY, respond with a short text message (e.g., "I could use a tool."). Do NOT use any tool.
        """
        final_llm_text_output = self.run(prompt)
        
        # DEBUG: Inspect memory steps
        # print(f"DEBUG Penguin MEMORY for {self.name} after run:")
        # for i, step in enumerate(self.memory.steps):
        #     print(f"  Step {i}: {step}")
        #     if hasattr(step, 'tool_calls') and step.tool_calls: print(f"    Tool Calls: {step.tool_calls}")
        #     if hasattr(step, 'observations'): print(f"    Observations: '{step.observations}' (type: {type(step.observations)})")
        #     if hasattr(step, 'content'): print(f"    Content: '{step.content}' (type: {type(step.content)})")


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
                                self.food += food_value_from_tool_output
                                tool_output_successfully_processed = True
                            except (ValueError, TypeError) as e:
                                print(f"  WARN: {self.name} 'find_food' output conversion error. Observation: '{step.observations}', Type: {type(step.observations)}, Error: {e}")
                        else:
                            print(f"  WARN: {self.name} 'find_food' called (ID {tc.id}), but 'observations' attribute missing or None on the step.")
                        break # Found and processed (or attempted to process) find_food
            if find_food_tool_called_by_llm: # If we identified the call, no need to check further steps for this *specific* tool call
                break 
        
        if find_food_tool_called_by_llm:
            if tool_output_successfully_processed:
                action_result = {
                    "action_type": "find_food", 
                    "method": find_food_method_argument, 
                    "food_found": food_value_from_tool_output,
                    "message": f"{self.name} used 'find_food' (method: {find_food_method_argument}), found {food_value_from_tool_output}. New food total: {self.food}."
                }
                print(f"  {self.name} (Self-Action): {action_result['message']}")
            else: 
                 action_result = {
                     "action_type": "error_processing_find_food_output", 
                     "message": f"{self.name} called 'find_food', but its output could not be processed. LLM final text: '{final_llm_text_output}'"
                 }
                 print(f"  WARN: {action_result['message']}")
        else: 
            # No find_food tool call by LLM, interpret its final text output
            text_to_parse = final_llm_text_output
            # If smolagents makes LLM use 'final_answer', its content is in action_output of the last step
            if self.memory.steps:
                last_step = self.memory.steps[-1]
                if hasattr(last_step, 'tool_calls') and last_step.tool_calls and \
                   last_step.tool_calls[0].name == 'final_answer' and \
                   hasattr(last_step, 'action_output'): # 'action_output' might hold the "answer" of final_answer
                    text_to_parse = str(last_step.action_output)

            response_lower = text_to_parse.lower()
            if "toy" in response_lower and ("need" in response_lower or "request" in response_lower or "want" in response_lower):
                action_result = {"action_type": "request_toy", "message": text_to_parse}
            elif "food" in response_lower and ("need" in response_lower or "request" in response_lower or "hungry" in response_lower or "want" in response_lower):
                action_result = {"action_type": "request_food", "message": text_to_parse}
            else: 
                action_result = {"action_type": "request_food", "message": f"Ambiguous request: '{text_to_parse}'. Defaulting to request food."}
            print(f"  {self.name} (Self-Action): Type: {action_result['action_type']}, Msg: '{action_result['message']}' (parsed from LLM text).")
            
        return action_result

def run_simulation(num_rounds=2, num_penguins=2):
    penguin_names = ['putzie', 'miercoles']
    scientist = ScientistAgent(initial_food_supply=15, refresh_interval=3)
    penguins = [PenguinAgent(f"{random.choice(penguin_names)}{i+1}") for i in range(num_penguins)]

    print(f"\n🐧🐟🛠️ Starting Simulation ({num_rounds} rounds, {num_penguins} penguins) 🛠️🐟🐧")
    for round_num in range(num_rounds):
        print(f"\n{'='*60}\nROUND {round_num + 1}\n{'='*60}")
        
        penguin_actions_this_round = {}
        for penguin in penguins:
            action = penguin.take_action()
            penguin_actions_this_round[penguin.name] = action
            
        print(f"\n--- Scientist's Interaction Phase (Round {round_num + 1}) ---")
        for penguin in penguins:
            action_details = penguin_actions_this_round[penguin.name]
            if action_details.get("action_type") != "find_food" or "error" in action_details.get("action_type","").lower() :
                scientist.respond_to_action(penguin, action_details)
            else:
                print(f"\n--- Scientist notes {penguin.name} successfully found its own food this turn. ---")

    print(f"\n{'='*60}\n🏁 FINAL SIMULATION STATE 🏁\n{'='*60}")
    print(f"Scientist: Food: {scientist.food_supply}, Toy: {scientist.toy_available} (Responded {scientist.turn_counter} times)")
    print("\nPenguins:")
    for p in penguins:
        print(f"  {p.name}: Food: {p.food}, Toy: {p.has_toy}")
        history = DISTRIBUTION_HISTORY.get(p.name, [])
        if history:
            food_hist = sum(h['food'] for h in history)
            toy_hist = sum(1 for h in history if h['has_toy'])
            print(f"    History (Scientist log for {p.name}): Decisions: {len(history)}, Total Food Intended: {food_hist}, Toy Intended: {toy_hist} times")


if __name__ == "__main__":
    run_simulation()