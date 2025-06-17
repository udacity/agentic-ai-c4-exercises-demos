from typing import Dict, List, Any, Optional
import os
import dotenv
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
import re
import json

dotenv.load_dotenv(dotenv_path="../.env")
openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)

class BookingSystem:
    def __init__(self):
        self.bookings: Dict[str, Dict[str, str]] = {}
    def check_availability(self, date: str, time: str) -> bool:
        if date not in self.bookings: return True
        return time not in self.bookings.get(date, {})
    def add_booking(self, date: str, time: str, customer: str) -> bool:
        if not self.check_availability(date, time): return False 
        if date not in self.bookings: self.bookings[date] = {}
        self.bookings[date][time] = customer
        return True
    def get_bookings(self, date: str) -> Dict[str, str]:
        return self.bookings.get(date, {})
booking_system = BookingSystem()

@tool
def check_booking_availability(date: str, time: str) -> str:
    """
    Check if a booking slot is available.

    Args:
        date (str): The date of the booking (YYYY-MM-DD).
        time (str): The time of the booking (HH:MM).

    Returns:
        str: A message indicating whether the booking slot is available.
    """
    if booking_system.check_availability(date, time):
        return f"The booking slot for {date} at {time} is available."
    return f"Sorry, the booking slot for {date} at {time} is not available."

@tool
def add_new_booking(date: str, time: str, customer: str) -> str:
    """
    Add a new booking to the system.

    Args:
        date (str): The date of the booking (YYYY-MM-DD).
        time (str): The time of the booking (HH:MM).
        customer (str): The name of the customer making the booking.

    Returns:
        str: A message confirming the booking or stating failure.
    """
    if booking_system.add_booking(date, time, customer):
        return f"Booking confirmed for {customer} on {date} at {time}."
    return f"Booking slot {date} at {time} is not available or booking failed for {customer}."

@tool
def get_all_bookings_for_date(date: str) -> str:
    """
    Get all bookings for a specific date. This tool should be used when a customer asks to see existing bookings for a date.

    Args:
        date (str): The date to retrieve bookings for (YYYY-MM-DD).

    Returns:
        str: String listing bookings or a 'no bookings found' message.
    """
    bookings = booking_system.get_bookings(date)
    if not bookings: return f"No bookings found for {date}."
    return f"Bookings for {date}: {json.dumps(bookings)}"

class Inventory:
    def __init__(self):
        self.stock: Dict[str, int] = {"skateboard": 20, "helmet": 30, "wheels": 50, "t-shirt": 25, "stickers": 100}
    def check_stock(self, item: str) -> int: 
        return self.stock.get(item.lower(), 0)
    def sell_item(self, item: str, quantity: int) -> bool:
        item_l = item.lower()
        if self.stock.get(item_l, 0) >= quantity:
            self.stock[item_l] -= quantity
            return True
        return False
inventory = Inventory()

@tool
def get_item_inventory_level(item: str) -> str:
    """
    Check the stock level of a specific item. Use this for inventory inquiries.

    Args:
        item (str): The item to check the stock level for (e.g., "skateboard", "helmet", "wheels", "t-shirt").

    Returns:
        str: A message indicating the stock level of the item.
    """
    stock = inventory.check_stock(item)
    return f"Stock level for {item}: {stock}."

@tool
def sell_item_from_inventory(item: str, quantity: int) -> str:
    """
    Sell a specified quantity of an item from the inventory. Use this when a customer wants to purchase an item.

    Args:
        item (str): The item to sell.
        quantity (int): The quantity to sell.

    Returns:
        str: A message indicating whether the item was sold successfully and the new stock level.
    """
    stock_before = inventory.check_stock(item)
    if inventory.sell_item(item, quantity):
        return f"Sold {quantity} of {item}. Stock was {stock_before}, now {inventory.check_stock(item)}."
    return f"Not enough {item} in stock (available: {stock_before}). Sale failed."

class EventSystem:
    def __init__(self):
        self.events: List[Dict[str, str]] = [] 
    def add_event(self, event_name: str, date: str, description: str) -> bool:
        self.events.append({"name": event_name, "date": date, "description": description})
        return True
    def list_events(self) -> List[Dict[str, str]]:
        return self.events
event_system = EventSystem()

class MaintenanceLog:
    def __init__(self):
        self.log_entries: List[Dict[str,str]] = []
        self.next_id = 1
    def add_entry(self, area: str, issue: str, reported_by: str) -> int:
        entry_id = self.next_id
        self.log_entries.append({"id": entry_id, "area": area, "issue": issue, "reported_by": reported_by, "status": "reported"})
        self.next_id +=1
        return entry_id
    def view_log(self) -> List[Dict[str,str]]:
        return self.log_entries
maintenance_log = MaintenanceLog()

@tool
def create_new_event(event_name: str, date: str, description: str) -> str:
    """
    Creates a new event in the skate park's schedule.

    Args:
        event_name (str): The name of the event.
        date (str): The date of the event (YYYY-MM-DD).
        description (str): A brief description of the event.

    Returns:
        str: A confirmation message.
    """
    # TODO: Implement this tool using event_system.add_event
    pass

@tool
def list_upcoming_events() -> str:
    """
    Lists all upcoming scheduled events.

    Returns:
        str: A string representation of the list of events, or a message if no events are scheduled.
    """
    # TODO: Implement this tool using event_system.list_events
    pass

@tool
def log_maintenance_request(area: str, issue_description: str, reported_by: str) -> str:
    """
    Logs a new maintenance request for an area in the skate park.

    Args:
        area (str): The park area requiring maintenance (e.g., "ramp section A", "grind rail 2").
        issue_description (str): A description of the maintenance issue.
        reported_by (str): Name of the person reporting the issue.

    Returns:
        str: A confirmation message with the request ID.
    """
    # TODO: Implement this tool using maintenance_log.add_entry
    pass

@tool
def view_maintenance_log() -> str:
    """
    Displays all current maintenance log entries.

    Returns:
        str: A string representation of the maintenance log, or a message if the log is empty.
    """
    # TODO: Implement this tool using maintenance_log.view_log
    pass

@tool
def submit_request_diagnosis(chosen_category: str, original_request_for_context: str) -> str:
    """
    Submits the diagnosed category of a customer's request. This tool is called by the CustomerSupportAgent's LLM.

    Args:
        chosen_category (str): The category determined by the LLM.
        original_request_for_context (str): The original user request.
    
    Returns:
        str: The chosen_category.
    """
    return chosen_category

class CustomerSupportAgent(ToolCallingAgent):
    def __init__(self, model_to_use: OpenAIServerModel):
        super().__init__(
            tools=[submit_request_diagnosis],
            model=model_to_use,
            name="customer_support_diagnoser",
            description="Agent that analyzes a customer's request and categorizes its primary intent."
        )
        self.possible_categories = [
            "Shop - Skateboard Inquiry", 
            "Park - Session Booking Inquiry", 
            "Park - List Bookings Inquiry", 
            "Shop - Gear repair or replacement Inquiry",
            "Park - Event Inquiry",             # NEW
            "Park - Maintenance Request",       # NEW
            "General Inquiry or Unknown"
        ]

    def get_llm_diagnosis(self, user_request: str) -> str:
        self.memory.steps = []
        prompt = f"""
        A customer stated: "{user_request}"
        Categorize this request into ONE of the following types: {json.dumps(self.possible_categories)}.
        Call 'submit_request_diagnosis' with your 'chosen_category' and the 'original_request_for_context'.
        Final text should be "Diagnosis submitted."
        """
        _ = self.run(prompt) 
        diagnosis_from_llm = "General Inquiry or Unknown" 
        for step in self.memory.steps:
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tc in step.tool_calls:
                    if tc.name == 'submit_request_diagnosis':
                        if hasattr(step, 'observations') and step.observations is not None:
                            diagnosis_from_llm = str(step.observations)
                        break
            if diagnosis_from_llm != "General Inquiry or Unknown" and diagnosis_from_llm in self.possible_categories:
                break
        if diagnosis_from_llm not in self.possible_categories: return "General Inquiry or Unknown"
        return diagnosis_from_llm

class Orchestrator(ToolCallingAgent):
    def __init__(self, model_to_use: OpenAIServerModel):
        self.customer_support_agent = CustomerSupportAgent(model_to_use)
        
        orchestrator_tools = [
            check_booking_availability, 
            add_new_booking, 
            get_all_bookings_for_date,
            get_item_inventory_level,
            sell_item_from_inventory,
            create_new_event,               # NEW
            list_upcoming_events,           # NEW
            log_maintenance_request,        # NEW
            view_maintenance_log            # NEW
        ]
        super().__init__(
            tools=orchestrator_tools,
            model=model_to_use,
            name="orchestrator",
            description="Orchestrator for skate park, shop, events, and maintenance."
        )

    def _get_final_answer_from_orchestrator_memory(self) -> str:
        for step in reversed(self.memory.steps):
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tc in step.tool_calls:
                    if tc.name == 'final_answer':
                        if hasattr(step, 'action_output') and step.action_output is not None:
                            return str(step.action_output)
                        elif hasattr(tc, 'arguments') and tc.arguments.get('answer') is not None:
                             return str(tc.arguments.get('answer'))
            if hasattr(step, 'observations') and step.observations is not None and \
               (not (hasattr(step, 'tool_calls') and step.tool_calls and hasattr(step.tool_calls[0], 'name') and step.tool_calls[0].name == 'final_answer')):
                return str(step.observations)
        return "Orchestrator: Could not determine a final response."

    def handle_customer_request(self, user_request: str) -> str:
        print(f"\nOrchestrator received request: '{user_request}'")
        
        diagnosis = self.customer_support_agent.get_llm_diagnosis(user_request)
        print(f"LLM Diagnosis from CustomerSupportAgent: '{diagnosis}'")

        self.memory.steps = []
        
        # TODO: Learner needs to expand this prompt to handle the new diagnosis categories
        # and guide the LLM to use the new tools for events and maintenance.
        orchestrator_prompt = f"""
        You are the main Orchestrator.
        Customer request: "{user_request}"
        Diagnosis: "{diagnosis}".
        Available tools: {json.dumps([t.name for t in self.tools])}.

        Based on the request and diagnosis, decide which tool to use.
        - For "{self.customer_support_agent.possible_categories[0]}" (Skateboard Inquiry), use 'get_item_inventory_level' or 'sell_item_from_inventory'.
        - For "{self.customer_support_agent.possible_categories[1]}" (Session Booking), use 'check_booking_availability' then 'add_new_booking'.
        - For "{self.customer_support_agent.possible_categories[2]}" (List Bookings), use 'get_all_bookings_for_date'.
        - For "{self.customer_support_agent.possible_categories[4]}" (Event Inquiry), use 'list_upcoming_events' or 'create_new_event' if details are provided for a new event.
        - For "{self.customer_support_agent.possible_categories[5]}" (Maintenance Request), use 'log_maintenance_request' or 'view_maintenance_log'.
        
        If diagnosis is "{self.customer_support_agent.possible_categories[3]}" (Gear repair), respond with 'final_answer': "Regarding your gear: please bring it to the shop for assessment."
        If diagnosis is "{self.customer_support_agent.possible_categories[6]}" (Unknown/General) or info is missing for tools, use 'final_answer' to ask for clarification or state inability to help.

        Extract arguments for tools from "{user_request}".
        Call tools sequentially if needed. Conclude with 'final_answer'.
        """
        _ = self.run(orchestrator_prompt)
        return self._get_final_answer_from_orchestrator_memory()

orchestrator = Orchestrator(model)

print("\n--- Test Your Implementation! ---\n")

requests = [
    "I want to book a skate session for 2024-09-01 at 15:00 for Carlos.",
    "Do you have any 'pro_model_deck' skateboards in stock?",
    "My helmet is cracked, can you fix it?",
    "What events are happening next month?",
    "The main ramp has a loose panel, someone could get hurt!",
    "I'd like to schedule a 'Beginner Skate Workshop' on 2024-09-15, it's for all ages.",
    "Show me the maintenance log." 
]

for i, req in enumerate(requests):
    print(f"\n--- Request {i+1}: '{req}' ---")
    response = orchestrator.handle_customer_request(req)
    print(f"Orchestrator's Final Response to Customer: {response}")