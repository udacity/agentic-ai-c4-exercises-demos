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
        # Sort events by date (optional, for nicer listing)
        try:
            self.events.sort(key=lambda x: x.get("date", "9999-99-99"))
        except TypeError: # Handle cases where date might not be sortable if malformed, though unlikely with YYYY-MM-DD
            pass
        return True
    def list_events(self) -> List[Dict[str, str]]:
        return self.events
event_system = EventSystem()

class MaintenanceLog:
    def __init__(self):
        self.log_entries: List[Dict[str,Any]] = [] # Changed to List[Dict[str,Any]] to include int ID
        self.next_id = 1
    def add_entry(self, area: str, issue: str, reported_by: str) -> int:
        entry_id = self.next_id
        self.log_entries.append({"id": entry_id, "area": area, "issue": issue, "reported_by": reported_by, "status": "reported"})
        self.next_id +=1
        return entry_id
    def view_log(self) -> List[Dict[str,Any]]: # Changed to List[Dict[str,Any]]
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
    if event_system.add_event(event_name, date, description):
        return f"Event '{event_name}' on {date} successfully created: {description}."
    return f"Failed to create event '{event_name}'." # Should not happen with current simple add_event

@tool
def list_upcoming_events() -> str:
    """
    Lists all upcoming scheduled events.

    Returns:
        str: A string representation of the list of events, or a message if no events are scheduled.
    """
    events = event_system.list_events()
    if not events:
        return "No upcoming events are currently scheduled."
    return f"Upcoming events: {json.dumps(events)}"

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
    request_id = maintenance_log.add_entry(area, issue_description, reported_by)
    return f"Maintenance request logged for '{area}' (Issue: '{issue_description}', Reported by: {reported_by}). Request ID: {request_id}."

@tool
def view_maintenance_log() -> str:
    """
    Displays all current maintenance log entries.

    Returns:
        str: A string representation of the maintenance log, or a message if the log is empty.
    """
    log = maintenance_log.view_log()
    if not log:
        return "The maintenance log is currently empty."
    return f"Maintenance Log: {json.dumps(log)}"

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
            "Park - Event Inquiry",
            "Park - Maintenance Request",
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
            create_new_event,
            list_upcoming_events,
            log_maintenance_request,
            view_maintenance_log
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
        
        orchestrator_prompt = f"""
        You are the main Orchestrator for a skate park, shop, events, and maintenance.
        A customer's request is: "{user_request}"
        The request has been diagnosed by our support agent with the category: "{diagnosis}".

        Based on the user request and this diagnosis, decide which of your available tools to use to best help the customer.
        
        Consider the diagnosis:
        - If "{diagnosis}" is "{self.customer_support_agent.possible_categories[0]}" (Skateboard Inquiry), you might use 'get_item_inventory_level' for items like 'skateboard', 'helmet', or 'wheels', or 'sell_item_from_inventory' if purchase details are clear.
        - If "{diagnosis}" is "{self.customer_support_agent.possible_categories[1]}" (Session Booking), attempt to extract date, time, and customer name from "{user_request}". If all details are present, you might first 'check_booking_availability', then 'add_new_booking'. If details are missing, use 'final_answer' to ask for them.
        - If "{diagnosis}" is "{self.customer_support_agent.possible_categories[2]}" (List Bookings), attempt to extract a date from "{user_request}". If a date is present, use 'get_all_bookings_for_date'. If no date, use 'final_answer' to ask for it.
        - If "{diagnosis}" is "{self.customer_support_agent.possible_categories[4]}" (Event Inquiry):
            - If the request is about listing events, use 'list_upcoming_events'.
            - If the request is about creating a new event and provides event_name, date, and description, use 'create_new_event'.
            - Otherwise, use 'final_answer' to ask for more details or provide general event info.
        - If "{diagnosis}" is "{self.customer_support_agent.possible_categories[5]}" (Maintenance Request):
            - If the request describes an issue and area, use 'log_maintenance_request'. Extract area, issue_description, and reported_by (assume customer name or default to "customer").
            - If the request is to view the log, use 'view_maintenance_log'.
            - Otherwise, use 'final_answer' to ask for more details.
        
        If the diagnosis is "{self.customer_support_agent.possible_categories[3]}" (Gear repair), provide a standard helpful response using the 'final_answer' tool: "Regarding your gear concern about '{user_request}': Please bring the item to our shop for a detailed assessment, or call us to discuss repair or replacement options."
        If the diagnosis is "{self.customer_support_agent.possible_categories[6]}" (Unknown/General), or if necessary information for other tools is missing and you need to ask for clarification, use the 'final_answer' tool with an appropriate message like: "I'm not entirely sure how to help with that. Could you please rephrase or provide more details for '{user_request}'?"

        Extract necessary arguments for any tool you call (like date, time, item, customer name, event_name, description, area, issue_description, reported_by) directly from the original user_request: "{user_request}".
        If a tool is used, its output (your observation) will be provided to you. You might need to call tools sequentially.
        Conclude by calling the 'final_answer' tool with your complete response to the customer.
        """
        _ = self.run(orchestrator_prompt)
        return self._get_final_answer_from_orchestrator_memory()

orchestrator = Orchestrator(model)

print("\n--- Test Your Implementation! ---\n")

requests = [
    "I want to book a skate session for 2024-09-01 at 15:00 for Carlos.",
    "Do you have any 'pro_model_deck' skateboards in stock?", # Assumes 'pro_model_deck' is a type of skateboard
    "My helmet is cracked, can you fix it?",
    "What events are happening next month?",
    "The main ramp has a loose panel, someone could get hurt! My name is Sarah.",
    "I'd like to schedule a 'Beginner Skate Workshop' on 2024-09-15, it's for all ages.",
    "Show me the maintenance log.",
    "Can I buy two helmets and a set of wheels?",
    "Book a session for 2024-09-01 at 15:00, it's for Leo." # Should fail as Carlos took it
]

for i, req in enumerate(requests):
    print(f"\n--- Request {i+1}: '{req}' ---")
    response = orchestrator.handle_customer_request(req)
    print(f"Orchestrator's Final Response to Customer: {response}")