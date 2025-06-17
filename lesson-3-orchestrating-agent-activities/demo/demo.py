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
        return not (date in self.bookings and time in self.bookings.get(date, {}))
    def add_booking(self, date: str, time: str, customer: str) -> bool:
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
    if booking_system.check_availability(date, time):
        if booking_system.add_booking(date, time, customer):
            return f"Booking confirmed for {customer} on {date} at {time}."
        return f"Failed to add booking for {customer} on {date} at {time} despite appearing available."
    return f"Booking slot {date} at {time} is not available for {customer}."

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
        self.stock: Dict[str, int] = {"skateboard": 20, "helmet": 30, "wheels": 50}
    def check_stock(self, item: str) -> int: return self.stock.get(item.lower(), 0)
    def sell_item(self, item: str, quantity: int) -> bool:
        item_l = item.lower()
        if self.stock.get(item_l, 0) >= quantity:
            self.stock[item_l] -= quantity; return True
        return False
inventory = Inventory()

@tool
def get_item_inventory_level(item: str) -> str:
    """
    Check the stock level of a specific item. Use this for inventory inquiries.

    Args:
        item (str): The item to check the stock level for (e.g., "skateboard", "helmet", "wheels").

    Returns:
        str: A message indicating the stock level of the item.
    """
    return f"Stock for {item}: {inventory.check_stock(item)}."

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

@tool
def submit_request_diagnosis(chosen_category: str, ) -> str:
    """
    Submits the diagnosed category of a customer's request. This tool is called by the CustomerSupportAgent's LLM.

    Args:
        chosen_category (str): The category determined by the LLM (e.g., "Shop - Skateboard Inquiry", "Park - Session Booking Inquiry", etc.).
    
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
            description="Agent that analyzes a customer's request and categorizes its primary intent using the 'submit_request_diagnosis' tool."
        )
        self.possible_categories = [
            "Shop - Skateboard Inquiry", 
            "Park - Session Booking Inquiry", 
            "Park - List Bookings Inquiry", 
            "Shop - Gear repair or replacement Inquiry", 
            "General Inquiry or Unknown"
        ]

    def get_llm_diagnosis(self, user_request: str) -> str:
        self.memory.steps = []
        prompt = f"""
        A customer stated: "{user_request}"
        Your task is to understand the customer's primary intent and categorize this request.
        Choose exactly ONE category from the following list that best fits the request:
        {json.dumps(self.possible_categories)}

        After determining the most appropriate category, you MUST call the 'submit_request_diagnosis' tool.
        Provide your chosen category as the 'chosen_category' argument,
        and the original user request as the 'original_request_for_context' argument.
        Your final text after the tool call should be simple, like "Diagnosis submitted."
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
        
        if diagnosis_from_llm not in self.possible_categories:
            return "General Inquiry or Unknown"
        return diagnosis_from_llm

class Orchestrator(ToolCallingAgent):
    def __init__(self, model_to_use: OpenAIServerModel):
        self.customer_support_agent = CustomerSupportAgent(model_to_use)
        
        orchestrator_tools = [
            check_booking_availability, 
            add_new_booking, 
            get_all_bookings_for_date,
            get_item_inventory_level,
            sell_item_from_inventory
        ]
        super().__init__(
            tools=orchestrator_tools,
            model=model_to_use,
            name="orchestrator",
            description="Orchestrator that handles customer requests by first diagnosing them and then using appropriate tools to fulfill the request or provide information."
        )

    def _get_final_answer_from_memory(self, agent_memory_steps: List[Any]) -> str:
        for step in reversed(agent_memory_steps):
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tc in step.tool_calls:
                    if tc.name == 'final_answer':
                        if hasattr(step, 'action_output') and step.action_output is not None:
                            return str(step.action_output)
                        elif hasattr(tc, 'arguments') and tc.arguments.get('answer') is not None:
                             return str(tc.arguments.get('answer'))
            if hasattr(step, 'observations') and step.observations is not None and \
               (not (hasattr(step, 'tool_calls') and step.tool_calls and step.tool_calls[0].name == 'final_answer')):
                return str(step.observations)
        return "Orchestrator: Could not determine a final response from the execution steps."

    def process_customer_request(self, user_request: str) -> str:
        print(f"\nOrchestrator received request: '{user_request}'")
        
        diagnosis = self.customer_support_agent.get_llm_diagnosis(user_request)
        print(f"LLM Diagnosis from CustomerSupportAgent: '{diagnosis}'")

        self.memory.steps = []
        
        orchestrator_prompt = f"""
        You are the main Orchestrator for a skate park and shop.
        A customer's request is: "{user_request}"
        The request has been diagnosed by our support agent with the category: "{diagnosis}".

        Based on the user request and this diagnosis, decide which of your available tools to use to best help the customer.
        Your available tools are:
        - 'check_booking_availability': Use for "{self.customer_support_agent.possible_categories[1]}" or "{self.customer_support_agent.possible_categories[2]}" if checking a specific slot.
        - 'add_new_booking': Use for "{self.customer_support_agent.possible_categories[1]}" if all details (date, time, customer name) are clear or can be inferred.
        - 'get_all_bookings_for_date': Use for "{self.customer_support_agent.possible_categories[2]}" if a date is provided.
        - 'get_item_inventory_level': Use for "{self.customer_support_agent.possible_categories[0]}" to check stock (e.g., for 'skateboard', 'helmet', or 'wheels').
        - 'sell_item_from_inventory': Use for "{self.customer_support_agent.possible_categories[0]}" if the customer wants to buy a specific item and quantity.
        
        If the diagnosis is "{self.customer_support_agent.possible_categories[3]}" (Gear repair) or "{self.customer_support_agent.possible_categories[4]}" (Unknown/General),
        or if the necessary information for other tools is missing from the user_request (e.g., no item for inventory check, no date/time for booking),
        then you should directly provide a helpful response or ask for clarification using the 'final_answer' tool.
        For gear repair, a standard response is: "Regarding your gear concern: Please bring the item to our shop for a detailed assessment, or call us to discuss repair or replacement options."
        For unknown inquiries, a standard response is: "I'm not entirely sure how to help with that. Could you please rephrase or provide more details?"

        Extract necessary arguments for the tools (like date, time, item, customer name) from the original user_request: "{user_request}".
        If a tool is used, its output will be an observation. You might need to call tools sequentially (e.g., check availability then book).
        Conclude by calling the 'final_answer' tool with your complete response to the customer.
        """
        _ = self.run(orchestrator_prompt)
        return self._get_final_answer_from_memory(self.memory.steps)

orchestrator = Orchestrator(model)

print("\n--- Demo in Action! ---\n")

requests = [
    "I want to book a skate session for 2024-08-15 at 14:00 for Alice.",
    "Can I rent a board for tomorrow?",
    "Do you have any skateboards in stock?",
    "My wheels are broken and I need a replacement.",
    "I'd like to book a session.",
    "Book a session for Bob on 2024-08-15 at 14:00.",
    "What bookings do you have for 2024-08-15?"
]

for i, req in enumerate(requests):
    print(f"\n--- Request {i+1}: '{req}' ---")
    response = orchestrator.process_customer_request(req)
    print(f"Orchestrator's Final Response to Customer: {response}")