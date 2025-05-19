from typing import Dict, List
import os
import dotenv
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from datetime import datetime, timedelta

dotenv.load_dotenv(dotenv_path="../.env")
openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)

class BookingSystem:
    def __init__(self):
        self.bookings = {}

    def check_availability(self, date, time):
        """Check if a session is available at a specific date and time."""
        if date not in self.bookings:
            return True
        if time not in self.bookings[date]:
            return True
        return False

    def add_booking(self, date, time, customer):
        """Add a booking to the system."""
        if date not in self.bookings:
            self.bookings[date] = {}
        self.bookings[date][time] = customer
        return True

    def get_bookings(self, date):
        """Get all bookings for a specific date."""
        if date in self.bookings:
            return self.bookings[date]
        return {}

booking_system = BookingSystem()

@tool
def check_booking_availability(date: str, time: str) -> str:
    """Check if a booking slot is available.

    Args:
        date: The date of the booking (YYYY-MM-DD).
        time: The time of the booking (HH:MM).

    Returns:
        A message indicating whether the booking slot is available.
    """
    if booking_system.check_availability(date, time):
        return "The booking slot is available."
    else:
        return "The booking slot is not available."

@tool
def add_new_booking(date: str, time: str, customer: str) -> str:
    """Add a new booking to the system.

    Args:
        date: The date of the booking (YYYY-MM-DD).
        time: The time of the booking (HH:MM).
        customer: The name of the customer making the booking.

    Returns:
        A message confirming the booking.
    """
    if booking_system.add_booking(date, time, customer):
        return f"Booking confirmed for {customer} on {date} at {time}."
    else:
        return "Failed to add booking."

@tool
def get_all_bookings(date: str) -> Dict:
    """Get all bookings for a specific date.

    Args:
        date: The date to retrieve bookings for (YYYY-MM-DD).

    Returns:
        A dictionary containing all bookings for the specified date.
    """
    bookings = booking_system.get_bookings(date)
    return bookings

class Inventory:
    def __init__(self):
        self.stock = {
            "skateboard": 20,
            "helmet": 30,
            "wheels": 50
        }

    def check_stock(self, item: str) -> int:
        """Check the stock level of an item.

        Args:
            item: The item to check the stock level for.

        Returns:
            The stock level of the item.
        """
        return self.stock.get(item, 0)

    def sell_item(self, item: str, quantity: int) -> bool:
        """Sell an item from the inventory.

        Args:
            item: The item to sell.
            quantity: The quantity to sell.

        Returns:
            True if the item was sold successfully, False otherwise.
        """
        if self.stock.get(item, 0) >= quantity:
            self.stock[item] -= quantity
            return True
        return False

inventory = Inventory()

@tool
def get_inventory_level(item: str) -> str:
    """Check the stock level of an item.

    Args:
        item: The item to check the stock level for.

    Returns:
        A message indicating the stock level of the item.
    """
    stock_level = inventory.check_stock(item)
    return f"The stock level for {item} is {stock_level}."

@tool
def sell_inventory_item(item: str, quantity: int) -> str:
    """Sell an item from the inventory.

    Args:
        item: The item to sell.
        quantity: The quantity to sell.

    Returns:
        A message indicating whether the item was sold successfully.
    """
    if inventory.sell_item(item, quantity):
        return f"{quantity} of {item} sold successfully."
    else:
        return f"Not enough {item} in stock."

class CustomerSupportAgent(ToolCallingAgent):
    """Agent for handling customer support requests."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[],
            model=model,
            name="customer_support_agent",
            description="Agent for handling customer support requests. Diagnose the issue and provide an initial response.",
        )

    def diagnose_issue(self, request: str) -> str:
        """Diagnose the customer's issue.

        Args:
            request: The customer's request.

        Returns:
            A diagnosis of the issue.
        """
        request_lower = request.lower()
        if "board" in request_lower or "skateboard" in request_lower:
            return "Shop - Skateboard Inquiry"
        elif "rent" in request_lower or "session" in request_lower or "booking" in request_lower:
            return "Park - Session Booking Inquiry"
        elif "broken" in request_lower or "damaged" in request_lower:
            return "Shop - Gear repair or replacement Inquiry"
        else:
            return "Unknown - needs escalation"

    def provide_initial_response(self, diagnosis: str) -> str:
        """Provide an initial response to the customer.

        Args:
            diagnosis: The diagnosis of the issue.

        Returns:
            An initial response to the customer.
        """
        responses = {
            "Shop - Skateboard Inquiry": "We have various boards! What kind are you looking for?",
            "Park - Session Booking Inquiry": "Great, what date and time are you looking for?",
            "Shop - Gear repair or replacement Inquiry": "Please describe the damage. We can repair or replace depending on the damage.",
            "Unknown - needs escalation": "Please wait while we connect you with the relevant agent."
        }
        return responses.get(diagnosis, "I am sorry, I don't understand. Please rephrase your request.")

class InventoryAgent(ToolCallingAgent):
    """Agent for managing inventory."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[get_inventory_level, sell_inventory_item],
            model=model,
            name="inventory_agent",
            description="Agent for managing inventory. Check stock levels and sell items.",
        )

class ParkManagementAgent(ToolCallingAgent):
    """Agent for managing park bookings."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[check_booking_availability, add_new_booking, get_all_bookings],
            model=model,
            name="park_management_agent",
            description="Agent for managing park bookings. Check availability and add bookings.",
        )

class Orchestrator(ToolCallingAgent):
    """Orchestrator agent for managing the skate park and shop."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[],
            model=model,
            name="orchestrator",
            description="Orchestrator agent for managing the skate park and shop. Handles customer requests and delegates to other agents.",
        )
        self.customer_support = CustomerSupportAgent(model)
        self.inventory = InventoryAgent(model)
        self.park_management = ParkManagementAgent(model)

    @tool
    def handle_request(user_request: str) -> str:
        """Handle a customer request and delegate to the appropriate agent.

        Args:
            user_request: The customer's request.

        Returns:
            A response to the customer.
        """
        diagnosis = self.customer_support.diagnose_issue(user_request)
        initial_response = self.customer_support.provide_initial_response(diagnosis)

        if "Shop" in diagnosis:
            if "Skateboard" in diagnosis:
                return self.inventory.run(f"""Check the stock level of skateboards and respond to the customer: {initial_response}""")
            elif "Gear" in diagnosis:
                return f"Delegate to repair services. {initial_response}"
            else:
                return "Cannot assist with your request directly. Please rephrase your request."

        elif "Park" in diagnosis:
            return self.park_management.run(f"""Check booking availability and respond to the customer: {initial_response}""")
        else:
            return "Cannot assist with your request directly. Please rephrase your request."

orchestrator = Orchestrator(model)

print("\n--- Demo in Action! ---\n")

request1 = "I want to book a skate session for 2024-07-28 at 10:00."
response1 = orchestrator.run(f"""{request1}""")
print(f"Response 1: {response1}")

request2 = "Do you have any skateboards?"
response2 = orchestrator.run(f"""{request2}""")
print(f"Response 2: {response2}")

request3 = "My helmet is broken!"
response3 = orchestrator.run(f"""{request3}""")
print(f"Response 3: {response3}")
