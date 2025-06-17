"""
Pasta Factory Exercise - Starter Code
====================================

In this exercise, you'll extend the Italian Pasta Factory multi-agent system 
to handle more complex scenarios with shared state coordination and proper
multi-agent orchestration patterns.

You'll need to:
1. Implement the missing production and custom recipe tools
2. Create the CustomPastaDesignerAgent
3. Build the proper Orchestrator using ToolCallingAgent
4. Add coordination tools that route requests between specialized agents

This demonstrates extending multi-agent systems with new capabilities while
maintaining proper orchestration patterns.
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import random
from dataclasses import dataclass, field, asdict

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)

# Load your OpenAI API key
import os
import dotenv
dotenv.load_dotenv(dotenv_path="../.env")
openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)

# Pasta Factory State Management

@dataclass
class PastaOrder:
    order_id: str
    pasta_shape: str
    quantity: float  # in kg
    status: str = "pending"  # pending, queued, completed, cancelled
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: int = 1  # 1 = normal, 2 = rush, 3 = emergency
    customer_notes: str = ""
    estimated_delivery_date: str = ""

@dataclass
class FactoryState:
    inventory: Dict[str, float] = field(default_factory=lambda: {
        "flour": 10.0,  # kg
        "water": 5.0,   # liters
        "eggs": 24,     # count
        "semolina": 8.0 # kg
    })
    production_queue: List[PastaOrder] = field(default_factory=list)
    pasta_recipes: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "spaghetti": {"flour": 0.2, "water": 0.1},
        "fettuccine": {"flour": 0.25, "water": 0.1},
        "penne": {"flour": 0.2, "water": 0.1},
        "ravioli": {"flour": 0.3, "water": 0.1, "eggs": 2},
        "lasagna": {"flour": 0.3, "water": 0.15, "eggs": 3}
    })
    custom_recipes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    order_counter: int = 0
    known_pasta_shapes: List[str] = field(default_factory=lambda: [
        "spaghetti", "fettuccine", "penne", "ravioli", "lasagna"
    ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inventory": self.inventory,
            "production_queue": [asdict(order) for order in self.production_queue],
            "pasta_recipes": self.pasta_recipes,
            "custom_recipes": self.custom_recipes
        }
    
    def update_known_pasta_shapes(self):
        """Update the list of known pasta shapes based on recipes."""
        self.known_pasta_shapes = list(self.pasta_recipes.keys()) + list(self.custom_recipes.keys())

# Initialize the shared factory state
factory_state = FactoryState()

# ======= Agent Tools =======

@tool
def check_pasta_recipe(pasta_shape: str) -> Dict[str, float]:
    """
    Check what ingredients are needed for a specific pasta shape.
    Returns a dictionary of ingredients and amounts needed per kg of pasta.
    """
    if pasta_shape in factory_state.pasta_recipes:
        return factory_state.pasta_recipes[pasta_shape]
    elif pasta_shape in factory_state.custom_recipes:
        return factory_state.custom_recipes[pasta_shape]
    return {}

@tool
def check_inventory() -> Dict[str, float]:
    """Check current inventory levels of all ingredients."""
    return factory_state.inventory

@tool
def generate_order_id() -> str:
    """Generate a unique order ID."""
    factory_state.order_counter += 1
    return f"ORD-{factory_state.order_counter:04d}"

@tool
def list_available_pasta_shapes() -> List[str]:
    """List all available pasta shapes that can be ordered."""
    return factory_state.known_pasta_shapes

@tool
def update_inventory(ingredient: str, amount: float) -> Dict[str, Any]:
    """
    Update the inventory amount for a specific ingredient.
    
    Args:
        ingredient: Name of the ingredient
        amount: New amount (will replace current amount)
        
    Returns:
        Status of the inventory update
    """
    if ingredient not in factory_state.inventory:
        return {
            "success": False,
            "message": f"Unknown ingredient: {ingredient}. Cannot update inventory."
        }
    
    old_amount = factory_state.inventory[ingredient]
    factory_state.inventory[ingredient] = amount
    
    return {
        "success": True,
        "message": f"Inventory updated: {ingredient} from {old_amount} to {amount}.",
        "ingredient": ingredient,
        "old_amount": old_amount,
        "new_amount": amount
    }

@tool
def check_production_capacity(days_ahead: int = 7) -> Dict[str, Any]:
    """
    Check the current production capacity and queue for the next X days.
    Returns information about queue size and estimated completion times.
    """
    queue_size = len(factory_state.production_queue)
    
    # Calculate the total production volume (in kg)
    total_volume = sum(order.quantity for order in factory_state.production_queue)
    
    # Simple capacity estimation: assume we can produce 10kg per day
    daily_capacity = 10.0  # kg per day
    days_to_complete = max(1, total_volume / daily_capacity)
    
    # Consider priority orders
    priority_orders = [o for o in factory_state.production_queue if o.priority > 1]
    priority_volume = sum(order.quantity for order in priority_orders)
    
    return {
        "queue_size": queue_size,
        "total_volume_kg": total_volume,
        "days_to_complete_current_queue": days_to_complete,
        "daily_capacity_kg": daily_capacity,
        "priority_orders": len(priority_orders),
        "priority_volume_kg": priority_volume
    }

# TODO: Implement the following tools

@tool
def add_to_production_queue(
    order_id: str,
    pasta_shape: str,
    quantity: float,
    priority: int = 1,
    customer_notes: str = ""
) -> Dict[str, Any]:
    """
    Add an order to the production queue.
    
    Args:
        order_id: Unique order identifier
        pasta_shape: Type of pasta to produce
        quantity: Amount in kg
        priority: Order priority (1=normal, 2=rush, 3=emergency)
        customer_notes: Additional notes from customer
        
    Returns:
        Status of the queuing operation with estimated delivery date
    """
    # TODO: Implement this function
    # 1. Verify that pasta_shape is valid using check_pasta_recipe
    # 2. Calculate required ingredients and check inventory availability
    # 3. Create PastaOrder object and add to factory_state.production_queue
    # 4. Calculate estimated delivery date based on priority and production capacity
    # 5. Update inventory by subtracting required ingredients
    # 6. Return success status with delivery date
    pass

@tool
def create_custom_pasta_recipe(
    pasta_name: str,
    ingredients: Dict[str, float]
) -> Dict[str, Any]:
    """
    Create a custom pasta recipe with specific ingredient ratios.
    
    Args:
        pasta_name: Name of the custom pasta
        ingredients: Dictionary mapping ingredient names to amounts needed per kg
        
    Returns:
        Status of the recipe creation
    """
    # TODO: Implement this function
    # 1. Validate ingredients exist in factory_state.inventory
    # 2. Check if recipe name already exists
    # 3. Add the custom recipe to factory_state.custom_recipes
    # 4. Update factory_state.known_pasta_shapes using update_known_pasta_shapes()
    # 5. Return success status with recipe details
    pass

@tool
def prioritize_order(order_id: str, new_priority: int) -> Dict[str, Any]:
    """
    Change the priority of an existing order in the queue.
    
    Args:
        order_id: ID of the order to update
        new_priority: New priority level (1=normal, 2=rush, 3=emergency)
        
    Returns:
        Status of the priority change
    """
    # TODO: Implement this function
    # 1. Validate priority level (1, 2, or 3)
    # 2. Find the order in factory_state.production_queue
    # 3. Update the order's priority
    # 4. Recalculate estimated delivery date based on new priority
    # 5. Return success status with new delivery date
    pass

# ======= Agents =======

class OrderProcessorAgent(ToolCallingAgent):
    """Agent responsible for processing customer order requests."""
    
    def __init__(self, model):
        super().__init__(
            tools=[check_pasta_recipe, generate_order_id, list_available_pasta_shapes],
            model=model,
            name="order_processor",
            description="Agent responsible for processing customer orders. Parses requests, identifies pasta shapes and quantities."
        )

class InventoryManagerAgent(ToolCallingAgent):
    """Agent responsible for managing ingredient inventory."""
    
    def __init__(self, model):
        super().__init__(
            tools=[check_inventory, check_pasta_recipe, update_inventory],
            model=model,
            name="inventory_manager",
            description="Agent responsible for tracking and managing ingredient inventory."
        )

class ProductionManagerAgent(ToolCallingAgent):
    """Agent responsible for managing the production queue."""
    
    def __init__(self, model):
        super().__init__(
            tools=[check_production_capacity],  # TODO: Add add_to_production_queue, prioritize_order
            model=model,
            name="production_manager",
            description="Agent responsible for managing production scheduling and prioritization."
        )

# TODO: Implement the CustomPastaDesignerAgent class
class CustomPastaDesignerAgent(ToolCallingAgent):
    """TODO: Agent responsible for designing custom pasta recipes."""
    
    def __init__(self, model):
        # TODO: Initialize with appropriate tools for custom pasta design
        super().__init__(
            tools=[],  # TODO: Add check_inventory, create_custom_pasta_recipe
            model=model,
            name="pasta_designer",
            description="TODO: Add description for custom pasta design specialist.",
        )

# ======= Orchestrator =======

# TODO: Create proper Orchestrator using ToolCallingAgent pattern
class Orchestrator(ToolCallingAgent):
    """TODO: Orchestrator that coordinates workflow between specialized agents."""
    
    def __init__(self, model):
        self.model = model
        
        # TODO: Initialize specialized agents
        # self.order_processor = OrderProcessorAgent(model)
        # self.inventory_manager = InventoryManagerAgent(model)
        # self.production_manager = ProductionManagerAgent(model)
        # self.pasta_designer = CustomPastaDesignerAgent(model)

        # TODO: Create coordination tools that route requests to different agents
        @tool
        def process_order_info(customer_request: str) -> str:
            """Process customer order information to extract details.
            
            Args:
                customer_request: The customer's order request
                
            Returns:
                Processed order information with pasta shape and quantity
            """
            # TODO: Route this request to the OrderProcessorAgent
            pass

        @tool
        def manage_inventory(order_details: str) -> str:
            """Check and manage inventory for an order.
            
            Args:
                order_details: Details about the order including pasta shape and quantity
                
            Returns:
                Inventory management result
            """
            # TODO: Route this request to the InventoryManagerAgent
            pass

        @tool
        def schedule_production(order_info: str, priority: int = 1) -> str:
            """Schedule production for an order.
            
            Args:
                order_info: Information about the order to schedule
                priority: Order priority (1=normal, 2=rush, 3=emergency)
                
            Returns:
                Production scheduling result with delivery date
            """
            # TODO: Route this request to the ProductionManagerAgent
            pass

        @tool
        def design_custom_pasta(customer_request: str) -> str:
            """Design a custom pasta recipe based on customer requirements.
            
            Args:
                customer_request: Customer's custom pasta request
                
            Returns:
                Custom pasta design result
            """
            # TODO: Route this request to the CustomPastaDesignerAgent
            pass

        super().__init__(
            tools=[],  # TODO: Add the coordination tools
            model=model,
            name="orchestrator",
            description="""
            TODO: Add description for orchestrating the pasta factory system
            by coordinating between specialized agents for order processing,
            inventory management, production scheduling, and custom pasta design.
            """,
        )
        
    def process_order(self, customer_request: str) -> str:
        """
        Process a customer order through coordinated agent workflow.
        """
        # TODO: Implement coordinated workflow
        # 1. Check if it's a custom pasta request
        # 2. Determine priority from customer language
        # 3. Use coordination tools to process through appropriate agents
        # 4. Return comprehensive response to customer
        pass

# ======= Main Demo =======

def run_demo():
    """Run a demonstration of the pasta factory system."""
    # TODO: Create orchestrator and test the multi-agent coordination
    # orchestrator = Orchestrator(model)
    
    print("Welcome to the Pasta Factory Multi-Agent System!")
    print("Initial Factory State:", json.dumps(factory_state.to_dict(), indent=2))
    
    orders = [
        "I'd like to order 2kg of spaghetti please. When can I get it?",
        "I need a custom pasta with extra semolina and no eggs. Can you make that?",
        "Rush order! We need 5kg of fettuccine for a catering event tomorrow!",
    ]
    
    for i, order in enumerate(orders):
        print(f"\n--- Processing Order {i+1} ---")
        print(f"Customer: {order}")
        
        # response = orchestrator.process_order(order)
        # print(f"Factory: {response}")
        print("Factory: [TODO: Implement orchestrator.process_order]")
        
    print("\n--- Final Factory State ---")
    print(json.dumps(factory_state.to_dict(), indent=2))
    print("\nDemo complete! This demonstrates multi-agent coordination with shared state management.")

if __name__ == "__main__":
    run_demo()