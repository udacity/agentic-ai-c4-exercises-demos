"""
Pasta Factory Exercise - Solution Code
====================================

This solution implements the custom pasta design and order prioritization 
features for the Italian Pasta Factory multi-agent system.
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import random
import re
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
    
    Args:
        pasta_shape: The name of the pasta shape to check the recipe for.
        
    Returns:
        A dictionary of ingredients and amounts needed per kg of pasta.
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
def check_production_capacity(days_ahead: int = 7) -> Dict[str, Any]:
    """
    Check the current production capacity and queue for the next X days.
    Returns information about queue size and estimated completion times.
    
    Args:
        days_ahead: Number of days ahead to check capacity for
        
    Returns:
        Dictionary with production capacity metrics
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
        Status of the queuing operation
    """
    # Verify that pasta_shape is valid
    recipe = check_pasta_recipe(pasta_shape)
    if not recipe:
        return {
            "success": False,
            "message": f"Pasta shape '{pasta_shape}' is not recognized. We don't have a recipe for it."
        }
    
    # Calculate required ingredients
    required_ingredients = {}
    for ingredient, amount in recipe.items():
        required_ingredients[ingredient] = amount * quantity
    
    # Check if we have enough inventory
    for ingredient, required in required_ingredients.items():
        if ingredient not in factory_state.inventory or factory_state.inventory[ingredient] < required:
            return {
                "success": False,
                "message": f"Not enough {ingredient} in inventory to produce {quantity}kg of {pasta_shape}."
            }
    
    # Calculate estimated delivery date based on production capacity
    capacity_info = check_production_capacity()
    
    # Priority orders get processed first, so adjust days_to_complete
    days_to_complete = capacity_info["days_to_complete_current_queue"]
    
    # Rush orders get priority
    if priority > 1:
        # Emergency orders get processed in 1 day
        if priority == 3:
            days_to_complete = 1
        # Rush orders get priority over normal orders
        elif priority == 2:
            days_to_complete = max(1, days_to_complete / 2)
    
    # Add 1 day for the order itself
    total_days = max(1, int(days_to_complete) + 1)
    
    # Calculate the delivery date
    delivery_date = (datetime.now() + timedelta(days=total_days)).strftime("%Y-%m-%d")
    
    # Create the order
    new_order = PastaOrder(
        order_id=order_id,
        pasta_shape=pasta_shape,
        quantity=quantity,
        status="queued",
        priority=priority,
        customer_notes=customer_notes,
        estimated_delivery_date=delivery_date
    )
    
    # Add to queue
    factory_state.production_queue.append(new_order)
    
    # Update inventory - subtract the required ingredients
    for ingredient, required in required_ingredients.items():
        factory_state.inventory[ingredient] -= required
    
    return {
        "success": True,
        "message": f"Order {order_id} added to production queue.",
        "estimated_delivery_date": delivery_date,
        "priority": priority,
        "status": "queued"
    }

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
    # Validate ingredients exist in inventory
    for ingredient in ingredients:
        if ingredient not in factory_state.inventory:
            return {
                "success": False,
                "message": f"Unknown ingredient: {ingredient}. We don't have this in our inventory."
            }
    
    # Check if recipe name already exists
    if pasta_name in factory_state.pasta_recipes or pasta_name in factory_state.custom_recipes:
        return {
            "success": False,
            "message": f"A recipe for '{pasta_name}' already exists."
        }
    
    # Add the custom recipe
    factory_state.custom_recipes[pasta_name] = ingredients
    
    # Update known pasta shapes
    factory_state.update_known_pasta_shapes()
    
    return {
        "success": True,
        "message": f"Custom pasta recipe '{pasta_name}' created successfully.",
        "recipe": ingredients
    }

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
    # Validate priority level
    if new_priority not in [1, 2, 3]:
        return {
            "success": False,
            "message": f"Invalid priority level: {new_priority}. Must be 1 (normal), 2 (rush), or 3 (emergency)."
        }
    
    # Find the order in the queue
    order_found = False
    for order in factory_state.production_queue:
        if order.order_id == order_id:
            order_found = True
            old_priority = order.priority
            
            # Update priority
            order.priority = new_priority
            
            # Recalculate estimated delivery date
            capacity_info = check_production_capacity()
            days_to_complete = capacity_info["days_to_complete_current_queue"]
            
            if new_priority > 1:
                # Emergency orders get processed in 1 day
                if new_priority == 3:
                    days_to_complete = 1
                # Rush orders get priority over normal orders
                elif new_priority == 2:
                    days_to_complete = max(1, days_to_complete / 2)
            
            # Add 1 day for the order itself
            total_days = max(1, int(days_to_complete) + 1)
            
            # Calculate the new delivery date
            new_delivery_date = (datetime.now() + timedelta(days=total_days)).strftime("%Y-%m-%d")
            order.estimated_delivery_date = new_delivery_date
            
            return {
                "success": True,
                "message": f"Order {order_id} priority updated from {old_priority} to {new_priority}.",
                "new_estimated_delivery_date": new_delivery_date
            }
    
    if not order_found:
        return {
            "success": False,
            "message": f"Order {order_id} not found in production queue."
        }

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
            tools=[check_production_capacity, add_to_production_queue, prioritize_order],
            model=model,
            name="production_manager",
            description="Agent responsible for managing production scheduling and prioritization."
        )

class CustomPastaDesignerAgent(ToolCallingAgent):
    """Agent responsible for designing custom pasta recipes."""
    
    def __init__(self, model):
        super().__init__(
            tools=[check_inventory, create_custom_pasta_recipe],
            model=model,
            name="pasta_designer",
            description="Agent responsible for creating custom pasta recipes based on customer requirements and available ingredients."
        )

# ======= Orchestrator =======

class Orchestrator(ToolCallingAgent):
    """Orchestrator that coordinates the multi-agent pasta factory system."""
    
    def __init__(self, model):
        self.model = model
        
        # Initialize specialized agents
        self.order_processor = OrderProcessorAgent(model)
        self.inventory_manager = InventoryManagerAgent(model)
        self.production_manager = ProductionManagerAgent(model)
        self.pasta_designer = CustomPastaDesignerAgent(model)

        @tool
        def process_order_info(customer_request: str) -> str:
            """Process customer order information to extract details.
            
            Args:
                customer_request: The customer's order request
                
            Returns:
                Processed order information with pasta shape and quantity
            """
            return self.order_processor.run(f"""
            The customer says: "{customer_request}"
            
            First, identify:
            1. What pasta shape they want
            2. How much they want (in kg)
            
            Then check if we make that pasta shape using check_pasta_recipe.
            Generate an order ID using generate_order_id.
            
            If the pasta shape is not one we make, you can use list_available_pasta_shapes to see what we offer.
            """)

        @tool
        def manage_inventory(order_details: str) -> str:
            """Check and manage inventory for an order.
            
            Args:
                order_details: Details about the order including pasta shape and quantity
                
            Returns:
                Inventory management result
            """
            return self.inventory_manager.run(f"""
            Order details: {order_details}
            
            Check if we have enough ingredients for this order:
            1. Use check_pasta_recipe to get the ingredient requirements
            2. Use check_inventory to verify we have sufficient ingredients
            3. Calculate if we have enough ingredients to fulfill this order.
            """)

        @tool
        def schedule_production(order_info: str, priority: int = 1) -> str:
            """Schedule production for an order.
            
            Args:
                order_info: Information about the order to schedule
                priority: Order priority (1=normal, 2=rush, 3=emergency)
                
            Returns:
                Production scheduling result with delivery date
            """
            return self.production_manager.run(f"""
            Order information: {order_info}
            Priority level: {priority}
            
            Schedule this order for production:
            1. Add the order to the production queue using add_to_production_queue
            2. Use the specified priority level
            3. Provide the customer with production status and delivery estimate
            """)

        @tool
        def design_custom_pasta(customer_request: str) -> str:
            """Design a custom pasta recipe based on customer requirements.
            
            Args:
                customer_request: Customer's custom pasta request
                
            Returns:
                Custom pasta design result
            """
            return self.pasta_designer.run(f"""
            The customer says: "{customer_request}"
            
            They want a custom pasta recipe. Design a custom pasta recipe based on their requirements:
            1. First, check what ingredients we have available using check_inventory
            2. Create a custom pasta recipe with appropriate ingredient ratios
            3. Name the pasta appropriately based on its characteristics
            4. Use the create_custom_pasta_recipe tool to save the recipe
            """)

        super().__init__(
            tools=[process_order_info, manage_inventory, schedule_production, design_custom_pasta],
            model=model,
            name="orchestrator",
            description="""
            You are the orchestrator for a pasta factory system.
            You coordinate between the order processor, inventory manager, production manager, and custom pasta designer.
            
            For customer orders, follow this workflow:
            1. Use process_order_info to understand what the customer wants
            2. If it's a custom pasta request, use design_custom_pasta first
            3. Use manage_inventory to check ingredient availability
            4. Use schedule_production to add to queue and get delivery date
            
            Determine priority level from customer language (rush, emergency, urgent = higher priority).
            Always provide clear responses to customers about their order status.
            """,
        )
        
    def process_order(self, customer_request: str) -> str:
        """
        Process a customer order from initial request through production queue.
        
        Args:
            customer_request: Natural language order request from customer
            
        Returns:
            Response to customer with order details and status
        """
        # Check if this is a custom pasta request
        is_custom_request = any(term in customer_request.lower() for term in 
                               ["custom", "special", "unique", "create", "design", "make", "new"])
        
        # Determine priority from customer language
        priority = 1  # default
        if any(term in customer_request.lower() for term in ["rush", "emergency", "urgent", "priority", "asap"]):
            if "emergency" in customer_request.lower():
                priority = 3
            else:
                priority = 2
        
        context = f"""
        Customer request: "{customer_request}"
        Is custom request: {is_custom_request}
        Priority level: {priority}
        
        Process this order by coordinating with our specialized agents:
        1. First process the order information to understand what they want
        2. If it's a custom pasta request, design the custom pasta first
        3. Check and manage inventory to ensure we can fulfill it
        4. Schedule production with appropriate priority and provide delivery information
        
        If at any step we cannot fulfill the order, explain why to the customer.
        """
        
        return self.run(context)

# ======= Main Demo =======

def run_demo():
    """Run a demonstration of the pasta factory system."""
    orchestrator = Orchestrator(model)
    
    print("Welcome to the Pasta Factory Multi-Agent System!")
    print("Initial Factory State:", json.dumps(factory_state.to_dict(), indent=2))
    
    # Simulate some customer orders
    orders = [
        "I'd like to order 2kg of spaghetti please. When can I get it?",
        "I need a custom pasta with extra semolina and no eggs. Can you make that?",
        "Rush order! We need 5kg of fettuccine for a catering event tomorrow!",
    ]
    
    for i, order in enumerate(orders):
        print(f"\n--- Processing Order {i+1} ---")
        print(f"Customer: {order}")
        
        response = orchestrator.process_order(order)
        print(f"Factory: {response}")
        
    print("\n--- Final Factory State ---")
    print(json.dumps(factory_state.to_dict(), indent=2))

if __name__ == "__main__":
    run_demo()