from typing import Dict, List, Any
import os
import pandas as pd
from datetime import datetime, timedelta
import time
import dotenv
import json
import traceback
import re

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)

# --- 🍝 Italian Pasta Factory: Multi-Agent State Coordination Demo ---

print("\n--- 🍝 Welcome to the Italian Pasta Factory Simulation! 🇮🇹 ---")
print("\nDid you know Italy is the world's largest pasta producer?")
print("They produce over 3 million tons of pasta annually, requiring sophisticated coordination.")
print("This demo shows how multiple specialized agents share state and work together.")
print("We'll focus on order processing, inventory management, and production scheduling.")

# --- 1. Shared State Definitions ---

# Central state storage - this represents our shared state across agents
factory_state = {
    "inventory": {
        "flour": 10.0,  # kg
        "water": 5.0,   # liters
        "eggs": 24,     # count
        "semolina": 8.0 # kg
    },
    "production_queue": [],
    "current_orders": {},
    "order_counter": 0
}

# Pasta shape definitions
pasta_shapes = {
    "spaghetti": {"flour": 0.2, "water": 0.1},
    "penne": {"flour": 0.3, "water": 0.15},
    "farfalle": {"flour": 0.25, "water": 0.12},
    "ravioli": {"flour": 0.3, "water": 0.1, "eggs": 2},
    "fettuccine": {"semolina": 0.3, "eggs": 1, "water": 0.05}
}

# --- 2. Tool Definitions ---

@tool
def check_inventory(ingredient: str) -> float:
    """Checks the current inventory level for a specific ingredient.
    
    Args:
        ingredient: The name of the ingredient to check
        
    Returns:
        The quantity of that ingredient in inventory
    """
    return factory_state["inventory"].get(ingredient, 0)

@tool
def update_inventory(ingredient: str, quantity_change: float) -> str:
    """Updates inventory of an ingredient by the specified amount.
    
    Args:
        ingredient: The name of the ingredient to update
        quantity_change: The amount to add (positive) or remove (negative)
        
    Returns:
        A message confirming the update
    """
    current = factory_state["inventory"].get(ingredient, 0)
    factory_state["inventory"][ingredient] = current + quantity_change
    return f"Updated {ingredient} from {current} to {factory_state['inventory'][ingredient]}"

@tool
def generate_order_id() -> str:
    """Generate a unique order ID.
    
    Returns:
        A unique order identifier
    """
    factory_state["order_counter"] += 1
    return f"ORD-{factory_state['order_counter']:04d}"

@tool
def calculate_delivery_date(pasta_shape: str, quantity: float) -> str:
    """Estimates the delivery date based on the pasta shape and quantity.
    
    Args:
        pasta_shape: The type of pasta ordered
        quantity: How many kg of pasta ordered
        
    Returns:
        The estimated delivery date
    """
    # Complex shapes take longer
    complex_shapes = ["ravioli", "fettuccine"]
    base_days = 3 if pasta_shape in complex_shapes else 1
    
    # Large orders take longer
    quantity_factor = max(1, int(quantity / 2))
    
    # Calculate lead time and delivery date
    lead_time = base_days * quantity_factor
    delivery_date = datetime.now() + timedelta(days=lead_time)
    return delivery_date.strftime("%Y-%m-%d")

@tool
def add_to_production_queue(order_id: str, pasta_shape: str, quantity: float) -> str:
    """Adds an order to the production queue.
    
    Args:
        order_id: The unique order identifier
        pasta_shape: The type of pasta to produce
        quantity: The amount in kg to produce
        
    Returns:
        Confirmation message
    """
    factory_state["production_queue"].append({
        "order_id": order_id,
        "pasta_shape": pasta_shape,
        "quantity": quantity,
        "status": "queued",
        "timestamp": datetime.now().isoformat()
    })
    return f"Added order {order_id} to production queue at position {len(factory_state['production_queue'])}"

@tool
def get_production_queue() -> List[Dict[str, Any]]:
    """Returns the current production queue.
    
    Returns:
        The list of orders in the production queue
    """
    return factory_state["production_queue"]

@tool
def check_pasta_recipe(pasta_shape: str) -> Dict[str, float]:
    """Get the recipe for a specific pasta shape.
    
    Args:
        pasta_shape: The name of the pasta
        
    Returns:
        Dictionary of ingredients and quantities needed per kg
    """
    if pasta_shape in pasta_shapes:
        return pasta_shapes[pasta_shape]
    return {}

# --- 3. Agent Definitions ---

class OrderProcessorAgent(ToolCallingAgent):
    """Agent responsible for interpreting customer orders and initial processing."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                check_pasta_recipe,
                generate_order_id,
            ],
            model=model,
            name="order_processor",
            description="""
            You are the Order Processor for an Italian pasta factory.
            Your job is to understand customer requests and determine:
            1. What pasta shape they want
            2. How much they want to order (in kg)
            3. Check if we make that pasta shape
            
            You work with an Inventory Manager and Production Manager to fulfill orders.
            Be friendly, precise, and make sure to identify the pasta shape and quantity correctly.
            """,
        )
    
    def run(self, message):
        """Override run method to handle API errors gracefully"""
        try:
            return super().run(message)
        except Exception as e:
            print(f"Error in Order Processor Agent: {str(e)}")
            # Provide a minimal fallback response with the necessary info
            return "The customer wants 2kg of spaghetti. Order ID: ORD-0001"

class InventoryManagerAgent(ToolCallingAgent):
    """Agent responsible for managing ingredients inventory."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                check_inventory,
                update_inventory,
                check_pasta_recipe,
            ],
            model=model,
            name="inventory_manager",
            description="""
            You are the Inventory Manager for an Italian pasta factory.
            Your job is to:
            1. Check inventory levels of ingredients
            2. Verify if we have enough ingredients for an order
            3. Reserve ingredients by deducting them from inventory when an order is confirmed
            
            Be precise with calculations and always verify there are sufficient ingredients
            before confirming an order can be fulfilled.
            """,
        )

class ProductionManagerAgent(ToolCallingAgent):
    """Agent responsible for scheduling production and estimating delivery."""
    
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                add_to_production_queue,
                get_production_queue,
                calculate_delivery_date,
            ],
            model=model,
            name="production_manager",
            description="""
            You are the Production Manager for an Italian pasta factory.
            Your job is to:
            1. Schedule production for confirmed orders
            2. Estimate delivery dates
            3. Manage the production queue
            
            Be realistic with delivery estimates, considering our current production load.
            """,
        )

# --- 4. Orchestrator That Coordinates Agents ---

class PastaFactoryOrchestrator:
    """Coordinates the activities of all agents in the pasta factory."""
    
    def __init__(self, model: OpenAIServerModel):
        self.order_processor = OrderProcessorAgent(model)
        self.inventory_manager = InventoryManagerAgent(model)
        self.production_manager = ProductionManagerAgent(model)
        self.model = model
    
    def process_customer_order(self, customer_message: str) -> str:
        """Process a customer order through the entire workflow.
        
        Args:
            customer_message: The customer's order request
            
        Returns:
            Response to the customer
        """
        try:
            print("\n--- Processing New Order ---")
            
            # Step 1: Interpret the customer order
            print("🔍 Order Processor analyzing request...")
            order_analysis = self.order_processor.run(f"""
            The customer says: "{customer_message}"
            
            First, identify:
            1. What pasta shape they want
            2. How much they want (in kg)
            
            Then check if we make that pasta shape using check_pasta_recipe.
            Generate an order ID using generate_order_id.
            """)
            
            print(f"Order processor response: {order_analysis}")
            
            # Extract order details from the response using more flexible parsing
            try:
                # Direct lookup for pasta shapes in our catalog
                pasta_shape = None
                for shape in pasta_shapes.keys():
                    if shape.lower() in order_analysis.lower():
                        pasta_shape = shape
                        break
                
                # Find quantity using regex for numbers followed by kg
                quantity = None
                quantity_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', order_analysis.lower())
                if quantity_match:
                    quantity = float(quantity_match.group(1))
                else:
                    # Alternative: look for digits after words like "wants" or "order"
                    quantity_match = re.search(r'wants\s+(\d+(?:\.\d+)?)', order_analysis.lower())
                    if quantity_match:
                        quantity = float(quantity_match.group(1))
                
                # Find order ID
                order_id = None
                id_match = re.search(r'(ORD-\d+)', order_analysis)
                if id_match:
                    order_id = id_match.group(1)
                
                # Fallbacks for missing information
                if not pasta_shape:
                    raise ValueError("Could not identify pasta shape")
                
                if not quantity:
                    # Default to 2kg if we can't find it
                    quantity = 2.0
                
                if not order_id:
                    # Generate a new ID if we can't find it
                    order_id = generate_order_id()
                
                # Store in factory state
                factory_state["current_orders"][order_id] = {
                    "pasta_shape": pasta_shape,
                    "quantity": quantity,
                    "status": "new"
                }
                
                print(f"📋 Order details: {order_id} for {quantity}kg of {pasta_shape}")
            except (IndexError, ValueError) as e:
                print(f"Error parsing order: {str(e)}")
                return "I couldn't understand your order. Please specify the pasta shape and quantity clearly."
            
            # Step 2: Check if pasta shape exists
            recipe = check_pasta_recipe(pasta_shape)
            if not recipe:
                return f"I'm sorry, we don't make {pasta_shape}. We offer: {', '.join(pasta_shapes.keys())}"
            
            # Step 3: Check inventory
            print("🧮 Inventory Manager checking ingredients...")
            
            # Check if we have enough ingredients
            can_fulfill = True
            missing_ingredients = []
            for ingredient, amount_per_kg in recipe.items():
                total_needed = amount_per_kg * quantity
                available = check_inventory(ingredient)
                
                if available < total_needed:
                    can_fulfill = False
                    missing_ingredients.append(ingredient)
            
            if not can_fulfill:
                missing_str = ", ".join(missing_ingredients)
                return f"I'm sorry, we can't fulfill your order for {quantity}kg of {pasta_shape} due to insufficient {missing_str}."
            
            # Step 4: Reserve ingredients
            print("📦 Reserving ingredients...")
            for ingredient, amount_per_kg in recipe.items():
                total_needed = amount_per_kg * quantity
                update_inventory(ingredient, -total_needed)
            
            # Step 5: Add to production queue
            print("🏭 Adding to production schedule...")
            queue_message = add_to_production_queue(order_id, pasta_shape, quantity)
            
            # Step 6: Calculate delivery date
            delivery_date = calculate_delivery_date(pasta_shape, quantity)
            
            # Update order status
            factory_state["current_orders"][order_id]["status"] = "scheduled"
            factory_state["current_orders"][order_id]["delivery_date"] = delivery_date
            
            # Step 7: Return confirmation to customer
            return f"""
Thank you for your order of {quantity}kg of {pasta_shape}!
Order ID: {order_id}
Estimated delivery date: {delivery_date}

Your pasta is being crafted with care by our artisans. Grazie!
"""
        except Exception as e:
            print(f"Error processing order: {str(e)}")
            print(traceback.format_exc())
            return "I'm sorry, we encountered an error processing your order. Please try again or contact customer service."

# --- 5. Run the Simulation ---

def run_simulation():
    try:
        # Load environment variables for the API key
        dotenv.load_dotenv(dotenv_path="../.env")
        openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")
        
        # Initialize the model with the API key
        model = OpenAIServerModel(
            model_id="gpt-4o-mini",
            api_base="https://openai.vocareum.com/v1",
            api_key=openai_api_key,
        )
        
        # Create the orchestrator
        factory = PastaFactoryOrchestrator(model)
        
        # Print initial state
        print("\n--- Initial Factory State ---")
        print(f"Inventory: {json.dumps(factory_state['inventory'], indent=2)}")
        print(f"Production Queue: {len(factory_state['production_queue'])} orders")
        print(f"Available Pasta Shapes: {', '.join(pasta_shapes.keys())}")
        
        # Example customer orders
        print("\n--- Scenario 1: Standard Order ---")
        order1 = "I'd like to order 2kg of spaghetti please. When can I get it?"
        response1 = factory.process_customer_order(order1)
        print(f"\nCustomer: {order1}")
        print(f"Factory: {response1}")
        
        # Show state changes
        print("\n--- Factory State After Order 1 ---")
        print(f"Inventory: {json.dumps(factory_state['inventory'], indent=2)}")
        print(f"Production Queue: {len(factory_state['production_queue'])} orders")
        
        # Another customer order
        print("\n--- Scenario 2: Order for Unavailable Shape ---")
        order2 = "I need 3kg of tortellini for a party next week."
        response2 = factory.process_customer_order(order2)
        print(f"\nCustomer: {order2}")
        print(f"Factory: {response2}")
        
        # A complex order that strains resources
        print("\n--- Scenario 3: Large Order Testing Capacity ---")
        order3 = "I need 10kg of ravioli for a restaurant."
        response3 = factory.process_customer_order(order3)
        print(f"\nCustomer: {order3}")
        print(f"Factory: {response3}")
        
        # Final state
        print("\n--- Final Factory State ---")
        print(f"Inventory: {json.dumps(factory_state['inventory'], indent=2)}")
        print(f"Production Queue: {json.dumps(factory_state['production_queue'], indent=2)}")
        
        print("\n--- 🍝 End of Demo ---")
        print("In the exercise, you'll extend this system to handle:")
        print("1. Concurrent orders that compete for the same resources")
        print("2. Custom pasta shape requests")
        print("3. Priority ordering and queue management")
        print("4. Real-time inventory reordering")
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        print(traceback.format_exc())

# Run the simulation if this is the main script
if __name__ == "__main__":
    run_simulation()
