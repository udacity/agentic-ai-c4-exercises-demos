# 🍝 Italian Pasta Factory Custom Orders: Solution

## Overview

`solution.py` is the completed version of `starter/starter.py`. It extends the Italian Pasta Factory multi-agent system with custom pasta recipes and order priorities. A `ToolCallingAgent` orchestrator coordinates four specialized agents, and every agent works on one shared factory state.

## Key Components

### Shared State

* **`PastaOrder`**: one order, with its ID, pasta shape, quantity in kg, status, priority (1 = normal, 2 = rush, 3 = emergency), customer notes and estimated delivery date.
* **`FactoryState`**: everything the agents share: the ingredient `inventory`, the `production_queue`, the standard `pasta_recipes`, the `custom_recipes`, an order counter and the list of `known_pasta_shapes`.
* **`factory_state`**: the single `FactoryState` instance that all tools read and modify.

### Tools

Agents never change `factory_state` directly. They call tools that enforce the business rules:

* `check_pasta_recipe`, `check_inventory`, `list_available_pasta_shapes`: read the state.
* `generate_order_id`: returns the next order ID (`ORD-0001`, `ORD-0002`, ...).
* `check_production_capacity`: estimates how long the current queue will take, at 10 kg per day.
* `add_to_production_queue`: checks that the pasta shape has a recipe and that the inventory holds enough of every ingredient. Only then does it queue the order and subtract the ingredients. If anything is missing, it changes nothing and returns `"success": False`.
* `create_custom_pasta_recipe`: accepts a recipe only if every ingredient is in the inventory and the name isn't taken, then adds it to `custom_recipes`.
* `prioritize_order`: changes the priority of a queued order and recalculates its delivery date.
* `update_inventory`: sets an ingredient to a new amount.

### Specialized Agents

Each one is a smolagents `ToolCallingAgent` with only the tools its job needs:

| Agent | Tools |
|---|---|
| `OrderProcessorAgent` | `check_pasta_recipe`, `generate_order_id`, `list_available_pasta_shapes` |
| `InventoryManagerAgent` | `check_inventory`, `check_pasta_recipe`, `update_inventory` |
| `ProductionManagerAgent` | `check_production_capacity`, `add_to_production_queue`, `prioritize_order` |
| `CustomPastaDesignerAgent` | `check_inventory`, `create_custom_pasta_recipe` |

### The Orchestrator

`Orchestrator` is itself a `ToolCallingAgent`. It creates the four specialized agents and gives itself four **coordination tools**, each of which passes a request to one agent with `.run()`:

| Coordination tool | Sends the request to |
|---|---|
| `process_order_info` | `OrderProcessorAgent`: identify the pasta shape and quantity, generate an order ID |
| `design_custom_pasta` | `CustomPastaDesignerAgent`: design and save a custom recipe |
| `manage_inventory` | `InventoryManagerAgent`: check that the ingredients are available |
| `schedule_production` | `ProductionManagerAgent`: queue the order with its priority and give a delivery date |

`process_order(customer_request)` prepares each order before handing it to the orchestrator:

1. it flags a custom request (words such as "custom", "special" or "design");
2. it sets the priority from the customer's words: "rush", "urgent", "priority" or "asap" give 2, "emergency" gives 3;
3. it passes the request, both flags and the workflow (understand the order, design the custom pasta first if one is requested, check the inventory, then schedule production) to `self.run()`, and the orchestrator decides which coordination tools to call. If an order can't be fulfilled, the orchestrator explains why to the customer.

## How the Shared State Stays Consistent

This solution doesn't use a lock. Two design choices keep the state consistent:

* only tools modify `factory_state`;
* `add_to_production_queue` checks the inventory and subtracts from it **in the same tool call**. So an order is either fully reserved or rejected with the inventory unchanged, even if an agent misjudged the inventory earlier in the conversation.

`run_demo()` processes the orders one after another, so two `add_to_production_queue` calls never overlap here. If they could run at the same moment (newer smolagents versions can run several tool calls from one step in parallel threads, for example), both could pass the inventory check before either one subtracted its ingredients. To rule that out, you would hold a `threading.Lock` around the check and the subtraction in `add_to_production_queue`. (`asyncio.Lock` only coordinates `async` code running on one event loop, so it wouldn't protect tools running in threads.)

## Running the Code

From the `solution` folder, run:

```bash
python solution.py
```

Like the starter and the demo, the script reads your API key from `UDACITY_OPENAI_API_KEY` and uses `gpt-4o-mini`. `run_demo()` processes three orders (a standard spaghetti order, a custom pasta request and a rush fettuccine order) and prints the factory state before and after.
