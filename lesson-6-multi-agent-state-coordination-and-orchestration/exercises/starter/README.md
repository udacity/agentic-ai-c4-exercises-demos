# 🍝 Italian Pasta Factory Multi-Agent Orchestration Exercise

## Overview

In this exercise, you'll extend the Italian Pasta Factory multi-agent system to handle custom pasta recipes and order prioritization using proper multi-agent orchestration patterns. This builds on the shared state management concepts from the demo and demonstrates how to maintain the smolagents orchestration architecture while adding new capabilities.

## Learning Objectives

By completing this exercise, you'll learn how to:
1. Implement tools that modify shared state across multiple agents
2. Create a new specialized agent that integrates with existing orchestration
3. Build a proper `ToolCallingAgent` orchestrator with coordination tools
4. Handle transaction-based state modifications through agent coordination
5. Extend multi-agent systems while maintaining clean orchestration patterns

## Task Description

You're tasked with extending the pasta factory to handle custom pasta recipes and priority ordering using the proper smolagents orchestration pattern. You'll need to:

### 1. Implement Missing Tools
- **`add_to_production_queue`**: Add orders to the production queue with priority handling and delivery date calculation
- **`create_custom_pasta_recipe`**: Create custom pasta recipes and update the known pasta shapes
- **`prioritize_order`**: Change the priority of existing orders and recalculate delivery dates

### 2. Create the CustomPastaDesignerAgent
- Implement the `CustomPastaDesignerAgent` class that inherits from `ToolCallingAgent`
- Add appropriate tools for custom pasta design (check_inventory, create_custom_pasta_recipe)
- Write a clear description of the agent's responsibilities

### 3. Build the Proper Orchestrator
- Make the `Orchestrator` inherit from `ToolCallingAgent` (not a custom class)
- Create coordination tools that route requests to specialized agents:
  - `process_order_info`: Route to OrderProcessorAgent
  - `manage_inventory`: Route to InventoryManagerAgent  
  - `schedule_production`: Route to ProductionManagerAgent
  - `design_custom_pasta`: Route to CustomPastaDesignerAgent
- Implement the `process_order` method that coordinates workflow between agents

### 4. Complete the Workflow Integration
- Handle custom pasta requests by first designing the recipe, then processing the order
- Implement priority detection from customer language (rush, emergency, urgent)
- Coordinate between multiple agents for complex workflows

## Architecture Pattern

Your solution must follow the established smolagents orchestration pattern:

```python
class Orchestrator(ToolCallingAgent):
    def __init__(self, model):
        # Initialize specialized agents
        self.agent_name = SpecializedAgent(model)
        
        @tool
        def coordination_tool(request: str) -> str:
            return self.agent_name.run(f"Process this request: {request}")
        
        super().__init__(tools=[coordination_tool], ...)
```

This ensures consistency with lessons 3 and 5 where the orchestrator coordinates workflow through tools that route to specialized agents.

## Getting Started

1. Review the starter code and understand the existing state management structure
2. Implement the three missing tools with proper state modifications
3. Create the `CustomPastaDesignerAgent` following the established agent pattern
4. Build the `Orchestrator` as a `ToolCallingAgent` with coordination tools
5. Implement the `process_order` workflow method
6. Test your implementation with the provided demo scenarios

## Evaluation Criteria

Your solution will be evaluated based on:
1. **Proper orchestration pattern**: Orchestrator must inherit from `ToolCallingAgent` and use coordination tools
2. **Correct state management**: Tools must properly modify the shared factory state
3. **Agent coordination**: Multiple agents must work together through the orchestrator
4. **Custom recipe functionality**: System must handle custom pasta design and ordering
5. **Priority handling**: Rush and emergency orders must be prioritized appropriately

## Key Requirements

- Orchestrator **must** inherit from `ToolCallingAgent` (not a custom class)
- Coordination **must** happen through tools that call `self.agent_name.run()`
- State modifications **must** maintain consistency across the system
- Custom pasta workflow **must** coordinate between designer and other agents
- Priority orders **must** affect delivery dates and queue position

## Hints

- Follow the same orchestration pattern used in lessons 3 and 5
- Use `factory_state.update_known_pasta_shapes()` after adding custom recipes
- Priority orders should get faster delivery dates (1 day for emergency, half normal time for rush)
- The orchestrator should detect custom requests and route to the pasta designer first
- Make sure to validate ingredient availability before creating custom recipes
- Test edge cases like inventory shortages and invalid priority levels