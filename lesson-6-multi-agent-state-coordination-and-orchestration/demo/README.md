# 🍝 Italian Pasta Factory Multi-Agent System Demo

## Overview

This demo showcases a multi-agent system for managing an Italian pasta factory's operations using shared state coordination. The system demonstrates how multiple specialized agents can work together while sharing and modifying a common state to handle customer orders, inventory management, and production scheduling.

## Key Concepts Demonstrated

- **Shared State Management**: All agents access and modify a centralized factory state
- **Specialized Agent Roles**: Each agent has a specific responsibility in the workflow
- **State Coordination**: Agents coordinate through a centralized orchestrator
- **Transaction-based State Changes**: State mutations are tracked through explicit operations

## Agents in the System

1. **Order Processor Agent**: Interprets customer requests and extracts order details
2. **Inventory Manager Agent**: Checks and updates ingredient levels
3. **Production Manager Agent**: Schedules production and calculates delivery dates

## How to Run

To run the demo:

```python
python demo.py
```

This will simulate several customer orders and demonstrate how the multi-agent system processes them while maintaining a consistent shared state.

## Learning Outcomes

By studying this demo, you'll understand:

1. How to design a shared state that multiple agents can access and modify
2. How to implement specialized agents with distinct responsibilities
3. How to coordinate agent activities through an orchestrator
4. How to handle error conditions and edge cases in a multi-agent system
5. How to implement transaction-based state changes for better traceability

These concepts are essential for building robust multi-agent systems that can handle complex workflows and maintain data consistency.
