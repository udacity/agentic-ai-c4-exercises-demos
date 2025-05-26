# 🍝 Italian Pasta Factory Custom Orders Exercise

## Overview

In this exercise, you'll extend the Italian Pasta Factory multi-agent system to handle custom pasta recipes and order prioritization. This builds on the shared state management concepts demonstrated in the demo and challenges you to implement more advanced agent coordination.

## Learning Objectives

By completing this exercise, you'll learn how to:
1. Implement tools that modify shared state across multiple agents
2. Create a new specialized agent that works with existing agents
3. Extend an orchestrator to coordinate more complex workflows
4. Handle transaction-based state modifications in a multi-agent system

## Task Description

You're tasked with extending the pasta factory to handle custom pasta recipes and priority ordering. You'll need to:

1. Implement the `add_to_production_queue`, `create_custom_pasta_recipe`, and `prioritize_order` tools
2. Create a new `CustomPastaDesignerAgent` that can design custom pasta recipes based on customer requirements
3. Extend the `PastaFactoryOrchestrator` to handle custom pasta requests and priority orders
4. Complete the order processing workflow to maintain proper state coordination

## Getting Started

1. Review the starter code to understand the existing agent architecture
2. Implement the TODO items in the starter.py file
3. Test your implementation with the provided demo function

## Evaluation Criteria

Your solution will be evaluated based on:
1. Correct implementation of shared state modifications
2. Proper coordination between agents through the orchestrator
3. Robust error handling and edge cases
4. Maintaining data consistency across the system

## Hints

- Make sure to check inventory before accepting orders
- When creating custom recipes, validate the ingredients
- Priority orders should be processed before normal orders
- The orchestrator needs to coordinate the flow of information between agents
- Pay special attention to how state changes are propagated through the system