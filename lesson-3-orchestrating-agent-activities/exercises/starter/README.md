# Exercise 3: Orchestrating Agent Activities

## Skate Park & Shop System in Kenya

### Overview

This exercise focuses on building a multi-agent system to manage operations of a Skate Park and Shop in Nairobi, Kenya. You'll implement orchestration patterns to coordinate different specialized agents, each responsible for a specific aspect of the business operations.

### Background

Skateboarding and action sports are rapidly growing in popularity across Africa. Kenya, with its vibrant culture and youthful population, is at the forefront of this movement. Your task is to build a system to manage a Skate Park and Shop in Nairobi, focusing on core operations through effective agent orchestration.

### Learning Objectives

By completing this exercise, you will:
- Implement orchestration patterns for coordinating multiple agents
- Create workflow management systems for business operations
- Handle dependencies between different agent activities
- Manage cross-cutting concerns across an agent ecosystem

### Task Description

You'll be working with the following agents:

1. **Customer Support Agent** - Handles customer inquiries and issues
2. **Inventory Agent** - Manages shop stock including skateboards, apparel, etc.
3. **Park Management Agent** - Deals with park bookings, events, and maintenance requests

Your implementation should handle:
- Customer inquiries about products and park facilities
- Inventory tracking and reordering processes
- Event scheduling and facility maintenance
- Coordination between all three agents for comprehensive customer service

### Getting Started

1. Review the starter code in `starter.py`
2. Implement the agent classes with their required functionality
3. Create the orchestration system to coordinate agent activities
4. Implement the workflows for common business operations

### Evaluation Criteria

Your solution will be evaluated based on:
1. Effective orchestration of agent activities
2. Proper workflow management across the system
3. Clear handling of dependencies between agents
4. Realistic simulation of the skate park business operations

### Hints

- Think about how customer requests might require multiple agents to fulfill
- Consider how inventory status affects both shop operations and park events
- Plan for error handling when one agent's activities depend on another's
- The orchestration system should maintain a coherent user experience
