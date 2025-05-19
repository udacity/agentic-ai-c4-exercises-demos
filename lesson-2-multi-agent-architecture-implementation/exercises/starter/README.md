# Exercise 2: Multi-Agent Architecture Implementation

## Antarctic Agents: Simulating Penguin and Scientist Interactions

### Overview

In this exercise, you will implement a multi-agent system simulating interactions between scientists and penguin colonies in Antarctica. You'll create specialized agents that represent different actors in this ecosystem, establish communication patterns between them, and implement the logic for their interactions.

### Learning Objectives

By completing this exercise, you will:
- Implement concrete agents based on architectural specifications
- Create messaging systems for communication between agents
- Establish coordination patterns between different agent types
- Handle state tracking and information sharing across the system

### Task Description

You'll be creating the following agents:
1. **PenguinColonyAgent** - Simulates a penguin colony's behaviors and needs
2. **ScientistAgent** - Represents field researchers studying the penguins
3. **ResearchStationAgent** - Coordinates research activities and resource allocation
4. **WeatherServiceAgent** - Provides weather forecasts that affect operations

Your implementation should handle:
- Regular observation cycles by scientists
- Resource management (food, equipment, research data)
- Weather impacts on research activities
- Communication between the research station and field teams

### Getting Started

1. Review the starter code in `starter.py`
2. Implement the agent classes with their required functionality
3. Create the messaging system for inter-agent communication
4. Complete the simulation system that coordinates agent activities

### Evaluation Criteria

Your solution will be evaluated based on:
1. Correct implementation of agent behaviors
2. Effective message passing between agents
3. Proper state management within each agent
4. Realistic simulation of the Antarctic research environment

### Hints

- Consider how weather conditions affect both penguins and scientists
- Think about different types of messages that need to be passed
- Remember that resources are limited in the Antarctic environment
- The research station needs to balance priorities across multiple research teams

