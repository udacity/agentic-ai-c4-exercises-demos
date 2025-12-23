# Exercise 5: Multi-Agent State Management with Purchase Tracking

## Colombian Fruit Market System

### Overview

This exercise focuses on building a multi-agent orchestration system that demonstrates proper state management across specialized agents. You'll implement purchase tracking functionality while maintaining the multi-agent coordination patterns from the demo.

### Background

Building on the Colombian fruit advisory system from the demo, you'll extend it with comprehensive purchase tracking capabilities. This represents a practical extension of state management that's common in real e-commerce systems where user preferences and transaction histories must be coordinated across multiple specialized agents.

### Learning Objectives

By completing this exercise, you will:
- Implement multi-agent orchestration with specialized agents
- Create tools that coordinate workflow between different agents
- Add purchase tracking and transaction analytics to a state management system
- Maintain state persistence across sessions using proper agent coordination

### Task Description

You'll be working with these specialized agents:

1. **FruitInfoAgent** - Provides information about Colombian fruits using `get_fruit_description`
2. **PreferenceAgent** - Manages user preferences and state persistence using `add_fruit_preference`, `get_user_preferences`, and `save_user_state`
3. **PurchaseAgent** - Handles purchase operations using `purchase_fruit`, `get_purchase_history`, and `get_purchase_summary`

### What You Need to Implement

**1. Complete the purchase_fruit() tool:**
- Record purchases with timestamps, quantities, and pricing
- Store purchase records in the user_states dictionary
- Return confirmation message with purchase details

**2. Complete the get_purchase_summary() tool:**
- Calculate total spending across all purchases
- Count total number of transactions
- Find most frequently purchased fruit
- Return dictionary with analytics

**3. Add purchase summary support:**
- Add `get_purchase_summary` tool to the PurchaseAgent
- Add 'summary' action support to the orchestrator's `handle_purchase` tool

### Key Requirements

Your solution must demonstrate:
- **Real orchestration** where the orchestrator coordinates workflow between specialized agents through tools
- **State persistence** where user preferences and purchase history survive across sessions
- **Purchase analytics** including total spending, transaction counts, and most purchased items
- **Coordinated workflows** where purchasing automatically updates preferences for new fruits

### Getting Started

1. Review the starter code in `starter.py` - it has the framework with TODO placeholders
2. Implement the `purchase_fruit()` tool to record transactions with timestamps and pricing
3. Implement the `get_purchase_summary()` tool to calculate analytics
4. Add `get_purchase_summary` to the PurchaseAgent's tools list
5. Add 'summary' action support to the orchestrator's `handle_purchase` tool

### Evaluation Criteria

Your solution will be evaluated based on:
1. Proper multi-agent orchestration (not single-agent tool usage)
2. Complete purchase tracking functionality with analytics
3. State persistence across orchestrator instances
4. Coordinated workflows between multiple specialized agents

### Hints

- Purchase records should include timestamps using `datetime.now().isoformat()`
- Purchase records should include quantities, prices, and total costs
- The purchase summary should use `Counter` to find most purchased fruits
- The orchestrator tools should actually call `self.agent_name.run()` to route to different agents