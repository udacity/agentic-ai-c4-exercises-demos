# Exercise 5: State Management in Multi-Agent Systems

## Healthcare System: Patient Data Management Across Clinical Teams

### Overview

In this exercise, you will implement a state management system for a healthcare organization that coordinates patient care across multiple clinical teams. You'll create mechanisms to store, share, and update patient information while maintaining data consistency and privacy.

### Learning Objectives

By completing this exercise, you will:
- Implement different state storage and sharing patterns
- Create consistency management across distributed agent states
- Design transactional operations that maintain data integrity
- Handle concurrent access and modifications to shared state

### Task Description

You'll be implementing a system with the following agents:

1. **Primary Care Provider Agent** - Manages routine patient care and referrals
2. **Specialist Agent** - Provides specialized medical expertise and treatments
3. **Pharmacy Agent** - Handles medication prescriptions and interactions
4. **Laboratory Agent** - Processes and reports diagnostic tests

Your implementation should handle:
- Patient record creation and updates
- Information sharing between care teams with appropriate access controls
- Medication prescription and interaction checking
- Scheduling and tracking of appointments and procedures
- Consistent views of patient data across all agents

### Getting Started

1. Review the starter code in `starter.py`
2. Implement the state management classes for storing patient information
3. Create the agent classes that interact with the shared state
4. Implement the transaction mechanisms to ensure data consistency

### Evaluation Criteria

Your solution will be evaluated based on:
1. Effective state storage and sharing implementation
2. Proper consistency management across the system
3. Robust transaction handling with appropriate rollbacks
4. Privacy-preserving information sharing between agents

### Hints

- Consider the different access levels required for patient information
- Think about how to handle conflicting updates from different specialists
- Plan for transaction rollbacks when operations cannot be completed
- Design your state structures to be easily queried by different agents