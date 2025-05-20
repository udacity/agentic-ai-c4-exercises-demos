# Multi-Agent RAG System - Demo

This directory contains the demonstration code for the Multi-Agent Retrieval Augmented Generation (RAG) system, specifically focused on insurance claims processing.

## Demo Overview

The `demo.py` file shows a working implementation of a multi-agent RAG system for processing insurance claims and handling customer complaints. Key components include:

- Vector similarity search for claim and knowledge retrieval
- Role-based access control for information privacy
- Multi-agent coordination between claim processors, medical reviewers, and customer service agents
- An orchestration layer that manages the workflow

## Key Components

- **VectorKnowledgeBase**: Uses TF-IDF embeddings to enable semantic search across insurance policy information
- **VectorClaimSearch**: Specialized retrieval for finding similar claims
- **Agents**: Specialized agents for different aspects of the claims process
- **Access Control**: Privacy-aware retrieval that filters information based on agent access level

## Running the Demo

To run the demo:

```bash
python demo.py
```

This will show a complete workflow of processing a customer complaint about a denied insurance claim.

## Understanding the RAG Implementation

This demo showcases how retrieval augmented generation enhances a multi-agent system by:

1. Grounding agent responses in domain-specific knowledge
2. Enabling semantic search across claims and policies
3. Providing agents with only the information they need
4. Ensuring privacy and security through access controls

For the exercise, you'll be extending this system with fraud detection capabilities, which will further demonstrate the power of RAG in a multi-agent environment.
