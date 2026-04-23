# Multi-Agent System Project Plan

## Beaver's Choice Paper Company - Inventory & Quoting System

### Project Overview

Build a multi-agent system (max 5 agents) to streamline inventory management, quote generation, and sales transactions for Beaver's Choice Paper Company. The system must handle text-based inputs/outputs and be tested against provided sample requests.

**Expected Duration:** ~6 hours
**Key Deliverables:** Workflow diagram, implementation script, test results, reflection report

---

## Phase 1: Planning & Design (Completion Target: 30 mins)

### 1.1 Review Project Requirements

- [x] Read overview.md to understand business context and challenges
- [x] Read instructions.md to understand step-by-step approach
- [x] Read rubric.md to understand evaluation criteria
- [x] Identify project constraints:
  - Maximum 5 agents
  - Text-based input/output only
  - Must handle inventory, quotes, and transactions

### 1.2 Review Starter Code

- [x] Open and examine project_starter.py
- [x] Document purpose of each helper function:
  - [x] `create_transaction()` - Purpose & usage
  - [x] `get_all_inventory()` - Purpose & usage
  - [x] `get_stock_level()` - Purpose & usage
  - [x] `get_supplier_delivery_date()` - Purpose & usage
  - [x] `get_cash_balance()` - Purpose & usage
  - [x] `generate_financial_report()` - Purpose & usage
  - [x] `search_quote_history()` - Purpose & usage
- [x] Review database schema and sample data
- [x] Understand the provided evaluation code stub

### 1.3 Design Multi-Agent Architecture

- [x] Define required agents (max 5):
  - [x] Orchestrator Agent - Responsibilities & capabilities
  - [x] Inventory Management Agent - Responsibilities & capabilities
  - [x] Quote Generation Agent - Responsibilities & capabilities
  - [x] Sales Finalization Agent - Responsibilities & capabilities
  - [x] (Optional) 5th Agent - Business Analyst Agent included as optional
- [x] Map agent responsibilities to ensure no overlap
- [x] Identify tools needed for each agent:
  - [x] Tool 1 with helper function reference
  - [x] Tool 2 with helper function reference
  - [x] Tool 3 with helper function reference
  - [x] Complete helper-function-to-agent mapping documented

### 1.4 Create Workflow Diagram

- [x] Choose diagramming tool (Mermaid or Diagrams.net)
- [x] Draft diagram showing:
  - [x] All agents and their responsibilities
  - [x] Tools associated with each agent
  - [x] Data flow between agents
  - [x] Orchestration logic
  - [x] Clear purpose labels for each tool
- [x] Save diagram as image file (diagram.png or diagram.svg)
- [x] Verify diagram clarity and completeness

---

## Phase 2: Setup & Configuration (Completion Target: 15 mins)

### 2.1 Select Agent Framework

- [x] Compare smolagents, pydantic-ai, and npcsh
- [x] Decision: **Selected Framework: pydantic-ai**
- [x] Justification for selection:
  - [x] Reason 1: Strong typed models and structured outputs improve reliability for orchestrator-to-worker handoffs.
  - [x] Reason 2: Clear tool input/output validation fits the database-driven helper function workflow.
  - [x] Reason 3: Better long-term maintainability and explainability for rubric-focused reporting.

### 2.2 Prepare Development Environment

- [x] Verify Python environment is active
- [x] Install selected framework:
  ```bash
  pip install pydantic-ai
  ```
- [x] Verify all dependencies installed
- [x] Confirm implementation source file: project/project_starter.py
- [x] Use project/project_starter.py as the primary executable and submission script

---

## Phase 3: Implementation (Completion Target: 2.5 hours)

### 3.1 Implement Base Agent Structure

- [x] Initialize chosen framework
- [x] Create orchestrator agent:
  - [x] Define orchestrator class/function
  - [x] Implement inquiry handling logic
  - [x] Implement task delegation to worker agents
  - [ ] Add context management for multi-turn conversations
- [ ] Create inventory management agent:
  - [x] Define agent class/function
  - [x] Implement inventory check tool
  - [x] Implement reorder decision logic
  - [x] Add stock level analysis
- [x] Create quote generation agent:
  - [x] Define agent class/function
  - [x] Implement quote history search tool
  - [x] Implement pricing strategy with bulk discounts
  - [x] Add quote generation logic
- [x] Create sales finalization agent:
  - [x] Define agent class/function
  - [x] Implement order fulfillment logic
  - [x] Implement inventory update on sale
  - [x] Add delivery timeline checking

### 3.2 Implement Tool Definitions

- [x] Map all helper functions to agent tools:
  - [x] `create_transaction()` - Implement tool wrapper
  - [x] `get_all_inventory()` - Implement tool wrapper
  - [x] `get_stock_level()` - Implement tool wrapper
  - [x] `get_supplier_delivery_date()` - Implement tool wrapper
  - [x] `get_cash_balance()` - Implement tool wrapper
  - [x] `generate_financial_report()` - Implement tool wrapper
  - [x] `search_quote_history()` - Implement tool wrapper
- [ ] Add tool descriptions and parameters
- [x] Test tool integration with agents
- [x] Verify all helper functions are utilized in at least one tool

### 3.3 Implement System Integration

- [x] Create main orchestration function/class
- [x] Implement request parsing and routing
- [x] Add error handling and validation:
  - [x] Invalid inventory checks
  - [x] Insufficient stock scenarios
  - [ ] Invalid quote requests
  - [x] Order processing errors
- [x] Implement agent communication patterns
- [ ] Add logging/debugging capabilities
- [x] Create response formatting for customer output

### 3.4 Code Quality & Best Practices

- [ ] Review and improve variable naming (snake_case)
- [ ] Review and improve function naming
- [ ] Add docstrings to all functions
- [ ] Add inline comments for complex logic
- [ ] Organize code into logical modules/sections
- [ ] Ensure no sensitive data leakage in outputs
- [ ] Verify customer-facing outputs are clear and justified

---

## Phase 4: Testing & Evaluation (Completion Target: 1.5 hours)

### 4.1 Prepare Test Data

- [x] Load quote_requests_sample.csv
- [x] Review sample requests for variety:
  - [x] Inventory queries
  - [x] Quote requests with different quantities
  - [x] Sales/order requests
  - [x] Edge cases (insufficient stock, etc.)

### 4.2 Run System Tests

- [x] Test with quote_requests_sample.csv
- [x] Generate test_results.csv with outputs
- [x] Document results for each request:
  - [x] Request type (inquiry/quote/order)
  - [x] Input details
  - [x] Agent response
  - [x] System action taken
  - [x] Outcome (success/failure/reason)

### 4.3 Validate Rubric Requirements Met

- [x] Verify at least 3 requests result in cash balance change
- [x] Verify at least 3 quote requests successfully fulfilled
- [x] Verify not all requests are fulfilled (with reasons documented)
- [x] Verify system handles inventory constraints properly
- [x] Verify quote generation logic is sound
- [x] Verify order fulfillment accuracy

### 4.4 Debug & Refine

- [x] Identify any failing test cases
- [x] Debug agent reasoning and decisions
- [x] Fix issues in tool integration
- [x] Verify improved functionality
- [x] Re-test until satisfactory performance

---

## Phase 5: Documentation & Reflection (Completion Target: 1 hour)

### 5.1 Create Reflection Report

- [x] Create or update reflection/analysis document (e.g., system_analysis.md)
- [x] Section 1: Agent Workflow Explanation
  - [x] Explain the workflow diagram in detail
  - [x] Describe each agent's role and responsibilities
  - [x] Explain decision-making process for architecture
  - [x] Justify why max agents chosen (e.g., why exactly 4 agents vs 5)
- [x] Section 2: Evaluation Results Analysis
  - [x] Reference test_results.csv findings
  - [x] Identify system strengths:
    - [x] Strength 1 documented
    - [x] Strength 2 documented
    - [x] Strength 3 documented
  - [x] Identify areas for improvement:
    - [x] Area 1 documented
    - [x] Area 2 documented
- [x] Section 3: Improvement Suggestions
  - [x] Suggestion 1 (detailed):
    - [x] What would be improved
    - [x] How to implement
    - [x] Expected benefit
  - [x] Suggestion 2 (detailed):
    - [x] What would be improved
    - [x] How to implement
    - [x] Expected benefit
  - [x] (Optional) Suggestion 3 included

### 5.2 Verify All Deliverables

- [x] ✅ Workflow Diagram
  - [x] File: project/diagram.png
  - [x] Shows all agents: YES
  - [x] Shows all tools: YES
  - [x] Shows data flow: YES
  - [x] Shows orchestration: YES
- [x] ✅ Implementation Script
  - [x] File: project/project_starter.py
  - [x] Readable variable names: YES
  - [x] Proper docstrings: PARTIAL
  - [x] Comments for complex logic: PARTIAL
  - [x] Modular structure: YES
  - [x] All 7 helper functions used: YES
- [x] ✅ Test Results
  - [x] File: test_results.csv
  - [x] ≥3 requests with cash balance change: YES
  - [x] ≥3 successful quote requests: YES
  - [x] Some unfulfilled requests with reasons: YES
- [x] ✅ Reflection Report
  - [x] File: docs/system_analysis.md
  - [x] Workflow explanation: YES
  - [x] Evaluation discussion: YES
  - [x] ≥2 improvement suggestions: YES

### 5.3 Final Review

- [ ] Run system one final time with test data
- [ ] Verify no errors in output
- [ ] Check all outputs follow rubric guidelines:
  - [ ] Transparent and explainable: YES / NO
  - [ ] No sensitive data exposed: YES / NO
  - [ ] Rationale provided for decisions: YES / NO
- [ ] Review code for any final improvements
- [ ] Ensure all documentation is clear and complete

---

## Optional: Going Above & Beyond (Bonus Features)

### Enhanced Features (if time permits)

- [ ] Implement customer negotiation agent
  - [ ] Analyze customer context from requests
  - [ ] Negotiate pricing/terms
  - [ ] Report negotiation outcomes
- [ ] Create terminal animation
  - [ ] Show real-time agent processing
  - [ ] Visualize data flow between agents
  - [ ] Display customer request journey
- [ ] Implement business advisor agent
  - [ ] Analyze all transactions
  - [ ] Identify trends and opportunities
  - [ ] Recommend operational improvements

---

## Summary of Deliverables

| Deliverable           | File/Location                | Status | Notes                                                                     |
| --------------------- | ---------------------------- | ------ | ------------------------------------------------------------------------- |
| Workflow Diagram      | project/diagram.png          | ✅     | Exported from workflow Mermaid diagram                                    |
| Architecture Options  | docs/architecture_options.md | ✅     | Alternatives separated and documented                                     |
| Implementation Script | project/project_starter.py   | 🟨     | Runnable multi-agent system with model-backed routing, quoting, and sales |
| Test Results          | test_results.csv             | ✅     | Full dataset evaluated; rubric metrics validated                          |
| Reflection Report     | docs/system_analysis.md      | ✅     | Workflow explanation, evaluation discussion, and improvements documented  |
| Plan Document         | docs/plan.md                 | ✅     | This document                                                             |

---

## Progress Checklist

**Phase 1 (Planning & Design):** ✅ 100%  
**Phase 2 (Setup & Configuration):** ✅ 100%  
**Phase 3 (Implementation):** 🟨 ~80% (routing plus inventory, quoting, and sales use pydantic-ai with fallbacks)  
**Phase 4 (Testing & Evaluation):** ✅ 100% (full-dataset run completed and rubric metrics validated)  
**Phase 5 (Documentation & Reflection):** ✅ 100% (system_analysis.md completed)

**Overall Project Completion:** 🟨 ~90%

---

## Notes & Observations

- **Architecture Selected:** Option 1 (Three-Agent Hub-and-Spoke) with optional Business Analyst agent
- **Architecture Artifacts Created:** `docs/architecture.md` and `docs/architecture_options.md`
- **Framework Selected:** pydantic-ai (installed in the active virtual environment)
- **Diagram Export Completed:** Workflow saved as `project/diagram.png` (source in `project/diagram.md` and `project/diagram.mmd`)
- **Implementation File Confirmed:** `project/project_starter.py` is the single source file submission
- **Execution Status:** `project/project_starter.py` runs successfully and generates test output
- **Model-Backed Agents Completed:** Orchestrator routing plus InventoryAgent, QuotingAgent, and SalesAgent decision flows
- **Remaining Major Implementation Gap:** Fine-tune routing stability and reduce occasional inventory underflow edge behavior
- **Key Constraint:** Maximum 5 agents - current design uses 4 agents

---

_Last Updated: 2026-04-23_  
_Next Phase Start Time: Final polish (code quality hardening and optional feature enhancements)_
