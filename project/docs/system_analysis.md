# System Analysis and Reflection

## 1) Workflow Explanation and Architecture Decisions

The implemented system follows a hub-and-spoke multi-agent architecture with one orchestrator and three worker agents.

- Orchestrator
  - Parses the request, routes work, and assembles the final customer response.
  - Uses a primary routing agent plus a routing evaluation agent to reduce misclassification risk.
- Inventory Agent
  - Checks stock levels and evaluates reorder needs.
  - Uses helper tools for stock and delivery timing.
- Quoting Agent
  - Produces quote totals and rationale using historical context and quantity logic.
- Sales Agent
  - Finalizes orders when fulfillable and records transactions.

Why this architecture was chosen:

1. It satisfies the project constraint of at most five agents while keeping responsibilities non-overlapping.
2. It makes orchestration explicit and testable (single entry point for delegation).
3. It maps directly to required business functions: inventory, quoting, and sales finalization.
4. It supports explainable customer responses because each worker returns structured rationale.

The workflow diagram in `project/diagram.mmd` and exported `project/diagram.png` reflects this routing and tool interaction model.

## 2) Evaluation Results (from test_results.csv)

Evaluation was run on the full sample request set and saved in `project/test_results.csv`.

Rubric-linked outcomes observed:

- Full dataset executed: 20/20 requests captured.
- At least three cash balance changes: met.
  - Cash balance increases across multiple requests (for example request IDs 1, 4, 5, 9).
- At least three quote requests successfully fulfilled: met.
  - The response log includes many successful quote outputs beginning with "Quote for ...".
- Not all requests fulfilled: met.
  - Multiple requests are explicitly unfulfilled with shortage and reorder ETA rationale.

Strengths identified:

1. Strong explainability in customer-facing outputs.
   - Quotes include pricing rationale; unfulfilled orders include shortage and estimated delivery date.
2. Reliable business-rule safety via fallbacks.
   - Routing, inventory, quote, and sales decisions all have deterministic fallback behavior.
3. Good helper-tool coverage and integration.
   - Required helper functions are actively used by worker logic and final reporting.

## 3) Improvement Suggestions

### Suggestion A: Add confidence-gated routing overrides

What to improve:

- The routing evaluator currently can override when it decides the initial route is incorrect, but the override policy can still be aggressive in edge cases.

How to implement:

- Extend the `RoutingEvaluation` schema with a `confidence` field (0.0-1.0).
- Apply override only when `confidence >= threshold` (for example 0.75).
- Log pre/post route decisions for auditability.

Expected benefit:

- Fewer unintended route switches and more stable behavior across similar prompts.

### Suggestion B: Make quote fulfillment state explicit in outputs

What to improve:

- While quote text is present, a structured "quote_fulfilled" flag is not written into `test_results.csv`.

How to implement:

- Return structured metadata from `handle_request` (or an internal result object) including:
  - `request_type_detected`
  - `quote_fulfilled` (bool)
  - `order_fulfilled` (bool)
  - `reason`
- Persist these fields as additional columns in `test_results.csv`.

Expected benefit:

- Direct rubric verification becomes mechanical and less ambiguous.

### Suggestion C: Add inventory reservation logic for quote-to-order consistency

What to improve:

- Back-to-back requests can consume stock unexpectedly, and some quote behavior may not reflect reservation windows.

How to implement:

- Add optional short-lived inventory reservation records for quote proposals.
- Expire reservations by date/time and confirm during finalization.

Expected benefit:

- Better consistency between quoted availability and order outcomes.

## 4) Summary

The current implementation satisfies the core evaluation requirements with full-dataset execution, measurable financial impact on multiple requests, successful quote generation, and explicit reasons for unfulfilled orders. The architecture remains modular and explainable, and the suggested improvements focus on precision, auditability, and operational robustness.
