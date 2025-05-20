```markdown
# Multi-Agent Pasta Factory

This project implements a multi-agent system simulating a pasta factory. It uses agents to process orders, manage inventory, and provide lead time estimates. The system includes conflict resolution and synchronization mechanisms to ensure consistent information across agents.

## Key Components

1.  **PastaShape Class:**
    *   Represents a pasta shape with its ingredients.

2.  **OrderProcessorAgent:**
    *   Processes customer orders.
    *   Determines the pasta shape requested.
    *   Checks the shape database to confirm if the requested shape can be made.
    *   Uses an LLM to interpret customer requests.

3.  **InventoryManagerAgent:**
    *   Manages the factory's inventory of ingredients.
    *   Checks ingredient availability based on order requirements.
    *   Updates the inventory after an order is processed (though update functionality is not fully implemented in this version).

4.  **GetCurrentDateTool:**
    *   A tool to retrieve the current date.

5.  **EstimateLeadTimeTool:**
    *   A tool to estimate the delivery lead time for an order.

6.  **PastaFactoryManager (AgentManager):**
    *   The central orchestrator of the system.
    *   Manages communication between the order processor and inventory manager.
    *   Uses a lock (`asyncio.Lock`) to synchronize access to shared resources and prevent race conditions.
    *   Coordinates order fulfillment:
        *   Receives customer orders.
        *   Queries the OrderProcessorAgent to identify the pasta shape.
        *   Queries the InventoryManagerAgent to check ingredient availability.
        *   If ingredients are available, estimates the lead time using the EstimateLeadTimeTool and confirms the order.
        *   Returns an appropriate response to the customer.

## How it Works

1.  **Initialization:**
    *   The `main()` function initializes the LLM (OpenAI), the inventory manager, the order processor, and the factory manager.
    *   Initial inventory is set.

2.  **Order Processing:**
    *   The `main()` function calls the `handle_order()` method of the `PastaFactoryManager`, passing in the customer order.
    *   The `PastaFactoryManager` uses the `asyncio.Lock` to ensure that operations are synchronized.
    *   The `OrderProcessorAgent` determines the requested pasta shape from the order.
    *   The `InventoryManagerAgent` checks for ingredient availability.
    *   If all ingredients are available, the `EstimateLeadTimeTool` is used.
    *   A confirmation message is returned, including the estimated delivery date.
    *   Error messages are returned if the order cannot be fulfilled (e.g., shape not available, missing ingredients).

3.  **Conflict Resolution and Synchronization:**
    *   **Synchronization:** The `asyncio.Lock` within the `PastaFactoryManager` synchronizes access to the shared resources of the agents, preventing potential race conditions. This ensures that the order processing and inventory checks occur in a controlled manner.
    *   **Conflict Resolution:**  The example does not explicitly demonstrate conflict resolution, but the design allows for it. For instance, if the InventoryManagerAgent's inventory data were to be updated by multiple concurrent requests, the `asyncio.Lock` would help prevent data corruption.  The `handle_order` method manages the workflow and can determine if the order can be fulfilled based on the shared data available from the agents.

4.  **Tools:**
    *   The `GetCurrentDateTool` and `EstimateLeadTimeTool` provide additional functionality, such as estimating the delivery date, allowing the system to be more sophisticated.

## Running the Code

1.  **Install Dependencies:**
    ```bash
    pip install smolagents
    pip install openai
    ```

2.  **Set OpenAI API Key:**
    *   Set the `OPENAI_API_KEY` environment variable to your OpenAI API key within the `main()` function.

3.  **Run the Script:**
    ```bash
    python your_script_name.py
    ```
```