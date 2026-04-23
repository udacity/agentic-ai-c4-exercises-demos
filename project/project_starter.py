import pandas as pd
import numpy as np
import os
import re
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from sqlalchemy import create_engine, Engine
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

dotenv.load_dotenv()

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Hard safety check: never allow sales to exceed available stock.
        if transaction_type == "sales":
            stock_snapshot = get_stock_level(item_name, date_str)
            available_units = int(stock_snapshot["current_stock"].iloc[0])
            if quantity > available_units:
                raise ValueError(
                    f"Insufficient stock for sale: requested {quantity}, available {available_units}."
                )

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            MAX(
                COALESCE(SUM(CASE
                    WHEN transaction_type = 'stock_orders' THEN units
                    WHEN transaction_type = 'sales' THEN -units
                    ELSE 0
                END), 0),
                0
            ) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.


class RequestIntent(BaseModel):
    request_type: str = Field(
        description="One of: inventory_inquiry, quote_request, sales_order, unknown"
    )
    item_name: Optional[str] = None
    quantity: Optional[int] = None
    rationale: str = ""


class AgentResult(BaseModel):
    status: str
    message: str
    details: Dict = Field(default_factory=dict)


class SalesDecision(BaseModel):
    status: str = Field(description="Either ok or rejected")
    rationale: str = Field(description="Reason for approving or rejecting the sale")
    fulfillable_units: int = Field(description="How many units can be fulfilled immediately")


class QuoteDecision(BaseModel):
    unit_price: float = Field(description="Quoted unit price after discount")
    discount_rate: float = Field(description="Discount rate as a decimal between 0 and 0.5")
    total_amount: float = Field(description="Final quoted total amount")
    rationale: str = Field(description="Short customer-safe explanation for the quote")


class InventoryDecision(BaseModel):
    status: str = Field(description="Either ok or pending")
    reorder_needed: bool = Field(description="Whether a reorder is required")
    shortage: int = Field(description="Units missing to satisfy the inventory need")
    rationale: str = Field(description="Short explanation of stock position and reorder need")


class RoutingEvaluation(BaseModel):
    final_request_type: str = Field(
        description="One of: inventory_inquiry, quote_request, sales_order, unknown"
    )
    should_override: bool = Field(description="Whether the initial router intent should be overridden")
    rationale: str = Field(description="Why the decision was kept or overridden")


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent
class InventoryAgent:
    """Inventory-focused worker agent with deterministic and model-assisted checks."""

    def __init__(self) -> None:
        self.inventory_agent = self._build_inventory_agent()

    def tool_get_stock_level(self, item_name: str, as_of_date: str) -> Dict[str, int]:
        stock_df = get_stock_level(item_name, as_of_date)
        return {
            "item_name": item_name,
            "current_stock": int(stock_df["current_stock"].iloc[0]),
        }

    def tool_get_all_inventory(self, as_of_date: str) -> Dict[str, int]:
        return get_all_inventory(as_of_date)

    def tool_get_supplier_delivery_date(self, input_date_str: str, quantity: int) -> str:
        return get_supplier_delivery_date(input_date_str, quantity)

    def _build_inventory_agent(self) -> Optional[Agent]:
        """Build and configure the inventory decision agent with tool registrations."""

        api_key = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("UDACITY_OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("UDACITY_OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

        if not api_key:
            return None

        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        model = OpenAIChatModel(model_name, provider=provider)

        inventory_agent = Agent(
            model=model,
            output_type=InventoryDecision,
            system_prompt=(
                "You are an inventory assessment agent for a paper supply company. "
                "Given current stock, minimum stock, and optionally requested quantity, decide whether a reorder is needed. "
                "Return status='pending' when a reorder is needed, otherwise status='ok'. "
                "Set shortage to the number of units missing relative to the requested quantity when one is provided, or 0 when no reorder is needed. "
                "Keep the rationale brief and operationally clear. "
                "You may use available tools when needed to validate stock and delivery information."
            ),
            retries=1,
            output_retries=1,
        )

        @inventory_agent.tool
        def tool_get_stock_level(ctx: RunContext[None], item_name: str, as_of_date: str) -> Dict[str, int]:
            return self.tool_get_stock_level(item_name, as_of_date)

        @inventory_agent.tool
        def tool_get_all_inventory(ctx: RunContext[None], as_of_date: str) -> Dict[str, int]:
            return self.tool_get_all_inventory(as_of_date)

        @inventory_agent.tool
        def tool_get_supplier_delivery_date(ctx: RunContext[None], input_date_str: str, quantity: int) -> str:
            return self.tool_get_supplier_delivery_date(input_date_str, quantity)

        return inventory_agent

    def _get_inventory_context(self, item_name: str, as_of_date: str) -> Dict[str, int]:
        """Collect current and minimum stock context required for inventory decisions."""

        stock_context = self.tool_get_stock_level(item_name, as_of_date)
        current_stock = int(stock_context["current_stock"])

        inventory_df = pd.read_sql(
            "SELECT min_stock_level FROM inventory WHERE item_name = :item_name LIMIT 1",
            db_engine,
            params={"item_name": item_name},
        )
        min_stock_level = int(inventory_df["min_stock_level"].iloc[0]) if not inventory_df.empty else 0

        return {
            "current_stock": current_stock,
            "min_stock_level": min_stock_level,
        }

    def _fallback_inventory_decision(
        self,
        current_stock: int,
        min_stock_level: int,
        needed_qty: Optional[int] = None,
    ) -> InventoryDecision:
        """Produce a deterministic inventory decision when model output is unavailable or invalid."""

        if needed_qty is None:
            if current_stock < min_stock_level:
                return InventoryDecision(
                    status="pending",
                    reorder_needed=True,
                    shortage=max(0, min_stock_level - current_stock),
                    rationale=(
                        f"Current stock is below the minimum stock threshold of {min_stock_level}."
                    ),
                )

            return InventoryDecision(
                status="ok",
                reorder_needed=False,
                shortage=0,
                rationale="Current stock is above the minimum stock threshold.",
            )

        shortage = max(0, needed_qty - current_stock)
        if shortage > 0:
            return InventoryDecision(
                status="pending",
                reorder_needed=True,
                shortage=shortage,
                rationale=(
                    f"Requested quantity exceeds available stock. Requested {needed_qty}, available {current_stock}."
                ),
            )

        return InventoryDecision(
            status="ok",
            reorder_needed=False,
            shortage=0,
            rationale="Sufficient inventory is available for the requested quantity.",
        )

    def _decide_inventory(
        self,
        item_name: str,
        as_of_date: str,
        current_stock: int,
        min_stock_level: int,
        needed_qty: Optional[int] = None,
    ) -> InventoryDecision:
        """Run model-assisted inventory reasoning with validation and safe fallback behavior."""

        if self.inventory_agent is None:
            return self._fallback_inventory_decision(current_stock, min_stock_level, needed_qty)

        prompt = (
            f"Item: {item_name}\n"
            f"As of date: {as_of_date}\n"
            f"Current stock: {current_stock}\n"
            f"Minimum stock level: {min_stock_level}\n"
            f"Requested quantity: {needed_qty if needed_qty is not None else 'not provided'}\n"
            "Assess inventory status and reorder need."
        )

        try:
            result = self.inventory_agent.run_sync(prompt)
            decision = result.output
            if decision.status not in {"ok", "pending"}:
                return self._fallback_inventory_decision(current_stock, min_stock_level, needed_qty)
            if decision.shortage < 0:
                return self._fallback_inventory_decision(current_stock, min_stock_level, needed_qty)
            if needed_qty is not None:
                expected_shortage = max(0, needed_qty - current_stock)
                if decision.reorder_needed != (expected_shortage > 0):
                    return self._fallback_inventory_decision(current_stock, min_stock_level, needed_qty)
                if decision.shortage != expected_shortage:
                    return self._fallback_inventory_decision(current_stock, min_stock_level, needed_qty)
            return decision
        except Exception:
            return self._fallback_inventory_decision(current_stock, min_stock_level, needed_qty)

    def check_inventory(self, item_name: str, as_of_date: str) -> AgentResult:
        """Handle inventory inquiry requests for a single item and date."""

        inventory_context = self._get_inventory_context(item_name, as_of_date)
        current_stock = inventory_context["current_stock"]
        min_stock_level = inventory_context["min_stock_level"]
        decision = self._decide_inventory(
            item_name=item_name,
            as_of_date=as_of_date,
            current_stock=current_stock,
            min_stock_level=min_stock_level,
        )

        message = f"Current stock for {item_name}: {current_stock}"
        if decision.reorder_needed:
            message += f". Reorder recommended: {decision.rationale}"

        return AgentResult(
            status=decision.status,
            message=message,
            details={
                "item_name": item_name,
                "current_stock": current_stock,
                "min_stock_level": min_stock_level,
                "reorder_needed": decision.reorder_needed,
                "shortage": decision.shortage,
                "rationale": decision.rationale,
            },
        )

    def maybe_reorder(self, item_name: str, needed_qty: int, as_of_date: str) -> AgentResult:
        """Assess shortage against requested quantity and return reorder guidance when needed."""

        inventory_context = self._get_inventory_context(item_name, as_of_date)
        current_stock = inventory_context["current_stock"]
        min_stock_level = inventory_context["min_stock_level"]
        decision = self._decide_inventory(
            item_name=item_name,
            as_of_date=as_of_date,
            current_stock=current_stock,
            min_stock_level=min_stock_level,
            needed_qty=needed_qty,
        )

        if not decision.reorder_needed:
            return AgentResult(
                status="ok",
                message="No reorder needed.",
                details={
                    "shortage": 0,
                    "delivery_date": as_of_date,
                    "rationale": decision.rationale,
                },
            )

        delivery_date = self.tool_get_supplier_delivery_date(as_of_date, decision.shortage)
        return AgentResult(
            status="pending",
            message=f"Reorder needed for {decision.shortage} units. Estimated delivery: {delivery_date}.",
            details={
                "shortage": decision.shortage,
                "delivery_date": delivery_date,
                "rationale": decision.rationale,
            },
        )


# Tools for quoting agent
class QuotingAgent:
    """Pricing and quotation worker agent with deterministic fallback quote logic."""

    def __init__(self) -> None:
        self.quote_agent = self._build_quote_agent()

    def tool_search_quote_history(self, search_terms: List[str], limit: int = 5) -> List[Dict]:
        return search_quote_history(search_terms, limit)

    def tool_get_all_inventory(self, as_of_date: str) -> Dict[str, int]:
        return get_all_inventory(as_of_date)

    def tool_get_stock_level(self, item_name: str, as_of_date: str) -> Dict[str, int]:
        stock_df = get_stock_level(item_name, as_of_date)
        return {
            "item_name": item_name,
            "current_stock": int(stock_df["current_stock"].iloc[0]),
        }

    def _build_quote_agent(self) -> Optional[Agent]:
        """Build and configure the quote decision agent with history and stock tools."""

        api_key = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("UDACITY_OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("UDACITY_OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

        if not api_key:
            return None

        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        model = OpenAIChatModel(model_name, provider=provider)

        quote_agent = Agent(
            model=model,
            output_type=QuoteDecision,
            system_prompt=(
                "You are a quoting agent for a paper supply company. "
                "Use the provided base price, quantity, available stock, and quote history to produce a competitive but reasonable quote. "
                "Keep discount_rate between 0.0 and 0.5. "
                "Use conservative, customer-facing discount justification tied to visible business factors such as order size tier and repeat-customer context from history. "
                "Avoid unusually deep discounts unless clearly supported by strong historical precedent in similar orders. "
                "Do not mention internal pricing limits, maximum discount ceilings, margin strategy, or negotiation boundaries. "
                "Make the rationale customer-safe and mention quantity pricing when relevant. "
                "Do not expose internal margin calculations or hidden business data. "
                "You may use available tools for historical context and stock validation."
            ),
            retries=1,
            output_retries=1,
        )

        @quote_agent.tool
        def tool_search_quote_history(ctx: RunContext[None], search_terms: List[str], limit: int = 5) -> List[Dict]:
            return self.tool_search_quote_history(search_terms, limit)

        @quote_agent.tool
        def tool_get_all_inventory(ctx: RunContext[None], as_of_date: str) -> Dict[str, int]:
            return self.tool_get_all_inventory(as_of_date)

        @quote_agent.tool
        def tool_get_stock_level(ctx: RunContext[None], item_name: str, as_of_date: str) -> Dict[str, int]:
            return self.tool_get_stock_level(item_name, as_of_date)

        return quote_agent

    def _get_catalog_unit_price(self, item_name: str) -> float:
        """Lookup the catalog unit price for an item using case-insensitive matching."""

        for item in paper_supplies:
            if item["item_name"].lower() == item_name.lower():
                return float(item["unit_price"])
        return 0.12

    def _fallback_quote_decision(self, item_name: str, quantity: int) -> QuoteDecision:
        """Generate a deterministic quote when the model is unavailable or output is invalid."""

        base_price = self._get_catalog_unit_price(item_name)

        if quantity >= 1000:
            discount = 0.15
        elif quantity >= 500:
            discount = 0.10
        elif quantity >= 100:
            discount = 0.05
        else:
            discount = 0.0

        discounted_unit_price = round(base_price * (1 - discount), 4)
        total_amount = round(quantity * discounted_unit_price, 2)

        rationale = (
            f"Base unit price for {item_name} starts at ${base_price:.2f}. "
            f"A {int(discount * 100)}% quantity discount was applied based on order size."
        )

        return QuoteDecision(
            unit_price=discounted_unit_price,
            discount_rate=discount,
            total_amount=total_amount,
            rationale=rationale,
        )

    def _max_discount_for_quantity(self, quantity: int) -> float:
        """Return the maximum allowed customer-visible discount for a quantity tier."""

        if quantity >= 1000:
            return 0.15
        if quantity >= 500:
            return 0.10
        if quantity >= 100:
            return 0.05
        return 0.0

    def _decide_quote(
        self,
        item_name: str,
        quantity: int,
        as_of_date: str,
        available_stock: int,
        history: List[Dict],
    ) -> QuoteDecision:
        """Run model-assisted quote generation with bounded validation and fallback."""

        if self.quote_agent is None:
            return self._fallback_quote_decision(item_name, quantity)

        base_price = self._get_catalog_unit_price(item_name)
        history_lines = []
        for idx, record in enumerate(history[:5], start=1):
            history_lines.append(
                f"{idx}. total_amount={record.get('total_amount')}, "
                f"job_type={record.get('job_type')}, order_size={record.get('order_size')}, "
                f"event_type={record.get('event_type')}"
            )
        history_summary = "\n".join(history_lines) if history_lines else "No relevant quote history found."

        prompt = (
            f"Item: {item_name}\n"
            f"Quantity: {quantity}\n"
            f"Available stock: {available_stock}\n"
            f"Base catalog unit price: {base_price:.2f}\n"
            f"As of date: {as_of_date}\n"
            f"Historical quote summary:\n{history_summary}\n"
            "Produce a quote decision."
        )

        try:
            result = self.quote_agent.run_sync(prompt)
            decision = result.output

            max_allowed_discount = self._max_discount_for_quantity(quantity)
            if not (0.0 <= decision.discount_rate <= max_allowed_discount):
                return self._fallback_quote_decision(item_name, quantity)
            if decision.unit_price <= 0 or decision.total_amount <= 0:
                return self._fallback_quote_decision(item_name, quantity)

            # Quote math must remain internally consistent for customer-facing responses.
            expected_total = round(quantity * decision.unit_price, 2)
            if abs(expected_total - round(decision.total_amount, 2)) > 0.01:
                return self._fallback_quote_decision(item_name, quantity)

            # Keep unit pricing in a plausible range relative to base catalog price.
            min_unit_price = round(base_price * (1 - max_allowed_discount), 4)
            if decision.unit_price < min_unit_price or decision.unit_price > round(base_price, 4):
                return self._fallback_quote_decision(item_name, quantity)

            return decision
        except Exception:
            return self._fallback_quote_decision(item_name, quantity)

    def generate_quote(self, item_name: str, quantity: int, as_of_date: str) -> AgentResult:
        """Create a customer-facing quote response with supporting context details."""

        history = self.tool_search_quote_history([item_name], limit=5)
        inventory_snapshot = self.tool_get_all_inventory(as_of_date)
        available_stock = int(inventory_snapshot.get(item_name, 0))
        decision = self._decide_quote(item_name, quantity, as_of_date, available_stock, history)

        return AgentResult(
            status="ok",
            message=f"Quote for {quantity} x {item_name}: ${decision.total_amount:.2f}",
            details={
                "item_name": item_name,
                "quantity": quantity,
                "total_amount": round(decision.total_amount, 2),
                "unit_price": round(decision.unit_price, 4),
                "discount_rate": decision.discount_rate,
                "available_stock": available_stock,
                "quote_explanation": decision.rationale,
                "history_matches": len(history),
            },
        )


# Tools for ordering agent
class SalesAgent:
    """Sales fulfillment worker agent with strict anti-oversell safety checks."""

    def __init__(self) -> None:
        self.decision_agent = self._build_sales_agent()

    def tool_get_stock_level(self, item_name: str, as_of_date: str) -> Dict[str, int]:
        stock_df = get_stock_level(item_name, as_of_date)
        return {
            "item_name": item_name,
            "current_stock": int(stock_df["current_stock"].iloc[0]),
        }

    def tool_create_transaction(
        self,
        item_name: str,
        transaction_type: str,
        quantity: int,
        price: float,
        date: str,
    ) -> int:
        return create_transaction(item_name, transaction_type, quantity, price, date)

    def tool_get_cash_balance(self, as_of_date: str) -> float:
        return get_cash_balance(as_of_date)

    def tool_generate_financial_report(self, as_of_date: str) -> Dict:
        return generate_financial_report(as_of_date)

    def _build_sales_agent(self) -> Optional[Agent]:
        """Build and configure the sales decision agent with transactional tools."""

        api_key = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("UDACITY_OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("UDACITY_OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

        if not api_key:
            return None

        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        model = OpenAIChatModel(model_name, provider=provider)

        sales_agent = Agent(
            model=model,
            output_type=SalesDecision,
            system_prompt=(
                "You are a sales finalization agent for a paper supply company. "
                "Given the requested quantity, available stock, quoted total price, and request date, decide whether the order should be fulfilled immediately. "
                "Return status='ok' only when the full requested quantity can be fulfilled now. "
                "Return status='rejected' if stock is insufficient. "
                "Set fulfillable_units to the number of units that can be fulfilled immediately without overselling. "
                "Do not approve partial fulfillment as ok. "
                "You may use available tools to verify stock and financial context."
            ),
            retries=1,
            output_retries=1,
        )

        @sales_agent.tool
        def tool_get_stock_level(ctx: RunContext[None], item_name: str, as_of_date: str) -> Dict[str, int]:
            return self.tool_get_stock_level(item_name, as_of_date)

        @sales_agent.tool
        def tool_create_transaction(
            ctx: RunContext[None],
            item_name: str,
            transaction_type: str,
            quantity: int,
            price: float,
            date: str,
        ) -> int:
            return self.tool_create_transaction(item_name, transaction_type, quantity, price, date)

        @sales_agent.tool
        def tool_get_cash_balance(ctx: RunContext[None], as_of_date: str) -> float:
            return self.tool_get_cash_balance(as_of_date)

        @sales_agent.tool
        def tool_generate_financial_report(ctx: RunContext[None], as_of_date: str) -> Dict:
            return self.tool_generate_financial_report(as_of_date)

        return sales_agent

    def _fallback_sales_decision(self, quantity: int, current_stock: int) -> SalesDecision:
        """Return deterministic sales approval/rejection based on current stock only."""

        if current_stock < quantity:
            return SalesDecision(
                status="rejected",
                rationale=(
                    f"Insufficient inventory for full fulfillment. Requested {quantity}, available {current_stock}."
                ),
                fulfillable_units=current_stock,
            )

        return SalesDecision(
            status="ok",
            rationale="Sufficient inventory is available to fulfill the full order immediately.",
            fulfillable_units=quantity,
        )

    def _decide_sale(
        self,
        item_name: str,
        quantity: int,
        total_price: float,
        current_stock: int,
        as_of_date: str,
    ) -> SalesDecision:
        """Run model-assisted fulfillment decisioning with strict output validation."""

        if self.decision_agent is None:
            return self._fallback_sales_decision(quantity, current_stock)

        prompt = (
            f"Item: {item_name}\n"
            f"Requested quantity: {quantity}\n"
            f"Available stock: {current_stock}\n"
            f"Quoted total price: {total_price:.2f}\n"
            f"Request date: {as_of_date}\n"
            "Decide whether to fulfill now or reject due to insufficient stock."
        )

        try:
            result = self.decision_agent.run_sync(prompt)
            decision = result.output
            if decision.status not in {"ok", "rejected"}:
                return self._fallback_sales_decision(quantity, current_stock)
            if decision.fulfillable_units < 0:
                return self._fallback_sales_decision(quantity, current_stock)
            if decision.fulfillable_units > current_stock:
                return self._fallback_sales_decision(quantity, current_stock)
            if decision.status == "ok" and decision.fulfillable_units < quantity:
                return self._fallback_sales_decision(quantity, current_stock)
            if decision.status == "ok" and current_stock < quantity:
                return self._fallback_sales_decision(quantity, current_stock)
            return decision
        except Exception:
            return self._fallback_sales_decision(quantity, current_stock)

    def finalize_sale(self, item_name: str, quantity: int, total_price: float, as_of_date: str) -> AgentResult:
        """Finalize sale transactions safely and reject any request that risks overselling."""

        stock_context = self.tool_get_stock_level(item_name, as_of_date)
        current_stock = int(stock_context["current_stock"])

        # Hard safety invariant: never allow overselling regardless of model output.
        if current_stock < quantity:
            return AgentResult(
                status="rejected",
                message=(
                    f"Order cannot be fulfilled: requested {quantity}, "
                    f"available {current_stock}."
                ),
                details={
                    "item_name": item_name,
                    "requested": quantity,
                    "available": current_stock,
                    "rationale": "Deterministic stock guard: insufficient inventory for full fulfillment.",
                    "fulfillable_units": max(current_stock, 0),
                },
            )

        decision = self._decide_sale(item_name, quantity, total_price, current_stock, as_of_date)

        if decision.status == "rejected":
            return AgentResult(
                status="rejected",
                message=(
                    f"Order cannot be fulfilled: requested {quantity}, "
                    f"available {current_stock}."
                ),
                details={
                    "item_name": item_name,
                    "requested": quantity,
                    "available": current_stock,
                    "rationale": decision.rationale,
                    "fulfillable_units": decision.fulfillable_units,
                },
            )

        try:
            transaction_id = self.tool_create_transaction(
                item_name=item_name,
                transaction_type="sales",
                quantity=quantity,
                price=total_price,
                date=as_of_date,
            )
        except Exception:
            return AgentResult(
                status="rejected",
                message=(
                    f"Order cannot be fulfilled: requested {quantity}, "
                    f"available {current_stock}."
                ),
                details={
                    "item_name": item_name,
                    "requested": quantity,
                    "available": current_stock,
                    "rationale": "Transaction safety guard blocked oversell.",
                    "fulfillable_units": max(current_stock, 0),
                },
            )
        cash_after = self.tool_get_cash_balance(as_of_date)

        return AgentResult(
            status="ok",
            message="Sale completed.",
            details={
                "transaction_id": transaction_id,
                "cash_balance": cash_after,
                "item_name": item_name,
                "quantity": quantity,
                "total_price": total_price,
                "rationale": decision.rationale,
            },
        )


# Set up your agents and create an orchestration agent that will manage them.
class Orchestrator:
    """Top-level coordinator that routes requests to inventory, quote, and sales workers."""

    def __init__(self) -> None:
        self.inventory_agent = InventoryAgent()
        self.quoting_agent = QuotingAgent()
        self.sales_agent = SalesAgent()
        self.router_agent = self._build_router_agent()
        self.routing_evaluator = self._build_routing_evaluator()

    def _build_router_agent(self) -> Optional[Agent]:
        """Build the primary intent router agent used for initial request classification."""

        api_key = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("UDACITY_OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("UDACITY_OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

        if not api_key:
            return None

        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        model = OpenAIChatModel(model_name, provider=provider)

        return Agent(
            model=model,
            output_type=RequestIntent,
            system_prompt=(
                "You are a routing agent for a paper supply company. "
                "Classify each incoming request as one of: inventory_inquiry, quote_request, sales_order, or unknown. "
                "Apply intent rules with these priorities: "
                "1) inventory_inquiry for stock/availability/lead-time questions when no explicit purchase confirmation is present; "
                "2) quote_request for quote/price/cost/budget questions, including requests that mention quantity; "
                "3) sales_order when user intent is to transact now (place order, confirm order, proceed with purchase, book it, finalize order). "
                "If a request contains both order language and pricing/availability questions, prefer quote_request or inventory_inquiry unless the user explicitly confirms to proceed now. "
                "Never classify as sales_order based only on quantity, product name, or urgent tone. "
                "Extract requested quantity if present. "
                "Extract the most likely item_name using exact catalog wording when clear. "
                "Do not invent unsupported details."
            ),
            retries=1,
            output_retries=1,
        )

    def _build_routing_evaluator(self) -> Optional[Agent]:
        """Build the routing evaluator used to review and override weak classifications."""

        api_key = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("UDACITY_OPENAI_MODEL", "gpt-4o-mini")
        base_url = os.getenv("UDACITY_OPENAI_BASE_URL", "https://openai.vocareum.com/v1")

        if not api_key:
            return None

        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        model = OpenAIChatModel(model_name, provider=provider)

        return Agent(
            model=model,
            output_type=RoutingEvaluation,
            system_prompt=(
                "You are a routing quality evaluator for a paper supply assistant. "
                "Review the incoming request text and an initial request_type classification. "
                "Return final_request_type as one of inventory_inquiry, quote_request, sales_order, or unknown. "
                "Override to sales_order only when there is clear transactional intent to buy now (for example: place order, confirm order, proceed with purchase, finalize order). "
                "If the request is exploratory, asks about price/cost/quote, or asks about stock/availability, prefer quote_request or inventory_inquiry. "
                "Set should_override=true only when the initial classification is clearly incorrect."
            ),
            retries=1,
            output_retries=1,
        )

    def _fallback_routing_guardrails(self, request_text: str, intent: RequestIntent) -> RequestIntent:
        """Apply deterministic keyword-based routing corrections when needed."""

        text = request_text.lower()

        quote_tokens = ["quote", "quotation", "pricing", "price", "cost", "estimate", "budget", "how much", "rate"]
        inventory_tokens = [
            "inventory",
            "stock",
            "available",
            "availability",
            "on hand",
            "lead time",
            "in stock",
            "delivery",
        ]
        strong_sales_tokens = [
            "place order",
            "book it",
            "confirm order",
            "confirm purchase",
            "proceed with purchase",
            "proceed with the order",
            "buy now",
            "finalize order",
            "go ahead with order",
            "i want to order now",
        ]
        weak_sales_tokens = ["order", "purchase", "buy"]
        question_tokens = ["can you", "could you", "do you", "is it", "what", "how", "?"]

        has_quote_signal = any(token in text for token in quote_tokens)
        has_inventory_signal = any(token in text for token in inventory_tokens)
        has_strong_sales_signal = any(token in text for token in strong_sales_tokens)
        has_weak_sales_signal = any(token in text for token in weak_sales_tokens)
        has_question_signal = any(token in text for token in question_tokens)

        if intent.request_type == "sales_order":
            if has_quote_signal and not has_strong_sales_signal:
                intent.request_type = "quote_request"
                intent.rationale = "Routing guardrail: pricing language present without explicit purchase confirmation."
            elif has_inventory_signal and not has_strong_sales_signal:
                intent.request_type = "inventory_inquiry"
                intent.rationale = "Routing guardrail: inventory language detected without explicit purchase confirmation."
            elif has_question_signal and has_weak_sales_signal and not has_strong_sales_signal:
                intent.request_type = "quote_request"
                intent.rationale = "Routing guardrail: order wording appears exploratory; routing to quote first."

        elif intent.request_type in {"unknown", "quote_request", "inventory_inquiry"}:
            if has_strong_sales_signal and not has_quote_signal and not has_inventory_signal:
                intent.request_type = "sales_order"
                intent.rationale = "Routing guardrail: explicit purchase confirmation detected."
            elif intent.request_type == "unknown":
                if has_quote_signal:
                    intent.request_type = "quote_request"
                    intent.rationale = "Routing guardrail: quote-related language detected."
                elif has_inventory_signal:
                    intent.request_type = "inventory_inquiry"
                    intent.rationale = "Routing guardrail: inventory-related language detected."
                elif has_weak_sales_signal and not has_question_signal:
                    intent.request_type = "sales_order"
                    intent.rationale = "Routing guardrail: transactional wording detected."

        return intent

    def _apply_routing_guardrails(self, request_text: str, intent: RequestIntent) -> RequestIntent:
        """Apply heuristic guardrails and optional evaluator-based routing overrides."""

        fallback_intent = self._fallback_routing_guardrails(request_text, intent)

        if self.routing_evaluator is None:
            return fallback_intent

        prompt = (
            f"Request text: {request_text}\n"
            f"Initial request_type: {intent.request_type}\n"
            f"Initial rationale: {intent.rationale or 'none'}\n"
            "Evaluate whether the initial classification should be kept or overridden."
        )

        try:
            result = self.routing_evaluator.run_sync(prompt)
            evaluation = result.output

            valid_types = {"inventory_inquiry", "quote_request", "sales_order", "unknown"}
            if evaluation.final_request_type not in valid_types:
                return fallback_intent

            if evaluation.should_override:
                fallback_intent.request_type = evaluation.final_request_type

            if evaluation.rationale:
                fallback_intent.rationale = f"Routing evaluator: {evaluation.rationale}"

            return fallback_intent
        except Exception:
            return fallback_intent

    def _fallback_parse_request(self, request_text: str) -> RequestIntent:
        """Parse request intent using lightweight regex heuristics when router is unavailable."""

        text = request_text.lower()

        quantity_match = re.search(r"(\d+)", text)
        quantity = int(quantity_match.group(1)) if quantity_match else None

        request_type = "unknown"
        if any(token in text for token in ["inventory", "stock", "available"]):
            request_type = "inventory_inquiry"
        elif any(token in text for token in ["quote", "price", "cost"]):
            request_type = "quote_request"
        elif any(token in text for token in ["order", "buy", "purchase"]):
            request_type = "sales_order"

        return RequestIntent(
            request_type=request_type,
            quantity=quantity,
            rationale="Fallback regex router used because pydantic-ai routing was unavailable.",
        )

    def parse_request(self, request_text: str) -> RequestIntent:
        """Parse and normalize request intent using router output plus fallback safeguards."""

        if self.router_agent is None:
            return self._fallback_parse_request(request_text)

        try:
            result = self.router_agent.run_sync(request_text)
            intent = result.output
            if intent.quantity is None:
                fallback_intent = self._fallback_parse_request(request_text)
                intent.quantity = fallback_intent.quantity
            intent = self._apply_routing_guardrails(request_text, intent)
            if not intent.rationale:
                intent.rationale = "pydantic-ai router classification"
            return intent
        except Exception:
            return self._fallback_parse_request(request_text)

    def _normalize_item_name(self, item_name: Optional[str]) -> Optional[str]:
        """Map free-form item mentions to canonical catalog names when possible."""

        if not item_name:
            return None

        supply_lookup = {supply["item_name"].lower(): supply["item_name"] for supply in paper_supplies}
        normalized = item_name.strip().lower()

        if normalized in supply_lookup:
            return supply_lookup[normalized]

        for candidate, actual_name in supply_lookup.items():
            if normalized in candidate or candidate in normalized:
                return actual_name

        return None

    def _extract_item_name(self, request_text: str) -> str:
        """Extract a likely catalog item from request text using simple keyword heuristics."""

        text = request_text.lower()
        # Narrow fallback heuristic: this is not a full catalog search and only handles a few common keywords.
        if "cardstock" in text:
            return "Cardstock"
        if "glossy" in text:
            return "Glossy paper"
        if "banner" in text:
            return "Banner paper"
        if "letter" in text:
            return "Letter-sized paper"
        return "A4 paper"

    def handle_request(self, request_text: str, request_date: str) -> str:
        """Route and fulfill a user request end-to-end through the worker agents."""

        intent = self.parse_request(request_text)
        item_name = self._normalize_item_name(intent.item_name) or self._extract_item_name(request_text)
        quantity = intent.quantity or 100

        if intent.request_type == "inventory_inquiry":
            result = self.inventory_agent.check_inventory(item_name, request_date)
            return result.message

        if intent.request_type == "quote_request":
            quote = self.quoting_agent.generate_quote(item_name, quantity, request_date)
            return f"{quote.message} {quote.details.get('quote_explanation', '')}"

        if intent.request_type == "sales_order":
            quote = self.quoting_agent.generate_quote(item_name, quantity, request_date)
            total_price = float(quote.details.get("total_amount", 0.0))
            result = self.sales_agent.finalize_sale(item_name, quantity, total_price, request_date)
            if result.status == "rejected":
                reorder = self.inventory_agent.maybe_reorder(item_name, quantity, request_date)
                return (
                    f"{quote.message} {quote.details.get('quote_explanation', '')} "
                    f"{result.message} {reorder.message}"
                )
            return f"{quote.message} {quote.details.get('quote_explanation', '')} {result.message}"

        return "I could not classify the request. Please ask about inventory, quotes, or orders."


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    orchestrator = Orchestrator()

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        response = orchestrator.handle_request(request_with_date, request_date)

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
