import json
from huggingface_hub import Agent
import pandas as pd
import numpy as np
import os
import time
import ast
import re

from smolagents import OpenAIServerModel, tool, ToolCallingAgent
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create an SQLite database
db_engine = create_engine(f"sqlite:///{os.path.join(SCRIPT_DIR, 'munder_difflin.db')}")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
    api_base="https://openai.vocareum.com/v1",
)

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",               "category": "paper",        "unit_price": 0.06},
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
        quote_requests_df = pd.read_csv(os.path.join(SCRIPT_DIR, "quote_requests.csv"))
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv(os.path.join(SCRIPT_DIR, "quotes.csv"))
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
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
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


"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent
@tool
def check_inventory_levels(as_of_date: str) -> Dict[str, int]:
    """
    Check inventory levels as of a specific date and return items that are below their minimum stock level.

    This function compares the current stock of each item against its defined minimum stock level
    and identifies which items need to be reordered.

    Args:
        as_of_date (str): The date (inclusive) for checking inventory levels, in ISO format (YYYY-MM-DD).

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels for items that are below minimum.
    """
    return get_all_inventory(as_of_date)

@tool
def check_delivery_date(order_date: str, quantity: int) -> str:
    """
    Tool function to check the estimated delivery date for a given order quantity and input date.

    This function serves as a wrapper around the `get_supplier_delivery_date` utility, allowing it to be called
    as a tool within an agentic system.

    Args:
        order_date (str): The starting date for the order in ISO format (YYYY-MM-DD).
        quantity (int): The number of units being ordered.

    Returns:
        str: The estimated delivery date in ISO format (YYYY-MM-DD).
    """
    return get_supplier_delivery_date(order_date, quantity)

@tool
def has_sufficient_stock(item_name: str, required_quantity: int, as_of_date: str) -> bool:
    """
    Tool function to check if there is sufficient stock of a specific item to fulfill a request.

    This function checks the current stock level of the specified item as of the given date and compares it
    against the required quantity to determine if the request can be fulfilled.

    Args:
        item_name (str): The name of the item to check.
        required_quantity (int): The quantity needed for the request.
        as_of_date (str): The date (inclusive) for checking stock levels, in ISO format (YYYY-MM-DD).

    Returns:
        bool: True if there is sufficient stock to fulfill the request, False otherwise.
    """
    stock_info = get_stock_level(item_name, as_of_date)
    current_stock = stock_info["current_stock"].iloc[0] if not stock_info.empty else 0
    return current_stock >= required_quantity

# Tools for quoting agent
@tool
def check_quote_history(search_terms: List[str]) -> List[Dict]:
    """
    Tool function to search historical quotes based on provided search terms.

    This function allows an agent to query past quotes that match specific keywords in the original customer request
    or the quote explanation. It serves as a wrapper around the `search_quote_history` utility function.

    Args:
        search_terms (List[str]): A list of keywords to search for in past quotes.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with relevant fields.
    """
    results = search_quote_history(search_terms)
    return results[0] if results else None

# Tools for ordering agent
@tool
def get_current_cash_balance(as_of_date: str) -> float:
    """
    Tool function to retrieve the current cash balance as of a specific date.

    This function serves as a wrapper around the `get_cash_balance` utility, allowing it to be called
    as a tool within an agentic system.

    Args:
        as_of_date (str): The date (inclusive) for which to retrieve the cash balance, in ISO format (YYYY-MM-DD).

    Returns:
        float: The current cash balance as of the given date.
    """
    return get_cash_balance(as_of_date)

@tool
def place_sales_order(item_name: str, quantity: int, price: float, transaction_date: str) -> int:
    """
    Tool function to record a sales transaction by creating a transaction in the database.
    
    This function serves as a wrapper around the `create_transaction` utility, allowing it to be called
    as a tool within an agentic system.

    Args:
        item_name (str): The name of the item being sold.
        quantity (int): The number of units being sold.
        price (float): The total price for the sales transaction.
        transaction_date (str): The date of the transaction in ISO format (YYYY-MM-DD).

    Returns:
        int: The ID of the newly created sales transaction.
    """
    return create_transaction(item_name, "sales", quantity, price, transaction_date)

@tool
def generate_financial_report_tool(as_of_date: str) -> Dict:
    """
    Tool function to generate a financial report as of a specific date.

    This function serves as a wrapper around the `generate_financial_report` utility, allowing it to be called
    as a tool within an agentic system.

    Args:
        as_of_date (str): The date (inclusive) for which to generate the report, in ISO format (YYYY-MM-DD).

    Returns:
        Dict: A dictionary containing the financial report fields such as cash balance, inventory value, etc.
    """
    return generate_financial_report(as_of_date)


# Set up your agents and create an orchestration agent that will manage them.
class InventoryAgent(ToolCallingAgent):
    def __init__(self, model) -> None:
        super().__init__(
            model=model, 
            tools=[check_inventory_levels, check_delivery_date, has_sufficient_stock],
            name="InventoryAgent",
            description="Agent responsible for managing inventory levels, checking stock availability, and estimating delivery dates."
        )

class QuotingAgent(ToolCallingAgent):
    def __init__(self, model, inventory_agent: InventoryAgent) -> None:
        super().__init__(
            model=model, 
            tools=[check_quote_history],
            name="QuotingAgent",
            description="Agent responsible for generating quotes based on customer requests and historical quote data.",
            managed_agents=[inventory_agent]
        )
    
    def create_quote(self, order_details: Dict) -> Dict:
        """
        Creates a quote based on the provided order details. First checks historical quotes for similar requests, then if none is found, 
        it generates a new quote if necessary.
        """
        response = self.run(
            f"""You are a quoting agent for Munder Difflin. Your task is to generate a quote for a customer request based on the order details provided.
            
            Given the following order details: {json.dumps(order_details)}, accomplish the following steps:

            Step 1. Search the quote history for any similar requests using the check_quote_history tool.
            Use the check_quote_history tool one time by extracting relevant keywords from the order details, such as job type, event type, and order size.
            If the result is valid, it will be a dictionary containing the original request, total amount, quote explanation, job type, order size, event type, and order date. If no similar quote is found, the result will be None.
            Note the pricing if the result is valid, as you may use it to inform your quote generation in the next step, but do not return it yet.
            Always use the check_quote_history tool first before attempting to generate a new quote, as historical quotes can provide valuable insights and help ensure consistency in pricing.
            Always proceed to step 2 to verify stock availability and delivery timelines, even if a similar quote is found in the history, as inventory levels and supplier lead times can impact the feasibility of fulfilling the request.

            Step 2. Check inventory levels and delivery timelines using the InventoryAgent, which has access to tools for checking current stock levels, minimum stock thresholds, and supplier delivery estimates.
            Ask the inventory agent to check if there is sufficient stock to fulfill the request based on the order details, and to estimate the delivery date for any items that need to be ordered from the supplier.
            CRITICAL: All delivery dates MUST be calculated from the request date provided in the order details, not from today's date or any other reference date.
            Exact task instructions for the inventory agent:
            - Extract the relevant item names and quantities from the order details.
            - For each item, use the has_sufficient_stock tool to check if there is enough stock to fulfill the required quantity as of the request date.
            - If any item does not have sufficient stock, use the check_delivery_date tool to estimate the delivery date for the required quantity of that item. CRITICAL: Pass the request date from order_details as the input to check_delivery_date, NOT today's date or any other date.

            Step 3. Based on the information gathered in steps 1 and 2, generate a quote for the customer request.
            CRITICAL LOGIC: Each item in the order must fall into EXACTLY ONE of these mutually exclusive categories:
            - FULFILLED: Item is in stock. Set "is_in_stock": true, "fulfilled": true, "delivery_date": today or request_date
            - OUT_OF_STOCK_WITH_DELIVERY: Item is out of stock but can be ordered. Set "is_in_stock": false, "fulfilled": false, "delivery_date": calculated supplier delivery date (MUST be in the future from request date)
            - CANNOT_FULFILL: Item cannot be fulfilled at all. Set "is_in_stock": false, "fulfilled": false, "delivery_date": null, "reason": explanation
            
            An item CANNOT be in both "fulfilled" and "out of stock" states simultaneously. This is a logical error that must be avoided.

            - If a similar quote was found in the history, use it as a reference for pricing, but adjust the quote as necessary based on current inventory availability and delivery timelines.
            - If no similar quote was found, generate a new quote based on the order details, inventory availability, and delivery timelines.
            - For each item in the order, use the stock availability and delivery information to determine which category it falls into (FULFILLED, OUT_OF_STOCK_WITH_DELIVERY, or CANNOT_FULFILL).
            - Ensure that the total amount for the quote is calculated correctly based on the included items and their prices (total price for each item = unit price * quantity, and total quote amount = sum of total prices for included items).
            - Provide a clear explanation for the quote, including any assumptions made, adjustments based on inventory or delivery constraints, and references to similar historical quotes if applicable.

            Return the generated quote as a JSON dictionary containing:
            - the request date
            - the needed by date
            - a list of items with their fulfillment status:
              * item_name
              * quantity
              * unit_price
              * total_price
              * is_in_stock: boolean
              * fulfilled: boolean (true ONLY if item is in stock and will be sent immediately)
              * delivery_date: ISO date (YYYY-MM-DD) or null if cannot fulfill. MUST be in future if out of stock.
              * reason: explanation if item cannot be fulfilled
            - the total amount (sum of prices for items that can be fulfilled)
            - the quote explanation

            The returned object should be a JSON dictionary and nothing else. Do not include any text outside of the JSON dictionary in your response.
            
            """
        )
        return response
    
class OrderingAgent(ToolCallingAgent): 
    def __init__(self, model) -> None:
        super().__init__(
            model=model, 
            tools=[get_current_cash_balance, place_sales_order, generate_financial_report_tool],
            name="OrderingAgent",
            description="Agent responsible for extracting order details and placing orders based on generated quotes and inventory status."
        )

    def extract_pertinent_order_details(self, request_details: Dict) -> Dict[str, int]:
        response = self.run(
            f"""You are an ordering agent for Munder Difflin. Your task is to extract the relevant order details from a customer request in order to inform the quoting and ordering process.

            Given the following customer request details: {json.dumps(request_details)}, accomplish the following steps:

            Extract the pertinent order details from the request, including:
            1. request date
            2. needed by date
            3. a list of items with their names and quantities needed for the order, as well as the unit price for each item if available in the request details. 
                This list should be structured as a list of dictionaries, where each dictionary contains the item name, quantity, and unit price
                The item_name should check the {paper_supplies} variable for an exact match to ensure that the item is recognized and can be processed correctly. If an item in the request does not have an exact match in the paper_supplies list, it should be flagged as an unrecognized item and handled accordingly in the quoting and ordering process.
                the quantity should be an integer representing the number of units needed for that item, and the unit price should be a float representing the price per unit for that item from the {paper_supplies} list if it is provided in the request details. If the unit price is not provided, it can be set to None or handled as needed in the quoting process.

            Sample
            {{
                "request_date": "2025-04-02",
                "needed_date": "2025-05-15",
                "items": [
                    {{
                        "item_name": "Glossy paper",
                        "quantity": 200,
                        "unit_price": 0.2
                    }},
                    {{
                        "item_name": "Cardstock",
                        "quantity": 100,
                        "unit_price": 0.15
                    }}
                ]
            }}

            The returned object should be a JSON dictionary and nothing else. Do not include any text outside of the JSON dictionary in your response.
            """
        )

        return response


    def place_order(self, quote: Dict) -> Dict[str, int]:
        """
        Places an order based on the provided quote. First checks the current cash balance to ensure sufficient funds, then records the sales transaction if the order can be fulfilled.
        
        CRITICAL: Only charges for items marked as "fulfilled": true. Items with "fulfilled": false 
        (out of stock, pending supplier delivery) are NOT charged at this time.
        """
        
        response = self.run(
            f"""You are an ordering agent for Munder Difflin. Your task is to place an order based on a generated quote and manage the financial transactions accordingly.

            CRITICAL FULFILLMENT & CHARGING RULES:
            ==========================================
            Rule 1 - ONLY CHARGE FOR FULFILLED ITEMS:
            - Items with "fulfilled": true = IN STOCK, READY TO SHIP, CHARGE CUSTOMER NOW
            - Items with "fulfilled": false = OUT OF STOCK, PENDING SUPPLIER DELIVERY, DO NOT CHARGE NOW
            
            Rule 2 - CALCULATE CORRECT TOTAL:
            - Total amount to charge = SUM of (unit_price * quantity) for ONLY items where "fulfilled": true
            - Do NOT include prices for items where "fulfilled": false in the charge total
            
            Rule 3 - VERIFY FULFILLMENT STATUS:
            - Examine each item in the quote's items list
            - Look at the "fulfilled" field for each item (true or false)
            - Items with "fulfilled": false should be listed as "out of stock" or "pending" in your response
            - Items with "fulfilled": true should be listed as "charged" or "successfully ordered" in your response

            Given the following quote details: {json.dumps(quote)}, accomplish the following steps:

            Step 1. ANALYZE FULFILLMENT STATUS:
            - Extract all items from the quote.
            - Separate items into two categories:
              a) FULFILLED ITEMS: Items where "fulfilled": true
              b) NON-FULFILLED ITEMS: Items where "fulfilled": false
            - Calculate the total amount to charge based ONLY on fulfilled items.

            Step 2. VERIFY CASH AVAILABILITY FOR FULFILLED ITEMS:
            - Use get_current_cash_balance tool to check the current cash balance.
            - Compare the current cash balance to the total amount calculated for FULFILLED ITEMS ONLY.
            - If the cash balance is sufficient to cover the fulfilled items, proceed to step 3.
            - If the cash balance is insufficient, return a response indicating the order cannot be placed due to insufficient funds and CRITICAL: Do NOT attempt to place any sales orders if funds are insufficient or if fulfilling the order would result in a negative cash balance.

            Step 3. RECORD SALES FOR FULFILLED ITEMS ONLY:
            - For EACH item where "fulfilled": true, call the place_sales_order tool with:
              * item_name (from the quote)
              * quantity (from the quote)
              * total_price (unit_price * quantity from the quote)
              * transaction_date (from the quote's request_date)
            - CRITICAL: Do NOT call place_sales_order for any items where "fulfilled": false
            - Each place_sales_order call records a real sale and deducts from cash balance

            Step 4. GENERATE UPDATED FINANCIAL REPORT:
            - After placing all sales orders for fulfilled items, use the generate_financial_report tool.
            - This reflects the new cash balance after charging for fulfilled items only.

            Step 5. PREPARE ORDER RESPONSE:
            Return a JSON dictionary containing:
            - "items_fulfilled": List of items that were charged (fulfilled: true)
            - "items_out_of_stock": List of items pending supplier delivery (fulfilled: false)
            - "items_cannot_fulfill": List of items that cannot be fulfilled at all
            - "amount_charged": Total amount charged (ONLY for fulfilled items)
            - "updated_cash_balance": New cash balance after charging for fulfilled items
            - "explanation": Clear explanation distinguishing what was charged vs. what is pending/unfulfilled

            CRITICAL VALIDATION:
            - Verify that "amount_charged" matches the sum of prices for items where "fulfilled": true
            - Verify that you called place_sales_order exactly once per fulfilled item (no more, no less)
            - Never charge for items where "fulfilled": false
            - Ensure that the updated cash balance reflects only the charges for fulfilled items
            - Do not place orders when funds are insufficient, and return an appropriate error message instead

            The returned object should be a JSON dictionary and nothing else. Do not include any text outside of the JSON dictionary in your response.
            """
        )
        
        return response
    
class CommunicationsAgent(ToolCallingAgent):
    def __init__(self, model) -> None:
        super().__init__(
            model=model, 
            tools=[],
            name="CommunicationsAgent",
            description="Agent responsible for managing communications with customers, including sending information on orders whether they were completed or not."
        )

    def _sanitize_customer_communication(self, message: str) -> str:
        """
        Post-processing guardrail that removes sensitive financial and internal information
        from customer communications before sending.
        
        This method implements regex-based filtering to prevent:
        - Cash/account balances (e.g., "cash balance is $50,000" or "Your updated cash balance is $46,052.20")
        - Transaction IDs (e.g., "Transaction ID: 12345")
        - Internal step-by-step reasoning (e.g., "Step 1 Analysis:", "Step 2 Communication Message:")
        - Dollar amounts in balance contexts
        
        Args:
            message (str): The raw message from CommunicationsAgent
            
        Returns:
            str: Sanitized message with sensitive information removed
        """
        sanitized = message
        
        # Pattern 1: Remove lines containing "cash balance" or "account balance" with any dollar amounts
        # Matches: "cash balance is $50,000" or "Your updated cash balance is $46,052.20"
        sanitized = re.sub(
            r'.*\b(?:cash\s+balance|account\s+balance|updated\s+cash\s+balance).*?\$[\d,]+\.?\d*.*?\n?',
            '',
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Pattern 2: Remove Transaction ID mentions
        # Matches: "Transaction ID: 12345" or "transaction id: ABC123"
        sanitized = re.sub(
            r'.*\b(?:transaction\s+id|transaction\s+ID)[\s:]*[\w\-]*.*?\n?',
            '',
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Pattern 3: Remove internal step-by-step reasoning markers
        # Matches: "Step 1 Analysis:", "Step 2 Communication Message:", etc.
        sanitized = re.sub(
            r'^(?:Step\s+\d+\s+(?:Analysis|Communication|Reasoning|Message):?).*?$\n?',
            '',
            sanitized,
            flags=re.MULTILINE | re.IGNORECASE
        )
        
        # Pattern 4: Remove standalone lines that contain "Your updated" followed by balance-related terms
        # Matches: "Your updated cash balance is $49,712.20"
        sanitized = re.sub(
            r'.*\byour\s+updated.*?(?:cash\s+balance|account\s+balance).*?\n?',
            '',
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Pattern 5: Remove standalone lines with balance context dollar amounts
        # Matches lines like: "Balance: $45,059.70" or "Cash: $51,000"
        sanitized = re.sub(
            r'^.*?(?:Balance|Cash|Account|Updated Balance)[\s:]*\$[\d,]+\.?\d*.*?$\n?',
            '',
            sanitized,
            flags=re.MULTILINE | re.IGNORECASE
        )
        
        # Pattern 6: Remove lines containing inventory report details
        # Matches: "Inventory Value:", "Financial Report:", etc.
        sanitized = re.sub(
            r'.*\b(?:inventory\s+value|financial\s+report|cash\s+on\s+hand)[\s:]*.*?\n?',
            '',
            sanitized,
            flags=re.IGNORECASE
        )
        
        # Pattern 7: Remove past-dated delivery date statements
        # Matches: "October 6, 2023", "2023-10-09", "October 22, 2023", "November 4, 2023" etc.
        # Captures lines mentioning delivery with years clearly in the past (2023 or earlier)
        sanitized = re.sub(
            r'.*\b(?:(?:delivery|deliver|available|will\s+be\s+ready|expected|estimated)\b.*?(?:2023|2022|2021|2020|2019|2018|2017|2016|2015|2014|2013|2012|2011|2010)).*?\n?',
            '',
            sanitized,
            flags=re.MULTILINE | re.IGNORECASE
        )
        
        # Pattern 7b: Also remove lines with impossible month-day date patterns like "October 6, 2023" when they appear in delivery contexts
        sanitized = re.sub(
            r'.*\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(?:\d{1,2},?\s+)?2023.*?\n?',
            '',
            sanitized,
            flags=re.MULTILINE | re.IGNORECASE
        )
        
        # Clean up extra blank lines that may have been left behind
        sanitized = re.sub(r'\n\n\n+', '\n\n', sanitized)
        sanitized = sanitized.strip()
        
        return sanitized

    def send_order_fulfillment_update(self, order_response: Dict) -> str:
        """
        Sends an update to the customer regarding the fulfillment status of their order based on the provided order response details.

        This function is responsible for communicating the outcome of the order processing back to the customer, including which items were fulfilled, any issues encountered, and the final status of the order.

        Args:
            order_response (Dict): A dictionary containing details about the order fulfillment status, including the original quote, items ordered, total amount, updated cash balance, and any issues or explanations related to the order processing.

        Returns:
            str: A string containing the communication details that would be sent to the customer, including a summary of the order fulfillment status, any relevant explanations, and the final outcome of the order processing.
        """
        response = self.run(
            f"""You are a communications agent for Munder Difflin. Your task is to manage communications with customers regarding the fulfillment status of their orders.

            CRITICAL LOGIC RULES - PREVENT CONTRADICTORY STATEMENTS:
            ========================================================
            RULE 1: An item CANNOT be BOTH "successfully ordered/fulfilled" AND "out of stock" in the same message.
            Each item falls into EXACTLY ONE category:
            - FULFILLED: Item is in stock and will be shipped. Say: "Successfully ordered: [item] [qty]"
            - OUT_OF_STOCK: Item is not available now. Say: "[Item] is currently out of stock. Estimated delivery: [DATE]"
            - CANNOT_FULFILL: Item cannot be ordered at all. Say: "[Item] could not be fulfilled due to [REASON]"
            If an item is fulfilled, do NOT say it is out of stock. If an item is out of stock, do NOT say it is fulfilled. If an item cannot be fulfilled, do NOT say it is fulfilled or out of stock. Avoid any language that mixes these categories for the same item.

            RULE 2: Delivery dates must be REALISTIC FUTURE dates from the order date.
            - Check all delivery dates in the order response. They MUST be in the future relative to the order date.
            - If a delivery date appears to be in the past (e.g., October 2023 for an order from April 2025), this is an ERROR. Do NOT use that date.
            - Only include delivery dates that are valid future dates (today or later).
            - Do not ask the customer for input on past dates; simply exclude any past-dated delivery information from the message and focus on valid future dates.

            RULE 3: Cash impact reflects fulfillment only.
            - If items are "successfully ordered" and charged to the customer, the cash balance should show a deduction.
            - If items are "out of stock", they should NOT be charged yet (no cash deduction).
            - Verify consistency: fulfilled items = cash deduction, out-of-stock items = no deduction.
            - If there is a mismatch (e.g., cash balance shows deduction but item is out of stock), this is an ERROR. Do NOT communicate that to the customer; instead, ensure your message reflects the correct fulfillment and charging status.
            - If the order cannot be fulfilled due to insufficient funds, communicate that clearly without mentioning specific cash balances.
            - If the order is fulfilled, communicate the successful charge without mentioning specific cash balances.
            - An order cannot have an charged amount if charges have not been made. Make sure to use language like "quote value" or "total amount for fulfilled items" rather than "charged amount" if the order is not actually charged yet.
            - For orders that have a delivery date in the future, do not say "charged" if the charge has not been processed yet. Instead, say "Your order for [item] [qty] is confirmed and will be charged when it ships."

            CRITICAL GUARDRAILS - DO NOT INCLUDE IN CUSTOMER MESSAGE:
            ========================================================
            You MUST NOT include any of the following in your customer communication:
            1. CASH BALANCES OR ACCOUNT BALANCES - Never mention "cash balance", "account balance", "updated balance", or "Your updated cash balance"
            2. DOLLAR AMOUNTS related to company finances or balances - Do not include amounts like "$50,000" or "$46,052.20" in balance contexts
            3. TRANSACTION IDs - Do not include "Transaction ID", "transaction id", or any transaction reference numbers
            4. INTERNAL FINANCIAL DETAILS - Do not include cash on hand, inventory value, or other internal financial metrics
            5. INTERNAL STEP-BY-STEP REASONING - Do not include "Step 1 Analysis", "Step 2 Communication", or other internal process steps
            6. INTERNAL THOUGHT PROCESS - Do not show your reasoning or analysis steps to the customer
            7. PAST-DATED DELIVERY DATES - Do not include delivery dates that are in the past relative to the order date (e.g., October 2023 for an order from April 2025), and do not ask for input from the customer about past dates, just fail gracefully by excluding those dates from the message.

            APPROVED INFORMATION TO INCLUDE:
            ================================
            You MAY include:
            - Item names and quantities that were fulfilled (in stock, shipped immediately)
            - The TOTAL ORDER AMOUNT (e.g., "Your order total is $65.00") - this is customer-facing pricing
            - Items that are out of stock with realistic estimated delivery dates (future dates only)
            - Items that could not be ordered and WHY (e.g., "insufficient stock", "missing pricing data")
            - Clear, empathetic explanations of any order processing issues
            - Professional next steps or recommendations

            Given the following order response details: {json.dumps(order_response)}, accomplish the following steps:

            Step 1. Analyze the order response details to determine the fulfillment status of the customer's order.
            - IMPORTANT: Carefully separate items that are FULFILLED from items that are OUT_OF_STOCK or CANNOT_FULFILL.
            - Verify all delivery dates are realistic future dates relative to the order date.
            
            Step 2. Craft a clear and informative communication message to the customer that summarizes the status of their order:
            - Section 1: Clearly list items that were successfully ordered and immediately fulfilled with quantities
            - Section 2: If applicable, list items that are out of stock. Do NOT say these items are "charged" or "successfully ordered". Instead, say they are "currently out of stock". Do not include any delivery date information for out-of-stock items.
            - Section 3: If applicable, list items that could not be fulfilled and explain why
            - Section 4: Include the total order amount (only for fulfilled items charged to customer)
            - CRITICAL: Never mix "successfully ordered" and "out of stock" language for the same item in the same sentence/paragraph.
            - Professional tone with appropriate empathy where needed

            REMINDER: Ensure your final message does NOT contain:
            - Any cash balances, account balances, transaction IDs, or internal financial details
            - Any contradictory statements (item cannot be both fulfilled AND out of stock)
            - Any past-dated delivery dates
            Focus ONLY on clear, consistent order status information relevant to the customer.

            Return ONLY the customer-facing message. Do not include any explanations, step-by-step analysis, or metadata. Just the message itself.
            """
        )
        
        # Apply post-processing guardrail to remove any sensitive information that may have slipped through
        sanitized_response = self._sanitize_customer_communication(response)
        
        return sanitized_response

class OrchestrationAgent(ToolCallingAgent):
    def __init__(self, model, quoting_agent: QuotingAgent, ordering_agent: OrderingAgent, communications_agent: CommunicationsAgent) -> None:
        super().__init__(
            model=model,
            tools=[],
            name="OrchestrationAgent",
            description="Agent responsible for orchestrating the overall workflow of processing customer requests, generating quotes, and placing orders by coordinating between the QuotingAgent and OrderingAgent."
        )
        self.quoting_agent = quoting_agent
        self.ordering_agent = ordering_agent
        self.communications_agent = communications_agent

    def process_request(self, request_details: Dict) -> Dict:
        """
        Processes a customer request by coordinating the quoting and ordering agents to generate a quote and place an order accordingly.

        Args:
            request_details (Dict): A dictionary containing the details of the customer request, such as job type, event type, order size, and any specific requirements.

        Returns:
            Dict: A dictionary containing the final outcome of processing the request, including the generated quote, order placement results, and any relevant explanations or issues encountered during the process.
        """
        # Step 1: Extract relevant information from the request details to inform the quoting process
        order: Dict[str, int] = self.ordering_agent.extract_pertinent_order_details(request_details)

        # Step 1: Generate a quote using the QuotingAgent based on the extracted order details
        quote = self.quoting_agent.create_quote(order)

        # Step 2: Place an order using the OrderingAgent based on the generated quote
        order_result = self.ordering_agent.place_order(quote)

        # Compile final response
        semi_final_response = {
            "quote": quote,
            "order_result": order_result,
        }

        final_response = self.communications_agent.send_order_fulfillment_update(semi_final_response)

        return final_response 

# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv(os.path.join(SCRIPT_DIR, "quote_requests_sample.csv"))
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
    orchestrator = OrchestrationAgent(
        model=model, 
        quoting_agent=QuotingAgent(model=model, inventory_agent=InventoryAgent(model=model)),
        ordering_agent=OrderingAgent(model=model),
        communications_agent=CommunicationsAgent(model=model)
    )

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

        # response = call_your_multi_agent_system(request_with_date)

        response = orchestrator.process_request(request_with_date)

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
