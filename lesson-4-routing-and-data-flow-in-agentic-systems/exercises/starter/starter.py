from typing import Dict, List
import os
import dotenv
import random
import time

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)

dotenv.load_dotenv(dotenv_path="../.env")
openai_api_key = os.getenv("UDACITY_OPENAI_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)

class BookingManager:
    def __init__(self):
        self.bookings = {}
        self.locations = ["Beijing Branch (北京分行)", "Shanghai Branch (上海分行)", "Guangzhou Branch (广州分行)"]
        self.special_services = {
            "VIP": ["deposit", "international_transfer"],
            "Regular": [], 
            "Student": ["loan"],
            "Senior": ["bill_payment"]
        }
        # Limited availability for services
        self.availability = {
            "deposit": 2,
            "postal": 2,
            "loan": 2,
            "bill_payment": 2,
            "international_transfer": 2,
            "general_inquiry": float('inf')  # unlimited
        }
        # Track proper routing decisions
        self.routing_accuracy = {
            "correct_service_type": 0,
            "total_requests": 0,
            "special_handling_applied": 0
        }
        # Customer profiles
        self.customer_profiles = {
            "Wang Xiaoming (王小明)": {"type": "VIP", "language": "Mandarin"},
            "Li Jiayi (李佳怡)": {"type": "Regular", "language": "Mandarin"},
            "Chen Student (陈学生)": {"type": "Student", "language": "Mandarin"},
            "Zhang Senior (张老先生)": {"type": "Senior", "language": "Cantonese"},
            "Ms. Qian (钱女士)": {"type": "VIP", "language": "English"},
            "Mr. Zhao (赵先生)": {"type": "Regular", "language": "Mandarin"}
        }

    def check_availability(self, service_type):
        return self.availability[service_type] > 0

    def add_booking(self, service_type, customer_name, expected_service=None):
        if service_type not in self.bookings:
            self.bookings[service_type] = []
            
        if not self.check_availability(service_type):
            return f"Sorry, no availability for {service_type} service. (很抱歉，{service_type}服务目前没有可用名额。)"
        
        # Track proper service identification
        if expected_service and service_type == expected_service:
            self.routing_accuracy["correct_service_type"] += 1
        
        # Update state
        self.bookings[service_type].append(customer_name)
        self.availability[service_type] -= 1
        
        # Get customer type
        customer_type = "Regular"
        for cust, profile in self.customer_profiles.items():
            if customer_name in cust:
                customer_type = profile["type"]
                break
                
        # Check if special handling is applied for eligible customers
        special_handling = ""
        if customer_type in self.special_services and service_type in self.special_services[customer_type]:
            special_handling = f" with {customer_type} priority service (享受{customer_type}优先服务)"
            self.routing_accuracy["special_handling_applied"] += 1
        
        # Random branch assignment
        branch = random.choice(self.locations)
        
        # Return bilingual response
        return f"{customer_name}'s {service_type} service booking is confirmed at {branch}{special_handling}. ({customer_name}的{service_type}服务预约已确认，地点在{branch}{special_handling}。)"

booking_manager = BookingManager()

@tool
def analyze_request(request: str) -> str:
    """Analyzes the customer request and identifies the service needed.
    This tool MUST return only the service type without additional text.

    Args:
        request: The customer's request.

    Returns:
        ONLY the identified service type (deposit, postal, loan, bill_payment, international_transfer, or general_inquiry).
    """
    # Map common request phrases to service types
    service_keywords = {
        "deposit": ["deposit", "save", "put money", "存", "存款", "存钱"],
        "postal": ["mail", "send", "package", "shipping", "邮寄", "包裹", "快递"],
        "loan": ["loan", "borrow", "贷款", "借钱"],
        "bill_payment": ["bill", "pay", "payment", "电费", "水费", "缴费", "账单"],
        "international_transfer": ["international", "transfer", "abroad", "foreign", "国际", "转账", "汇款"],
        "general_inquiry": ["hours", "information", "question", "when", "where", "how", "什么时候", "在哪里", "怎么样", "如何"]
    }
    
    request_lower = request.lower()
    for service, keywords in service_keywords.items():
        for keyword in keywords:
            if keyword.lower() in request_lower:
                return service
    
    # Default to general inquiry if no match
    return "general_inquiry"

@tool
def handle_deposit(customer_name: str) -> str:
    """Handles deposit requests.

    Args:
        customer_name: The name of the customer.

    Returns:
        A message indicating the deposit process has been initiated.
    """
    return booking_manager.add_booking("deposit", customer_name, "deposit")

@tool
def handle_postal(customer_name: str) -> str:
    """Handles postal service requests.

    Args:
        customer_name: The name of the customer.

    Returns:
        A message indicating the package details have been collected.
    """
    return booking_manager.add_booking("postal", customer_name, "postal")

@tool
def handle_loan(customer_name: str) -> str:
    """Handles loan application requests.

    Args:
        customer_name: The name of the customer.

    Returns:
        A message indicating the loan application process has started.
    """
    return booking_manager.add_booking("loan", customer_name, "loan")

@tool
def handle_bill_payment(customer_name: str) -> str:
    """Handles bill payment requests.

    Args:
        customer_name: The name of the customer.

    Returns:
        A message indicating the bill payment process has started.
    """
    return booking_manager.add_booking("bill_payment", customer_name, "bill_payment")

@tool
def handle_international_transfer(customer_name: str) -> str:
    """Handles international transfer requests.

    Args:
        customer_name: The name of the customer.

    Returns:
        A message indicating the international transfer process has started.
    """
    return booking_manager.add_booking("international_transfer", customer_name, "international_transfer")

@tool
def handle_general_inquiry(customer_name: str) -> str:
    """Handles general inquiry requests.

    Args:
        customer_name: The name of the customer.

    Returns:
        A message indicating redirection to the general information desk.
    """
    booking_manager.routing_accuracy["correct_service_type"] += 1
    return f"{customer_name} has been redirected to the general information desk. (已将{customer_name}转接至一般咨询服务台。)"

class RequestAnalyzer(ToolCallingAgent):
    """Agent that analyzes customer requests."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[analyze_request],
            model=model,
            name="request_analyzer",
            description="""You are a request analyzer agent for a Chinese Postal Bank. 
            Your job is to identify what service a customer needs from their request.
            
            IMPORTANT: You MUST ONLY use the analyze_request tool to classify the request.
            NEVER provide explanations or direct responses to the customer.
            
            When using the analyze_request tool, you must identify the request as one of these categories:
            - deposit (for money deposits)
            - postal (for sending packages/mail)
            - loan (for loan applications/inquiries)
            - bill_payment (for bill payments)
            - international_transfer (for international money transfers)
            - general_inquiry (for general questions/information)
            
            DO NOT give explanations or respond directly to the user.""",
        )

class ServiceRouter(ToolCallingAgent):
    """Agent that routes customer requests to the appropriate service agent."""

    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                handle_deposit,
                handle_postal,
                handle_loan,
                handle_bill_payment,
                handle_international_transfer,
                handle_general_inquiry,
            ],
            model=model,
            name="service_router",
            description="""You are a service router agent for a Chinese Postal Bank.
            Your job is to route the customer to the appropriate service handler based on the service type.
            
            IMPORTANT: You MUST ONLY use the appropriate handle_* tool. 
            DO NOT add additional explanations or messages.
            
            You will receive a direct instruction with the service and customer name to process.
            Simply execute the correct tool and return the result.""",
        )

class ChineseBankPostOfficeAgent(ToolCallingAgent):
    """Agent for the Chinese Bank Post Office."""

    def __init__(self, model: OpenAIServerModel):
        self.request_analyzer = RequestAnalyzer(model)
        self.service_router = ServiceRouter(model)
        super().__init__(
            tools=[],
            model=model,
            name="bank_post_office_agent",
            description="Agent for the Chinese Postal Bank that analyzes and routes customer requests.",
        )

    def handle_customer_request(self, customer_name: str, request: str, expected_service: str) -> str:
        """Handles a customer request by analyzing it and routing it to the appropriate service agent.

        Args:
            customer_name: The name of the customer.
            request: The customer's request.
            expected_service: The service we expect this request to be routed to.

        Returns:
            The response from the service agent.
        """
        # Track total requests
        booking_manager.routing_accuracy["total_requests"] += 1
        
        # Get the service type from the request analyzer with explicit prompt instructions
        analyzer_prompt = f"""
        You are a service classifier that ONLY returns service type categories.
        
        Analyze this customer request: "{request}"
        
        Choose EXACTLY ONE service type from this list:
        - deposit
        - postal
        - loan
        - bill_payment
        - international_transfer
        - general_inquiry
        
        Return ONLY the service type as a single word. Do not include any explanation, 
        additional text, or punctuation. Just the service type alone.
        """
        
        service_type = self.request_analyzer.run(analyzer_prompt).strip().lower()
        
        # Clean up response to ensure it's just a service type
        for valid_type in ["deposit", "postal", "loan", "bill_payment", "international_transfer", "general_inquiry"]:
            if valid_type in service_type:
                service_type = valid_type
                break
        else:
            # Default if no valid type found
            service_type = "general_inquiry"
        
        # Construct the router prompt with explicit instructions
        router_prompt = f"""
        Execute the handle_{service_type} tool with this customer name: '{customer_name}'
        
        Do not add any explanation or additional text. Just execute the tool.
        """
        
        # Route to the service handler with explicit prompt
        return self.service_router.run(router_prompt)

def print_state():
    """Prints the current state of the system, showing what's been accomplished."""
    print("\n" + "=" * 80)
    print("CURRENT SYSTEM STATE")
    print("=" * 80)
    
    # Service availability
    print("\nRemaining Service Availability:")
    for service, count in booking_manager.availability.items():
        print(f"  - {service}: {'∞' if count == float('inf') else count} slots")
    
    # Bookings made
    print("\nBookings Completed:")
    for service, customers in booking_manager.bookings.items():
        if customers:
            print(f"  - {service}: {', '.join(customers)}")
    
    # Performance metrics
    accuracy = 0
    if booking_manager.routing_accuracy["total_requests"] > 0:
        accuracy = (booking_manager.routing_accuracy["correct_service_type"] / 
                   booking_manager.routing_accuracy["total_requests"]) * 100
    
    print("\nPerformance Metrics:")
    print(f"  - Request Routing Accuracy: {accuracy:.1f}%")
    print(f"  - Special Customer Handling Applied: {booking_manager.routing_accuracy['special_handling_applied']} times")
    
    # Overall success assessment
    if accuracy >= 80 and booking_manager.routing_accuracy["special_handling_applied"] >= 2:
        print("\n✅ SUCCESS: The system demonstrated proper routing and special case handling!")
    else:
        print("\n⚠️ The system needs improvement in request routing or special case handling.")
    print("=" * 80)

"""
"""

# TODO: Add an analyze_urgency tool that detects urgency in customer requests
@tool
def analyze_urgency(request: str) -> bool:
    """Analyzes whether a customer request is urgent.
    
    Args:
        request: The customer's request text.
        
    Returns:
        True if the request is urgent, False otherwise.
    """
    # Implement urgency detection logic here
    pass

# TODO: Create a UrgencyDetector agent that uses the analyze_urgency tool
class UrgencyDetector(ToolCallingAgent):
    """Agent that detects urgency in customer requests."""
    
    def __init__(self, model: OpenAIServerModel):
        # TODO: Initialize the urgency detector agent with appropriate tools and description
        pass

# TODO: Update the BookingManager to handle urgent requests
# - Add an "urgent_requests_handled" counter to routing_accuracy metrics
# - Modify add_booking to accept an is_urgent parameter and provide priority handling

# TODO: Enhance the ChineseBankPostOfficeAgent to include urgency detection in its workflow
# - Add the UrgencyDetector agent as a component
# - Modify handle_customer_request to check for urgency before analyzing service type
# - Ensure urgent requests receive priority handling

# Example test case to implement for your solution:
urgent_test_case = {
    "name": "Emergency Customer (紧急客户)",
    "request": "I urgently need to transfer money for a medical emergency! (我急需汇款处理医疗紧急情况！)",
    "expected_service": "international_transfer",
    "urgent": True
}

if __name__ == "__main__":
    bank_post_office_agent = ChineseBankPostOfficeAgent(model)
    
    print("🏦 Chinese Postal Bank Service Demo (中国邮政银行服务示例) 🏦\n")
    print("This demo demonstrates proper routing and data flow between agents.\n")

    # Initial state
    print("INITIAL STATE:")
    for service, count in booking_manager.availability.items():
        print(f"  - {service}: {count} slots available")
    print("-" * 60)

    # Test cases with expected service mapping - English (Chinese)
    test_cases = [
        {
            "name": "Wang Xiaoming (王小明)",
            "request": "I need to deposit money into my account. (我需要存一些钱到我的账户。)",
            "expected_service": "deposit",
            "metadata": "VIP customer eligible for special handling"
        },
        {
            "name": "Li Jiayi (李佳怡)",
            "request": "I want to send a package to Shanghai. (我想邮寄一个包裹到上海。)", 
            "expected_service": "postal",
            "metadata": "Regular customer"
        },
        {
            "name": "Chen Student (陈学生)",
            "request": "How do I apply for a student loan? (我该如何申请学生贷款？)", 
            "expected_service": "loan",
            "metadata": "Student customer eligible for special handling"
        },
        {
            "name": "Zhang Senior (张老先生)",
            "request": "I need help paying my electricity bill. (我需要帮助支付我的电费。)", 
            "expected_service": "bill_payment",
            "metadata": "Senior customer eligible for special handling"
        },
        {
            "name": "Ms. Qian (钱女士)",
            "request": "I want to transfer money to my son in Canada. (我想给我在加拿大的儿子转账。)", 
            "expected_service": "international_transfer",
            "metadata": "VIP customer eligible for special handling"
        },
        {
            "name": "Mr. Zhao (赵先生)",
            "request": "What are the business hours for the Beijing branch? (北京分行的营业时间是什么时候？)", 
            "expected_service": "general_inquiry",
            "metadata": "Regular customer"
        }
    ]
    
    # Add your urgent test case to the test cases when your implementation is ready
    # test_cases.append(urgent_test_case)
    
    for case in test_cases:
        print(f"\nProcessing request from {case['name']}:")
        print(f"Request: \"{case['request']}\"")
        print(f"Customer info: {case['metadata']}")
        
        # Run the request
        response = bank_post_office_agent.handle_customer_request(case['name'], case['request'], case['expected_service'])
        
        # Display response
        print(f"Response: {response}")
        
        # Show mini-state change after each request
        print("\nService availability after this request:")
        for service, count in booking_manager.availability.items():
            if service == case['expected_service']:
                print(f"  - {service}: {'∞' if count == float('inf') else count} slots")
        print("-" * 60)
        
        # Small delay for readability
        time.sleep(0.5)
    
    # Print final state and success metrics
    print_state()