from typing import Dict, List, Any, Optional
import os
import dotenv
import random
import time
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
import json

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
        self.locations = ["Beijing Branch (北京分行)", 
                          "Shanghai Branch (上海分行)",
                          "Guangzhou Branch (广州分行)"]
        self.special_services = {
            "VIP": ["deposit", "international_transfer"],
            "Regular": [], 
            "Student": ["loan"],
            "Senior": ["bill_payment"]
        }
        self.availability = {
            "deposit": 2, "postal": 2, "loan": 2, "bill_payment": 2,
            "international_transfer": 2, "general_inquiry": float('inf')
        }
        self.routing_accuracy = {
            "correct_service_type": 0, "total_requests": 0, "special_handling_applied": 0
        }
        self.customer_profiles = {
            "Wang Xiaoming (王小明)": {"type": "VIP", "language": "Mandarin"},
            "Li Jiayi (李佳怡)": {"type": "Regular", "language": "Mandarin"},
            "Chen Student (陈学生)": {"type": "Student", "language": "Mandarin"},
            "Zhang Senior (张老先生)": {"type": "Senior", "language": "Cantonese"},
            "Ms. Qian (钱女士)": {"type": "VIP", "language": "English"},
            "Mr. Zhao (赵先生)": {"type": "Regular", "language": "Mandarin"}
        }

    def check_availability(self, service_type: str) -> bool:
        service_type_lower = service_type.lower()
        return self.availability.get(service_type_lower, 0) > 0

    def add_booking(self, service_type: str, customer_name: str) -> str:
        service_type_lower = service_type.lower()
        
        if not self.check_availability(service_type_lower):
            return f"Sorry, no availability for {service_type_lower} service. (很抱歉，{service_type_lower}服务目前没有可用名额。)"
        
        if service_type_lower not in self.bookings:
            self.bookings[service_type_lower] = []
        
        self.bookings[service_type_lower].append(customer_name)
        if self.availability.get(service_type_lower, 0) != float('inf'):
            self.availability[service_type_lower] -= 1
        
        customer_type = "Regular"
        for cust, profile in self.customer_profiles.items():
            if customer_name.lower() in cust.lower():
                customer_type = profile["type"]
                break
                
        special_handling = ""
        if customer_type in self.special_services and service_type_lower in self.special_services[customer_type]:
            special_handling = f" with {customer_type} priority service (享受{customer_type}优先服务)"
            self.routing_accuracy["special_handling_applied"] += 1
        
        branch = random.choice(self.locations)
        
        confirmation_parts = [
            f"{customer_name}'s {service_type_lower} service booking is confirmed at {branch}{special_handling}.",
            f"({customer_name}的{service_type_lower}服务预约已确认，地点在{branch}{special_handling}。)"
        ]
        return " ".join(confirmation_parts)

booking_manager = BookingManager()

@tool
def handle_deposit_request(customer_name: str) -> str:
    """
    Handles a deposit request for the given customer by booking a 'deposit' service.

    Args:
        customer_name (str): The name of the customer making the deposit.

    Returns:
        str: A confirmation message or status of the deposit booking.
    """
    return booking_manager.add_booking("deposit", customer_name)

@tool
def handle_postal_request(customer_name: str) -> str:
    """
    Handles a postal service request for the given customer by booking a 'postal' service.

    Args:
        customer_name (str): The name of the customer using postal services.

    Returns:
        str: A confirmation message or status of the postal service booking.
    """
    return booking_manager.add_booking("postal", customer_name)

@tool
def handle_loan_request(customer_name: str) -> str:
    """
    Handles a loan application or inquiry for the given customer by booking a 'loan' service.

    Args:
        customer_name (str): The name of the customer applying for/inquiring about a loan.

    Returns:
        str: A confirmation message or status of the loan process initiation.
    """
    return booking_manager.add_booking("loan", customer_name)

@tool
def handle_bill_payment_request(customer_name: str) -> str:
    """
    Handles a bill payment request for the given customer by booking a 'bill_payment' service.

    Args:
        customer_name (str): The name of the customer paying a bill.

    Returns:
        str: A confirmation message or status of the bill payment process.
    """
    return booking_manager.add_booking("bill_payment", customer_name)

@tool
def handle_international_transfer_request(customer_name: str) -> str:
    """
    Handles an international money transfer request for the given customer by booking an 'international_transfer' service.

    Args:
        customer_name (str): The name of the customer making an international transfer.

    Returns:
        str: A confirmation message or status of the international transfer process.
    """
    return booking_manager.add_booking("international_transfer", customer_name)

@tool
def handle_general_inquiry_request(customer_name: str, original_request: str) -> str:
    """
    Handles a general inquiry from the given customer. 
    This tool provides a direct response for general questions not fitting other service categories.

    Args:
        customer_name (str): The name of the customer.
        original_request (str): The customer's original general inquiry.

    Returns:
        str: A helpful response or direction for the general inquiry.
    """
    customer_profile = {}
    for cust_key, profile_val in booking_manager.customer_profiles.items():
        if customer_name.lower() in cust_key.lower():
            customer_profile = profile_val
            break
    lang = customer_profile.get("language", "Mandarin")
    
    if lang == "English":
        return f"Thank you for your inquiry, {customer_name}. For general questions like '{original_request}', please refer to our FAQ or a bank representative can assist you shortly."
    elif lang == "Cantonese":
        return f"{customer_name}，多謝你嘅查詢。關於一般問題，好似「{original_request}」，請參考我哋嘅常見問題，或者稍後銀行職員會協助你。"
    else: 
        return f"{customer_name}，感谢您的咨询。关于一般问题，例如“{original_request}”，请参考我们的常见问题解答，或者稍后银行代表将为您提供帮助。"

class RequestAnalysisAgent(ToolCallingAgent):
    def __init__(self, model_to_use: OpenAIServerModel):
        super().__init__(
            tools=[], 
            model=model_to_use,
            name="request_analysis_agent",
            description="Analyzes customer requests and directly outputs the categorized service type as its final answer."
        )
        self.possible_service_types = [
            "deposit", "postal", "loan", "bill_payment", 
            "international_transfer", "general_inquiry"
        ]

    def get_service_type_from_llm(self, user_request: str) -> str:
        self.memory.steps = []
        prompt = f"""
        A customer stated in a Chinese Postal Bank context: "{user_request}"
        Your ONLY task is to identify the primary banking or postal service the customer needs.
        Choose exactly ONE service type from this list: {json.dumps(self.possible_service_types)}.
        
        You MUST then use the 'final_answer' tool. The 'answer' argument for the 'final_answer' tool
        should be ONLY the chosen service type string (e.g., "deposit", "postal").
        Do not add any other text, explanation, or conversational filler. Just the service type.
        Example: If the request is for a deposit, your 'final_answer' tool call should have arguments: {json.dumps({'answer': 'deposit'})}
        """
        _ = self.run(prompt)

        diagnosed_service = "general_inquiry" 
        for step in reversed(self.memory.steps):
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tc in step.tool_calls:
                    if tc.name == 'final_answer':
                        if hasattr(tc, 'arguments') and tc.arguments.get('answer') is not None:
                            candidate_service = str(tc.arguments.get('answer')).lower().strip()
                            if candidate_service in self.possible_service_types:
                                return candidate_service 
                        elif hasattr(step, 'action_output') and step.action_output is not None: # Fallback
                            candidate_service = str(step.action_output).lower().strip()
                            if candidate_service in self.possible_service_types:
                                return candidate_service
                        break 
            if any(tc.name == 'final_answer' for tc in getattr(step, 'tool_calls', [])):
                break
        
        return "general_inquiry"


class ChineseBankPostOfficeAgent(ToolCallingAgent): 
    def __init__(self, model_to_use: OpenAIServerModel):
        self.request_analyzer = RequestAnalysisAgent(model_to_use)
        super().__init__(
            tools=[
                handle_deposit_request, handle_postal_request, handle_loan_request,
                handle_bill_payment_request, handle_international_transfer_request,
                handle_general_inquiry_request
            ],
            model=model_to_use,
            name="chinese_bank_post_office_orchestrator",
            description="""Orchestrator for the Chinese Postal Bank. 
            It first analyzes a customer request to determine service type, 
            then calls the appropriate handler tool using its own LLM."""
        )

    def _get_final_response_from_orchestrator_memory(self) -> str:
        for step in reversed(self.memory.steps):
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tc in step.tool_calls:
                    if tc.name == 'final_answer':
                        if hasattr(step, 'action_output') and step.action_output is not None:
                            return str(step.action_output)
                        elif hasattr(tc, 'arguments') and tc.arguments.get('answer') is not None:
                             return str(tc.arguments.get('answer'))
            if hasattr(step, 'observations') and step.observations is not None:
                return str(step.observations)
        return "Orchestrator: Could not determine a final response from its execution."


    def handle_customer_request(self, customer_name: str, request: str, expected_service_for_metric: str) -> str:
        booking_manager.routing_accuracy["total_requests"] += 1
        
        print(f"\n--- Orchestrator processing request from {customer_name} ---")
        print(f"Original Request: \"{request}\"")

        diagnosed_service_type = self.request_analyzer.get_service_type_from_llm(request)
        print(f"LLM Diagnosed Service Type by RequestAnalysisAgent: '{diagnosed_service_type}' (Expected for metric: '{expected_service_for_metric}')")

        if diagnosed_service_type.lower() == expected_service_for_metric.lower():
            booking_manager.routing_accuracy["correct_service_type"] +=1

        self.memory.steps = []
        orchestrator_prompt = f"""
        You are the main Orchestrator for the Chinese Postal Bank.
        A customer named '{customer_name}' made a request: "{request}"
        This request has been analyzed and categorized as needing the service type: '{diagnosed_service_type}'.

        Your task is to call the correct handler tool based on the diagnosed_service_type.
        Your available tools are: 'handle_deposit_request', 'handle_postal_request', 'handle_loan_request', 'handle_bill_payment_request', 'handle_international_transfer_request', 'handle_general_inquiry_request'.

        Based on the diagnosed_service_type ('{diagnosed_service_type}'):
        - If 'deposit', call 'handle_deposit_request' with customer_name='{customer_name}'.
        - If 'postal', call 'handle_postal_request' with customer_name='{customer_name}'.
        - If 'loan', call 'handle_loan_request' with customer_name='{customer_name}'.
        - If 'bill_payment', call 'handle_bill_payment_request' with customer_name='{customer_name}'.
        - If 'international_transfer', call 'handle_international_transfer_request' with customer_name='{customer_name}'.
        - If 'general_inquiry', call 'handle_general_inquiry_request' with customer_name='{customer_name}' and original_request='{request}'.

        After calling the appropriate handler tool, you MUST use the 'final_answer' tool.
        The 'answer' for 'final_answer' should be the EXACT string output (observation) you received from the handler tool.
        Do not add any extra text or explanation around it. Just the observation.
        """
        _ = self.run(orchestrator_prompt)
        
        final_response = self._get_final_response_from_orchestrator_memory()
        print(f"Orchestrator's LLM action result (final_answer or observation): {final_response}")
        return final_response

def print_state():
    print("\n" + "=" * 80 + "\nFINAL SYSTEM STATE\n" + "=" * 80)
    print("\nRemaining Service Availability:")
    for service, count in booking_manager.availability.items():
        print(f"  - {service}: {'∞' if count == float('inf') else count} slots")
    print("\nBookings Completed:")
    if not any(booking_manager.bookings.values()):
        print("  No bookings were made in this session.")
    else:
        for service, customers in booking_manager.bookings.items():
            if customers: print(f"  - {service}: {', '.join(customers)}")
    
    accuracy = 0
    if booking_manager.routing_accuracy["total_requests"] > 0:
        accuracy = (booking_manager.routing_accuracy["correct_service_type"] / 
                   booking_manager.routing_accuracy["total_requests"]) * 100
    
    print("\nPerformance Metrics:")
    print(f"  - Total Requests Processed: {booking_manager.routing_accuracy['total_requests']}")
    print(f"  - Correct Service Type Diagnosed & Routed (metric): {booking_manager.routing_accuracy['correct_service_type']}")
    print(f"  - Request Routing Accuracy: {accuracy:.1f}%")
    print(f"  - Special Customer Handling Applied: {booking_manager.routing_accuracy['special_handling_applied']} times")
    
    if accuracy >= 80 and booking_manager.routing_accuracy["special_handling_applied"] >= 2:
        print("\n✅ SUCCESS: The system demonstrated effective LLM-driven routing and special case handling!")
    else:
        print("\n⚠️  The system may need improvement in LLM-driven request routing or special case handling based on these metrics.")
    print("=" * 80)

if __name__ == "__main__":
    bank_post_office_agent = ChineseBankPostOfficeAgent(model)
    
    print("🏦 Chinese Postal Bank Service Demo (中国邮政银行服务示例) 🏦\n")

    test_cases = [
        {"name": "Wang Xiaoming (王小明)", "request": "I need to deposit money into my account. (我需要存一些钱到我的账户。)", "expected_service": "deposit", "metadata": "VIP customer"},
        {"name": "Li Jiayi (李佳怡)", "request": "I want to send a package to Shanghai. (我想邮寄一个包裹到上海。)", "expected_service": "postal", "metadata": "Regular customer"},
        {"name": "Chen Student (陈学生)", "request": "How do I apply for a student loan? (我该如何申请学生贷款？)", "expected_service": "loan", "metadata": "Student customer"},
        {"name": "Zhang Senior (张老先生)", "request": "I need to help paying my electricity bill. (我需要帮助支付我的电费。)", "expected_service": "bill_payment", "metadata": "Senior customer"},
        {"name": "Ms. Qian (钱女士)", "request": "I want to transfer money to my son in Canada. (我想给我在加拿大的儿子转账。)", "expected_service": "international_transfer", "metadata": "VIP customer"},
        {"name": "Mr. Zhao (赵先生)", "request": "What are the business hours for the Beijing branch? (北京分行的营业时间是什么时候？)", "expected_service": "general_inquiry", "metadata": "Regular customer"}
    ]
    
    for case in test_cases:
        response = bank_post_office_agent.handle_customer_request(case['name'], case['request'], case['expected_service'])
        time.sleep(0.5) 
    
    print_state()