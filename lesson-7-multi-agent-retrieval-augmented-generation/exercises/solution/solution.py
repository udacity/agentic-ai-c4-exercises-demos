# Insurance Claims RAG System Exercise - Solution

from typing import Dict, List, Any, Optional, Union, Set
import random
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)
import os
import dotenv

# Note: Make sure to set up your .env file with your API key before running
dotenv.load_dotenv(dotenv_path='../.env')
openai_api_key = os.getenv('UDACITY_OPENAI_API_KEY')

model = OpenAIServerModel(
    model_id='gpt-4o-mini',
    api_base='https://openai.vocareum.com/v1',
    api_key=openai_api_key,
)

# Import core components from the demo file
from demo.demo import (
    PrivacyLevel, AccessControl, Claim, PatientRecord, ComplaintRecord, 
    Database, VectorKnowledgeBase, VectorClaimSearch,
    DataGenerator, database, vector_kb, vector_claim_search,
    search_knowledge_base, retrieve_claim_history, get_claim_details,
    get_patient_info, find_similar_claims, submit_complaint,
    respond_to_complaint, get_complaint_history, process_new_claim,
    ClaimProcessingAgent, CustomerServiceAgent, MedicalReviewAgent,
    ComplaintResolutionOrchestrator
)

# STEP 1: Create a knowledge base of fraud patterns
# Common fraud patterns based on industry knowledge
fraud_patterns = [
    {
        'pattern_id': 'FP-001',
        'pattern_name': 'Rapid Claim Submission',
        'description': 'Multiple claims submitted for the same patient within a short time period (often within 7-14 days)',
        'indicators': 'multiple claims, same patient, short timeframe, different providers',
        'severity': 'medium',
        'privacy_level': PrivacyLevel.AGENT
    },
    {
        'pattern_id': 'FP-002',
        'pattern_name': 'Procedure Upcoding',
        'description': 'Billing for more complex procedures than what was actually performed',
        'indicators': 'expensive procedures, pattern of similar claims, inconsistent with patient history',
        'severity': 'high',
        'privacy_level': PrivacyLevel.AGENT
    },
    {
        'pattern_id': 'FP-003',
        'pattern_name': 'Duplicate Billing',
        'description': 'Multiple claims for the same service on the same date',
        'indicators': 'identical service date, identical procedure, multiple claims',
        'severity': 'high',
        'privacy_level': PrivacyLevel.AGENT
    },
    {
        'pattern_id': 'FP-004',
        'pattern_name': 'Phantom Billing',
        'description': 'Billing for services that were never provided',
        'indicators': 'no supporting documentation, patient denies receiving service',
        'severity': 'critical',
        'privacy_level': PrivacyLevel.AGENT
    },
    {
        'pattern_id': 'FP-005',
        'pattern_name': 'Unusual Procedure Frequency',
        'description': 'Higher than normal frequency of certain procedures',
        'indicators': 'repeated identical procedures, statistically anomalous frequency',
        'severity': 'medium',
        'privacy_level': PrivacyLevel.AGENT
    },
    {
        'pattern_id': 'FP-006',
        'pattern_name': 'Service Date Manipulation',
        'description': 'Altering service dates to avoid coverage limitations or maximize reimbursement',
        'indicators': 'claims near policy expiration, suspicious service date patterns',
        'severity': 'high',
        'privacy_level': PrivacyLevel.AGENT
    },
    {
        'pattern_id': 'FP-007',
        'pattern_name': 'Provider Shopping',
        'description': 'Patient visits multiple providers for the same condition to obtain multiple prescriptions or services',
        'indicators': 'multiple providers, similar services, short timeframe',
        'severity': 'medium',
        'privacy_level': PrivacyLevel.AGENT
    },
]

# STEP 2: Implement a vector-based fraud pattern detector
class FraudPatternDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.fraud_patterns = []
        self.pattern_vectors = None
        self.is_initialized = False
        
    def update_patterns(self, fraud_patterns):
        """Update the fraud patterns database with vector embeddings"""
        self.fraud_patterns = fraud_patterns
        
        # Extract text content for vectorization
        texts = []
        for pattern in fraud_patterns:
            # Combine pattern fields for better text representation
            text = f"{pattern['pattern_name']} {pattern['description']} {pattern['indicators']}"
            texts.append(text)
            
        # Create the vector representations
        if texts:
            self.pattern_vectors = self.vectorizer.fit_transform(texts)
            self.is_initialized = True
        
    def detect_fraud_indicators(self, claim, patient_history=None, access_level=PrivacyLevel.AGENT):
        """
        Detect fraud indicators for a claim using vector similarity and rule-based methods
        
        Args:
            claim: The claim to analyze
            patient_history: Optional list of previous claims for this patient
            access_level: The access level of the requester
            
        Returns:
            Dictionary with fraud indicators and risk assessment
        """
        # Ensure we have initialized patterns
        if not self.is_initialized:
            return {
                'claim_id': claim.id if hasattr(claim, 'id') else 'unknown',
                'error': 'Fraud pattern detector not initialized',
                'fraud_risk': 'unknown'
            }
            
        # Prepare a text representation of the claim for vector similarity
        claim_text = f"Procedure {claim.procedure_code} Amount {claim.amount} Patient {claim.patient_id} Status {claim.status}"
        
        # Add patient history context if available
        if patient_history:
            recent_procedures = []
            for history_claim in patient_history:
                if history_claim['id'] != claim.id:  # Skip the current claim
                    recent_procedures.append(history_claim['procedure_code'])
                    claim_text += f" Recent {history_claim['procedure_code']}"
        
        # Vectorize the claim
        claim_vector = self.vectorizer.transform([claim_text])
        
        # Calculate similarity to known fraud patterns
        similarities = cosine_similarity(claim_vector, self.pattern_vectors).flatten()
        
        # Identify top matching patterns
        matches = []
        for i, score in enumerate(similarities):
            if score > 0.1:  # Threshold for considering it a match
                pattern = self.fraud_patterns[i]
                # Check access control
                if AccessControl.can_access(access_level, pattern['privacy_level']):
                    matches.append({
                        'pattern_id': pattern['pattern_id'],
                        'pattern_name': pattern['pattern_name'],
                        'severity': pattern['severity'],
                        'similarity_score': float(score)
                    })
        
        # Add rule-based fraud indicators
        rule_indicators = self._apply_fraud_rules(claim, patient_history)
        
        # Calculate overall fraud risk
        fraud_risk = self._calculate_fraud_risk(matches, rule_indicators)
        
        return {
            'claim_id': claim.id,
            'matching_patterns': sorted(matches, key=lambda x: x['similarity_score'], reverse=True),
            'rule_based_indicators': rule_indicators,
            'fraud_risk': fraud_risk
        }
    
    def _apply_fraud_rules(self, claim, patient_history=None):
        """Apply rule-based fraud detection"""
        indicators = []
        
        # Check for unusually high amount
        if claim.amount > 400:  # Threshold based on typical claim amounts
            indicators.append({
                'rule': 'high_amount',
                'description': 'Claim amount is significantly higher than average',
                'confidence': 0.6
            })
            
        # Check for duplicate claims if history available
        if patient_history:
            # Look for claims with same procedure code in recent history
            similar_procedure_claims = [
                c for c in patient_history 
                if c['procedure_code'] == claim.procedure_code and c['id'] != claim.id
            ]
            
            if len(similar_procedure_claims) >= 2:
                indicators.append({
                    'rule': 'repeat_procedure',
                    'description': f'Procedure {claim.procedure_code} claimed multiple times recently',
                    'confidence': 0.7
                })
        
        return indicators
    
    def _calculate_fraud_risk(self, pattern_matches, rule_indicators):
        """Calculate overall fraud risk based on matches and rules"""
        if not pattern_matches and not rule_indicators:
            return 'low'
            
        # Calculate pattern-based risk score
        pattern_score = 0
        for match in pattern_matches:
            severity_multiplier = {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8,
                'critical': 1.0
            }.get(match['severity'], 0.5)
            
            pattern_score += match['similarity_score'] * severity_multiplier
        
        # Calculate rule-based risk score
        rule_score = sum(indicator['confidence'] for indicator in rule_indicators)
        
        # Combine scores
        total_score = pattern_score + rule_score
        
        # Classify risk
        if total_score < 0.3:
            return 'low'
        elif total_score < 0.7:
            return 'medium'
        elif total_score < 1.2:
            return 'high'
        else:
            return 'critical'

# Initialize the fraud detector
fraud_detector = FraudPatternDetector()
fraud_detector.update_patterns(fraud_patterns)

# STEP 3: Implement a tool for fraud detection
@tool
def check_claim_for_fraud(claim_id: str, access_level: str = PrivacyLevel.AGENT) -> Dict:
    """
    Check a claim for potential fraud indicators.
    
    Args:
        claim_id: The claim ID to check
        access_level: The access level of the requester
        
    Returns:
        Dictionary containing fraud assessment results
    """
    # Get the claim
    claim_data = database.get_claim(claim_id, access_level)
    if not claim_data:
        return {
            'success': False,
            'error': 'Claim not found or access denied'
        }
    
    # Get the actual claim object
    claim = database.claims[claim_id]
    
    # Get patient history for context
    patient_id = claim.patient_id
    patient_claims = database.get_patient_claims(patient_id, access_level)
    
    # Run fraud detection
    fraud_analysis = fraud_detector.detect_fraud_indicators(claim, patient_claims, access_level)
    
    return {
        'success': True,
        'claim_id': claim_id,
        'fraud_analysis': fraud_analysis,
        'recommendation': _get_fraud_recommendation(fraud_analysis['fraud_risk'])
    }

def _get_fraud_recommendation(risk_level):
    """Generate a recommendation based on fraud risk level"""
    if risk_level == 'low':
        return 'Proceed with normal processing'
    elif risk_level == 'medium':
        return 'Flag for manual review before approving'
    elif risk_level == 'high':
        return 'Requires investigator review before processing'
    elif risk_level == 'critical':
        return 'Suspend claim and initiate fraud investigation'
    else:
        return 'Requires further assessment'

# STEP 4: Create a FraudDetectionAgent
class FraudDetectionAgent(ToolCallingAgent):
    """Agent for detecting potential fraud in insurance claims."""
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                check_claim_for_fraud, 
                get_claim_details,
                get_patient_info,
                retrieve_claim_history,
                search_knowledge_base
            ],
            model=model,
            name='fraud_investigator',
            description="""Agent responsible for detecting potential fraud in insurance claims.
            You have AGENT level access to the database.
            Your main job is to assess claims for potential fraud indicators,
            flag suspicious claims, and provide recommendations.
            Use the fraud detection tools and search for any relevant information
            that might help assess the legitimacy of claims.
            """
        )
        self.access_level = PrivacyLevel.AGENT

# STEP 5: Update the orchestrator to include fraud detection
class EnhancedOrchestrator(ComplaintResolutionOrchestrator):
    """Enhanced orchestrator that includes fraud detection in the workflow."""
    def __init__(self, model: OpenAIServerModel):
        super().__init__(model)
        self.fraud_detector = FraudDetectionAgent(model)
        
        @tool
        def handle_claim_with_fraud_check(claim_data: Dict) -> Dict:
            """
            Process a new claim with integrated fraud detection.
            
            Args:
                claim_data: The claim data to process
                
            Returns:
                Dictionary containing the claim processing result with fraud assessment
            """
            # Step 1: Process the new claim
            process_result = self.claim_processor.run(
                f"""
                Process this new claim:
                Patient ID: {claim_data['patient_id']}
                Service Date: {claim_data['service_date']}
                Procedure Code: {claim_data['procedure_code']}
                Amount: ${claim_data['amount']}
                
                Use the process_new_claim tool.
                """
            )
            
            # Extract claim_id from the result
            claim_id = None
            if hasattr(process_result, 'tool_calls') and process_result.tool_calls:
                for call in process_result.tool_calls:
                    if call.name == 'process_new_claim' and 'claim_id' in call.arguments:
                        claim_id = call.arguments['claim_id']
            
            if not claim_id:
                return {
                    'success': False,
                    'error': 'Failed to process claim'
                }
            
            # Step 2: Run fraud detection on the new claim
            fraud_result = self.fraud_detector.run(
                f"""
                Analyze claim {claim_id} for potential fraud.
                
                First, get details about the claim using get_claim_details tool.
                Then check for fraud indicators using check_claim_for_fraud tool.
                Provide a detailed assessment of any potential fraud risks.
                """
            )
            
            # Extract fraud analysis
            fraud_analysis = {'fraud_risk': 'unknown'}
            if hasattr(fraud_result, 'tool_calls') and fraud_result.tool_calls:
                for call in fraud_result.tool_calls:
                    if call.name == 'check_claim_for_fraud' and 'fraud_analysis' in call.arguments:
                        fraud_analysis = call.arguments['fraud_analysis']
            
            # Get the claim details
            claim = database.get_claim(claim_id, PrivacyLevel.ADMIN)
            
            return {
                'success': True,
                'claim_id': claim_id,
                'claim_status': claim['status'],
                'decision_reason': claim['decision_reason'],
                'fraud_risk': fraud_analysis.get('fraud_risk', 'unknown'),
                'recommendation': _get_fraud_recommendation(fraud_analysis.get('fraud_risk', 'unknown'))
            }
            
        # Add the new tool to the orchestrator
        self.tools.append(handle_claim_with_fraud_check)

# STEP 6: Function to demonstrate the fraud detection capabilities
def demonstrate_fraud_detection():
    """
    Run a demonstration of the fraud detection capabilities.
    """
    # Create orchestrator with fraud detection
    orchestrator = EnhancedOrchestrator(model)
    
    print("Generating a legitimate claim for processing...")
    # Generate a legitimate claim
    legitimate_claim = {
        'patient_id': random.choice(list(database.patients.keys())),
        'service_date': f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        'procedure_code': random.choice(['71020', '81003', '85025']),  # Choose common procedures
        'amount': random.uniform(50, 200)  # Reasonable amount
    }
    
    print(f"Processing legitimate claim: {json.dumps(legitimate_claim, indent=2)}")
    legitimate_result = orchestrator.run(
        f"""
        Process this claim and check for fraud:
        Patient ID: {legitimate_claim['patient_id']}
        Service Date: {legitimate_claim['service_date']}
        Procedure Code: {legitimate_claim['procedure_code']}
        Amount: ${legitimate_claim['amount']:.2f}
        
        Use the handle_claim_with_fraud_check tool.
        """
    )
    
    print("\n" + "="*50 + "\n")
    
    print("Now generating a suspicious claim with fraud indicators...")
    # Generate a suspicious claim (high amount, unusual procedure)
    # Find a patient with existing claims for this example
    patients_with_claims = {
        patient_id: len(patient.claim_ids)
        for patient_id, patient in database.patients.items()
        if patient.claim_ids
    }
    
    if patients_with_claims:
        # Get patient with most claims
        patient_id = max(patients_with_claims.items(), key=lambda x: x[1])[0]
    else:
        # If no patients with claims, just pick a random one
        patient_id = random.choice(list(database.patients.keys()))
    
    suspicious_claim = {
        'patient_id': patient_id,
        'service_date': f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        'procedure_code': '43239',  # Expensive procedure
        'amount': random.uniform(800, 1200)  # Unusually high amount
    }
    
    print(f"Processing suspicious claim: {json.dumps(suspicious_claim, indent=2)}")
    suspicious_result = orchestrator.run(
        f"""
        Process this claim and check for fraud:
        Patient ID: {suspicious_claim['patient_id']}
        Service Date: {suspicious_claim['service_date']}
        Procedure Code: {suspicious_claim['procedure_code']}
        Amount: ${suspicious_claim['amount']:.2f}
        
        Use the handle_claim_with_fraud_check tool.
        """
    )
    
    print("\nFraud detection demonstration completed.")
    return True

if __name__ == '__main__':
    # Initialize and populate database
    print('Initializing and populating database...')
    DataGenerator.populate_database(num_patients=20, num_claims=50, num_complaints=10)
    print(f"Database contains {len(database.patients)} patients, {len(database.claims)} claims, and {len(database.complaints)} complaints")
    
    # Run the fraud detection demo
    print('\n=== Insurance Claim Fraud Detection Demo ===\n')
    demonstrate_fraud_detection()