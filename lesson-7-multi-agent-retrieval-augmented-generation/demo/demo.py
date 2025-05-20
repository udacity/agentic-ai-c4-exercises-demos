from typing import Dict, List, Any, Optional, Union, Set
import random
import time
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

dotenv.load_dotenv(dotenv_path='../.env')
openai_api_key = os.getenv('UDACITY_OPENAI_API_KEY')

model = OpenAIServerModel(
    model_id='gpt-4o-mini',
    api_base='https://openai.vocareum.com/v1',
    api_key=openai_api_key,
)

# Data privacy levels - define who can access what data
class PrivacyLevel:
    PUBLIC = 'public'  # Anyone can access
    CUSTOMER = 'customer'  # Only customer and admins can access
    AGENT = 'agent'  # Only agents and admins can access  
    FINANCIAL = 'financial'  # Only financial dept and admins can access
    ADMIN = 'admin'  # Only admins can access

# Define access control
class AccessControl:
    @staticmethod
    def can_access(requester_level: str, data_level: str) -> bool:
        """Check if the requester has access to the data based on privacy levels."""
        # Admin can access everything
        if requester_level == PrivacyLevel.ADMIN:
            return True
            
        # Same level access is allowed
        if requester_level == data_level:
            return True
            
        # Financial can access agent data
        if requester_level == PrivacyLevel.FINANCIAL and data_level == PrivacyLevel.AGENT:
            return True
            
        # Agent can access customer data
        if requester_level == PrivacyLevel.AGENT and data_level == PrivacyLevel.CUSTOMER:
            return True
            
        # Anyone can access public data
        if data_level == PrivacyLevel.PUBLIC:
            return True
            
        # By default, access is denied
        return False

class Claim:
    def __init__(self, patient_id: int, service_date: str, procedure_code: str, amount: float, 
                 privacy_level: str = PrivacyLevel.AGENT):
        self.patient_id = patient_id
        self.service_date = service_date
        self.procedure_code = procedure_code
        self.amount = amount
        self.status = 'pending'  # pending, approved, denied
        self.decision_reason = ''
        self.id = f'CLM-{random.randint(100000, 999999)}'
        self.timestamp = datetime.now().isoformat()
        self.privacy_level = privacy_level
        self.complaint_history = []

    def __str__(self):
        return f"Claim(id={self.id}, patient_id={self.patient_id}, service_date='{self.service_date}', procedure_code='{self.procedure_code}', amount={self.amount}, status='{self.status}')"

    def to_dict(self, requester_level: str = PrivacyLevel.ADMIN) -> Dict:
        """Return claim data based on requester's access level"""
        base_data = {
            'id': self.id,
            'patient_id': self.patient_id,
            'service_date': self.service_date,
            'status': self.status,
        }
        
        # Add sensitive data only if requester has access
        if AccessControl.can_access(requester_level, self.privacy_level):
            base_data.update({
                'procedure_code': self.procedure_code,
                'amount': self.amount,
                'decision_reason': self.decision_reason,
                'timestamp': self.timestamp
            })
            
        # Add complaint history only for customer service or admin
        if requester_level in [PrivacyLevel.AGENT, PrivacyLevel.ADMIN]:
            base_data['complaint_history'] = self.complaint_history
            
        return base_data
        
    def add_complaint(self, complaint_text: str):
        """Add a complaint to the claim history"""
        self.complaint_history.append({
            'timestamp': datetime.now().isoformat(),
            'text': complaint_text
        })

class PatientRecord:
    def __init__(self, patient_id: int, name: str, policy_number: str, 
                 contact_info: Dict, medical_history: List[Dict] = None,
                 privacy_level: str = PrivacyLevel.CUSTOMER):
        self.patient_id = patient_id
        self.name = name
        self.policy_number = policy_number
        self.contact_info = contact_info
        self.medical_history = medical_history or []
        self.privacy_level = privacy_level
        self.claim_ids = set()  # Associated claim IDs
        
    def to_dict(self, requester_level: str = PrivacyLevel.ADMIN) -> Dict:
        """Return patient data based on requester's access level"""
        # Basic info accessible to customer service
        base_data = {
            'patient_id': self.patient_id,
            'name': self.name,
            'policy_number': self.policy_number,
        }
        
        # Add contact info for customer-level access
        if AccessControl.can_access(requester_level, PrivacyLevel.CUSTOMER):
            base_data['contact_info'] = self.contact_info
            
        # Add medical history only for medical staff or admin
        if AccessControl.can_access(requester_level, PrivacyLevel.AGENT):
            base_data['medical_history'] = self.medical_history
            base_data['claim_ids'] = list(self.claim_ids)
            
        return base_data

class ComplaintRecord:
    def __init__(self, complaint_id: str, patient_id: int, claim_id: str, 
                 description: str, status: str = 'open',
                 privacy_level: str = PrivacyLevel.AGENT):
        self.complaint_id = complaint_id
        self.patient_id = patient_id
        self.claim_id = claim_id
        self.description = description
        self.status = status
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.resolution = ''
        self.privacy_level = privacy_level
        self.response_history = []
        
    def to_dict(self, requester_level: str = PrivacyLevel.ADMIN) -> Dict:
        """Return complaint data based on requester's access level"""
        base_data = {
            'complaint_id': self.complaint_id,
            'patient_id': self.patient_id,
            'claim_id': self.claim_id,
            'status': self.status,
            'created_at': self.created_at,
        }
        
        # Add full details if requester has access
        if AccessControl.can_access(requester_level, self.privacy_level):
            base_data.update({
                'description': self.description,
                'updated_at': self.updated_at,
                'resolution': self.resolution,
                'response_history': self.response_history
            })
            
        return base_data
        
    def add_response(self, response_text: str, responder: str):
        """Add a response to the complaint history"""
        self.response_history.append({
            'timestamp': datetime.now().isoformat(),
            'text': response_text,
            'responder': responder
        })
        self.updated_at = datetime.now().isoformat()
        
    def resolve(self, resolution_text: str):
        """Mark the complaint as resolved"""
        self.status = 'resolved'
        self.resolution = resolution_text
        self.updated_at = datetime.now().isoformat()

class Database:
    def __init__(self):
        self.claims = {}  # claim_id -> Claim
        self.patients = {}  # patient_id -> PatientRecord
        self.complaints = {}  # complaint_id -> ComplaintRecord
        self.procedure_codes = {
            '99214': 'Office visit, established patient',
            '71020': 'Chest X-ray, two views',
            '81003': 'Urinalysis, automated, without microscopy',
            '85025': 'Complete blood count (CBC) with differential',
            '93000': 'Electrocardiogram (ECG)',
            '97110': 'Physical therapy, therapeutic exercises',
            '43239': 'Upper GI endoscopy, biopsy',
            '99283': 'Emergency department visit, moderate severity',
            '11100': 'Biopsy of skin lesion',
            '36415': 'Collection of venous blood by venipuncture'
        }
        
    def add_claim(self, claim: Claim):
        """Add a claim to the database"""
        self.claims[claim.id] = claim
        # Link claim to patient if patient exists
        if claim.patient_id in self.patients:
            self.patients[claim.patient_id].claim_ids.add(claim.id)
    
    def add_patient(self, patient: PatientRecord):
        """Add a patient to the database"""
        self.patients[patient.patient_id] = patient
        
    def add_complaint(self, complaint: ComplaintRecord):
        """Add a complaint to the database"""
        self.complaints[complaint.complaint_id] = complaint
        # Add the complaint to the claim's history
        if complaint.claim_id in self.claims:
            self.claims[complaint.claim_id].add_complaint(complaint.description)
    
    def get_claim(self, claim_id: str, requester_level: str) -> Optional[Dict]:
        """Get a claim if requester has access"""
        if claim_id in self.claims:
            claim = self.claims[claim_id]
            if AccessControl.can_access(requester_level, claim.privacy_level):
                return claim.to_dict(requester_level)
        return None
        
    def get_patient(self, patient_id: int, requester_level: str) -> Optional[Dict]:
        """Get a patient if requester has access"""
        if patient_id in self.patients:
            patient = self.patients[patient_id]
            if AccessControl.can_access(requester_level, patient.privacy_level):
                return patient.to_dict(requester_level)
        return None
        
    def get_complaint(self, complaint_id: str, requester_level: str) -> Optional[Dict]:
        """Get a complaint if requester has access"""
        if complaint_id in self.complaints:
            complaint = self.complaints[complaint_id]
            if AccessControl.can_access(requester_level, complaint.privacy_level):
                return complaint.to_dict(requester_level)
        return None
        
    def get_patient_claims(self, patient_id: int, requester_level: str) -> List[Dict]:
        """Get all claims for a patient if requester has access"""
        result = []
        if patient_id in self.patients:
            patient = self.patients[patient_id]
            if AccessControl.can_access(requester_level, patient.privacy_level):
                for claim_id in patient.claim_ids:
                    claim_data = self.get_claim(claim_id, requester_level)
                    if claim_data:
                        result.append(claim_data)
        return result
        
    def search_claims(self, query: Dict, requester_level: str) -> List[Dict]:
        """Search claims based on criteria if requester has access"""
        results = []
        for claim in self.claims.values():
            # Skip if requester doesn't have access
            if not AccessControl.can_access(requester_level, claim.privacy_level):
                continue
                
            match = True
            for key, value in query.items():
                if hasattr(claim, key) and getattr(claim, key) != value:
                    match = False
                    break
            
            if match:
                results.append(claim.to_dict(requester_level))
                
        return results
        
    def search_similar_claims(self, claim_dict: Dict, requester_level: str, threshold: float = 0.7) -> List[Dict]:
        """Find similar claims using the specified embedding similarity threshold"""
        results = []
        
        # Extract key fields for comparison
        procedure_code = claim_dict.get('procedure_code', '')
        amount = claim_dict.get('amount', 0)
        patient_id = claim_dict.get('patient_id', 0)
        
        for claim in self.claims.values():
            # Skip if requester doesn't have access
            if not AccessControl.can_access(requester_level, claim.privacy_level):
                continue
                
            # Calculate similarity score
            score = 0
            
            # Same procedure code is a strong signal
            if claim.procedure_code == procedure_code:
                score += 0.5
                
            # Similar amount is also relevant
            amount_diff = abs(claim.amount - amount) if amount > 0 else float('inf')
            if amount_diff < 50:
                score += 0.3
            elif amount_diff < 100:
                score += 0.1
                
            # Previous claims for same patient slightly increase relevance
            if claim.patient_id == patient_id:
                score += 0.1
                
            # If score exceeds threshold, add to results
            if score >= threshold:
                results.append({
                    'claim': claim.to_dict(requester_level),
                    'similarity_score': score
                })
                
        # Sort by similarity score, descending
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)

# Vector embedding for knowledge base
class VectorKnowledgeBase:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.kb_entries = []
        self.kb_vectors = None
        
    def add_entries(self, entries):
        """Add multiple entries to the knowledge base"""
        self.kb_entries = entries
        # Extract text content for vectorization
        texts = [f"{entry['topic']} {entry['content']}" for entry in entries]
        # Create the vector representations
        self.kb_vectors = self.vectorizer.fit_transform(texts)
    
    def search(self, query, access_level, threshold=0.15):
        """Search knowledge base with vector similarity"""
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, self.kb_vectors).flatten()
        
        # Filter results by similarity threshold and access level
        results = []
        for i, score in enumerate(similarities):
            entry = self.kb_entries[i]
            if score >= threshold and AccessControl.can_access(access_level, entry['privacy_level']):
                results.append({
                    **entry,
                    'similarity_score': float(score)
                })
                
        # Sort by similarity score (descending)
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)

# Vector embedding for claims
class VectorClaimSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.claims = []
        self.claim_vectors = None
        self.is_initialized = False
        
    def update_claims(self, claims_list):
        """Update the vectorized claims database"""
        self.claims = claims_list
        # Extract text content for vectorization by combining relevant fields
        texts = []
        for claim in claims_list:
            # Combine procedure code with decision reason and other text fields
            text = f"{claim.procedure_code} {database.procedure_codes.get(claim.procedure_code, '')} {claim.decision_reason}"
            # Add complaint text if available
            if claim.complaint_history:
                for complaint in claim.complaint_history:
                    text += f" {complaint.get('text', '')}"
            texts.append(text)
            
        # Create the vector representations if we have data
        if texts:
            self.claim_vectors = self.vectorizer.fit_transform(texts)
            self.is_initialized = True
            
    def search(self, query_claim, requester_level, threshold=0.2):
        """Search claims with vector similarity"""
        # Lazy initialization if needed
        if not self.is_initialized:
            self.update_claims(list(database.claims.values()))
            
        # Prepare query text
        procedure_desc = database.procedure_codes.get(query_claim.get('procedure_code', ''), '')
        query_text = f"{query_claim.get('procedure_code', '')} {procedure_desc}"
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query_text])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, self.claim_vectors).flatten()
        
        # Filter results by similarity threshold and access level
        results = []
        for i, score in enumerate(similarities):
            claim = self.claims[i]
            # Skip if requester doesn't have access
            if not AccessControl.can_access(requester_level, claim.privacy_level):
                continue
                
            if score >= threshold:
                # Add original rules to boost similarity when appropriate
                boost = 0
                
                # Same procedure code is a strong signal
                if claim.procedure_code == query_claim.get('procedure_code', ''):
                    boost += 0.2
                    
                # Similar amount is also relevant
                if 'amount' in query_claim and query_claim['amount'] > 0:
                    amount_diff = abs(claim.amount - float(query_claim['amount']))
                    if amount_diff < 50:
                        boost += 0.1
                    elif amount_diff < 100:
                        boost += 0.05
                    
                # Previous claims for same patient slightly increase relevance
                if claim.patient_id == query_claim.get('patient_id', 0):
                    boost += 0.1
                
                # Combine similarity score with boost
                final_score = min(1.0, score + boost)
                
                results.append({
                    'claim': claim.to_dict(requester_level),
                    'similarity_score': float(final_score)
                })
                
        # Sort by similarity score, descending
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)

# Initialize vector claim search
vector_claim_search = VectorClaimSearch()

# Initialize the vector knowledge base
vector_kb = VectorKnowledgeBase()

# Mock data generator
class DataGenerator:
    @staticmethod
    def generate_claim() -> Claim:
        """Generate a random claim"""
        patient_id = random.randint(1000, 2000)
        service_date = '2024-07-' + str(random.randint(10, 28)).zfill(2)
        procedure_code = random.choice(list(database.procedure_codes.keys()))
        amount = round(random.uniform(50.00, 500.00), 2)
        return Claim(patient_id, service_date, procedure_code, amount)
        
    @staticmethod
    def generate_patient(patient_id: int = None) -> PatientRecord:
        """Generate a random patient record"""
        if patient_id is None:
            patient_id = random.randint(1000, 2000)
            
        first_names = ['James', 'Mary', 'Robert', 'Patricia', 'John', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        policy_number = f"POL-{random.randint(10000, 99999)}"
        
        contact_info = {
            'email': f"{name.lower().replace(' ', '.')}@example.com",
            'phone': f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            'address': f"{random.randint(100, 999)} Main St, Anytown, ST {random.randint(10000, 99999)}"
        }
        
        return PatientRecord(patient_id, name, policy_number, contact_info)
        
    @staticmethod
    def generate_complaint(claim: Claim) -> ComplaintRecord:
        """Generate a random complaint for a claim"""
        complaint_reasons = [
            'The claim was denied without proper explanation.',
            'I was told this procedure would be covered by my insurance.',
            'The amount charged is much higher than what was quoted to me.',
            'I\'ve been waiting for resolution for over 30 days.',
            'I was billed for services I didn\'t receive.',
            'My doctor says this procedure is medically necessary but it was denied.',
            'I\'ve paid my deductible already, why am I being charged?',
            'Similar claims have been approved for me in the past.',
            'The explanation of benefits is confusing and unclear.',
            'I was not notified that this provider was out-of-network.'
        ]
        
        complaint_id = f"CMPL-{random.randint(1000, 9999)}"
        description = random.choice(complaint_reasons)
        
        return ComplaintRecord(
            complaint_id=complaint_id,
            patient_id=claim.patient_id,
            claim_id=claim.id,
            description=description
        )
        
    @staticmethod
    def populate_database(num_patients: int = 50, num_claims: int = 200, num_complaints: int = 40):
        """Populate the database with random data"""
        # Generate patients
        for _ in range(num_patients):
            patient = DataGenerator.generate_patient()
            database.add_patient(patient)
            
        patient_ids = list(database.patients.keys())
        
        # Generate claims
        for _ in range(num_claims):
            # Assign to a random existing patient
            patient_id = random.choice(patient_ids)
            
            claim = DataGenerator.generate_claim()
            claim.patient_id = patient_id
            
            # Set random status (with 2% approval rate)
            if random.random() < 0.02:  # 2% approval rate
                claim.status = 'approved'
                claim.decision_reason = 'Meets coverage criteria.'
            else:
                claim.status = 'denied'
                claim.decision_reason = random.choice([
                    'Service not covered under current plan.',
                    'Insufficient documentation provided.',
                    'Exceeds coverage limits.',
                    'Experimental/investigational procedure.',
                    'Pre-authorization required but not obtained.',
                    'Out-of-network provider.',
                    'Non-medically necessary.',
                    'Duplicate claim.',
                    'Patient not eligible on date of service.',
                    'Covered by another insurance plan.'
                ])
                
            database.add_claim(claim)
            
        # Generate complaints for some denied claims
        denied_claims = [c for c in database.claims.values() if c.status == 'denied']
        
        for _ in range(min(num_complaints, len(denied_claims))):
            claim = random.choice(denied_claims)
            complaint = DataGenerator.generate_complaint(claim)
            database.add_complaint(complaint)
            
            # Generate a few responses for some complaints
            if random.random() < 0.7:
                responses = [
                    'We have received your complaint and are reviewing it.',
                    'We need additional information to process your request.',
                    'We\'re consulting with the medical review team.',
                    'Your case has been escalated to a supervisor.'
                ]
                
                num_responses = random.randint(1, 3)
                for i in range(num_responses):
                    complaint.add_response(random.choice(responses), 'Customer Service Rep')
                    
                # Some complaints get resolved
                if random.random() < 0.3:
                    resolutions = [
                        'After review, we\'ve decided to approve your claim.',
                        'We\'ll process a partial payment for covered services.',
                        'We\'re standing by our initial determination.',
                        'We apologize for the confusion and have corrected the error.'
                    ]
                    complaint.resolve(random.choice(resolutions))

# Initialize global database
database = Database()

# RAG Tools
@tool
def search_knowledge_base(query: str, access_level: str = PrivacyLevel.AGENT) -> Dict:
    """
    Search the knowledge base for information about insurance policies, procedures, etc.
    using vector embedding similarity.
    
    Args:
        query: The search query string
        access_level: The access level of the requester
        
    Returns:
        Dictionary containing search results with similarity scores
    """
    # Knowledge base entries
    kb = [
        {
            'topic': 'claim denial',
            'content': 'Claims may be denied for various reasons including: service not covered under plan, insufficient documentation, exceeding coverage limits, experimental procedures, lack of pre-authorization, out-of-network providers, non-medical necessity, duplicate claims, eligibility issues, or coverage by another plan.',
            'privacy_level': PrivacyLevel.PUBLIC
        },
        {
            'topic': 'appeal process',
            'content': 'Patients have the right to appeal claim denials. The process involves submitting a written request with supporting documentation within 60 days of the denial. Appeals are reviewed by medical professionals not involved in the initial decision.',
            'privacy_level': PrivacyLevel.PUBLIC
        },
        {
            'topic': 'coverage verification',
            'content': 'Agents should verify patient eligibility and coverage before providing definitive information. This includes checking effective dates, plan limitations, deductibles, and coordination of benefits with other insurers.',
            'privacy_level': PrivacyLevel.AGENT
        },
        {
            'topic': 'payment policies',
            'content': 'Standard processing time for approved claims is 30 days. Electronic payments are issued weekly. Paper checks are issued bi-weekly. Payments under $5 are held until the cumulative amount exceeds the minimum threshold.',
            'privacy_level': PrivacyLevel.FINANCIAL
        },
        {
            'topic': 'procedure codes',
            'content': json.dumps(database.procedure_codes),
            'privacy_level': PrivacyLevel.AGENT
        },
        {
            'topic': 'insurance policy limitations',
            'content': 'All insurance policies have coverage limitations and exclusions. Common limitations include annual maximums, lifetime maximums for certain procedures, waiting periods for specific treatments, and exclusions for pre-existing conditions.',
            'privacy_level': PrivacyLevel.PUBLIC
        },
        {
            'topic': 'out-of-network coverage',
            'content': 'When patients receive care from providers outside their network, they typically face higher out-of-pocket costs. Some plans offer no coverage for out-of-network care except in emergencies, while others cover a percentage but with higher deductibles.',
            'privacy_level': PrivacyLevel.PUBLIC
        },
        {
            'topic': 'claim resubmission',
            'content': 'Denied claims can be resubmitted with additional documentation within 90 days of the initial denial. All resubmissions must include the original claim number and clearly indicate what additional information is being provided.',
            'privacy_level': PrivacyLevel.AGENT
        }
    ]
    
    # Ensure vector KB is initialized
    if not vector_kb.kb_entries:
        vector_kb.add_entries(kb)
    
    # Perform vector search
    results = vector_kb.search(query, access_level)
    
    return {
        'query': query,
        'results_count': len(results),
        'results': results
    }

@tool
def retrieve_claim_history(patient_id: int, access_level: str = PrivacyLevel.AGENT) -> Dict:
    """
    Retrieve claim history for a patient.
    
    Args:
        patient_id: The patient ID to retrieve claims for
        access_level: The access level of the requester
        
    Returns:
        Dictionary containing patient claims
    """
    claims = database.get_patient_claims(patient_id, access_level)
    
    return {
        'patient_id': patient_id,
        'claims_count': len(claims),
        'claims': claims
    }

@tool
def get_claim_details(claim_id: str, access_level: str = PrivacyLevel.AGENT) -> Dict:
    """
    Get detailed information about a specific claim.
    
    Args:
        claim_id: The claim ID to retrieve
        access_level: The access level of the requester
        
    Returns:
        Dictionary containing claim details or error message
    """
    claim = database.get_claim(claim_id, access_level)
    
    if claim:
        return {
            'success': True,
            'claim': claim
        }
    else:
        return {
            'success': False,
            'error': 'Claim not found or access denied'
        }

@tool
def get_patient_info(patient_id: int, access_level: str = PrivacyLevel.AGENT) -> Dict:
    """
    Get patient information.
    
    Args:
        patient_id: The patient ID to retrieve
        access_level: The access level of the requester
        
    Returns:
        Dictionary containing patient details or error message
    """
    patient = database.get_patient(patient_id, access_level)
    
    if patient:
        return {
            'success': True,
            'patient': patient
        }
    else:
        return {
            'success': False,
            'error': 'Patient not found or access denied'
        }

@tool
def find_similar_claims(claim: Dict, access_level: str = PrivacyLevel.AGENT) -> Dict:
    """
    Find similar claims in the database using vector embedding similarity.
    
    Args:
        claim: The claim to find similar claims for
        access_level: The access level of the requester
        
    Returns:
        Dictionary containing similar claims with similarity scores
    """
    # Ensure vector claim search is initialized with the latest claims
    if not vector_claim_search.is_initialized:
        vector_claim_search.update_claims(list(database.claims.values()))
    
    # Perform vector-based semantic search
    similar = vector_claim_search.search(claim, access_level)
    
    return {
        'query_claim': claim,
        'results_count': len(similar),
        'similar_claims': similar
    }

@tool
def submit_complaint(patient_id: int, claim_id: str, description: str) -> Dict:
    """
    Submit a new complaint about a claim.
    
    Args:
        patient_id: The patient ID
        claim_id: The claim ID being complained about
        description: Description of the complaint
        
    Returns:
        Dictionary containing the result of the complaint submission
    """
    # Verify claim exists
    claim = database.get_claim(claim_id, PrivacyLevel.ADMIN)
    if not claim:
        return {
            'success': False,
            'error': 'Claim not found'
        }
        
    # Create complaint
    complaint_id = f"CMPL-{random.randint(1000, 9999)}"
    complaint = ComplaintRecord(
        complaint_id=complaint_id,
        patient_id=patient_id,
        claim_id=claim_id,
        description=description
    )
    
    database.add_complaint(complaint)
    
    return {
        'success': True,
        'message': 'Complaint submitted successfully',
        'complaint_id': complaint_id
    }

@tool
def respond_to_complaint(complaint_id: str, response: str, responder: str, resolve: bool = False) -> Dict:
    """
    Respond to a complaint.
    
    Args:
        complaint_id: The complaint ID
        response: Text of the response
        responder: Name/role of the person responding
        resolve: Whether to mark the complaint as resolved
        
    Returns:
        Dictionary containing the result of the response submission
    """
    # Get complaint with admin access to ensure we find it
    complaint_data = database.get_complaint(complaint_id, PrivacyLevel.ADMIN)
    if not complaint_data:
        return {
            'success': False,
            'error': 'Complaint not found'
        }
        
    complaint = database.complaints[complaint_id]
    complaint.add_response(response, responder)
    
    if resolve:
        complaint.resolve(response)
        
    return {
        'success': True,
        'message': 'Response added successfully',
        'complaint_status': complaint.status
    }

@tool
def get_complaint_history(complaint_id: str, access_level: str = PrivacyLevel.AGENT) -> Dict:
    """
    Get history of a complaint including all responses.
    
    Args:
        complaint_id: The complaint ID
        access_level: The access level of the requester
        
    Returns:
        Dictionary containing complaint history
    """
    complaint = database.get_complaint(complaint_id, access_level)
    
    if complaint:
        return {
            'success': True,
            'complaint': complaint
        }
    else:
        return {
            'success': False,
            'error': 'Complaint not found or access denied'
        }

@tool
def process_new_claim(claim_data: Dict) -> Dict:
    """
    Process a new health insurance claim.
    
    Args:
        claim_data: The claim data to process
        
    Returns:
        Dictionary containing the processing result
    """
    # Create a new claim object
    claim = Claim(
        patient_id=claim_data['patient_id'],
        service_date=claim_data['service_date'],
        procedure_code=claim_data['procedure_code'],
        amount=claim_data['amount']
    )
    
    # Use the 2% approval rate
    is_approved = random.random() < 0.02
    
    if is_approved:
        claim.status = 'approved'
        claim.decision_reason = 'Meets coverage criteria.'
        result_message = f"Claim {claim.id} approved. Amount: ${claim.amount:.2f}"
    else:
        claim.status = 'denied'
        claim.decision_reason = random.choice([
            'Service not covered under current plan.',
            'Insufficient documentation provided.',
            'Exceeds coverage limits.',
            'Experimental/investigational procedure.',
            'Pre-authorization required but not obtained.',
            'Out-of-network provider.',
            'Non-medically necessary.',
            'Duplicate claim.',
            'Patient not eligible on date of service.',
            'Covered by another insurance plan.'
        ])
        result_message = f"Claim {claim.id} denied. Reason: {claim.decision_reason}"
    
    # Add claim to database
    database.add_claim(claim)
    
    return {
        'success': True,
        'claim_id': claim.id,
        'status': claim.status,
        'decision_reason': claim.decision_reason,
        'message': result_message
    }

# Agent classes with RAG capabilities
class ClaimProcessingAgent(ToolCallingAgent):
    """Agent for processing health insurance claims with RAG capabilities."""
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[process_new_claim, find_similar_claims, get_claim_details, search_knowledge_base],
            model=model,
            name='claim_processor',
            description="""Agent responsible for processing health insurance claims.
            You have access to the claims database but can only access data at the AGENT privacy level.
            Always search for similar claims and reference knowledge base when making decisions.
            Remember that most claims (98%) should be denied based on company policy.
            """,
        )
        self.access_level = PrivacyLevel.AGENT

class CustomerServiceAgent(ToolCallingAgent):
    """Agent for handling customer inquiries and complaints with RAG capabilities."""
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                get_patient_info, 
                get_claim_details,
                submit_complaint,
                get_complaint_history,
                respond_to_complaint,
                search_knowledge_base,
                retrieve_claim_history
            ],
            model=model,
            name='customer_service',
            description="""Agent responsible for handling customer inquiries and complaints.
            You have CUSTOMER level access to the database.
            You can access basic patient info and their claims, submit complaints on their behalf,
            and provide information about insurance policies.
            Always be empathetic and helpful, especially when dealing with denied claims.
            """,
        )
        self.access_level = PrivacyLevel.CUSTOMER

class MedicalReviewAgent(ToolCallingAgent):
    """Agent for medical review of claims with RAG capabilities."""
    def __init__(self, model: OpenAIServerModel):
        super().__init__(
            tools=[
                get_claim_details,
                get_patient_info,
                find_similar_claims,
                search_knowledge_base,
                respond_to_complaint
            ],
            model=model,
            name='medical_reviewer',
            description="""Agent responsible for medical review of claims.
            You have AGENT level access to the database.
            You focus on reviewing denied claims that have received complaints,
            and can provide medical justification for decisions.
            Use the knowledge base and similar claims history to inform your decisions.
            """,
        )
        self.access_level = PrivacyLevel.AGENT

class ComplaintResolutionOrchestrator(ToolCallingAgent):
    """Orchestrates the complaint resolution workflow using RAG."""
    def __init__(self, model: OpenAIServerModel):
        self.model = model
        self.customer_service = CustomerServiceAgent(model)
        self.medical_reviewer = MedicalReviewAgent(model)
        self.claim_processor = ClaimProcessingAgent(model)

        @tool
        def handle_customer_complaint(patient_id: int, complaint_text: str, claim_id: str = None) -> Dict:
            """
            Handle a customer complaint about a claim.
            
            Args:
                patient_id: The patient ID making the complaint
                complaint_text: The text of the complaint
                claim_id: Optional claim ID if known
                
            Returns:
                Dictionary containing the complaint handling result
            """
            # Step 1: If claim_id not provided, try to find relevant claims
            if not claim_id:
                claims_result = self.customer_service.run(
                    f"""
                    We've received a complaint from patient ID {patient_id}. 
                    The complaint says: "{complaint_text}"
                    
                    First, retrieve the patient's claim history using retrieve_claim_history tool.
                    Then identify which claim they're referring to.
                    """
                )
                
                # Parse the response to extract the claim_id using text content from the agent
                text_content = str(claims_result)
                if 'CLM-' in text_content:
                    start_idx = text_content.find('CLM-')
                    end_idx = start_idx + 10  # CLM- plus 6 digits
                    potential_claim_id = text_content[start_idx:end_idx]
                    if potential_claim_id.startswith('CLM-') and len(potential_claim_id) >= 9:
                        claim_id = potential_claim_id
            
            # If we still don't have a claim_id, we can't proceed
            if not claim_id:
                return {
                    'success': False,
                    'error': 'Could not identify which claim the complaint is about',
                    'recommendation': 'Please provide a specific claim ID'
                }
            
            # Step 2: Submit the complaint
            complaint_result = self.customer_service.run(
                f"""
                Submit a complaint for patient {patient_id} about claim {claim_id}.
                The complaint says: "{complaint_text}"
                
                Use the submit_complaint tool.
                """
            )
            
            # Extract complaint_id from the result
            complaint_id = None
            if hasattr(complaint_result, 'tool_calls') and complaint_result.tool_calls:
                for call in complaint_result.tool_calls:
                    if call.name == 'submit_complaint' and 'complaint_id' in call.arguments:
                        complaint_id = call.arguments['complaint_id']
            
            # If no complaint_id in tool_calls, try to extract from the text
            if not complaint_id:
                text_content = str(complaint_result)
                if 'CMPL-' in text_content:
                    start_idx = text_content.find('CMPL-')
                    end_idx = start_idx + 9  # CMPL- plus 4 digits
                    potential_id = text_content[start_idx:end_idx]
                    if potential_id.startswith('CMPL-'):
                        complaint_id = potential_id
            
            # Still no complaint_id, we really can't proceed
            if not complaint_id:
                return {
                    'success': False,
                    'error': 'Failed to register complaint',
                    'message': str(complaint_result)
                }
            
            # Step 3: Get medical review
            medical_review = self.medical_reviewer.run(
                f"""
                Review complaint {complaint_id} about claim {claim_id} for patient {patient_id}.
                The complaint says: "{complaint_text}"
                
                First, get claim details using get_claim_details tool.
                Then search for similar claims using find_similar_claims tool.
                Also search knowledge_base for relevant information.
                
                Based on your analysis, provide a response to the complaint.
                Use respond_to_complaint tool to record your response.
                """
            )
            
            # Extract medical review info for reference
            medical_response = ''
            if hasattr(medical_review, 'tool_calls') and medical_review.tool_calls:
                for call in medical_review.tool_calls:
                    if call.name == 'respond_to_complaint' and 'response' in call.arguments:
                        medical_response = call.arguments['response']
            
            # Step 4: Get customer service to provide a final response
            final_response = self.customer_service.run(
                f"""
                Provide a final response to complaint {complaint_id} about claim {claim_id}.
                
                First, get the complaint history using get_complaint_history tool.
                Consider the medical review already provided.
                
                Provide a final response that is empathetic and clear about the decision.
                Use respond_to_complaint tool with resolve=true to finish handling this complaint.
                """
            )
            
            # Extract final decision and message
            resolution = ''
            if hasattr(final_response, 'tool_calls') and final_response.tool_calls:
                for call in final_response.tool_calls:
                    if call.name == 'respond_to_complaint' and 'response' in call.arguments:
                        resolution = call.arguments['response']
            
            # If we still don't have a resolution, check the complaint in the database
            if not resolution:
                complaint_info = database.get_complaint(complaint_id, PrivacyLevel.ADMIN)
                if complaint_info and complaint_info.get('status') == 'resolved':
                    resolution = complaint_info.get('resolution', '')
            
            # If still no resolution, extract from final_response text
            if not resolution:
                text_content = str(final_response)
                if 'apologize' in text_content.lower() or 'review' in text_content.lower():
                    # Extract a portion of the response text as the resolution
                    sentences = text_content.split('.')
                    if len(sentences) > 1:
                        resolution = '. '.join(sentences[:3]) + '.'
            
            # If still no resolution, use a default message
            if not resolution:
                resolution = 'We have reviewed your complaint and our decision is to uphold the original claim determination based on your policy coverage.'
            
            return {
                'success': True,
                'complaint_id': complaint_id,
                'claim_id': claim_id,
                'resolution': resolution
            }

        @tool
        def generate_random_complaint() -> Dict:
            """Generate a random customer complaint for demonstration purposes."""
            # Get a random denied claim
            denied_claims = [c for c in database.claims.values() if c.status == 'denied']
            if not denied_claims:
                return {
                    'success': False,
                    'error': 'No denied claims in database'
                }
                
            claim = random.choice(denied_claims)
            
            # Generate a complaint
            complaint_reasons = [
                f"I don't understand why my claim {claim.id} was denied. Can you please explain?",
                f"I believe my claim {claim.id} was denied in error. My doctor says this is a covered procedure.",
                f"This is outrageous! How can you deny claim {claim.id} when I've been paying premiums for years?",
                f"I was told procedure {claim.procedure_code} would be covered by my insurance, but claim {claim.id} was denied.",
                f"I need help with claim {claim.id}. I can't afford to pay this out of pocket and I don't understand the denial."
            ]
            
            complaint_text = random.choice(complaint_reasons)
            
            return {
                'success': True,
                'patient_id': claim.patient_id,
                'claim_id': claim.id,
                'complaint_text': complaint_text
            }

        super().__init__(
            tools=[handle_customer_complaint, generate_random_complaint],
            model=model,
            name='orchestrator',
            description="""You are an orchestrator that manages the insurance complaint resolution workflow.
            You coordinate between customer service, medical review, and claim processing agents.
            Your focus is on handling the 98% of claims that get denied and resolving customer complaints efficiently.
            """,
        )

def run_demo():
    # Initialize and populate database
    print('Initializing and populating database...')
    DataGenerator.populate_database(num_patients=20, num_claims=50, num_complaints=10)
    print(f"Database contains {len(database.patients)} patients, {len(database.claims)} claims, and {len(database.complaints)} complaints")
    
    # Create orchestrator
    orchestrator = ComplaintResolutionOrchestrator(model)
    
    # Run the complaint handling demo
    print('\n=== Insurance Claim Complaint Resolution Demo ===\n')
    
    # Generate a random complaint
    print('Generating a random complaint...')
    complaint_result = orchestrator.run(
        'Generate a random complaint for us to handle in this demo.'
    )
    
    # Extract complaint data from the result
    complaint_data = None
    if hasattr(complaint_result, 'tool_calls') and complaint_result.tool_calls:
        for call in complaint_result.tool_calls:
            if call.name == 'generate_random_complaint':
                complaint_data = call.arguments
    
    if not complaint_data or not complaint_data.get('success', False):
        print('Failed to generate a valid complaint. Demo cannot continue.')
        return
    
    print(f"Generated complaint: {complaint_data['complaint_text']}")
    print(f"For patient ID: {complaint_data['patient_id']}")
    print(f"About claim ID: {complaint_data['claim_id']}")
    
    # Handle the complaint
    print('\nNow handling the complaint through our agentic RAG system...')
    resolution_result = orchestrator.run(
        f"""
        Handle this customer complaint:
        - Patient ID: {complaint_data['patient_id']}
        - Claim ID: {complaint_data['claim_id']}
        - Complaint: "{complaint_data['complaint_text']}"
        
        Use the handle_customer_complaint tool.
        """
    )
    
    # Extract resolution info from tool calls
    resolution_info = None
    if hasattr(resolution_result, 'tool_calls') and resolution_result.tool_calls:
        for call in resolution_result.tool_calls:
            if call.name == 'handle_customer_complaint':
                resolution_info = call.arguments
                break
    
    # Check if we got a successful resolution
    if resolution_info and resolution_info.get('success', False):
        complaint_id = resolution_info.get('complaint_id', 'Unknown')
        resolution_text = resolution_info.get('resolution', 'No specific resolution provided')
        
        print('\n=== Final Resolution ===\n')
        print(f"Complaint ID: {complaint_id}")
        print(f"Claim ID: {resolution_info.get('claim_id')}")
        print('\nResolution:')
        print(resolution_text)
        return True  # Return success
    else:
        # Extract error information if available
        error_message = 'Unknown error occurred during complaint handling'
        if resolution_info and 'error' in resolution_info:
            error_message = resolution_info['error']
            
        print('\n=== Complaint Handling Failed ===\n')
        print(f"Error: {error_message}")
        
        # If we have a recommendation, display it
        if resolution_info and 'recommendation' in resolution_info:
            print(f"Recommendation: {resolution_info['recommendation']}")
        return False  # Return failure

if __name__ == '__main__':
    demo_result = run_demo()
    if demo_result is not False:  # Only show this message if the demo didn't explicitly fail
        print('\nDemo completed successfully! Thank you for trying the Multi-Agent RAG System.')
