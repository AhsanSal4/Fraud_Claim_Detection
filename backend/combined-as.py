import os
import json
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import uuid
import hashlib

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================
# DATA CLASSES FOR STRUCTURED DATA
# =====================================================

@dataclass
class ClaimData:
    """Structured claim data"""
    claim_id: str
    months_as_customer: int
    age: int
    policy_number: str
    policy_bind_date: str
    policy_state: str
    policy_csl: str
    policy_deductable: float
    policy_annual_premium: float
    umbrella_limit: float
    insured_zip: int
    insured_sex: str
    insured_education_level: str
    insured_occupation: str
    insured_hobbies: str
    insured_relationship: str
    capital_gains: float
    capital_loss: float
    incident_date: str
    incident_type: str
    collision_type: str
    incident_severity: str
    authorities_contacted: str
    incident_state: str
    incident_city: str
    incident_location: str
    incident_hour_of_the_day: int
    number_of_vehicles_involved: int
    property_damage: str
    bodily_injuries: int
    witnesses: int
    police_report_available: str
    total_claim_amount: float
    injury_claim: float
    property_claim: float
    vehicle_claim: float
    auto_make: str
    auto_model: str
    auto_year: int
    fraud_reported: str = "N"
    created_at: str = None
    updated_at: str = None

    def __post_init__(self):
        if not self.claim_id:
            self.claim_id = self.generate_claim_id()
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

    def generate_claim_id(self) -> str:
        """Generate unique claim ID"""
        data_string = f"{self.policy_number}{self.incident_date}{self.total_claim_amount}"
        hash_object = hashlib.md5(data_string.encode())
        return f"CLAIM_{hash_object.hexdigest()[:8].upper()}"

@dataclass
class FraudAnalysisResult:
    """Structured fraud analysis result"""
    claim_id: str
    rule_based_score: float
    catboost_probability: float
    combined_score: float
    ai_fraud_score: float
    explanation: str
    action: str
    follow_up_questions: List[str]
    risk_level: str
    reasons: List[str]
    analysis_timestamp: str = None
    processing_time_ms: float = None

    def __post_init__(self):
        if not self.analysis_timestamp:
            self.analysis_timestamp = datetime.now().isoformat()

# =====================================================
# FIREBASE MANAGER
# =====================================================

class FirebaseManager:
    """Handles all Firebase operations"""
    
    def __init__(self):
        self.db = None
        self.initialize_firebase()

    def initialize_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase app is already initialized
            if not firebase_admin._apps:
                cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH', 'insurance-fraud-detectio-a6526-firebase-adminsdk-fbsvc-9a0f74002a.json')
                
                if not os.path.exists(cred_path):
                    raise FileNotFoundError(f"Firebase credentials file not found: {cred_path}")
                
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': os.getenv('FIREBASE_PROJECT_ID', 'insurance-fraud-detection')
                })
            
            self.db = firestore.client()
            logger.info("✅ Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Firebase initialization failed: {str(e)}")
            raise

    def save_claim(self, claim_data: ClaimData) -> bool:
        """Save claim data to Firestore"""
        try:
            claim_dict = asdict(claim_data)
            doc_ref = self.db.collection('claims').document(claim_data.claim_id)
            doc_ref.set(claim_dict)
            logger.info(f"✅ Claim saved: {claim_data.claim_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving claim: {str(e)}")
            return False

    def save_analysis_result(self, result: FraudAnalysisResult) -> bool:
        """Save fraud analysis result to Firestore"""
        try:
            result_dict = asdict(result)
            doc_ref = self.db.collection('fraud_analyses').document(result.claim_id)
            doc_ref.set(result_dict)
            logger.info(f"✅ Analysis result saved: {result.claim_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving analysis result: {str(e)}")
            return False

    def get_claim(self, claim_id: str) -> Optional[Dict]:
        """Retrieve claim data from Firestore"""
        try:
            doc_ref = self.db.collection('claims').document(claim_id)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"❌ Error retrieving claim: {str(e)}")
            return None

    def get_analysis_result(self, claim_id: str) -> Optional[Dict]:
        """Retrieve analysis result from Firestore"""
        try:
            doc_ref = self.db.collection('fraud_analyses').document(claim_id)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            logger.error(f"❌ Error retrieving analysis result: {str(e)}")
            return None

    def get_claims_by_policy(self, policy_number: str) -> List[Dict]:
        """Get all claims for a specific policy"""
        try:
            claims_ref = self.db.collection('claims')
            query = claims_ref.where('policy_number', '==', policy_number)
            docs = query.stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"❌ Error retrieving claims by policy: {str(e)}")
            return []

    def get_high_risk_claims(self, threshold: float = 70.0) -> List[Dict]:
        """Get all high-risk claims above threshold"""
        try:
            analyses_ref = self.db.collection('fraud_analyses')
            query = analyses_ref.where('combined_score', '>=', threshold)
            docs = query.stream()
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.error(f"❌ Error retrieving high-risk claims: {str(e)}")
            return []

    def update_claim_status(self, claim_id: str, status: str) -> bool:
        """Update claim status (e.g., 'approved', 'rejected', 'under_investigation')"""
        try:
            doc_ref = self.db.collection('claims').document(claim_id)
            doc_ref.update({
                'status': status,
                'updated_at': datetime.now().isoformat()
            })
            logger.info(f"✅ Claim status updated: {claim_id} -> {status}")
            return True
        except Exception as e:
            logger.error(f"❌ Error updating claim status: {str(e)}")
            return False

# =====================================================
# IMPROVED FRAUD DETECTION SYSTEM
# =====================================================

class ImprovedFraudDetectionSystem:
    """Enhanced fraud detection system with Firebase integration"""
    
    def __init__(self):
        self.firebase = FirebaseManager()
        self.catboost_model = None
        self.categorical_features = None
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.load_models()

    def load_models(self):
        """Load pre-trained models"""
        try:
            self.catboost_model = joblib.load("models/catboost_model.pkl")
            self.categorical_features = joblib.load("models/categorical_features.pkl")
            logger.info("✅ Models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading models: {str(e)}")
            raise

    def preprocess_input(self, user_df: pd.DataFrame) -> pd.DataFrame:
        """Safe preprocessing for CatBoost input"""
        user_df = user_df.copy()
        
        # Ensure all features exist
        for col in self.catboost_model.feature_names_:
            if col not in user_df.columns:
                user_df[col] = 'Unknown' if col in self.categorical_features else 0
        
        # Keep only required features in order
        user_df = user_df[self.catboost_model.feature_names_]
        
        # Fill missing values
        for col in self.categorical_features:
            user_df.loc[:, col] = user_df[col].fillna('Unknown').astype(str)
        for col in user_df.columns:
            if col not in self.categorical_features:
                user_df.loc[:, col] = user_df[col].fillna(user_df[col].median())
        
        return user_df

    def get_catboost_prediction(self, user_df: pd.DataFrame) -> Dict:
        """Get fraud prediction from CatBoost model"""
        try:
            X_processed = self.preprocess_input(user_df)
            prob = self.catboost_model.predict_proba(X_processed)[:, 1][0]
            pred = 'y' if prob >= 0.5 else 'n'
            return {
                "fraud_prediction": pred,
                "fraud_probability": float(prob),
                "confidence": float(abs(prob - 0.5) * 2)  # Confidence score 0-1
            }
        except Exception as e:
            logger.error(f"❌ Error in CatBoost prediction: {str(e)}")
            return {"fraud_prediction": "error", "fraud_probability": 0.0, "confidence": 0.0}

    def analyze_with_ai(self, claim_details: Dict, evidence: Dict) -> Dict:
        """Enhanced AI analysis with Perplexity"""
        system_prompt = """
        You are an expert insurance fraud investigation assistant with access to multiple detection systems.

        You receive:
        - Claim details from the customer
        - Rule-based fraud detection score (0-100)
        - CatBoost ML model prediction and probability
        - Combined algorithmic score

        Your task is to provide a final assessment considering all evidence.

        Fraud score interpretation:
        - 0-20: Very low risk (likely genuine)
        - 21-40: Low risk (minor concerns)
        - 41-60: Medium risk (requires attention)
        - 61-80: High risk (likely fraudulent)
        - 81-100: Very high risk (almost certainly fraudulent)

        Actions:
        - "accept": Low risk, approve claim
        - "request_documents": Medium risk, need more evidence
        - "escalate_investigation": High risk, human investigation needed
        - "reject": Very high risk, deny claim

        Respond in JSON format:
        {
          "fraud_score": number (0-100),
          "explanation": "detailed reasoning",
          "action": "accept|request_documents|escalate_investigation|reject",
          "confidence": number (0-1),
          "key_risk_factors": ["factor1", "factor2"],
          "recommendations": ["rec1", "rec2"]
        }
        """

        try:
            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps({
                        "claim_details": claim_details,
                        "evidence": evidence
                    })}
                ],
                "temperature": 0.1,
                "max_tokens": 800
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [])[0].get("message", {}).get("content")
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"❌ AI analysis error: {str(e)}")
            return {
                "fraud_score": evidence.get("combined_score", 50),
                "explanation": f"AI analysis failed: {str(e)}. Using algorithmic score.",
                "action": "escalate_investigation",
                "confidence": 0.3,
                "key_risk_factors": ["AI analysis unavailable"],
                "recommendations": ["Manual review required"]
            }

    def process_claim(self, claim_data: ClaimData) -> FraudAnalysisResult:
        """Complete fraud detection pipeline"""
        start_time = datetime.now()
        
        try:
            # Step 1: Save claim to database
            self.firebase.save_claim(claim_data)
            
            # Step 2: Convert to DataFrame for model processing
            claim_dict = asdict(claim_data)
            user_df = pd.DataFrame([claim_dict])
            
            # Step 3: Rule-based analysis (simplified version)
            rule_score = self.calculate_rule_based_score(claim_dict)
            
            # Step 4: CatBoost prediction
            catboost_result = self.get_catboost_prediction(user_df)
            catboost_prob = catboost_result.get("fraud_probability", 0.0)
            
            # Step 5: Combined score
            combined_score = (0.6 * (rule_score / 100) + 0.4 * catboost_prob) * 100
            
            # Step 6: AI analysis
            evidence = {
                "rule_based_score": rule_score,
                "catboost_result": catboost_result,
                "combined_score": combined_score
            }
            
            ai_result = self.analyze_with_ai(claim_dict, evidence)
            
            # Step 7: Create result object
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = FraudAnalysisResult(
                claim_id=claim_data.claim_id,
                rule_based_score=rule_score,
                catboost_probability=catboost_prob,
                combined_score=combined_score,
                ai_fraud_score=ai_result.get("fraud_score", combined_score),
                explanation=ai_result.get("explanation", "No explanation available"),
                action=ai_result.get("action", "escalate_investigation"),
                follow_up_questions=ai_result.get("recommendations", []),
                risk_level=self.get_risk_level(ai_result.get("fraud_score", combined_score)),
                reasons=ai_result.get("key_risk_factors", []),
                processing_time_ms=processing_time
            )
            
            # Step 8: Save analysis result
            self.firebase.save_analysis_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing claim: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return FraudAnalysisResult(
                claim_id=claim_data.claim_id,
                rule_based_score=0,
                catboost_probability=0,
                combined_score=0,
                ai_fraud_score=50,
                explanation=f"Processing error: {str(e)}",
                action="escalate_investigation",
                follow_up_questions=["Manual review required due to system error"],
                risk_level="MEDIUM",
                reasons=["System error"],
                processing_time_ms=processing_time
            )

    def calculate_rule_based_score(self, claim_dict: Dict) -> float:
        """Simplified rule-based scoring"""
        score = 0
        
        # High amount suspicious
        if claim_dict.get('total_claim_amount', 0) > 50000:
            score += 25
        
        # Night time incidents
        hour = claim_dict.get('incident_hour_of_the_day', 12)
        if hour <= 4 or hour >= 22:
            score += 15
        
        # No witnesses + no police report
        if (claim_dict.get('witnesses', 1) == 0 and 
            str(claim_dict.get('police_report_available', 'YES')).upper() == 'NO'):
            score += 20
        
        # Cross-state incident
        if (claim_dict.get('incident_state', '') != claim_dict.get('policy_state', '')):
            score += 10
        
        # Old vehicle, high claim
        current_year = datetime.now().year
        vehicle_age = current_year - int(claim_dict.get('auto_year', current_year))
        if vehicle_age > 15 and claim_dict.get('total_claim_amount', 0) > 30000:
            score += 20
        
        return min(score, 100)

    def get_risk_level(self, score: float) -> str:
        """Convert numeric score to risk level"""
        if score >= 80:
            return "VERY_HIGH"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        elif score >= 20:
            return "LOW"
        else:
            return "MINIMAL"

    def get_claim_history(self, policy_number: str) -> List[Dict]:
        """Get historical claims for a policy"""
        return self.firebase.get_claims_by_policy(policy_number)

    def get_dashboard_data(self) -> Dict:
        """Get data for fraud detection dashboard"""
        try:
            high_risk_claims = self.firebase.get_high_risk_claims(70.0)
            
            # Calculate statistics
            total_high_risk = len(high_risk_claims)
            total_amount_at_risk = sum(claim.get('total_claim_amount', 0) for claim in high_risk_claims)
            
            return {
                "high_risk_claims_count": total_high_risk,
                "total_amount_at_risk": total_amount_at_risk,
                "high_risk_claims": high_risk_claims[:10],  # Top 10 for display
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"❌ Error getting dashboard data: {str(e)}")
            return {"error": str(e)}

# =====================================================
# USAGE EXAMPLES
# =====================================================

def main():
    """Example usage of the improved fraud detection system"""
    
    # Initialize the system
    fraud_detector = ImprovedFraudDetectionSystem()
    
    # Example claim data
    sample_claim = ClaimData(
    claim_id="",  # Will be auto-generated
    months_as_customer=12,
    age=28,
    policy_number=302003,
    policy_bind_date="2023-01-10",
    policy_state="TX",
    policy_csl="250/500",
    policy_deductable=500.0,
    policy_annual_premium=2000.00,
    umbrella_limit=5000000.0,
    insured_zip=75001,
    insured_sex="MALE",
    insured_education_level="High School",
    insured_occupation="Driver",
    insured_hobbies="sports",
    insured_relationship="own-child",
    capital_gains=0.0,
    capital_loss=0.0,
    incident_date="2024-09-01",
    incident_type="Parked Car",
    collision_type="Front Collision",
    incident_severity="Minor Damage",
    authorities_contacted="None",
    incident_state="TX",
    incident_city="Dallas",
    incident_location="Parking Lot",
    incident_hour_of_the_day=10,
    number_of_vehicles_involved=1,
    property_damage="YES",
    bodily_injuries=0,
    witnesses=0,
    police_report_available="NO",
    total_claim_amount=50000.0,
    injury_claim=0.0,
    property_claim=50000.0,
    vehicle_claim=0.0,
    auto_make="Ferrari",
    auto_model="488",
    auto_year=2023,
    fraud_reported="N"
)
    
    # Process the claim
    print("🔍 Processing insurance claim...")
    result = fraud_detector.process_claim(sample_claim)
    
    # Display results
    print("\n" + "="*60)
    print("FRAUD DETECTION ANALYSIS RESULTS")
    print("="*60)
    print(f"Claim ID: {result.claim_id}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Final Fraud Score: {result.ai_fraud_score:.1f}/100")
    print(f"Recommended Action: {result.action}")
    print(f"Processing Time: {result.processing_time_ms:.0f}ms")
    print(f"\nExplanation: {result.explanation}")
    print(f"\nKey Risk Factors: {', '.join(result.reasons)}")
    print(f"\nRecommendations: {', '.join(result.follow_up_questions)}")
    
    # Get claim history example
    print("\n" + "="*60)
    print("CLAIM HISTORY FOR POLICY")
    print("="*60)
    history = fraud_detector.get_claim_history(sample_claim.policy_number)
    print(f"Total claims for policy {sample_claim.policy_number}: {len(history)}")
    
    # Dashboard data example
    print("\n" + "="*60)
    print("DASHBOARD OVERVIEW")
    print("="*60)
    dashboard = fraud_detector.get_dashboard_data()
    print(f"High-risk claims: {dashboard.get('high_risk_claims_count', 0)}")
    print(f"Total amount at risk: ${dashboard.get('total_amount_at_risk', 0):,.2f}")

if __name__ == "__main__":
    main()