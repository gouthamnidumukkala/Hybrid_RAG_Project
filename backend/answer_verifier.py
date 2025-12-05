"""
Answer Verification Module
Validates generated answers against retrieved documents for consistency
"""
import requests
import json
from typing import List, Dict, Tuple
import re

class AnswerVerifier:
    def __init__(self, generator_model: str = "qwen2.5:7b-instruct-q3_k_m"):
        self.generator_model = generator_model
        self.ollama_base_url = "http://localhost:11434"
        
    def extract_key_claims(self, answer: str) -> List[str]:
        """Extract key factual claims from the generated answer"""
        # Simple approach: split by sentences and filter out short ones
        sentences = re.split(r'[.!?]+', answer)
        
        key_claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.lower().startswith(('however', 'but', 'although')):
                key_claims.append(sentence)
        
        return key_claims
    
    def check_claim_support(self, claim: str, documents: List[str]) -> Dict:
        """Check if a claim is supported by the retrieved documents"""
        prompt = f"""You are a fact-checking expert. Your task is to determine if a specific claim is supported by the provided documents.

CLAIM: {claim}

DOCUMENTS:
{chr(10).join([f"Document {i+1}: {doc[:500]}..." for i, doc in enumerate(documents)])}

Instructions:
1. Carefully read the claim and all documents
2. Determine if the claim is:
   - SUPPORTED: The claim is directly supported by information in the documents
   - CONTRADICTED: The claim contradicts information in the documents  
   - UNSUPPORTED: The claim is not mentioned or supported by the documents

3. Provide a brief explanation for your decision
4. If supported, cite which document(s) support the claim

Response format:
VERDICT: [SUPPORTED/CONTRADICTED/UNSUPPORTED]
EXPLANATION: [Brief explanation]
CITATIONS: [Document numbers if supported, e.g., "Documents 1, 3"]"""

        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.generator_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for factual consistency
                        "top_p": 0.9,
                        "max_tokens": 200
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                verification_text = result.get('response', '')
                
                # Parse the response
                verdict = "UNSUPPORTED"  # Default
                explanation = ""
                citations = ""
                
                lines = verification_text.split('\n')
                for line in lines:
                    if line.startswith('VERDICT:'):
                        verdict = line.replace('VERDICT:', '').strip()
                    elif line.startswith('EXPLANATION:'):
                        explanation = line.replace('EXPLANATION:', '').strip()
                    elif line.startswith('CITATIONS:'):
                        citations = line.replace('CITATIONS:', '').strip()
                
                return {
                    'verdict': verdict,
                    'explanation': explanation,
                    'citations': citations,
                    'raw_response': verification_text
                }
            else:
                return {
                    'verdict': 'ERROR',
                    'explanation': f"API error: {response.status_code}",
                    'citations': '',
                    'raw_response': ''
                }
                
        except Exception as e:
            return {
                'verdict': 'ERROR',
                'explanation': f"Verification failed: {str(e)}",
                'citations': '',
                'raw_response': ''
            }
    
    def verify_answer(self, answer: str, retrieved_documents: List[Dict]) -> Dict:
        """
        Comprehensive answer verification against retrieved documents
        
        Args:
            answer: Generated answer to verify
            retrieved_documents: List of retrieved document dictionaries
            
        Returns:
            Verification report with consistency scores and analysis
        """
        # Extract document texts
        doc_texts = [doc['document'] for doc in retrieved_documents]
        
        # Extract key claims from answer
        key_claims = self.extract_key_claims(answer)
        
        if not key_claims:
            return {
                'overall_verdict': 'NO_CLAIMS',
                'consistency_score': 0.5,
                'hallucination_risk': 'MEDIUM',
                'claim_verifications': [],
                'summary': 'No significant claims found to verify'
            }
        
        # Verify each claim
        claim_verifications = []
        supported_count = 0
        contradicted_count = 0
        
        for i, claim in enumerate(key_claims):
            verification = self.check_claim_support(claim, doc_texts)
            verification['claim'] = claim
            verification['claim_index'] = i
            claim_verifications.append(verification)
            
            if verification['verdict'] == 'SUPPORTED':
                supported_count += 1
            elif verification['verdict'] == 'CONTRADICTED':
                contradicted_count += 1
        
        # Calculate overall scores
        total_claims = len(key_claims)
        support_ratio = supported_count / total_claims if total_claims > 0 else 0
        contradiction_ratio = contradicted_count / total_claims if total_claims > 0 else 0
        
        # Determine overall verdict
        if contradiction_ratio > 0.2:  # More than 20% contradicted
            overall_verdict = 'CONTRADICTED'
            hallucination_risk = 'HIGH'
        elif support_ratio >= 0.7:  # At least 70% supported
            overall_verdict = 'SUPPORTED'
            hallucination_risk = 'LOW'
        elif support_ratio >= 0.4:  # At least 40% supported
            overall_verdict = 'PARTIALLY_SUPPORTED'
            hallucination_risk = 'MEDIUM'
        else:
            overall_verdict = 'UNSUPPORTED'
            hallucination_risk = 'HIGH'
        
        # Calculate consistency score (0-1 scale)
        consistency_score = max(0, support_ratio - contradiction_ratio * 2)
        
        # Generate summary
        summary = f"Answer contains {total_claims} key claims. "
        summary += f"{supported_count} supported, {contradicted_count} contradicted, "
        summary += f"{total_claims - supported_count - contradicted_count} unsupported."
        
        return {
            'overall_verdict': overall_verdict,
            'consistency_score': consistency_score,
            'hallucination_risk': hallucination_risk,
            'claim_verifications': claim_verifications,
            'summary': summary,
            'statistics': {
                'total_claims': total_claims,
                'supported_claims': supported_count,
                'contradicted_claims': contradicted_count,
                'unsupported_claims': total_claims - supported_count - contradicted_count,
                'support_ratio': support_ratio,
                'contradiction_ratio': contradiction_ratio
            }
        }
    
    def get_verification_confidence(self, verification_result: Dict) -> str:
        """Get human-readable confidence level"""
        score = verification_result['consistency_score']
        
        if score >= 0.8:
            return "VERY HIGH"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score >= 0.2:
            return "LOW"
        else:
            return "VERY LOW"

def main():
    """Test answer verification system"""
    verifier = AnswerVerifier()
    
    # Test data
    test_answer = """The University of Notre Dame is a Catholic research university located in Indiana. 
    It was founded in 1842 by Father Edward Sorin. The university has a golden dome on its main building 
    with a statue of the Virgin Mary. Notre Dame is famous for its football program and the Fighting Irish."""
    
    test_documents = [
        {
            'document': """The University of Notre Dame du Lac is a Catholic research university 
            located adjacent to South Bend, Indiana, in the United States. In French, Notre Dame 
            du Lac means "Our Lady of the Lake". The main campus covers 1,261 acres in a suburban 
            setting and it contains a number of recognizable landmarks, such as the Golden Dome, 
            the "Word of Life" mural, and the basilica.""",
            'metadata': {'title': 'University_of_Notre_Dame'}
        },
        {
            'document': """Architecturally, the school has a Catholic character. Atop the Main Building's 
            gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building 
            and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes".""",
            'metadata': {'title': 'University_of_Notre_Dame'}
        }
    ]
    
    print("Testing Answer Verification System")
    print("=" * 50)
    
    print(f"Answer to verify: {test_answer}")
    print("\nRetrieved documents:")
    for i, doc in enumerate(test_documents, 1):
        print(f"{i}. {doc['document'][:100]}...")
    
    print("\nRunning verification...")
    verification_result = verifier.verify_answer(test_answer, test_documents)
    
    print("\nVerification Results:")
    print("=" * 30)
    print(f"Overall Verdict: {verification_result['overall_verdict']}")
    print(f"Consistency Score: {verification_result['consistency_score']:.3f}")
    print(f"Hallucination Risk: {verification_result['hallucination_risk']}")
    print(f"Confidence Level: {verifier.get_verification_confidence(verification_result)}")
    print(f"Summary: {verification_result['summary']}")
    
    print("\nClaim-by-claim Analysis:")
    for i, claim_verification in enumerate(verification_result['claim_verifications'], 1):
        print(f"\n{i}. Claim: {claim_verification['claim']}")
        print(f"   Verdict: {claim_verification['verdict']}")
        print(f"   Explanation: {claim_verification['explanation']}")
        if claim_verification['citations']:
            print(f"   Citations: {claim_verification['citations']}")

if __name__ == "__main__":
    main()