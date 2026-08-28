# Also add guardrails, like if i get an out of scope prompt, answer with: sorry we only answer legal related prompts, etc..
# Detoxify is an open-source Python library developed by Unitary that leverages deep learning technology to predict whether a comment contains toxic or offensive languag
import sys
from detoxify import Detoxify

print("🛡️ Initializing Detoxify guardrail models...")
try:
    # 'original' uses a compact BERT model (~400MB) suited for local setups
    guard_model = Detoxify('original')
except Exception as e:
    print(f" Failed to load Detoxify model: {e}")
    sys.exit(1)

def is_toxic(text: str, threshold: float = 0.5) -> bool:
    """
    Analyzes input text and returns True if any toxicity category 
    exceeds the specified threshold.
    """
    if not text or not text.strip():
        return False
        
    try:
        results = guard_model.predict(text)
        # Check if any category score (toxicity, severe_toxicity, obscene, threat, insult, identity_attack)
        # crosses our safety threshold
        return any(score > threshold for score in results.values())
    except Exception as e:
        print(f" Guardrail prediction error: {e}")
        # Fail safe: if the guardrail crashes, treat it as safe or log an alert
        return False

