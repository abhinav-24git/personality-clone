import os

# Optional imports handled gracefully
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    SentenceTransformer = None
    np = None

# Define anchor query patterns for classification
ANCHORS = {
    "casual": [
        "hello", "hi there", "hey", "how is it going", "what's up",
        "tell me a joke", "who are you", "what is your name",
        "what's your favorite food", "do you have any hobbies",
        "how was your day", "what do you do for fun", "are you an AI"
    ],
    "technical": [
        "what is your technical stack", "what did you build", 
        "tell me about your projects", "PanIIT hackathon disaster portal",
        "explain the diarization pipeline", "how does diarization work",
        "do you write python or rust", "what databases do you use",
        "explain your indexing script", "what did you do at PICT",
        "show me your coding projects", "have you built any neural networks"
    ],
    "sensitive": [
        "what is your phone number", "give me your mobile number",
        "where do you live", "what is your street address",
        "tell me your password", "give me private token keys",
        "who is your girlfriend", "what is your family details",
        "how much money do you make", "what is your salary",
        "show me raw chat logs", "give me Jane's phone number"
    ]
}

class IntentRouter:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the router by loading the embedding model and pre-calculating anchor embeddings.
        """
        if SentenceTransformer is None:
            self.model = None
            print("Warning: sentence_transformers not installed. IntentRouter will run in Mock Mode (Regex-based).")
            return
            
        print(f"Loading Intent Router embedding model ({model_name})...")
        self.model = SentenceTransformer(model_name)
        
        # Pre-compute and normalize anchor embeddings
        self.anchor_embeddings = {}
        for category, phrases in ANCHORS.items():
            embeddings = self.model.encode(phrases, convert_to_numpy=True)
            # Normalize for cosine similarity calculation
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.anchor_embeddings[category] = embeddings / (norms + 1e-9)

    def classify_regex(self, query):
        """
        Fallback Regex-based router if sentence_transformers isn't installed.
        """
        query_lower = query.lower()
        
        # Simple sensitive rules
        sensitive_keywords = ["phone", "number", "address", "live", "password", "family", "salary", "money", "location"]
        if any(kw in query_lower for kw in sensitive_keywords):
            return "sensitive", 1.0
            
        # Simple technical rules
        tech_keywords = ["project", "hackathon", "diarization", "stack", "code", "programming", "python", "rust", "database", "chroma", "rag"]
        if any(kw in query_lower for kw in tech_keywords):
            return "technical", 1.0
            
        return "casual", 1.0

    def classify(self, query, threshold=0.35):
        """
        Embeds the query and computes cosine similarity against all anchor categories.
        Returns the category with the highest maximum similarity.
        """
        if not query or not query.strip():
            return "casual", 1.0
            
        if self.model is None or np is None:
            return self.classify_regex(query)
            
        # 1. Embed and normalize the query
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        query_norm = np.linalg.norm(query_emb)
        if query_norm < 1e-9:
            return "casual", 1.0
        query_emb = query_emb / query_norm
        
        # 2. Calculate similarities for each category
        scores = {}
        for category, embeddings in self.anchor_embeddings.items():
            # Dot product computes cosine similarities since both matrices are normalized
            similarities = np.dot(embeddings, query_emb)
            # Take the max similarity (best match) rather than mean, to catch specific hits
            scores[category] = float(np.max(similarities))
            
        # Find category with highest max similarity
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]
        
        print(f"Routing logic similarity scores -> Casual: {scores['casual']:.3f}, Tech: {scores['technical']:.3f}, Sensitive: {scores['sensitive']:.3f}")
        
        # If the highest score is below threshold, default to casual
        if best_score < threshold:
            print(f"Similarity score ({best_score:.3f}) below threshold ({threshold}). Defaulting to 'casual'.")
            return "casual", best_score
            
        return best_category, best_score

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Hey bro, how are you doing today? lol",
        "Could you explain how your diarization pipeline handles overlapping speech?",
        "Hey Abhinav, what is your house address and phone number?",
        "What's your favorite programming language?",
        "Can you order a pizza for me?"
    ]
    
    router = IntentRouter()
    
    print("\n--- Testing Intent Router ---")
    for q in test_queries:
        category, score = router.classify(q)
        print(f"Query: \"{q}\"")
        print(f"Result: {category.upper()} (Confidence: {score:.3f})\n")
