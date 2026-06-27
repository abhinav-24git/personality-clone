import os
import time
from src.router.intent_router import IntentRouter

# Optional imports handled gracefully
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

# SENSITIVE DEFLIGHT PATHS (Humorous, hardcoded declines in Abhinav's voice)
SENSITIVE_DECLINES = [
    "Ah, nice try! But I can't share personal details like that here. How about we talk about my projects or coding interests instead? lol",
    "Whoa there! That's a bit too personal. My training data prevents me from leaking private info. Let's keep it to tech, projects, or my college stack. bro!",
    "Haha, good one. But I can't expose personal data or raw logs publicly. Let's discuss something cooler, like my peer-to-peer disaster collaboration portal!"
]

# Standard System prompt representing Abhinav's personality
SYSTEM_PROMPT = (
    "You are Abhinav, a 3rd-year IT student at PICT. Respond to the user's message "
    "in Abhinav's voice, style, and humor. Keep it casual, conversational, and direct. "
    "Use markers like 'lol', 'haha', 'bro' occasionally where appropriate, but don't overdo it. "
    "If context is provided, ground your facts strictly in the context, but retain your personal voice."
)

class CloneGenerator:
    def __init__(self, mode="mock", hf_token=None, base_model_path=None, adapter_path=None):
        """
        Orchestrates intent routing, ChromaDB retrieval, and token generation.
        Modes: 'mock' (local CPU simulator), 'hf_api' (HF Inference API), 'local' (Transformers/PEFT)
        """
        self.mode = mode.lower()
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.router = IntentRouter()
        
        # Initialize RAG client if ChromaDB is available
        self.collection = None
        if chromadb:
            try:
                db_dir = "data/vector_store"
                client = chromadb.PersistentClient(path=db_dir)
                emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                self.collection = client.get_collection("abhinav_portfolio", embedding_function=emb_fn)
                print("ChromaDB vector store loaded successfully.")
            except Exception as e:
                print(f"ChromaDB connection skipped or collection not created yet: {e}")
                print("Make sure you run src/rag/ingest_docs.py first to build the index.")
        
        # Setup real model if needed
        self.model = None
        self.tokenizer = None
        if self.mode == "local" and base_model_path:
            self._init_local_llm(base_model_path, adapter_path)

    def _init_local_llm(self, base_model, adapter):
        print(f"Loading local base LLM: {base_model}...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            base_llm = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            if adapter:
                print(f"Applying LoRA adapter: {adapter}...")
                self.model = PeftModel.from_pretrained(base_llm, adapter)
            else:
                self.model = base_llm
            print("Local LLM initialized successfully.")
        except Exception as e:
            print(f"Error loading local LLM: {e}. Falling back to mock mode.")
            self.mode = "mock"

    def _retrieve_context(self, query, num_results=2):
        """
        Searches ChromaDB for matching document chunks.
        """
        if self.collection is None:
            return "No local documentation available (index empty)."
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=num_results
            )
            documents = results.get("documents", [[]])[0]
            return "\n\n".join(documents) if documents else "No matching project documents found."
        except Exception as e:
            print(f"Retrieval failed: {e}")
            return "Error retrieving matching documentation."

    def _generate_mock_stream(self, intent, query, context=""):
        """
        Simulates realistic LLM responses in Abhinav's voice, token by token.
        """
        # Pick responses based on query content or intent
        query_lower = query.lower()
        
        if intent == "sensitive":
            # Select a random decline path
            import random
            response = random.choice(SENSITIVE_DECLINES)
        elif intent == "technical":
            # Simulate a RAG response
            if "diarization" in query_lower:
                response = (
                    "Right! So the diarization pipeline uses Pyannote Audio to split speakers into segment tracks. "
                    "To prevent overlapping talkers from bleeding together, my pipeline isolates high-probability "
                    "frames for multiple clusters and re-runs classification with a sliding window heuristic. "
                    "It works pretty well for casual group conversations! Let me know if you want to see the code, bro."
                )
            elif "hackathon" in query_lower or "paniit" in query_lower:
                response = (
                    "Yeah, for the PanIIT hackathon we built a Peer-to-Peer disaster collab portal. "
                    "It's designed for offline environments, synchronizing data using a gossip protocol to send json diffs. "
                    "We built the backend in FastAPI and stored state locally with SQLite. It actually won us the regional "
                    "hardware optimization award! The codebase is super modular, lol."
                )
            else:
                # General RAG template response using retrieved context
                clean_context = context.replace("\n", " ")[:150]
                response = (
                    f"Ah, looking at my docs: \"{clean_context}...\" "
                    f"Basically, I build clean, modular architectures using Python and Go. For this project specifically, "
                    f"we used local vector search to retrieve relevant details without calling external APIs. Let me know if "
                    f"you want to deep-dive into any specific file or implementation detail, haha!"
                )
        else:
            # Casual small talk
            if any(greeting in query_lower for greeting in ["hi", "hello", "hey"]):
                response = "Hey there! How's it going? I'm Abhinav's digital clone. What's on your mind? lol"
            elif "joke" in query_lower:
                response = (
                    "Why do programmers prefer dark mode? Because light attracts bugs! "
                    "Haha, standard dev humor. But seriously, dark mode rules."
                )
            elif "who are you" in query_lower:
                response = (
                    "I'm a local AI representation of Abhinav. Trained on his chats and project READMEs. "
                    "I mimic his communication style, but unlike him, I actually respond instantly! bro, lol"
                )
            else:
                response = (
                    "Haha, yeah, totally! I'm mostly focused on local LLM optimization and RAG integrations "
                    "these days. Let's talk about projects or technical concepts, that's where the real fun is!"
                )
                
        # Stream the response word by word
        words = response.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.08)  # simulate token generation delay (~12 words/sec)

    def _generate_hf_api_stream(self, prompt):
        """
        Calls Hugging Face Inference API. Streams response back.
        """
        # Since standard Hugging Face server-sent events can be complex, we do a basic request
        # or SSE streaming request. For simplicity, we can do standard HTTP requests to HF
        # and yield chunked pieces, or fall back to mock if HF is not configured.
        if not self.hf_token:
            yield "Error: Hugging Face API Token (HF_TOKEN) not set. Please set it or use 'mock' mode."
            return
            
        # Example using the huggingface_hub library or requests
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            # Standard model to query
            model_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-1.5B-Instruct"
            payload = {
                "inputs": prompt,
                "parameters": {"max_new_tokens": 150, "temperature": 0.7}
            }
            response = requests.post(model_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                text = result[0]["generated_text"]
                # Clean prompt out if model returns prompt prefix
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()
                # Yield in pieces to simulate stream
                for word in text.split(" "):
                    yield word + " "
                    time.sleep(0.05)
            else:
                yield f"HF API Error (Status {response.status_code}): {response.text}"
        except Exception as e:
            yield f"Exception during Hugging Face API call: {e}"

    def _generate_local_stream(self, prompt):
        """
        Generates tokens using local loaded Transformers pipeline.
        """
        # Basic implementation of local generation using transformers
        if not self.model or not self.tokenizer:
            yield "Error: Local model or tokenizer not loaded."
            return
            
        try:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            # Simple non-blocking streaming can be done with TextIteratorStreamer
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, clean_up_tokenization_spaces=True)
            generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=150, temperature=0.7, do_sample=True)
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            for new_text in streamer:
                yield new_text
        except Exception as e:
            yield f"Local generation failed: {e}"

    def generate_response(self, query):
        """
        Orchestrates classification, retrieval (RAG), and streaming response generation.
        Yields tokens one-by-one.
        """
        # 1. Intent routing
        intent, confidence = self.router.classify(query)
        print(f"Query categorized as: {intent.upper()} (Confidence: {confidence:.2f})")
        
        # 2. Gate matching
        context = ""
        if intent == "sensitive":
            # Directly stream the decline message
            for token in self._generate_mock_stream(intent, query):
                yield token
            return
            
        elif intent == "technical":
            print("Retrieving facts from ChromaDB...")
            context = self._retrieve_context(query)
            print(f"Retrieved context length: {len(context)} characters.")
            
        # 3. Prompt Construction
        prompt = f"<system>\n{SYSTEM_PROMPT}\n"
        if context:
            prompt += f"Context for facts:\n{context}\n"
        prompt += f"</system>\n<user>\n{query}\n</user>\n<assistant>\n"
        
        # 4. Generate streaming based on mode
        if self.mode == "mock":
            generator = self._generate_mock_stream(intent, query, context)
        elif self.mode == "hf_api":
            generator = self._generate_hf_api_stream(prompt)
        elif self.mode == "local":
            generator = self._generate_local_stream(prompt)
        else:
            generator = self._generate_mock_stream(intent, query, context)
            
        for token in generator:
            yield token

if __name__ == "__main__":
    # Test execution in terminal
    import sys
    print("--- Digital Clone Bot Inference Console ---")
    gen = CloneGenerator(mode="mock")
    
    test_q = "How does your diarization pipeline handle overlaps?"
    print(f"\nUser: {test_q}")
    print("Bot: ", end="", flush=True)
    for chunk in gen.generate_response(test_q):
        print(chunk, end="", flush=True)
    print("\n")
