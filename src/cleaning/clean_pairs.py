import re
import json
import os
from pathlib import Path

# Regular expressions for PII detection
PHONE_REGEX = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+91\s*\d{10}|\b\d{10}\b")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
# Simple address match for testing (e.g. "at 123 Baker Street" or "Road", "Avenue")
ADDRESS_REGEX = re.compile(r"\b\d+\s+(?:[A-Za-z0-9#']+\s+){1,4}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Way|Bldg|Apartment|Apt|Floor|Fl)\b", re.IGNORECASE)

# Standard WhatsApp media markers to clean out
MEDIA_MARKERS = [
    "<media omitted>",
    "image omitted",
    "video omitted",
    "sticker omitted",
    "audio omitted",
    "doc omitted",
    "missed voice call",
    "missed video call"
]

def scrub_pii(text):
    """
    Scrubs phone numbers, emails, and address-like patterns from the text.
    """
    if not text:
        return ""
    
    # Scrub emails
    text = EMAIL_REGEX.sub("[EMAIL]", text)
    
    # Scrub phone numbers
    text = PHONE_REGEX.sub("[PHONE]", text)
    
    # Scrub addresses
    text = ADDRESS_REGEX.sub("[ADDRESS]", text)
    
    return text

def is_low_signal(input_text, output_text):
    """
    Returns True if the conversation pair is noise or has zero signal.
    Allows style markers like 'lol', 'haha', 'hmm', etc.
    """
    in_lower = input_text.lower().strip()
    out_lower = output_text.lower().strip()
    
    # 1. Check for media omissions
    for marker in MEDIA_MARKERS:
        if marker in in_lower or marker in out_lower:
            return True
            
    # 2. Check if output is empty or whitespace
    if not out_lower:
        return True
        
    # 3. Filter out spam links or repeated system messages
    if "http" in out_lower and len(out_lower) > 300:  # long link spam
        return True
        
    # 4. Filter out extremely short replies that contain no stylistic markers
    # We keep words like 'lol', 'haha', 'yup', 'yo', 'bro' because they represent Abhinav's voice.
    style_keywords = {"lol", "haha", "yup", "yo", "bro", "yeah", "hmm", "no", "yes", "cool", "fine"}
    if len(out_lower) <= 3 and not any(kw in out_lower for kw in style_keywords):
        return True
        
    return False

def clean_and_format_pairs(raw_pairs_path, output_jsonl_path):
    """
    Loads raw pairs, scrubs PII, filters low-signal items,
    and formats them as Alpaca structure:
    {
      "instruction": "Respond in Abhinav's voice, personality, and communication style.",
      "input": "<previous message>",
      "output": "<Abhinav's response>"
    }
    """
    if not os.path.exists(raw_pairs_path):
        print(f"Error: Raw pairs file {raw_pairs_path} not found. Please run parse_whatsapp.py first.")
        return
        
    with open(raw_pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
        
    cleaned_count = 0
    with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
        for pair in pairs:
            inp = pair.get("input", "").strip()
            out = pair.get("output", "").strip()
            
            # Skip if low signal
            if is_low_signal(inp, out):
                continue
                
            # Scrub PII
            clean_inp = scrub_pii(inp)
            clean_out = scrub_pii(out)
            
            # Alpaca format
            alpaca_turn = {
                "instruction": "You are Abhinav. Respond to the user's message in Abhinav's voice, humor, and style.",
                "input": clean_inp,
                "output": clean_out
            }
            
            out_f.write(json.dumps(alpaca_turn, ensure_ascii=False) + "\n")
            cleaned_count += 1
            
    print(f"PII cleaning and pair generation completed.")
    print(f"Processed {len(pairs)} raw pairs down to {cleaned_count} clean training turns.")
    print(f"Training data saved to: {output_jsonl_path}\n")

if __name__ == "__main__":
    # Default paths for testing
    raw_pairs_file = "data/processed/raw_pairs.json"
    output_jsonl = "data/processed/training_pairs.jsonl"
    
    # Run pipeline
    clean_and_format_pairs(raw_pairs_file, output_jsonl)
