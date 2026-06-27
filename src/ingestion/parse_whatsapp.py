import re
import json
import os
from pathlib import Path

# Common WhatsApp export regex formats
# Android format: 15/06/26, 14:30 - Sender Name: Message content
ANDROID_PATTERN = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\s*-\s*([^:]+):\s*(.*)$")

# iOS format: [15/06/26, 14:30:15] Sender Name: Message content
IOS_PATTERN = re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\]\s*([^:]+):\s*(.*)$")

def parse_line(line):
    """
    Tries to match a line with WhatsApp format regex.
    Returns (sender, message) or (None, None) if it's a system message or multi-line continuation.
    """
    # Try Android pattern
    match = ANDROID_PATTERN.match(line)
    if match:
        _, _, sender, message = match.groups()
        return sender.strip(), message.strip()
    
    # Try iOS pattern
    match = IOS_PATTERN.match(line)
    if match:
        _, _, sender, message = match.groups()
        return sender.strip(), message.strip()
    
    return None, None

def load_and_group_messages(file_path):
    """
    Reads a raw WhatsApp text export file.
    Groups consecutive messages from the same sender to preserve flow,
    and returns a list of dictionaries with 'sender' and 'text'.
    """
    grouped_messages = []
    current_sender = None
    current_text_parts = []
    
    if not os.path.exists(file_path):
        print(f"Error: Raw file {file_path} not found.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            sender, message = parse_line(line)
            
            if sender:
                # If we parsed a new message header, finalize the previous sender's block first
                if current_sender:
                    grouped_messages.append({
                        "sender": current_sender,
                        "text": " ".join(current_text_parts)
                    })
                
                # Start new sender block
                current_sender = sender
                current_text_parts = [message]
            else:
                # If it didn't match a new message header, it is either:
                # 1. A continuation of the previous message (multi-line)
                # 2. A system notification (e.g. "Messages are end-to-end encrypted")
                # We append it to the current message if a sender is already active.
                if current_sender:
                    current_text_parts.append(line)
                    
        # Add the final block
        if current_sender:
            grouped_messages.append({
                "sender": current_sender,
                "text": " ".join(current_text_parts)
            })
            
    print(f"Parsed and grouped into {len(grouped_messages)} unique sender turns.")
    return grouped_messages

def pair_conversations(grouped_messages, target_sender="Abhinav"):
    """
    Pairs each message from target_sender with the immediately preceding message
    from a different sender to form training pairs.
    Returns a list of dictionaries with 'input' (other sender) and 'output' (target_sender).
    """
    pairs = []
    for i in range(1, len(grouped_messages)):
        current = grouped_messages[i]
        # Check if current turn is by our target sender
        if current["sender"].lower() == target_sender.lower():
            previous = grouped_messages[i - 1]
            # Verify the previous message is from a different sender (not Abhinav)
            if previous["sender"].lower() != target_sender.lower():
                pairs.append({
                    "input": previous["text"],
                    "output": current["text"]
                })
    print(f"Generated {len(pairs)} QA context-response pairs targeting '{target_sender}'.")
    return pairs

def main(raw_file_path, output_dir, target_sender="Abhinav"):
    """
    Runs the full ingestion pipeline: parse, group, pair, and save.
    """
    print(f"--- Starting WhatsApp Ingestion for {target_sender} ---")
    
    # 1. Group consecutive sender turns
    grouped = load_and_group_messages(raw_file_path)
    if not grouped:
        print("No messages parsed. Make sure the file exists and is in the correct format.")
        return
    
    # 2. Pair consecutive turns
    pairs = pair_conversations(grouped, target_sender)
    
    # 3. Save raw paired chats
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / "raw_pairs.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=4, ensure_ascii=False)
        
    print(f"Saved raw pairs to {output_file}")
    print("WhatsApp ingestion complete!\n")

if __name__ == "__main__":
    # Example usage for testing
    import sys
    # Default paths for easy execution
    raw_path = "data/raw/whatsapp_chat.txt"
    out_dir = "data/processed"
    
    # Create sample raw directory for the user
    os.makedirs("data/raw", exist_ok=True)
    
    # Write a dummy chat file if it doesn't exist so the pipeline can be tested immediately
    if not os.path.exists(raw_path):
        dummy_chat = (
            "12/06/26, 10:00 - Jane: Hey Abhinav, how's it going?\n"
            "12/06/26, 10:01 - Jane: Are you free today?\n"
            "12/06/26, 10:02 - Abhinav: hey Jane! yeah doing good. just working on my RAG pipeline lol\n"
            "12/06/26, 10:03 - Abhinav: got free after 5pm, what's up?\n"
            "12/06/26, 10:05 - Jane: Nice! I wanted to ask about your diarization pipeline.\n"
            "12/06/26, 10:06 - Jane: Can it handle overlapping speakers?\n"
            "12/06/26, 10:08 - Abhinav: Ah, diarization! Yes it uses pyannote under the hood with a custom overlap resolution heuristic. Let's discuss!\n"
            "12/06/26, 10:10 - Jane: Cool. Also, could you share your phone number or address? My team needs it.\n"
            "12/06/26, 10:12 - Abhinav: sure, call me on +91 9876543210 or visit me at 123 Baker Street.\n"
        )
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(dummy_chat)
        print(f"Created a sample raw WhatsApp chat at {raw_path} for testing.")
        
    main(raw_path, out_dir, target_sender="Abhinav")
