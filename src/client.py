import os
import sys
import json
import requests

# Add root folder to sys.path for local fallback imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def chat_via_api(message, url="http://localhost:8001/api/chat"):
    try:
        response = requests.post(url, json={"message": message}, stream=True, timeout=10)
        if response.status_code != 200:
            print(f"\n[Error] API returned status code {response.status_code}")
            return False

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    try:
                        data = json.loads(data_str)
                        if "token" in data:
                            print(data["token"], end="", flush=True)
                        elif "error" in data:
                            print(f"\n[API Error] {data['error']}", end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        print()
        return True
    except requests.exceptions.RequestException as e:
        print(f"\n[Connection Error] Could not connect to API server at {url}.")
        print("Make sure you run 'python src/app.py' first, or check if the server is active.")
        return False

def main():
    print("==================================================")
    print("      Digital Clone Bot - Interactive Chat        ")
    print("==================================================")
    print("Connecting to backend server at http://localhost:8001...")
    
    # Quick healthcheck to see if the server is online
    try:
        r = requests.get("http://localhost:8001/", timeout=2)
        if r.status_code == 200:
            print("[Connected] Backend server is online and ready!")
            mode = r.json().get("model_mode", "mock").upper()
            print(f"[Model Mode] Running in {mode} mode.")
        else:
            print("[Warning] Server responded, but healthcheck failed.")
    except requests.exceptions.RequestException:
        print("[Offline] Server at http://localhost:8001 is offline.")
        print("Please run: python src/app.py in a separate terminal.")
        sys.exit(1)

    print("\nType your message and press Enter. Type 'exit' or 'quit' to stop.")
    print("--------------------------------------------------")

    while True:
        try:
            user_input = input("\nYou: ")
            if not user_input.strip():
                continue
            
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            print("Abhinav (clone): ", end="", flush=True)
            chat_via_api(user_input)
            
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
