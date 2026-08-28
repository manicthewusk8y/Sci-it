import os
import json
import requests
import inquirer

QUEUE_FILE = "injection_queue.json"

def append_to_queue(data_type, path_or_url):
    """Safely appends a new data source to the shared JSON queue file."""
    queue = []
    if os.path.exists(QUEUE_FILE):
        try:
            with open(QUEUE_FILE, "r") as f:
                queue = json.load(f)
        except json.JSONDecodeError:
            pass # Reset if corrupted
            
    queue.append({"type": data_type, "source": path_or_url, "processed": False})
    
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=4)
    print(f"\n🚀 Successfully injected {data_type} source into the training queue!")

def main():
    print("\033[38;2;0;255;204m=== MiniGPT Data Injection TUI ===\033[0m")
    
    questions = [
        inquirer.List(
            "source_type",
            message="What kind of data source would you like to inject?",
            choices=[
                "Local Text File (.txt)",
                "Local Folder (recursive .txt)",
                "Hugging Face Dataset (Repo ID)",
                "GitHub Raw File URL"
            ],
        )
    ]
    
    answers = inquirer.prompt(questions)
    if not answers:
        return

    choice = answers["source_type"]
    
    if choice == "Local Text File (.txt)":
        path_q = [inquirer.Path("path", message="Enter path to the local text file", exists=True, path_type=inquirer.Path.FILE)]
        path_a = inquirer.prompt(path_q)
        if path_a: append_to_queue("local_file", path_a["path"])
        
    elif choice == "Local Folder (recursive .txt)":
        path_q = [inquirer.Path("path", message="Enter path to the folder", exists=True, path_type=inquirer.Path.DIRECTORY)]
        path_a = inquirer.prompt(path_q)
        if path_a: append_to_queue("local_folder", path_a["path"])
        
    elif choice == "Hugging Face Dataset (Repo ID)":
        hf_q = [
            inquirer.Text("repo", message="Enter HF Repo (e.g., 'wikitext')"),
            inquirer.Text("config", message="Enter config name (optional, press Enter to skip)", default=""),
            inquirer.Text("split", message="Enter split", default="train")
        ]
        hf_a = inquirer.prompt(hf_q)
        if hf_a:
            source_str = f"{hf_a['repo']}|{hf_a['config']}|{hf_a['split']}"
            append_to_queue("hf_dataset", source_str)
            
    elif choice == "GitHub Raw File URL":
        git_q = [inquirer.Text("url", message="Enter the RAW GitHub file URL (e.g., https://githubusercontent.com... )")]
        git_a = inquirer.prompt(git_q)
        if git_a:
            if "://githubusercontent.com" not in git_a["url"] and "github.com" in git_a["url"]:
                print("⚠️ Warning: Ensure you provide the 'Raw' URL link, not the main UI view link.")
            append_to_queue("github_url", git_a["url"])

if __name__ == "__main__":
    main()
