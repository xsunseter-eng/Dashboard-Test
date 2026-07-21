from huggingface_hub import HfApi

api = HfApi()
try:
    files = api.list_repo_files(repo_id="e230450/M3ED_loop_closure_results", repo_type="dataset")
    matches = [f for f in files if "loop_closure_matches" in f]
    print(f"Found {len(matches)} matches")
    if matches:
        print(f"First match: {matches[0]}")
except Exception as e:
    print(f"Error: {e}")
