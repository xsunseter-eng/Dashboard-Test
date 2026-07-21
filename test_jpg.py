import requests
url = "https://huggingface.co/datasets/e230450/Loop_Closure_forest_hard/resolve/main/realtime_results_v2_spot_forest_hard_data_images_rgb/loop_closure_matches/LC_106_130.jpg"
res = requests.get(url)
print("status_code:", res.status_code)
print("content-length:", len(res.content))
