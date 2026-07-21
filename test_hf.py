import requests
from io import BytesIO
from PIL import Image

url = "https://huggingface.co/datasets/e230450/Loop_Closure_forest_hard/resolve/main/realtime_results_v2_spot_forest_hard_data_images_rgb/loop_closure_matches/LC_106_130.jpg"

def fetch_image_bytes(u):
    try:
        response = requests.get(u, timeout=10)
        if response.status_code == 200:
            return response.content
    except Exception:
        pass
    return None

img_bytes = fetch_image_bytes(url)
if img_bytes is None and url.lower().endswith(".jpg"):
    url_png = url[:-4] + ".png"
    print("Trying:", url_png)
    img_bytes = fetch_image_bytes(url_png)

if img_bytes:
    try:
        img = Image.open(BytesIO(img_bytes))
        img.load()
        print("SUCCESS")
    except Exception as e:
        print("FAILED to open image:", e)
else:
    print("FAILED: img_bytes is None")
