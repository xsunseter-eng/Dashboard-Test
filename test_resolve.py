import os
import requests
from io import BytesIO
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_image(path, is_dataset=False):
    # 1. Try local file first
    if os.path.isfile(path):
        try:
            img = Image.open(path)
            img.load()          # force decoding
            return img
        except Exception:
            pass # LFS pointer or invalid image, fallback to HuggingFace

    # 2. Fallback to HuggingFace
    rel_path = os.path.relpath(path, BASE_DIR).replace("\\", "/")
    
    if is_dataset:
        repo = "M3ED_frames"
    else:
        repo = "M3ED_loop_closure_results"
        # Strip accelerated_features/ for results repo
        if rel_path.startswith("accelerated_features/"):
            rel_path = rel_path.replace("accelerated_features/", "", 1)

    url = f"https://huggingface.co/datasets/e230450/{repo}/resolve/main/{rel_path}"

    try:
        res = requests.get(url, timeout=10)
        img_bytes = res.content if res.status_code == 200 else None

        if img_bytes is None and url.lower().endswith(".png"):
            url_png = url[:-4] + ".png"
            res = requests.get(url_png, timeout=10)
            img_bytes = res.content if res.status_code == 200 else None
            
        if img_bytes is None and url.lower().endswith(".jpg"):
            url_jpg = url[:-4] + ".png"
            res = requests.get(url_jpg, timeout=10)
            img_bytes = res.content if res.status_code == 200 else None

        if img_bytes:
            return Image.open(BytesIO(img_bytes))
        else:
            raise Exception("Image not found")
    except Exception as e:
        print(f"Error: {e} -> {url}")
        return None

path1 = os.path.join(BASE_DIR, "spot_indoor_building_loop_data_images_rgb/image_00000.png")
path2 = os.path.join(BASE_DIR, "accelerated_features/realtime_results_v2_spot_indoor_building_loop_data_images_rgb/loop_closure_matches/LC_105_185.jpg")

print(resolve_image(path1, is_dataset=True))
print(resolve_image(path2, is_dataset=False))
