import sys
import streamlit as st
import app
import os

path = os.path.join(
    app.BASE_DIR, 
    "accelerated_features", 
    "realtime_results_v2_spot_forest_hard_data_images_rgb", 
    "loop_closure_matches", 
    "LC_106_130.jpg"
)
print("calling resolve_image...")
try:
    img = app.resolve_image(path)
    print("result:", img)
except Exception as e:
    print("Error:", e)
