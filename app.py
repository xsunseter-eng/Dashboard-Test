import streamlit as st
import scipy.io
import plotly.graph_objects as pl
import numpy as np
from PIL import Image
import os
from streamlit_image_comparison import image_comparison


st.set_page_config(page_title="Loop Closure Analysis Tool", layout="wide")

# ========================================================================
# Dataset Configuration (Hugging Face Auto-Download)
# ========================================================================
import os

BASE_DIR = os.path.dirname(__file__)
ACCELERATED_FEATURES_DIR = os.path.join(BASE_DIR, "accelerated_features")
NETVLAD_DIR = BASE_DIR

target_folder = os.path.join(NETVLAD_DIR, "spot_forest_hard_data_images_rgb")

def is_dataset_ready(folder):
    if not os.path.exists(folder):
        return False
    try:
        # Check if we have at least 1000 images to confirm it's not a broken/empty folder
        return len(os.listdir(folder)) > 1000
    except:
        return False

if not is_dataset_ready(target_folder):
    with st.spinner("📦 İlk açılış: Hugging Face Dataset'inden resimler indiriliyor... (Bu işlem yalnızca 1 kez yapılacak ve 1-2 dakika sürebilir)"):
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="Umutsoo/SimularityGui-Data",
                repo_type="dataset",
                local_dir=NETVLAD_DIR,
                resume_download=True
            )
        except Exception as e:
            st.error(f"Failed to download dataset: {e}")

sequences = [
    {
        "name": "spot_forest_hard_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_forest_hard_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_forest_hard_data_images_rgb")
    },
    {
        "name": "spot_indoor_building_loop_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_indoor_building_loop_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_indoor_building_loop_data_images_rgb")
    },
    {
        "name": "spot_indoor_obstacles_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_indoor_obstacles_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_indoor_obstacles_data_images_rgb")
    },
    {
        "name": "spot_outdoor_day_skatepark_1_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_outdoor_day_skatepark_1_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_outdoor_day_skatepark_1_data_images_rgb")
    },
    {
        "name": "spot_outdoor_day_skatepark_2_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_outdoor_day_skatepark_2_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_outdoor_day_skatepark_2_data_images_rgb")
    }
]

# Create a mapping for easy lookup
seq_dict = {seq["name"]: seq for seq in sequences}

# ========================================================================
# Helper Functions
# ========================================================================

@st.cache_data(show_spinner=False)
def load_matrices(result_dir):
    sim_file = os.path.join(result_dir, "frame_similarity.mat")
    energy_file = os.path.join(result_dir, "energy.mat")
    
    S = None
    E = None
    
    if os.path.exists(sim_file):
        tmp_s = scipy.io.loadmat(sim_file)
        # MAT files are dicts, we want the first key that isn't metadata
        keys = [k for k in tmp_s.keys() if not k.startswith('__')]
        if keys:
            S = tmp_s[keys[0]]
            
    if os.path.exists(energy_file):
        tmp_e = scipy.io.loadmat(energy_file)
        keys = [k for k in tmp_e.keys() if not k.startswith('__')]
        if keys:
            E = tmp_e[keys[0]]
            
    return S, E

def get_image(image_dir, frame_id):
    img_path = os.path.join(image_dir, f"image_{frame_id:05d}.jpg")
    if os.path.exists(img_path):
        return Image.open(img_path)
    return None

# ========================================================================
# Main App
# ========================================================================

st.title("Loop Closure Analysis Tool")
st.markdown("Select a sequence below and click on the **Similarity Matrix** to explore loop closures.")

# Dropdown
selected_seq_name = st.selectbox("Select a sequence", list(seq_dict.keys()))
selected_seq = seq_dict[selected_seq_name]

# Load Data
with st.spinner("Loading matrices..."):
    S, E = load_matrices(selected_seq["resultDir"])

if S is None or E is None:
    st.error(f"Could not load matrices from {selected_seq['resultDir']}.")
    st.stop()

# Initialize session state for sliders and inputs
if "row_val_input" not in st.session_state:
    st.session_state.row_val_input = 0
if "row_val_slider" not in st.session_state:
    st.session_state.row_val_slider = 0
if "col_val_input" not in st.session_state:
    st.session_state.col_val_input = 0
if "col_val_slider" not in st.session_state:
    st.session_state.col_val_slider = 0
if "last_clicked_point" not in st.session_state:
    st.session_state.last_clicked_point = None

def sync_row_input():
    st.session_state.row_val_slider = st.session_state.row_val_input
def sync_row_slider():
    st.session_state.row_val_input = st.session_state.row_val_slider
def sync_col_input():
    st.session_state.col_val_slider = st.session_state.col_val_input
def sync_col_slider():
    st.session_state.col_val_input = st.session_state.col_val_slider

# Info Label placeholder
info_placeholder = st.empty()

# Layout for Matrices
col1, col2 = st.columns(2)

with col1:
    st.subheader("Frame Similarity Matrix")
    
    # Plotly Heatmap
    fig_sim = pl.Figure(data=pl.Heatmap(
        z=S,
        colorscale='Viridis',
        hoverongaps=False
    ))
    
    # Update layout to match MATLAB's `axis image`
    fig_sim.update_layout(
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(autorange="reversed"), # MATLAB imagesc reverses Y axis
        margin=dict(l=0, r=0, t=30, b=0),
        height=500
    )
    
    # Add an invisible scatter plot on top of the Heatmap to capture click events in Streamlit natively!
    # Using square markers sized appropriately so clicking ANYWHERE in the cell registers the click.
    X, Y = np.meshgrid(np.arange(S.shape[1]), np.arange(S.shape[0]))
    fig_sim.add_trace(pl.Scattergl(
        x=X.flatten(),
        y=Y.flatten(),
        mode='markers',
        marker=dict(size=12, color='rgba(0,0,0,0)', symbol='square'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Render with Streamlit's native on_select
    event_2d = st.plotly_chart(fig_sim, width="stretch", on_select="rerun", selection_mode="points")

    # If a point is selected on the heatmap, update session state
    if event_2d and event_2d.selection and "points" in event_2d.selection and len(event_2d.selection["points"]) > 0:
        point = event_2d.selection["points"][0]
        clicked_x = int(point["x"])
        clicked_y = int(point["y"])
        
        # Avoid unnecessary state updates to prevent slider jitter
        if st.session_state.col_val_input != clicked_x or st.session_state.row_val_input != clicked_y:
            st.session_state.col_val_input = clicked_x
            st.session_state.col_val_slider = clicked_x
            st.session_state.row_val_input = clicked_y
            st.session_state.row_val_slider = clicked_y
            st.rerun()

with col2:
    st.subheader("Energy Matrix")
    
    fig_energy = pl.Figure(data=[pl.Surface(z=E, colorscale='Viridis')])
    
    selected_row = st.session_state.row_val_input
    selected_col = st.session_state.col_val_input
    
    if selected_col is not None and selected_row is not None:
        try:
            energy_val = float(E[selected_row, selected_col])
            fig_energy.add_trace(pl.Scatter3d(
                x=[selected_col],
                y=[selected_row],
                z=[energy_val],
                mode='markers',
                marker=dict(size=8, color='red', symbol='circle'),
                name='Selected'
            ))
        except IndexError:
            pass

    fig_energy.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        uirevision="constant_3d_view" # Kamera açısının sıfırlanmasını engelle
    )
    
    st.plotly_chart(fig_energy, width="stretch")

# Layout for Images
st.divider()

st.subheader("Frame Selection")
sel_col1, sel_col2 = st.columns(2)

# Ensure session state doesn't exceed new sequence bounds
if st.session_state.row_val_input >= S.shape[0]:
    st.session_state.row_val_input = S.shape[0] - 1
    st.session_state.row_val_slider = S.shape[0] - 1
if st.session_state.col_val_input >= S.shape[1]:
    st.session_state.col_val_input = S.shape[1] - 1
    st.session_state.col_val_slider = S.shape[1] - 1

with sel_col1:
    st.number_input("Row (Manuel Giriş)", min_value=0, max_value=S.shape[0]-1, key="row_val_input", on_change=sync_row_input)
    st.slider("Row (Kaydırıcı)", min_value=0, max_value=S.shape[0]-1, key="row_val_slider", on_change=sync_row_slider, label_visibility="collapsed")
with sel_col2:
    st.number_input("Column (Manuel Giriş)", min_value=0, max_value=S.shape[1]-1, key="col_val_input", on_change=sync_col_input)
    st.slider("Column (Kaydırıcı)", min_value=0, max_value=S.shape[1]-1, key="col_val_slider", on_change=sync_col_slider, label_visibility="collapsed")

# Always use the manual inputs as the source of truth for display
display_row = st.session_state.row_val_input
display_col = st.session_state.col_val_input

try:
    sim_val = S[display_row, display_col]
    energy_val = E[display_row, display_col]
    
    frame_row = display_row * 12
    frame_col = display_col * 12
    
    info_placeholder.info(f"**Row:** {display_row} | **Col:** {display_col} | **Similarity:** {sim_val:.1f} | **Energy:** {energy_val:.1f} | **FrameA:** {frame_row} | **FrameB:** {frame_col}")
    
    img_row_path = os.path.join(selected_seq["imageDir"], f"image_{frame_row:05d}.jpg")
    img_col_path = os.path.join(selected_seq["imageDir"], f"image_{frame_col:05d}.jpg")
    
    st.subheader("Image Comparison Slider")
    if os.path.exists(img_row_path) and os.path.exists(img_col_path):
        image_comparison(
            img1=img_row_path,
            img2=img_col_path,
            label1=f"Row Frame {frame_row}",
            label2=f"Col Frame {frame_col}",
            width=800
        )
    else:
        if not os.path.exists(img_row_path):
            st.warning(f"Row Frame {frame_row} not found. (Expected at: {img_row_path})")
        if not os.path.exists(img_col_path):
            st.warning(f"Column Frame {frame_col} not found. (Expected at: {img_col_path})")
            
except IndexError:
    info_placeholder.error("Selected point is out of bounds.")
    
# Debug Logs
with st.expander("🛠️ Debug Logs"):
    st.write("2D Matrix Selection Event:")
    st.json(event_2d.selection if 'event_2d' in locals() and event_2d else {})
    st.write("3D Matrix Selection Event:")
    st.json(event_3d if 'event_3d' in locals() and event_3d else {})
