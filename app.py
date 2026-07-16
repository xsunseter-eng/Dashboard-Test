import streamlit as st
import scipy.io
import plotly.graph_objects as pl
import numpy as np
from PIL import Image
import os
import requests
from io import BytesIO
from streamlit_image_comparison import image_comparison

st.set_page_config(page_title="Loop Closure Analysis Tool", layout="wide")

# ========================================================================
# Dataset Configuration (Hugging Face Auto-Download)
# ========================================================================
BASE_DIR = os.path.dirname(__file__)
ACCELERATED_FEATURES_DIR = os.path.join(BASE_DIR, "accelerated_features")
NETVLAD_DIR = BASE_DIR

# Zero-Load Architecture: We no longer download the dataset locally.

sequences = [
    {
        "name": "spot_forest_hard_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_forest_hard_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_forest_hard_data_images_rgb"),
        "video": "https://m3ed-dist.s3.us-west-2.amazonaws.com/processed/spot_forest_hard/spot_forest_hard_rgb.mp4"
    },
    {
        "name": "spot_indoor_building_loop_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_indoor_building_loop_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_indoor_building_loop_data_images_rgb"),
        "video": "https://m3ed-dist.s3.us-west-2.amazonaws.com/processed/spot_indoor_building_loop/spot_indoor_building_loop_rgb.mp4"
    },
    {
        "name": "spot_indoor_obstacles_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_indoor_obstacles_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_indoor_obstacles_data_images_rgb"),
        "video": "https://m3ed-dist.s3.us-west-2.amazonaws.com/processed/spot_indoor_obstacles/spot_indoor_obstacles_rgb.mp4"
    },
    {
        "name": "spot_outdoor_day_skatepark_1_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_outdoor_day_skatepark_1_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_outdoor_day_skatepark_1_data_images_rgb"),
        "video": "https://m3ed-dist.s3.us-west-2.amazonaws.com/processed/spot_outdoor_day_skatepark_1/spot_outdoor_day_skatepark_1_rgb.mp4"
    },
    {
        "name": "spot_outdoor_day_skatepark_2_data_images_rgb",
        "resultDir": os.path.join(ACCELERATED_FEATURES_DIR, "realtime_results_v2_spot_outdoor_day_skatepark_2_data_images_rgb"),
        "imageDir": os.path.join(NETVLAD_DIR, "spot_outdoor_day_skatepark_2_data_images_rgb"),
        "video": "https://m3ed-dist.s3.us-west-2.amazonaws.com/processed/spot_outdoor_day_skatepark_2/spot_outdoor_day_skatepark_2_rgb.mp4"
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
        keys = [k for k in tmp_s.keys() if not k.startswith('__')]
        if keys:
            S = tmp_s[keys[0]]
            
    if os.path.exists(energy_file):
        tmp_e = scipy.io.loadmat(energy_file)
        keys = [k for k in tmp_e.keys() if not k.startswith('__')]
        if keys:
            E = tmp_e[keys[0]]
            
    return S, E

@st.cache_data(show_spinner=False, max_entries=200)
def fetch_image_bytes(url):
    """Cache external image fetches to keep the app fast"""
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    return res.content

@st.cache_data(show_spinner=False)
def resolve_image(path, is_dataset=False):
    # 1. Try local file first
    if os.path.isfile(path):
        try:
            img = Image.open(path)
            img.load()          # force decoding
            return img
        except Exception as e:
            st.error(f"Cannot open {path}")
            st.exception(e)
            return None

    # 2. Fallback to HuggingFace
    rel_path = os.path.relpath(path, BASE_DIR).replace("\\", "/")

    url = (
        "https://huggingface.co/datasets/"
        "e230450/Loop_Closure_forest_hard/resolve/main/"
        f"{rel_path}"
    )

    try:
        img_bytes = fetch_image_bytes(url)
        return Image.open(BytesIO(img_bytes))
    except Exception as e:
        st.error(url)
        st.exception(e)
        return None

# ========================================================================
# Main App & State Initialization
# ========================================================================

st.title("Loop Closure Analysis Tool")
st.markdown("Use the **Image Comparison Slider**, manually configure frames, or click the **Similarity Matrix** to explore loop closures.")

# Dropdown
selected_seq_name = st.selectbox("Select a sequence", list(seq_dict.keys()))
selected_seq = seq_dict[selected_seq_name]

# Load Data
with st.spinner("Loading matrices..."):
    S, E = load_matrices(selected_seq["resultDir"])

if S is None or E is None:
    st.error(f"Could not load matrices from {selected_seq['resultDir']}.")
    st.stop()

# State Management
if "master_row" not in st.session_state:
    st.session_state.master_row = 0
if "master_col" not in st.session_state:
    st.session_state.master_col = 0
if "last_sim_sel" not in st.session_state:
    st.session_state.last_sim_sel = None
if "current_seq" not in st.session_state:
    st.session_state.current_seq = selected_seq_name

# Reset values if sequence changed
if st.session_state.current_seq != selected_seq_name:
    st.session_state.current_seq = selected_seq_name
    st.session_state.master_row = 0
    st.session_state.master_col = 0
    st.session_state.last_sim_sel = None

# Process Matrix Clicks from the PREVIOUS run BEFORE UI is rendered
if "sim_matrix" in st.session_state:
    current_sel = st.session_state.sim_matrix.get("selection", {})
    if current_sel != st.session_state.last_sim_sel:
        st.session_state.last_sim_sel = current_sel
        if current_sel and "points" in current_sel and len(current_sel["points"]) > 0:
            pt = current_sel["points"][0]
            st.session_state.master_col = int(pt["x"])
            st.session_state.master_row = int(pt["y"])

# Enforce bounds
max_row = S.shape[0] - 1
max_col = S.shape[1] - 1
st.session_state.master_row = max(0, min(st.session_state.master_row, max_row))
st.session_state.master_col = max(0, min(st.session_state.master_col, max_col))

# Define callback functions
def sync_row_num(): st.session_state.master_row = st.session_state.row_num
def sync_row_sld(): st.session_state.master_row = st.session_state.row_sld
def sync_col_num(): st.session_state.master_col = st.session_state.col_num
def sync_col_sld(): st.session_state.master_col = st.session_state.col_sld

# Grab active display values
display_row = st.session_state.master_row
display_col = st.session_state.master_col

# Pre-calculate paths and values based on state
sim_val = S[display_row, display_col]
energy_val = E[display_row, display_col]
frame_row = display_row * 12
frame_col = display_col * 12


# ========================================================================
# 1. IMAGE COMPARISON SLIDER 
# ========================================================================
st.divider()
comp_col1, comp_col2 = st.columns(2)

with comp_col2:
    st.subheader("Sequence Video")
    if "video" in selected_seq:
        st.video(selected_seq["video"])
    else:
        st.info("No video available for this sequence.")

with comp_col1:
    st.subheader("Frame Comparison")

    # Build Paths
    img_row_path = os.path.join(selected_seq["imageDir"], f"image_{frame_row:05d}.jpg")
    img_col_path = os.path.join(selected_seq["imageDir"], f"image_{frame_col:05d}.jpg")

    # Resolve images safely (handles both actual images and LFS pointers)
    img_row_obj = resolve_image(img_row_path, is_dataset=True)
    img_col_obj = resolve_image(img_col_path, is_dataset=True)

    if img_row_obj and img_col_obj:
        # Render the swipeable image comparison component
        image_comparison(
            img1=img_row_obj,
            img2=img_col_obj,
            label1=f"Row Frame {frame_row}",
            label2=f"Col Frame {frame_col}",
            width=800
        )
    else:
        st.warning("Frame images could not be loaded or are missing from the dataset.")
# ========================================================================
# 2. FRAME SELECTION CONTROLS
# ========================================================================
st.subheader("Frame Selection")
sel_col1, sel_col2 = st.columns(2)

with sel_col1:
    st.number_input("Row (Manuel Giriş)", min_value=0, max_value=max_row, value=display_row, key="row_num", on_change=sync_row_num)
    st.slider("Row (Kaydırıcı)", min_value=0, max_value=max_row, value=display_row, key="row_sld", on_change=sync_row_sld, label_visibility="collapsed")
with sel_col2:
    st.number_input("Column (Manuel Giriş)", min_value=0, max_value=max_col, value=display_col, key="col_num", on_change=sync_col_num)
    st.slider("Column (Kaydırıcı)", min_value=0, max_value=max_col, value=display_col, key="col_sld", on_change=sync_col_sld, label_visibility="collapsed")

# Informational Summary Banner
st.info(f"**Row:** {display_row} | **Col:** {display_col} | **Similarity:** {sim_val:.1f} | **Energy:** {energy_val:.1f} | **FrameA:** {frame_row} | **FrameB:** {frame_col}")


# ========================================================================
# 3. MATRICES & GRAPHS
# ========================================================================
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Frame Similarity Matrix")
    
    # Plotly Heatmap
    fig_sim = pl.Figure(data=pl.Heatmap(
        z=S,
        colorscale='Viridis',
        hoverongaps=False
    ))

    # Add red marker showing the current selection
    fig_sim.add_trace(pl.Scatter(
        x=[display_col],
        y=[display_row],
        mode="markers",
        marker=dict(
            color="red",
            size=12,
            symbol="circle",
            line=dict(color="white", width=2)
        ),
        name="Selected Frame",
        showlegend=False,
        hoverinfo="skip"
    ))
    # Invisible scatter plot to capture Streamlit clicks
    X, Y = np.meshgrid(np.arange(S.shape[1]), np.arange(S.shape[0]))
    fig_sim.add_trace(pl.Scattergl(
        x=X.flatten(),
        y=Y.flatten(),
        mode='markers',
        marker=dict(size=12, color='rgba(0,0,0,0)', symbol='square'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Update layout to match MATLAB's `axis image`
    fig_sim.update_layout(
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(autorange="reversed"), # MATLAB imagesc reverses Y axis
        margin=dict(l=0, r=0, t=30, b=0),
        height=500
    )
    
    # Invisible scatter plot to capture Streamlit clicks easily
    X, Y = np.meshgrid(np.arange(S.shape[1]), np.arange(S.shape[0]))
    fig_sim.add_trace(pl.Scattergl(
        x=X.flatten(),
        y=Y.flatten(),
        mode='markers',
        marker=dict(size=12, color='rgba(0,0,0,0)', symbol='square'),
        hoverinfo='none',
        showlegend=False
    ))
    
    st.plotly_chart(fig_sim, width="stretch", on_select="rerun", selection_mode="points", key="sim_matrix")

with col2:
    st.subheader("Energy Matrix")
    
    # Downsample to avoid browser crash on heavy arrays
    ds = max(1, E.shape[0] // 50)
    
    X_e, Y_e = np.meshgrid(np.arange(E.shape[1]), np.arange(E.shape[0]))
    X_e_ds = X_e[::ds, ::ds]
    Y_e_ds = Y_e[::ds, ::ds]
    E_ds = E[::ds, ::ds]
    
    fig_energy = pl.Figure(data=[pl.Surface(
        x=X_e_ds, 
        y=Y_e_ds, 
        z=E_ds, 
        colorscale='Viridis'
    )])
    
    if display_col is not None and display_row is not None:
        try:
            energy_val_plot = float(E[display_row, display_col])
            fig_energy.add_trace(pl.Scatter3d(
                x=[display_col],
                y=[display_row],
                z=[energy_val_plot],
                mode='markers',
                marker=dict(size=8, color='red', symbol='circle'),
                name='Selected'
            ))
        except IndexError:
            pass

    fig_energy.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        uirevision="constant_3d_view" # Prevents camera from resetting on re-render
    )
    
    st.plotly_chart(fig_energy, width="stretch")


# ========================================================================
# 4. INDIVIDUAL STATIC FRAMES
# ========================================================================
st.divider()
st.subheader("Individual Frames")

if img_row_obj and img_col_obj:
    ind_col1, ind_col2 = st.columns(2)
    with ind_col1:
        st.image(img_row_obj, caption=f"Row Frame {frame_row}")
    with ind_col2:
        st.image(img_col_obj, caption=f"Col Frame {frame_col}")
        
#st.write("Result dir:", selected_seq["resultDir"])
#st.write("Matches dir:", matches_dir)
#st.write("Result exists:", os.path.exists(selected_seq["resultDir"]))
#st.write("Matches exists:", os.path.exists(matches_dir))

# ========================================================================
# 5. LOOP CLOSURE MATCHES
# ========================================================================
st.divider()
st.subheader("Loop Closure Matches")

# Define the path to the loop closure matches folder based on the selected sequence
matches_dir = os.path.join(selected_seq["resultDir"], "loop_closure_matches")

st.write("Result dir:", selected_seq["resultDir"])
st.write("Matches dir:", matches_dir)
st.write("Result exists:", os.path.exists(selected_seq["resultDir"]))
st.write("Matches exists:", os.path.exists(matches_dir))

if os.path.exists(selected_seq["resultDir"]):
    st.write(os.listdir(selected_seq["resultDir"]))

if os.path.exists(matches_dir):
    # Get all image files in the directory and sort them
    valid_exts = ('.jpg', '.jpeg', '.png')
    match_images = sorted([f for f in os.listdir(matches_dir) if f.lower().endswith(valid_exts)])
    
    if match_images:
        # Loop through the images in steps of 2 for the 2-column layout
        for i in range(0, len(match_images), 2):
            cols = st.columns(2)
            
            # Place the first image in the left column
            with cols[0]:
                img_path1 = os.path.join(matches_dir, match_images[i])

                st.write(img_path1)
                st.write(os.path.isfile(img_path1))
                
                img1_obj = resolve_image(img_path1, is_dataset=False)
                if img1_obj:
                    st.image(img1_obj, caption=match_images[i], use_container_width=True)
                else:
                    st.error(f"Could not load {match_images[i]}")
            
            # Place the second image in the right column (if it exists)
            with cols[1]:
                if i + 1 < len(match_images):
                    img_path2 = os.path.join(matches_dir, match_images[i + 1])
                    img2_obj = resolve_image(img_path2, is_dataset=False)
                    if img2_obj:
                        st.image(img2_obj, caption=match_images[i + 1], use_container_width=True)
                    else:
                        st.error(f"Could not load {match_images[i+1]}")
    else:
        st.info("No images found in the loop_closure_matches folder.")
else:
    st.info("No loop_closure_matches folder found for this sequence.")
