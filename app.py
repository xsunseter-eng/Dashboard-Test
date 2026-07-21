import streamlit as st
import scipy.io
import plotly.graph_objects as pl
import plotly.express as px
import pandas as pd
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
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
NETVLAD_DIR = BASE_DIR

# ========================================================================
# SEQUENCE VIDEO LINKS (HOCA BURADAN LİNKLERİ DÜZENLEYEBİLİR)
# ========================================================================
SEQUENCE_VIDEOS = {
    "spot_indoor_obstacles": "https://www.youtube.com/watch?v=0AtS6m-pwZI",
    "spot_forest_hard": "https://www.youtube.com/watch?v=MVPtSsw274s",
    "spot_indoor_building_loop": "https://www.youtube.com/watch?v=oV3owInlcCY",
    "spot_outdoor_day_skatepark_1": "https://www.youtube.com/watch?v=M5C0eX6gyNo",
    "spot_outdoor_day_skatepark_2": "https://www.youtube.com/watch?v=PcSp6wdFgU8",
}

def get_sequence_info(folder_name, version_dir):
    base_name = folder_name
    if base_name.startswith("realtime_results_v2_"):
        base_name = base_name.replace("realtime_results_v2_", "")
    if base_name.endswith("_data_images_rgb"):
        base_name = base_name.replace("_data_images_rgb", "")
    
    if base_name == "indoor_obstacles":
        base_name = "spot_indoor_obstacles"

    image_dir = os.path.join(NETVLAD_DIR, f"{base_name}_data_images_rgb")
    video_url = SEQUENCE_VIDEOS.get(base_name, f"https://m3ed-dist.s3.us-west-2.amazonaws.com/processed/{base_name}/{base_name}_rgb.mp4")
    
    return {
        "name": folder_name,
        "resultDir": os.path.join(version_dir, folder_name),
        "imageDir": image_dir,
        "video": video_url
    }

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
        except Exception:
            pass # LFS pointer or invalid image, fallback to HuggingFace

    # 2. Fallback to HuggingFace
    rel_path = os.path.relpath(path, BASE_DIR).replace("\\", "/")
    
    if is_dataset:
        repo = "M3ED_frames"
    else:
        repo = "M3ED_loop_closure_results"
        # Strip Results/v1/ because HF dataset root contains its children
        if rel_path.startswith("Results/v1/"):
            rel_path = rel_path.replace("Results/v1/", "", 1)

    url = f"https://huggingface.co/datasets/e230450/{repo}/resolve/main/{rel_path}"

    try:
        try:
            img_bytes = fetch_image_bytes(url)
        except Exception:
            img_bytes = None

        if img_bytes is None and url.lower().endswith((".jpg", ".jpeg")):
            url_png = url.rsplit(".", 1)[0] + ".png"
            try:
                img_bytes = fetch_image_bytes(url_png)
            except Exception:
                img_bytes = None
            
        if img_bytes:
            return Image.open(BytesIO(img_bytes))
        else:
            raise Exception("Image not found")
    except Exception as e:
        st.error(f"Could not load image from HF: {url}")
        return None
# ========================================================================
# Main App & State Initialization
# ========================================================================

# Inject Custom CSS for Academic Formatting
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Times New Roman', Times, serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Times New Roman', Times, serif !important;
    }
    .main-title {
        text-align: center;
        font-size: 3rem !important;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 20px;
        border-bottom: 2px solid #000;
        padding-bottom: 10px;
    }
    /* IEEE Style Tables */
    [data-testid="stDataFrame"] table {
        border-top: 2px solid black !important;
        border-bottom: 2px solid black !important;
        border-collapse: collapse !important;
    }
    [data-testid="stDataFrame"] th {
        border-bottom: 1px solid black !important;
        border-top: none !important;
        background-color: transparent !important;
        font-weight: bold !important;
    }
    [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
        border-left: none !important;
        border-right: none !important;
        background-color: white !important;
    }
    /* Divider Customization */
    hr {
        border-top: 1px solid black !important;
        margin: 1.5em 0 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Loop Closure Analysis Tool</div>", unsafe_allow_html=True)
st.markdown("Use the **Image Comparison Slider**, manually configure frames, or click the **Similarity Matrix** to explore loop closures.")

if not os.path.exists(RESULTS_DIR):
    st.error(f"Results directory not found at {RESULTS_DIR}")
    st.stop()

# Dynamic version selection
versions = sorted([d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))])
if not versions:
    st.error("No versions found in Results directory.")
    st.stop()

# Dropdowns side-by-side
sel_col1, sel_col2 = st.columns([1, 3])
with sel_col1:
    selected_version = st.selectbox("Select Version", versions)

version_dir = os.path.join(RESULTS_DIR, selected_version)
seq_names = sorted([d for d in os.listdir(version_dir) if os.path.isdir(os.path.join(version_dir, d))])

if not seq_names:
    st.warning(f"No sequences found in {selected_version}")
    st.stop()

seq_dict = {name: get_sequence_info(name, version_dir) for name in seq_names}

with sel_col2:
    selected_seq_name = st.selectbox("Select a sequence", list(seq_dict.keys()))

selected_seq = seq_dict[selected_seq_name]

# ========================================================================
# DASHBOARD SUMMARY & VIDEO (NEW GUI ELEMENTS)
# ========================================================================
st.divider()

# 1. Full-width Video
st.header("Loop Closure GUI")
demo_video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
st.video(demo_video_url)

# 2. Figure and Table Side-by-Side
summary_col1, summary_col2 = st.columns(2)

try:
    df_summary = pd.read_csv("dashboard_summary.csv")
    
    with summary_col1:
        st.subheader("Figure")
        df_summary["Pair"] = df_summary["Frame1"].astype(str) + " - " + df_summary["Frame2"].astype(str)
        fig_summary = px.bar(
            df_summary, 
            x="Pair", 
            y=["Raw Matches", "Fundamental Inliers", "Pose Inliers"],
            barmode="group"
        )
        st.plotly_chart(fig_summary, use_container_width=True)
        
    with summary_col2:
        st.subheader("Table")
        st.dataframe(df_summary, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load dashboard_summary.csv: {e}")

st.divider()
st.header("Image Retrival")

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
    img_row_path = os.path.join(selected_seq["imageDir"], f"image_{frame_row:05d}.png")
    img_col_path = os.path.join(selected_seq["imageDir"], f"image_{frame_col:05d}.png")

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
        st.image(img_row_obj, caption=f"Fig. A: Row Frame {frame_row}")
    with ind_col2:
        st.image(img_col_obj, caption=f"Fig. B: Col Frame {frame_col}")
        


# ========================================================================
# 5. LOOP CLOSURE MATCHES
# ========================================================================
st.divider()
st.header("Retrival Results")

# Define the path to the loop closure matches folder based on the selected sequence
matches_dir = os.path.join(selected_seq["resultDir"], "loop_closure_matches")

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

                img1_obj = resolve_image(img_path1, is_dataset=False)
                if img1_obj:
                    st.image(img1_obj, caption=f"Fig: {match_images[i]}", use_container_width=True)
                else:
                    st.error(f"Could not load {match_images[i]}")
            
            # Place the second image in the right column (if it exists)
            with cols[1]:
                if i + 1 < len(match_images):
                    img_path2 = os.path.join(matches_dir, match_images[i + 1])
                    img2_obj = resolve_image(img_path2, is_dataset=False)
                    if img2_obj:
                        st.image(img2_obj, caption=f"Fig: {match_images[i + 1]}", use_container_width=True)
                    else:
                        st.error(f"Could not load {match_images[i+1]}")
    else:
        st.info("No images found in the loop_closure_matches folder.")
else:
    st.info("No loop_closure_matches folder found for this sequence.")

# ========================================================================
# 6. GEOMETRY CHECK
# ========================================================================
st.divider()
st.header("Geometry Check")

st.markdown("#### GUI Output")
gui_images = sorted([f for f in os.listdir(selected_seq["resultDir"]) if f.startswith("realtime_dashboard_") and f.endswith(".png")])
if gui_images:
    for img_file in gui_images:
        img_path = os.path.join(selected_seq["resultDir"], img_file)
        img_obj = resolve_image(img_path, is_dataset=False)
        if img_obj:
            st.image(img_obj, caption=f"Fig: {img_file}", use_container_width=True)
        else:
            st.error(f"Could not load {img_file}")
else:
    st.info("No GUI Output images found for this sequence.")


st.markdown("#### Overall Summary")
import pandas as pd
all_csvs = []
for seq_name in seq_names:
    csv_path = os.path.join(version_dir, seq_name, "dashboard_summary.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df.insert(0, "Sequence", seq_name)
            all_csvs.append(df)
        except Exception as e:
            st.warning(f"Could not read CSV for {seq_name}: {e}")

if all_csvs:
    master_df = pd.concat(all_csvs, ignore_index=True)
    st.dataframe(master_df, use_container_width=True)
    
    # Sort the dataframe by Coverage Frame1 (%) to make piecewise linear plots look correct
    if "Coverage Frame1 (%)" in master_df.columns:
        master_df_sorted = master_df.sort_values(by="Coverage Frame1 (%)")
        
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            if "Rotation Error (deg)" in master_df_sorted.columns:
                fig1 = px.line(master_df_sorted, x="Coverage Frame1 (%)", y="Rotation Error (deg)", color="Sequence", markers=True, title="Rotation Error vs Coverage")
                st.plotly_chart(fig1, use_container_width=True)
            
        with plot_col2:
            if "Translation Error (deg)" in master_df_sorted.columns:
                fig2 = px.line(master_df_sorted, x="Coverage Frame1 (%)", y="Translation Error (deg)", color="Sequence", markers=True, title="Translation Error vs Coverage")
                st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No dashboard_summary.csv files found in this version.")
