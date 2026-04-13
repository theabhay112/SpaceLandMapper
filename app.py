# SpaceLandMapper - Streamlit Deployment App
# This file is responsible for loading the final trained land-classification model,
# accepting a satellite-style image from the user, and providing either:
# 1. a single-image prediction, or
# 2. a grid-based land distribution map.
# The model used here is the final EfficientNetB0-based classifier selected
# after comparison with the baseline CNN.

import io
from collections import Counter

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageDraw
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image


########################################
# Page configuration
# This section sets the page title and
# uses a wide layout for better display.
########################################
st.set_page_config(
    page_title="Land Classification AI System",
    layout="wide"
)

########################################
# Model and class settings
# MODEL_PATH points to the final saved model.
# MODEL_INPUT_SIZE defines the image size used by EfficientNet.
# CLASS_LABELS stores the ten EuroSAT classes.
# CLASS_COLOR_MAP links each class to a display color.
########################################
MODEL_PATH = "Outputs/Models/efficientnet_frozen_best.keras"
MODEL_INPUT_SIZE = (224, 224)

CLASS_LABELS = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

CLASS_COLOR_MAP = {
    "AnnualCrop": (255, 215, 0),
    "Forest": (34, 139, 34),
    "HerbaceousVegetation": (144, 238, 144),
    "Highway": (128, 128, 128),
    "Industrial": (105, 105, 105),
    "Pasture": (189, 183, 107),
    "PermanentCrop": (218, 165, 32),
    "Residential": (220, 20, 60),
    "River": (30, 144, 255),
    "SeaLake": (0, 0, 139),
}

########################################
# Helper functions
# These functions support color formatting,
# image export, and image resizing.
########################################
def rgb_to_hex(rgb):
    # Convert RGB tuple into hex format for the legend.
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def pil_to_download_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    # Convert a PIL image into bytes so it can be downloaded.
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    buffer.seek(0)
    return buffer.read()


def resize_large_image(img: Image.Image, max_width: int = 1024) -> Image.Image:
    # Resize large uploaded images to reduce prediction time
    # while keeping the original aspect ratio.
    width, height = img.size
    if width <= max_width:
        return img

    scale = max_width / width
    new_size = (int(width * scale), int(height * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS)

########################################
# Model loading
# This function loads the final trained model.
# compile=False is used because the app only
# performs inference, not training.
########################################
@st.cache_resource
def load_model(model_path: str):
    # Load the saved EfficientNet model once
    # and reuse it during Streamlit reruns.
    return tf.keras.models.load_model(model_path, compile=False)


model = load_model(MODEL_PATH)

########################################
# Image preprocessing and prediction
# These functions prepare images for the model
# and generate class predictions.
########################################
def preprocess_pil_for_model(img: Image.Image) -> np.ndarray:
    # Resize and preprocess a PIL image so it matches
    # the input format expected by EfficientNet.
    img_resized = img.resize(MODEL_INPUT_SIZE)
    img_array = keras_image.img_to_array(img_resized)
    img_array = preprocess_input(img_array)
    return img_array


def predict_single_image(img: Image.Image, model):
    # Predict the main land-use class for a full image.
    # Also returns the confidence score and top 3 predictions.
    img_array = preprocess_pil_for_model(img)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_LABELS[pred_idx]
    confidence = float(preds[pred_idx])

    top3_idx = np.argsort(preds)[-3:][::-1]
    top3 = [(CLASS_LABELS[i], float(preds[i])) for i in top3_idx]

    return pred_class, confidence, top3


def split_image_into_tiles(img: Image.Image, tile_size: int):
    # Split a large image into smaller square tiles.
    # Returns the tile images and their x/y positions.
    width, height = img.size
    tiles = []
    tile_meta = []

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = img.crop((x, y, min(x + tile_size, width), min(y + tile_size, height)))
            tiles.append(tile)
            tile_meta.append({"x": x, "y": y})

    return tiles, tile_meta


def predict_tiles_batch(tiles, model, batch_size: int = 32):
    # Predict all image tiles in batches to improve speed.
    tile_arrays = [preprocess_pil_for_model(tile) for tile in tiles]
    tile_arrays = np.stack(tile_arrays, axis=0)

    preds = model.predict(tile_arrays, batch_size=batch_size, verbose=0)
    pred_indices = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)

    pred_classes = [CLASS_LABELS[i] for i in pred_indices]
    confidences = [float(c) for c in confidences]

    return pred_classes, confidences


def classify_grid(img: Image.Image, model, tile_size: int = 64, batch_size: int = 32):
    # This function is responsible for grid-based classification.
    # It splits the image into tiles, predicts each tile,
    # and stores class and confidence information.
    tiles, tile_meta = split_image_into_tiles(img, tile_size)
    pred_classes, confidences = predict_tiles_batch(tiles, model, batch_size=batch_size)

    tile_details = []
    for meta, pred_class, confidence in zip(tile_meta, pred_classes, confidences):
        tile_details.append({
            "x": meta["x"],
            "y": meta["y"],
            "predicted_class": pred_class,
            "confidence": confidence
        })

    return tile_details

########################################
# Map generation
# These functions transform tile predictions
# into visual outputs for the user.
########################################
def create_color_map(img: Image.Image, tile_details, tile_size: int = 64, draw_grid: bool = True):
    # Create a pure color classification map where each tile
    # is filled with the predicted class color.
    width, height = img.size
    color_map = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(color_map)

    for tile in tile_details:
        x, y = tile["x"], tile["y"]
        pred_class = tile["predicted_class"]
        color = CLASS_COLOR_MAP.get(pred_class, (0, 0, 0))

        draw.rectangle(
            [x, y, min(x + tile_size, width), min(y + tile_size, height)],
            fill=color
        )

        if draw_grid:
            draw.rectangle(
                [x, y, min(x + tile_size, width), min(y + tile_size, height)],
                outline=(255, 255, 255),
                width=1
            )

    return color_map


def create_overlay_map(img: Image.Image, tile_details, tile_size: int = 64, alpha: int = 110, draw_grid: bool = True):
    # Create a semi-transparent overlay map on top of the
    # original image so predictions can be compared visually.
    width, height = img.size
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for tile in tile_details:
        x, y = tile["x"], tile["y"]
        rgb = CLASS_COLOR_MAP.get(tile["predicted_class"], (0, 0, 0))
        rgba = (rgb[0], rgb[1], rgb[2], alpha)

        draw.rectangle(
            [x, y, min(x + tile_size, width), min(y + tile_size, height)],
            fill=rgba
        )

        if draw_grid:
            draw.rectangle(
                [x, y, min(x + tile_size, width), min(y + tile_size, height)],
                outline=(255, 255, 255, 180),
                width=1
            )

    return Image.alpha_composite(base, overlay).convert("RGB")


def calculate_percentages(tile_details):
    # Calculate tile counts and class percentages
    # to summarise land distribution in the image.
    classes = [tile["predicted_class"] for tile in tile_details]
    counts = Counter(classes)
    total = sum(counts.values())

    percentages = {
        cls: round((count / total) * 100, 2)
        for cls, count in counts.items()
    }

    return counts, percentages


def show_legend():
    # Display the class-color legend so users can
    # interpret the generated map correctly.
    st.subheader("Legend")
    for cls in CLASS_LABELS:
        color_hex = rgb_to_hex(CLASS_COLOR_MAP[cls])
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; margin-bottom:6px;">
                <div style="
                    width:18px;
                    height:18px;
                    background:{color_hex};
                    border:1px solid #333;
                    margin-right:8px;
                "></div>
                <div>{cls}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

########################################
# Streamlit user interface
# This section handles user input, mode selection,
# and result visualisation.
########################################
st.title("Land Classification AI System")
st.caption("Deep learning-based land-use classification using EfficientNetB0 and tile-based mapping.")

mode = st.radio(
    "Choose analysis mode",
    ["Single Prediction", "Grid Mapping"],
    horizontal=True
)

with st.sidebar:
    st.header("Settings")
    tile_size = st.selectbox("Tile size", [32, 64, 128], index=1)
    draw_grid = st.checkbox("Show grid lines", value=True)
    overlay_alpha = st.slider("Overlay transparency", min_value=50, max_value=200, value=110, step=10)
    max_preview_width = st.selectbox("Max analysis image width", [768, 1024, 1280], index=1)

uploaded_file = st.file_uploader("Upload a satellite-style image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    resized_img = resize_large_image(img, max_width=max_preview_width)

    if img.size != resized_img.size:
        st.info(f"Large image detected. Resized from {img.size} to {resized_img.size} for faster grid analysis.")

    ########################################
    # Single-image prediction mode
    # Predict one dominant class for the full image.
    ########################################
    if mode == "Single Prediction":
        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.image(resized_img, caption="Uploaded Image", use_column_width=True)

        with col2:
            if st.button("Predict Image"):
                with st.spinner("Running prediction..."):
                    pred_class, confidence, top3 = predict_single_image(resized_img, model)

                st.success(f"Prediction: {pred_class} ({confidence:.2%})")
                st.subheader("Top 3 Predictions")
                for label, score in top3:
                    st.write(f"{label}: {score:.2%}")

    ########################################
    # Grid-mapping mode
    # Split the image into tiles, classify each tile,
    # and display land distribution across the image.
    ########################################
    else:
        if st.button("Run Grid Mapping"):
            with st.spinner("Classifying image tiles..."):
                tile_details = classify_grid(
                    resized_img,
                    model,
                    tile_size=tile_size,
                    batch_size=32
                )

                color_map = create_color_map(
                    resized_img,
                    tile_details,
                    tile_size=tile_size,
                    draw_grid=draw_grid
                )

                overlay_map = create_overlay_map(
                    resized_img,
                    tile_details,
                    tile_size=tile_size,
                    alpha=overlay_alpha,
                    draw_grid=draw_grid
                )

                counts, percentages = calculate_percentages(tile_details)

            col1, col2 = st.columns(2)

            with col1:
                st.image(resized_img, caption="Original Image", use_column_width=True)

            with col2:
                st.image(overlay_map, caption="Overlay Land Classification Map", use_column_width=True)

            st.image(color_map, caption="Pure Color Grid Map", use_column_width=True)

            stats_col, legend_col = st.columns([1.4, 1])

            with stats_col:
                st.subheader("Land Distribution")
                for cls, pct in sorted(percentages.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"{cls}: {pct}%")

                st.subheader("Tile Counts")
                for cls, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"{cls}: {count}")

            with legend_col:
                show_legend()

            st.subheader("Download Results")
            dl_col1, dl_col2 = st.columns(2)

            with dl_col1:
                st.download_button(
                    label="Download Overlay Map",
                    data=pil_to_download_bytes(overlay_map),
                    file_name="overlay_land_classification_map.png",
                    mime="image/png"
                )

            with dl_col2:
                st.download_button(
                    label="Download Color Grid Map",
                    data=pil_to_download_bytes(color_map),
                    file_name="color_grid_land_classification_map.png",
                    mime="image/png"
                )