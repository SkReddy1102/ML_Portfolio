from pathlib import Path

import PIL.Image
import streamlit as st
from ultralytics import YOLO

import settings


st.set_page_config(
    page_title="Farm Insect Classification Using YOLOv8",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_classifier(model_path: Path):
    return YOLO(model_path)


st.title("Farm Insect Classification Using YOLOv8")

try:
    model = load_classifier(settings.CUSTOM_MODEL)
except Exception as ex:
    st.error(f"Unable to load classifier model: {settings.CUSTOM_MODEL}")
    st.error(ex)
    st.stop()

st.sidebar.header("Image Upload")
confidence = st.sidebar.slider("Minimum Confidence", 0, 100, 25) / 100
source_img = st.sidebar.file_uploader(
    "Choose an insect image",
    type=("jpg", "jpeg", "png", "bmp", "webp"),
)

col1, col2 = st.columns(2)

with col1:
    if source_img is None:
        image = PIL.Image.open(settings.DEFAULT_IMAGE)
        st.image(image, caption="Default Image", use_container_width=True)
    else:
        image = PIL.Image.open(source_img)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    if source_img is None:
        st.info("Upload an insect image to classify it.")
    elif st.sidebar.button("Classify Insect", type="primary"):
        result = model.predict(image, verbose=False)[0]

        if result.probs is None:
            st.error("The loaded model did not return classification probabilities.")
            st.stop()

        top_class_id = int(result.probs.top1)
        top_confidence = float(result.probs.top1conf)
        insect_name = model.names[top_class_id]

        st.image(image, caption="Classified Image", use_container_width=True)

        if top_confidence < confidence:
            st.warning(
                f"Top prediction is below the selected confidence threshold: "
                f"{top_confidence:.1%}"
            )

        st.subheader("Detected Insect")
        st.success(f"{insect_name} ({top_confidence:.1%} confidence)")

        if result.probs.top5:
            st.caption("Top predictions")
            for class_id in result.probs.top5[:5]:
                score = float(result.probs.data[class_id])
                st.write(f"{model.names[int(class_id)]}: {score:.1%}")
    else:
        st.info("Click **Classify Insect** to run the YOLOv8 classifier.")
