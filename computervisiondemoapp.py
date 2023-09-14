import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from trainer import create_model
from tensorflow import argmax

# model creation

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights("./checkpoints/my_checkpoint").expect_partial()


def main():
    if "button_id" not in st.session_state:
        st.session_state["button_id"] = ""
    if "color_to_label" not in st.session_state:
        st.session_state["color_to_label"] = {}
    PAGES = {
        # "About": about,
        "Basic example": full_app,
        # "Get center coords of circles": center_circle_app,
        # "Color-based image annotation": color_annotation_app,
        # "Download Base64 encoded PNG": png_export,
        # "Compute the length of drawn arcs": compute_arc_length,
    }
    page = st.sidebar.selectbox("Page:", options=list(PAGES.keys()))
    PAGES[page]()

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://twitter.com/andfanilo">@andfanilo</a></h6>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="margin: 0.75em 0;"><a href="https://www.buymeacoffee.com/andfanilo" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a></div>',
            unsafe_allow_html=True,
        )


def full_app():
    st.sidebar.header("Configuration")
    st.markdown(
        """
    Dessinez sur le canvas ci-dessous, et obtenez la prédiction du modèle entraîné !
    * Vous pouvez configerer l’épaisseur du trait dans la barre latérale (ne pas trop réduire)
    * vous pouvez dessiner votre chiffre en plusieurs traits successifs, annuler les derniers traits faits…
    * cliquez sur le bouton pour obtenir la prédiction du modèle
    """
    )

    zoomfactor = 5
    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:",
        ("freedraw",),
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, zoomfactor)
    if drawing_mode == "point":
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFF")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=28 * zoomfactor,
        width=28 * zoomfactor,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )

    # Do something interesting with the image

    def make_prediction():
        st.image(canvas_result.image_data)
        img_data = canvas_result.image_data[:, :, 0].reshape(
            28 * zoomfactor, 28 * zoomfactor, 1
        )
        zoomed_img = cv2.resize(img_data, None, fx=1 / zoomfactor, fy=1 / zoomfactor)

        to_predict = np.array([zoomed_img])
        prediction = model.predict(to_predict)
        st.write("### La prédiction :")
        st.write(np.argmax(prediction, axis=1))
        st.write("### Les probabilités estimées pour chaque chiffre")
        st.write(prediction)

    if st.button("Obtenir la prédiction"):
        make_prediction()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Streamlit Drawable Canvas Demo", page_icon=":pencil2:"
    )
    st.title("Neural Network Digit Recognition Demo")
    st.sidebar.subheader("Configuration")
    main()
