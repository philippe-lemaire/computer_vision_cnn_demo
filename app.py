import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import pandas as pd
from trainer import create_model


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
            "<h6>Made with love by Philippe du Tubà</h6>",
            unsafe_allow_html=True,
        )


def full_app():
    st.sidebar.header("Configuration")
    st.markdown(
        """
    Dessinez sur le canvas ci-dessous, et obtenez la prédiction du modèle entraîné !
    * vous pouvez dessiner votre chiffre en plusieurs traits successifs, annuler les derniers traits faits…
    * cliquez sur le bouton pour obtenir la prédiction du modèle
    """
    )

    zoomfactor = 10
    # Specify canvas parameters in application
    drawing_mode = "freedraw"

    stroke_width = st.sidebar.slider("Stroke width: ", 2, 25, 2 * zoomfactor)
    stroke_color = "#FFF"
    bg_color = "#000"
    # bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    # realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=None,
        update_streamlit=True,
        height=28 * zoomfactor,
        width=28 * zoomfactor,
        drawing_mode=drawing_mode,
        # point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="full_app",
    )

    # Do something interesting with the image

    def make_prediction():
        # st.image(canvas_result.image_data)
        img_data = canvas_result.image_data[:, :, 0].reshape(
            28 * zoomfactor, 28 * zoomfactor, 1
        )
        zoomed_img = cv2.resize(img_data, None, fx=1 / zoomfactor, fy=1 / zoomfactor)

        to_predict = np.array([zoomed_img])
        prediction = model.predict(to_predict)
        st.write("### La prédiction :")
        st.write(f"## {str(np.argmax(prediction, axis=1)[0])}")
        st.write("### Les probabilités estimées pour chaque chiffre")
        st.write(prediction)

    if st.button("Obtenir la prédiction"):
        make_prediction()


if __name__ == "__main__":
    title = "Neural Network Digit vision demo"
    st.set_page_config(page_title=title, page_icon=":pencil2:")
    st.title(title)
    st.sidebar.subheader("Configuration")
    main()
