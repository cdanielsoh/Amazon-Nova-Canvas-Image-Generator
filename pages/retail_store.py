import streamlit as st
from utils.canvas import NovaGenerator
import base64
import tempfile
from random import randint
import io
from PIL import Image, ImageDraw

# Initialize generator and session state
generator = NovaGenerator()
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.images = {}

# Streamlit UI
st.title("Product Visualization Pipeline")
st.write("Generate product images and transform them into marketing assets")

# Step 1: Initial Product Generation
with st.expander("1. Generate Base Product Image", expanded=st.session_state.step == 1):
    product_prompt = st.text_input("Product description:",
                                   "A sleek modern toaster with chrome finish")

    if st.button("Generate Product Image"):
        with st.spinner("Creating base product image..."):
            try:
                task_type = 'TEXT_IMAGE'
                params = {
                    'text': product_prompt,
                    'width': 1280,
                    'height': 720,
                    'num_images': 1,
                    'seed': randint(0, 858993459)
                }
                result = generator.generate_image(task_type, params)
                st.session_state.images['original'] = result[0]
                st.session_state.step = 2
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

    if 'original' in st.session_state.images:
        st.image(base64.b64decode(st.session_state.images['original']),
                 caption="Initial Product Image")

# Step 2: Background Removal
if st.session_state.step >= 2:
    with st.expander("2. Remove Background", expanded=st.session_state.step == 2):
        if st.button("Remove Background"):
            with st.spinner("Removing background..."):
                try:
                    task_type = 'BACKGROUND_REMOVAL'
                    params = {
                        'image': st.session_state.images['original']
                    }
                    result = generator.generate_image(task_type, params)
                    st.session_state.images['no_bg'] = result[0]
                    st.session_state.step = 3
                except Exception as e:
                    st.error(f"Background removal failed: {str(e)}")

        if 'no_bg' in st.session_state.images:
            st.image(base64.b64decode(st.session_state.images['no_bg']),
                     caption="Background Removed")

# Step 3: Add New Background (Original version without aspect ratio change)
if st.session_state.step >= 3:
    with st.expander("3. Create Marketing Background", expanded=st.session_state.step == 3):
        bg_prompt = st.text_input("Background description:",
                                  "Modern kitchen counter with morning coffee and fresh flowers")

        if st.button("Generate New Background"):
            with st.spinner("Creating enhanced background..."):
                try:
                    task_type = 'OUTPAINTING'
                    params = {
                        'text': 'A' + product_prompt + 'on a ' + bg_prompt,
                        'image': st.session_state.images['no_bg'],
                        'maskPrompt': product_prompt,
                        'outPaintingMode': "PRECISE",
                        'num_images': 1,
                        'quality': 'standard',
                        'cfg_scale': 6.5,
                        'seed': randint(0, 858993459)
                    }
                    result = generator.generate_image(task_type, params)
                    st.session_state.images['background'] = result[0]
                    st.session_state.step = 4
                except Exception as e:
                    st.error(f"Background generation failed: {str(e)}")

        if 'background' in st.session_state.images:
            st.image(base64.b64decode(st.session_state.images['background']),
                     caption="Product with New Background")

# Step 4: Video Generation
if st.session_state.step >= 4:
    with st.expander("4. Create Marketing Reel", expanded=st.session_state.step == 4):
        text_prompt = st.text_input("Video prompt: ",
                                    "Dolly forward")

        if st.button("Generate Marketing Reel"):
            with st.spinner("Creating video reel..."):
                try:
                    params = {
                        "image": st.session_state.images['background'],
                        "text": text_prompt,
                        "seed": randint(0, 858993459),
                    }

                    video_b64 = generator.generate_video(params)
                    video_bytes = base64.b64decode(video_b64)
                    st.session_state.video = video_bytes

                except Exception as e:
                    st.error(f"Video generation failed: {str(e)}")
                    st.write("Full error:", e)

        if 'video' in st.session_state:
            st.video(st.session_state.video)

            # Video download
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(st.session_state.video)
                st.download_button(
                    label="Download Marketing Reel",
                    data=st.session_state.video,
                    file_name="product_reel.mp4",
                    mime="video/mp4"
                )
