import streamlit as st
from PIL import Image
from utils.canvas import NovaGenerator
import base64
import io


def main():
    st.title("Amazon Nova Canvas Image Generator")

    generator = NovaGenerator()

    # Task type selection
    task_type = st.selectbox(
        "Select Generation Task",
        ["TEXT_IMAGE", "COLOR_GUIDED_GENERATION", "INPAINTING", "OUTPAINTING",
         "IMAGE_VARIATION", "BACKGROUND_REMOVAL", "TEXT_VIDEO"]
    )

    # Common parameters
    if task_type != "TEXT_VIDEO":
        st.header("Image Configuration")
        col1, col2 = st.columns(2)
        with col1:
            if task_type in ["TEXT_IMAGE", "COLOR_GUIDED_GENERATION"]:
                width = st.number_input("Width", min_value=320, max_value=4096, value=1024, step=64)
            if task_type != "BACKGROUND_REMOVAL":
                quality = st.selectbox("Quality", ["standard", "premium"])
                cfg_scale = st.slider("CFG Scale", min_value=1.1, max_value=10.0, value=6.5)

        with col2:
            if task_type in ["TEXT_IMAGE", "COLOR_GUIDED_GENERATION"]:
                height = st.number_input("Height", min_value=320, max_value=4096, value=1024, step=64)
            if task_type != "BACKGROUND_REMOVAL":
                seed = st.number_input("Seed (Optional)",
                                       min_value=0,
                                       max_value=4294967295,
                                       value=12,
                                       help="Random seed to use for image generation")
                num_images = st.slider("Number of Images", min_value=1, max_value=5, value=1)

    # Task-specific parameters
    st.header("Task Parameters")
    params = {}

    if task_type == "TEXT_IMAGE":
        params["text"] = st.text_area("Enter prompt")
        params["negativeText"] = st.text_area("Enter negative prompt (optional)")
        use_condition = st.checkbox("Use conditioning image")
        if use_condition:
            condition_image = st.file_uploader("Upload conditioning image", type=["png", "jpg", "jpeg"])
            control_mode = st.selectbox("Control Mode", ["CANNY_EDGE", "SEGMENTATION"])
            control_strength = st.slider("Control Strength", min_value=0.2, max_value=1.0, value=0.7, step=0.1)

            if condition_image:
                params["conditionImage"] = base64.b64encode(condition_image.read()).decode('utf-8')
                params["controlMode"] = control_mode
                params["controlStrength"] = control_strength

    elif task_type == "INPAINTING":
        base_image = st.file_uploader("Upload base image", type=["png", "jpg", "jpeg"])
        if base_image:
            params["image"] = base64.b64encode(base_image.read()).decode('utf-8')

        params["text"] = st.text_area("Enter inpainting prompt")
        params["negativeText"] = st.text_input("Negative prompt (optional)")

        mask_option = st.radio("Mask Input Type", ["Image Mask", "Text Prompt"])
        if mask_option == "Image Mask":
            mask_image = st.file_uploader("Upload mask image", type=["png", "jpg", "jpeg"])
            if mask_image:
                params["maskImage"] = base64.b64encode(mask_image.read()).decode('utf-8')
        else:
            params["maskPrompt"] = st.text_input("Enter mask prompt (e.g., 'the dog's face')")

    elif task_type == "OUTPAINTING":
        base_image = st.file_uploader("Upload base image", type=["png", "jpg", "jpeg"])
        if base_image:
            params["image"] = base64.b64encode(base_image.read()).decode('utf-8')

        params["outPaintingMode"] = st.radio("Outpainting Mode", ["DEFAULT", "PRECISE"])
        params["text"] = st.text_area("Describe the complete scene after editing:")

        mask_option = st.radio("Mask Input Type", ["Image Mask", "Text Prompt"])
        if mask_option == "Image Mask":
            mask_image = st.file_uploader("Upload mask image (white=generate areas)", type=["png", "jpg", "jpeg"])
            if mask_image:
                params["maskImage"] = base64.b64encode(mask_image.read()).decode('utf-8')
        else:
            params["maskPrompt"] = st.text_input("Enter mask prompt (e.g., 'expand background left')")

    elif task_type == "IMAGE_VARIATION":
        ref_images = st.file_uploader("Upload reference images", type=["png", "jpg", "jpeg"],
                                      accept_multiple_files=True)
        if ref_images:
            params["images"] = [base64.b64encode(img.read()).decode('utf-8') for img in ref_images]
        prompt = st.text_area("Enter prompt (optional)")
        if prompt:
            params["text"] = prompt

    elif task_type == "COLOR_GUIDED_GENERATION":
        params["text"] = st.text_area("Enter prompt")
        params["colors"] = st.text_input("Enter hex colors (comma-separated, e.g., #FF0000,#00FF00)")
        ref_image = st.file_uploader("Upload reference image (optional)", type=["png", "jpg", "jpeg"])
        if ref_image:
            params["referenceImage"] = base64.b64encode(ref_image.read()).decode('utf-8')

    elif task_type == "BACKGROUND_REMOVAL":
        image = st.file_uploader("Upload image for background removal", type=["png", "jpg", "jpeg"])
        if image:
            params["image"] = base64.b64encode(image.read()).decode('utf-8')

    elif task_type == "TEXT_VIDEO":
        st.header("Video Generation Parameters")

        # Nova Reel requires 1280x720 resolution
        st.info("Nova Reel generates videos at a fixed resolution of 1280x720 pixels, 24fps, and 6 seconds duration.")

        # Prompt for video generation
        params["text"] = st.text_area("Enter video prompt (max 512 characters)",
                                      max_chars=512,
                                      help="Describe the scene and camera movements. Place camera movement descriptions at the start or end.")

        # Option to use an image as starting point
        use_image = st.checkbox("Use reference image as starting point")
        if use_image:
            ref_image = st.file_uploader("Upload reference image (must be 1280x720)", type=["png", "jpg", "jpeg"])
            if ref_image:
                # Optional validation of image dimensions
                image = Image.open(ref_image)
                if image.size != (1280, 720):
                    st.warning("Image should be exactly 1280x720 pixels.")

                params["image"] = base64.b64encode(ref_image.getvalue()).decode('utf-8')

        # Seed for reproducibility
        params["seed"] = st.number_input("Seed (Optional)",
                                         min_value=0,
                                         max_value=4294967295,
                                         value=12,
                                         help="Random seed for reproducible video generation")

    if st.button("Generate" + (" Video" if task_type == "TEXT_VIDEO" else " Image")):
        try:
            # Build generation parameters
            generation_params = {
                'taskType': task_type,
                'width': width if task_type in ["TEXT_IMAGE", "COLOR_GUIDED_GENERATION"] else None,
                'height': height if task_type in ["TEXT_IMAGE", "COLOR_GUIDED_GENERATION"] else None,
                'quality': quality if task_type not in ["BACKGROUND_REMOVAL", "TEXT_VIDEO"] else None,
                'cfg_scale': cfg_scale if task_type not in ["BACKGROUND_REMOVAL", "TEXT_VIDEO"] else None,
                'num_images': num_images if task_type not in ["BACKGROUND_REMOVAL", "TEXT_VIDEO"] else 1,
                'seed': seed if task_type not in ["BACKGROUND_REMOVAL", "TEXT_VIDEO"] else None
            }

            # Remove None values
            generation_params = {k: v for k, v in generation_params.items() if v is not None}

            # Add task-specific parameters
            generation_params.update(params)

            if task_type == "TEXT_VIDEO":
                with st.spinner("Generating video... This may take around 3 minutes."):
                    video_b64 = generator.generate_video(params=generation_params)

                    if video_b64:
                        video_bytes = base64.b64decode(video_b64)
                        st.video(video_bytes)

            else:
                # Existing image generation code
                result = generator.generate_image(
                    task_type=task_type,
                    params=generation_params
                )


                # Handle different return types
                images = result if isinstance(result, list) else [result]

                # Display results
                for idx, img_data in enumerate(images):
                    image = Image.open(io.BytesIO(base64.b64decode(img_data)))
                    st.image(image, caption=f"Generated Image {idx + 1}")

                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='PNG')
                    st.download_button(
                        label=f"Download Image {idx + 1}",
                        data=img_bytes.getvalue(),
                        file_name=f'generated_{task_type.lower()}_{idx}.png',
                        mime="image/png"
                    )

        except Exception as e:
            st.error(f"Error generating image: {str(e)}")


if __name__ == "__main__":
    main()
