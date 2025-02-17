import streamlit as st
from PIL import Image
from utils.canvas import NovaCanvasGenerator
import base64
import io


def main():
    st.title("Amazon Nova Canvas Image Generator")

    generator = NovaCanvasGenerator()

    # Task type selection
    task_type = st.selectbox(
        "Select Generation Task",
        ["TEXT_IMAGE", "INPAINTING", "OUTPAINTING", "IMAGE_VARIATION", "COLOR_GUIDED_GENERATION"]
    )

    # Common parameters
    st.header("Image Configuration")
    col1, col2 = st.columns(2)
    with col1:
        width = st.number_input("Width", min_value=320, max_value=4096, value=1024, step=64)
        quality = st.selectbox("Quality", ["standard", "premium"])
    with col2:
        height = st.number_input("Height", min_value=320, max_value=4096, value=1024, step=64)
        cfg_scale = st.slider("CFG Scale", min_value=1.1, max_value=10.0, value=6.5)

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
        params["image"] = st.file_uploader("Upload base image", type=["png", "jpg", "jpeg"])
        params["text"] = st.text_area("Enter inpainting prompt")

        # Mask selection
        mask_option = st.radio("Mask Input Type", ["Image Mask", "Text Prompt"])

        if mask_option == "Image Mask":
            params["maskImage"] = st.file_uploader("Upload mask image", type=["png", "jpg", "jpeg"])
            if params.get("maskImage"):
                # Option to process mask image
                if st.checkbox("Process mask image (remove background)"):
                    mask_base64 = base64.b64encode(params["maskImage"].read()).decode('utf-8')
                    params["maskImage"] = generator.generate_image('BACKGROUND_REMOVAL', {'image': mask_base64})[0]
        else:
            params["maskPrompt"] = st.text_input("Enter mask prompt (e.g., 'the dog's face')")

        # Validation
        if not params.get("maskImage") and not params.get("maskPrompt"):
            st.error("You must provide either a mask image or mask prompt")
        elif params.get("maskImage") and params.get("maskPrompt"):
            st.error("You can only use either mask image or mask prompt, not both")



    elif task_type == "OUTPAINTING":
        params["image"] = st.file_uploader("Upload base image", type=["png", "jpg", "jpeg"])

        # Mask selection
        mask_option = st.radio("Mask Input Type", ["Image Mask", "Text Prompt"])
        params["outPaintingMode"] = st.radio("Outpainting Mode", ["DEFAULT", "PRECISE"])

        if mask_option == "Image Mask":
            params["maskImage"] = st.file_uploader("Upload mask image", type=["png", "jpg", "jpeg"])
            if params.get("maskImage"):
                # Option to use background removal on mask
                if st.checkbox("Process mask image"):
                    mask_base64 = base64.b64encode(params["maskImage"].read()).decode('utf-8')
                    params["maskImage"] = generator.generate_image('BACKGROUND_REMOVAL', {'image': mask_base64})[0]
        else:
            params["maskPrompt"] = st.text_input("Enter mask prompt (e.g., 'expand left side by 20%')")

        # Validation
        if not params.get("maskImage") and not params.get("maskPrompt"):
            st.error("You must provide either a mask image or mask prompt")
        elif params.get("maskImage") and params.get("maskPrompt"):
            st.error("You can only use either mask image or mask prompt, not both")


    elif task_type == "IMAGE_VARIATION":
        params["images"] = st.file_uploader("Upload reference images", type=["png", "jpg", "jpeg"],
                                            accept_multiple_files=True)
        params["text"] = st.text_area("Enter prompt (optional)")

    elif task_type == "COLOR_GUIDED_GENERATION":
        params["text"] = st.text_area("Enter prompt")
        params["colors"] = st.text_input("Enter hex colors (comma-separated, e.g., #FF0000,#00FF00)")

        # Add reference image option
        reference_image = st.file_uploader("Upload reference image (optional)", type=["png", "jpg", "jpeg"])
        if reference_image:
            params["referenceImage"] = base64.b64encode(reference_image.read()).decode('utf-8')

    if st.button("Generate Image"):
        try:
            # Collect all parameters
            generation_params = {
                'width': width,
                'height': height,
                'quality': quality,
                'cfg_scale': cfg_scale,
                'num_images': num_images
            }

            # Add task-specific parameters
            generation_params.update(params)

            # Process uploaded images if present
            if "image" in params and params["image"]:
                generation_params["image"] = base64.b64encode(params["image"].read()).decode('utf-8')
            if "maskImage" in params and params["maskImage"]:
                generation_params["maskImage"] = base64.b64encode(params["maskImage"].read()).decode('utf-8')
            if "images" in params and params["images"]:
                generation_params["images"] = [base64.b64encode(img.read()).decode('utf-8') for img in params["images"]]

            # Generate image
            response = generator.generate_image(
                task_type=task_type,
                params=generation_params
            )

            # Display generated image
            image_data = base64.b64decode(response)
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption="Generated Image")

        except Exception as e:
            st.error(f"Error generating image: {str(e)}")


if __name__ == "__main__":
    main()
