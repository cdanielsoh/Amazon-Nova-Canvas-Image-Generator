import boto3
import base64
import json
import io
import uuid
import time
from typing import List, Union
from PIL import Image


class NovaGenerator:
    def __init__(self):
        self.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
        self.image_model_id = 'amazon.nova-canvas-v1:0'
        self.video_model_id = 'amazon.nova-reel-v1:0'  # Hypothetical video model ID

    def _ensure_valid_image_format(self, image_base64):
        """
        Ensures the image is in a format acceptable to Nova Canvas.
        Converts PNG with transparency to JPEG with white background.
        """
        try:
            # Decode base64 to image
            image_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_data))

            # Check if image has transparency
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                # Convert to RGB with white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                else:
                    background.paste(img)
                img = background

            # Save as JPEG (no transparency)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)

            # Return base64 encoded JPEG
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            print(f"Error processing image: {e}")
            return image_base64  # Return original if processing fails

    def generate_image(self, task_type: str, params: dict) -> Union[List[str], str]:
        """
        Delegator function that routes to the specialized image generation functions.
        Maintains backward compatibility with existing code.
        """
        # Extract common parameters from the params dict
        num_images = params.get('num_images', 1)
        quality = params.get('quality', 'standard')
        cfg_scale = params.get('cfg_scale', 6.5)
        seed = params.get('seed', 0)

        # Route to the appropriate specialized function based on task_type
        if task_type == 'TEXT_IMAGE':
            return self.text_image(
                text=params.get('text', ''),
                negative_text=params.get('negativeText', ''),
                num_images=num_images,
                quality=quality,
                cfg_scale=cfg_scale,
                seed=seed,
                height=params.get('height'),
                width=params.get('width'),
                condition_image=params.get('conditionImage'),
                control_mode=params.get('controlMode', 'CANNY_EDGE'),
                control_strength=params.get('controlStrength', 0.7)
            )
        elif task_type == 'INPAINTING':
            return self.inpainting(
                image=params.get('image', ''),
                text=params.get('text', ''),
                negative_text=params.get('negativeText', ''),
                mask_image=params.get('maskImage'),
                mask_prompt=params.get('maskPrompt'),
                num_images=num_images,
                quality=quality,
                cfg_scale=cfg_scale,
                seed=seed
            )
        elif task_type == 'OUTPAINTING':
            return self.outpainting(
                image=params.get('image', ''),
                text=params.get('text', ''),
                mask_image=params.get('maskImage'),
                mask_prompt=params.get('maskPrompt'),
                outpainting_mode=params.get('outPaintingMode', 'DEFAULT'),
                num_images=num_images,
                quality=quality,
                cfg_scale=cfg_scale,
                seed=seed
            )
        elif task_type == 'COLOR_GUIDED_GENERATION':
            colors = params.get('colors', '')
            if isinstance(colors, str):
                colors = colors.split(',')
            return self.color_guided_generation(
                text=params.get('text', ''),
                width=params.get('width'),
                height=params.get('height'),
                colors=colors,
                reference_image=params.get('referenceImage'),
                num_images=num_images,
                quality=quality,
                cfg_scale=cfg_scale,
                seed=seed
            )
        elif task_type == 'BACKGROUND_REMOVAL':
            return self.background_removal(params.get('image', ''))
        elif task_type == 'IMAGE_VARIATION':
            return self.image_variation(
                images=params.get('images', []),
                text=params.get('text', ''),
                num_images=num_images,
                quality=quality,
                cfg_scale=cfg_scale,
                seed=seed
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def background_removal(self, image: str) -> str:
        """
        Remove background from an image.

        Args:
            image (str): Base64 encoded image

        Returns:
            str: Base64 encoded image with removed background
        """
        # No need to process image format - background removal can work with transparency
        request_body = {
            'taskType': 'BACKGROUND_REMOVAL',
            'backgroundRemovalParams': {
                'image': image
            }
        }

        response = self.bedrock_runtime.invoke_model(
            modelId=self.image_model_id,
            body=json.dumps(request_body),
            accept='application/json',
            contentType='application/json'
        )

        result = json.loads(response['body'].read())
        return result.get('images', '')

    def text_image(self, text: str, negative_text: str = '', num_images: int = 1,
                   width: int = 1280, height: int = 720,
                   quality: str = 'standard', cfg_scale: float = 6.5, seed: int = 0,
                   condition_image: str = None, control_mode: str = 'CANNY_EDGE',
                   control_strength: float = 0.7) -> List[str]:
        """
        Generate images from text prompt.

        Args:
            text (str): Text prompt
            negative_text (str): Negative prompt
            num_images (int): Number of images to generate
            quality (str): Quality setting
            cfg_scale (float): CFG scale
            seed (int): Random seed
            condition_image (str): Optional control image (Base64)
            control_mode (str): Mode for controlled generation
            control_strength (float): Strength of control

        Returns:
            List[str]: List of Base64 encoded images
        """
        request_body = {
            'taskType': 'TEXT_IMAGE',
            'imageGenerationConfig': {
                'width': width,
                'height': height,
                'numberOfImages': num_images,
                'quality': quality,
                'cfgScale': cfg_scale,
                'seed': seed
            },
            'textToImageParams': {
                'text': text
            }
        }

        if negative_text:
            request_body['textToImageParams']['negativeText'] = negative_text

        if condition_image:
            # Process the condition image to ensure compatibility
            condition_image = self._ensure_valid_image_format(condition_image)
            request_body['textToImageParams'].update({
                'conditionImage': condition_image,
                'controlMode': control_mode,
                'controlStrength': control_strength
            })

        response = self.bedrock_runtime.invoke_model(
            modelId=self.image_model_id,
            body=json.dumps(request_body),
            accept='application/json',
            contentType='application/json'
        )

        result = json.loads(response['body'].read())
        return result.get('images', [])

    def inpainting(self, image: str, text: str = '', negative_text: str = '',
                   mask_image: str = None, mask_prompt: str = None,
                   num_images: int = 1, quality: str = 'standard',
                   cfg_scale: float = 6.5, seed: int = 0) -> List[str]:
        """
        Perform inpainting on an image.

        Args:
            image (str): Base64 encoded image
            text (str): Text prompt
            negative_text (str): Negative prompt
            mask_image (str): Optional mask image (Base64)
            mask_prompt (str): Optional mask prompt
            num_images (int): Number of images to generate
            quality (str): Quality setting
            cfg_scale (float): CFG scale
            seed (int): Random seed

        Returns:
            List[str]: List of Base64 encoded images
        """
        # Convert PNG with transparency to JPEG for inpainting
        image = self._ensure_valid_image_format(image)

        request_body = {
            'taskType': 'INPAINTING',
            'imageGenerationConfig': {
                'numberOfImages': num_images,
                'quality': quality,
                'cfgScale': cfg_scale,
                'seed': seed
            },
            'inPaintingParams': {
                'image': image,
                'text': text
            }
        }

        if negative_text:
            request_body['inPaintingParams']['negativeText'] = negative_text

        if mask_image:
            # Process the mask image to ensure compatibility
            mask_image = self._ensure_valid_image_format(mask_image)
            request_body['inPaintingParams']['maskImage'] = mask_image
        elif mask_prompt:
            request_body['inPaintingParams']['maskPrompt'] = mask_prompt

        response = self.bedrock_runtime.invoke_model(
            modelId=self.image_model_id,
            body=json.dumps(request_body),
            accept='application/json',
            contentType='application/json'
        )

        result = json.loads(response['body'].read())
        return result.get('images', [])

    def outpainting(self, image: str, text: str = '', mask_image: str = None,
                    mask_prompt: str = None, outpainting_mode: str = 'DEFAULT',
                    num_images: int = 1, quality: str = 'standard',
                    cfg_scale: float = 6.5, seed: int = 0) -> List[str]:
        """
        Perform outpainting on an image.

        Args:
            image (str): Base64 encoded image
            text (str): Text prompt
            mask_image (str): Optional mask image (Base64)
            mask_prompt (str): Optional mask prompt
            outpainting_mode (str): Mode for outpainting
            num_images (int): Number of images to generate
            quality (str): Quality setting
            cfg_scale (float): CFG scale
            seed (int): Random seed

        Returns:
            List[str]: List of Base64 encoded images
        """
        # Convert PNG with transparency to JPEG for outpainting
        image = self._ensure_valid_image_format(image)

        request_body = {
            'taskType': 'OUTPAINTING',
            'imageGenerationConfig': {
                'numberOfImages': num_images,
                'quality': quality,
                'cfgScale': cfg_scale,
                'seed': seed
            },
            'outPaintingParams': {
                'image': image,
                'text': text,
                'outPaintingMode': outpainting_mode
            }
        }

        if mask_image:
            # Process the mask image to ensure compatibility
            mask_image = self._ensure_valid_image_format(mask_image)
            request_body['outPaintingParams']['maskImage'] = mask_image
        elif mask_prompt:
            request_body['outPaintingParams']['maskPrompt'] = mask_prompt

        response = self.bedrock_runtime.invoke_model(
            modelId=self.image_model_id,
            body=json.dumps(request_body),
            accept='application/json',
            contentType='application/json'
        )

        result = json.loads(response['body'].read())
        return result.get('images', [])

    def color_guided_generation(self, text: str, colors: List[str],
                                width: int, height: int, reference_image: str = None,
                                num_images: int = 1, quality: str = 'standard',
                                cfg_scale: float = 6.5, seed: int = 0) -> List[str]:
        """
        Generate images with specified color palette.

        Args:
            text (str): Text prompt
            colors (List[str]): List of color codes
            reference_image (str): Optional reference image (Base64)
            num_images (int): Number of images to generate
            quality (str): Quality setting
            cfg_scale (float): CFG scale
            seed (int): Random seed

        Returns:
            List[str]: List of Base64 encoded images
        """
        request_body = {
            'taskType': 'COLOR_GUIDED_GENERATION',
            'imageGenerationConfig': {
                'width': width,
                'height': height,
                'numberOfImages': num_images,
                'quality': quality,
                'cfgScale': cfg_scale,
                'seed': seed
            },
            'colorGuidedGenerationParams': {
                'text': text,
                'colors': colors if isinstance(colors, list) else colors.split(',')
            }
        }

        if reference_image:
            # Process the reference image to ensure compatibility
            reference_image = self._ensure_valid_image_format(reference_image)
            request_body['colorGuidedGenerationParams']['image'] = reference_image

        response = self.bedrock_runtime.invoke_model(
            modelId=self.image_model_id,
            body=json.dumps(request_body),
            accept='application/json',
            contentType='application/json'
        )

        result = json.loads(response['body'].read())
        return result.get('images', [])

    def image_variation(self, images: List[str], text: str = '',
                        num_images: int = 1, quality: str = 'standard',
                        cfg_scale: float = 6.5, seed: int = 0) -> List[str]:
        """
        Generate variations of the input images.

        Args:
            images (List[str]): List of Base64 encoded images
            text (str): Optional text prompt
            num_images (int): Number of images to generate
            quality (str): Quality setting
            cfg_scale (float): CFG scale
            seed (int): Random seed

        Returns:
            List[str]: List of Base64 encoded images
        """
        # Process images to ensure format compatibility
        processed_images = [self._ensure_valid_image_format(img) for img in images]

        request_body = {
            'taskType': 'IMAGE_VARIATION',
            'imageGenerationConfig': {
                'numberOfImages': num_images,
                'quality': quality,
                'cfgScale': cfg_scale,
                'seed': seed
            },
            'imageVariationParams': {
                'images': processed_images,
                'text': text
            }
        }

        response = self.bedrock_runtime.invoke_model(
            modelId=self.image_model_id,
            body=json.dumps(request_body),
            accept='application/json',
            contentType='application/json'
        )

        result = json.loads(response['body'].read())
        return result.get('images', [])


    def generate_video(self, params: dict) -> str:
        """Generate video from sequence of images using asynchronous invocation"""

        # Set up the model input structure
        model_input = {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {
                "text": params.get('text'),
                "images": [
                    {
                        "format": "png",
                        "source": {
                            "bytes": params.get('image')
                        }
                    }
                ]
                },
            "videoGenerationConfig": {
                "durationSeconds": 6,
                "fps": 24,
                "dimension": "1280x720",
                "seed": params.get('seed')
            },
        }

        # Use the S3 bucket you have configured
        s3_bucket = "chanhos-misc"
        s3_key_prefix = "nova-videos"

        # Start asynchronous job
        response = self.bedrock_runtime.start_async_invoke(
            modelId="amazon.nova-reel-v1:0",
            modelInput=model_input,
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": f"s3://{s3_bucket}/{s3_key_prefix}/"
                }
            },
        )

        job_id = response.get("invocationArn", "").split('/')[-1]
        print(f"Started video generation job: {job_id}")

        # Poll for job completion
        while True:
            job_response = self.bedrock_runtime.get_async_invoke(
                invocationArn=response["invocationArn"]
            )
            status = job_response["status"]

            if status == "Completed":
                # Download from S3
                s3_client = boto3.client('s3')
                bucket_uri = job_response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
                video_path = f"{bucket_uri}/output.mp4"
                bucket_and_key = video_path[5:].split("/", 1)
                bucket = bucket_and_key[0]
                key = bucket_and_key[1] if len(bucket_and_key) > 1 else None

                try:
                    obj = s3_client.get_object(Bucket=bucket, Key=key)
                    video_data = obj["Body"].read()
                    return base64.b64encode(video_data).decode('utf-8')
                except Exception as e:
                    raise Exception(f"Error downloading video: {str(e)}")

            elif status == "FAILED":
                error_message = job_response.get("failureMessage", "Unknown error")
                raise Exception(f"Video generation failed: {error_message}")

            time.sleep(10)  # Check every 10 seconds
