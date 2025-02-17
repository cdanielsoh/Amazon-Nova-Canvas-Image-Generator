import boto3
import base64
import json


class NovaCanvasGenerator:
    def __init__(self):
        self.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
        self.model_id = 'amazon.nova-canvas-v1:0'

    def encode_image(self, image_path):
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_image(self, task_type, params):
        request_body = {
            'taskType': task_type
        }

        # Add imageGenerationConfig for non-background removal tasks
        if task_type != 'BACKGROUND_REMOVAL':
            request_body['imageGenerationConfig'] = {
                'width': params.get('width', 1024),
                'height': params.get('height', 1024),
                'numberOfImages': params.get('num_images', 1),
                'quality': params.get('quality', 'standard'),
                'cfgScale': params.get('cfg_scale', 6.5)
            }

        # Add task-specific parameters
        if task_type == 'TEXT_IMAGE':
            request_body['textToImageParams'] = {
                'text': params['text'],
                'negativeText': params.get('negativeText', ' ')
            }
            # Add conditioning image parameters if provided
            if 'conditionImage' in params:
                request_body['textToImageParams'].update({
                    'conditionImage': params['conditionImage'],
                    'controlMode': params.get('controlMode', 'CANNY_EDGE'),
                    'controlStrength': params.get('controlStrength', 0.7)
                })

        # In the generate_image method's inPaintingParams section:
        elif task_type == 'INPAINTING':
            request_body['inPaintingParams'] = {
                'text': params['text'],
                'image': params['image']
            }

            # Add either maskImage or maskPrompt, not both
            if 'maskImage' in params:
                request_body['inPaintingParams']['maskImage'] = params['maskImage']
            elif 'maskPrompt' in params:
                request_body['inPaintingParams']['maskPrompt'] = params['maskPrompt']


        elif task_type == 'OUTPAINTING':
            request_body['outPaintingParams'] = {
                'image': params['image'],
                'outPaintingMode': params['outPaintingMode']
            }

            # Add either maskImage or maskPrompt, not both
            if 'maskImage' in params:
                request_body['outPaintingParams']['maskImage'] = params['maskImage']
            elif 'maskPrompt' in params:
                request_body['outPaintingParams']['maskPrompt'] = params['maskPrompt']
            print(request_body)

        elif task_type == 'IMAGE_VARIATION':
            request_body['imageVariationParams'] = {
                'text': params.get('text', ' '),
                'images': params['images']
            }
        elif task_type == 'COLOR_GUIDED_GENERATION':
            request_body['colorGuidedGenerationParams'] = {
                'text': params['text'],
                'colors': params['colors'].split(',')
            }
            if 'referenceImage' in params:
                request_body['colorGuidedGenerationParams']['image'] = params['referenceImage']

        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body)
        )

        # Return all generated images
        return json.loads(response['body'].read())['images'][0]