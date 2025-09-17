from google import genai
from google.genai import types
from PIL import Image
import json
import re
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, get_type_hints
import inspect
import re
import requests

api_key = 'AIzaSyCe8-OJT-zr8HNSbUnnXcAjGJS2OIUsi14'

client = genai.Client(api_key=api_key)

def pil_image_to_bytes(pil_image: Image.Image, format: str = "JPEG") -> tuple[bytes, str]:
    """
    Convert PIL Image to bytes and determine MIME type.
    
    Args:
        pil_image: PIL Image object
        format: Image format (JPEG, PNG, etc.)
    
    Returns:
        Tuple of (image_bytes, mime_type)
    """
    buffer = BytesIO()
    
    # Convert RGBA to RGB for JPEG
    if format.upper() == "JPEG" and pil_image.mode in ('RGBA', 'LA', 'P'):
        # Create a white background
        background = Image.new('RGB', pil_image.size, (255, 255, 255))
        if pil_image.mode == 'P':
            pil_image = pil_image.convert('RGBA')
        background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
        pil_image = background
    
    pil_image.save(buffer, format=format)
    image_bytes = buffer.getvalue()
    
    mime_type = f"image/{format.lower()}"
    if format.upper() == "JPEG":
        mime_type = "image/jpeg"
    
    return image_bytes, mime_type

def bytes_to_pil_image(image_bytes: bytes) -> Image.Image:
    """
    Convert bytes to PIL Image.
    
    Args:
        image_bytes: Image data as bytes
    
    Returns:
        PIL Image object
    """
    return Image.open(BytesIO(image_bytes))

def get_function_info_as_string(func):
    """
    Get function information as a JSON string in the specified format.
    
    Returns:
        str: JSON string with function details
    """
    # Get function name
    func_name = func.__name__
    
    # Get docstring for description
    description = func.__doc__.strip() if func.__doc__ else "No description available"
    
    # Get type hints
    hints = get_type_hints(func)
    
    # Get signature for parameters
    sig = inspect.signature(func)
    
    # Build the result dictionary
    result = {
        func_name: {
            "tool_description": description
        }
    }
    
    # Add parameters with their types
    for param_name, param in sig.parameters.items():
        param_type = hints.get(param_name, "Any")
        
        # Convert type to string representation
        if hasattr(param_type, '__name__'):
            type_str = param_type.__name__
        else:
            type_str = str(param_type)
        
        # Add default value if exists
        if param.default != param.empty:
            type_str += f" (default: {param.default})"
        
        result[func_name][param_name] = type_str
    
    # Convert to JSON string
    return json.dumps(result, indent=2)


def generate_style(base_image_of_person_url: str, item_image_urls: list[str], item_category_list: list[str], client=client)-> Image.Image:
  
  """Function that generate picture of the person in the base image with the clothes and accessories given."""
  
#   Example usage:
#   {
#     "base_image_url":
#   "https://i8.amplience.net/i/Lindex/3003139_80_PS_MF/menstrosa-med-medium-absorption-hog-midja-female-engineering?w=1200&h=1600&fmt=auto&qlt=90&fmt.jp2.qlt=50&sm=c",
#     "item_image_urls": [
#       "https://i8.amplience.net/i/Lindex/3007143_9618_PS_MF/brun-finstickad-topp?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
# "https://i8.amplience.net/i/Lindex/3000057_7268_PS_MF?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
# "https://i8.amplience.net/i/Lindex/3007476_8665_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c"
#     ],
#     "item_categories": [
#       "top", "bottom", "accessory"
#     ]
#   }

  


  headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
  }

  item_image_urls.insert(0, base_image_of_person_url)

  image_list = []
  for image_url in item_image_urls:
    response = requests.get(image_url, headers=headers)

    # Create an in-memory binary stream from the downloaded content
    image_bytes = BytesIO(response.content)
    pil_image = Image.open(image_bytes)
    image_list.append(pil_image)

  prompt = (
      "Please dress the person in the first picture with "
  )

  for item in range(len(item_category_list)):
    prompt += f"the {item_category_list[item]} from picture {item + 2} "

  image_list.insert(0, prompt)

  response = client.models.generate_content(
      model="gemini-2.5-flash-image-preview",
      contents=image_list,
  )

  for part in response.candidates[0].content.parts:
      if part.text is not None:
          print(part.text)
      elif part.inline_data is not None:
          image = Image.open(BytesIO(part.inline_data.data))

  return image


def execute_function_with_args(function_data: Dict[str, Dict[str, Any]], function_registry: Dict[str, Callable] = None):
    """
    Execute a function with provided arguments from a dictionary.
    
    Args:
        function_data: Dictionary with function name as key and arguments as nested dict
        function_registry: Optional dictionary of available functions {name: function}
    
    Returns:
        The result of the function execution
    """
    
    if not function_data:
        raise ValueError("No function data provided")
    
    # Get the function name and arguments
    function_name = list(function_data.keys())[0]
    arguments = function_data[function_name]
    
    # Find the function
    target_function = None
    
    if function_registry and function_name in function_registry:
        target_function = function_registry[function_name]
    else:
        # Try to find in globals
        if function_name in globals():
            target_function = globals()[function_name]
        else:
            raise ValueError(f"Function '{function_name}' not found")
    
    if not callable(target_function):
        raise ValueError(f"'{function_name}' is not callable")
    
    try:
        # Execute the function with the provided arguments
        result = target_function(**arguments)
        return result
    except TypeError as e:
        raise ValueError(f"Error calling function '{function_name}': {e}")
    except Exception as e:
        raise RuntimeError(f"Error executing function '{function_name}': {e}")

def call_llm(model, client, system_prompt, messages, enable_image_generation=False):
    """
    Generic LLM calling function that can be rewritten for different APIs.
    For Google API: extracts system prompt and converts message format.
    
    Args:
        model: Model name/ID
        client: API client object
        system_prompt: System prompt string
        messages: List of messages in standard format [{"role": "system/user/assistant", "content": "..."}]
        enable_image_generation: Whether to enable image generation in response
    
    Returns:
        Tuple of (response_text, generated_images_list)
    """
    # GOOGLE API IMPLEMENTATION:
    # Convert standard format to Google format
    contents = []
    
    for msg in messages:
        if msg["role"] == "user":
            # Handle user messages with potential image content
            parts = []
            
            if isinstance(msg["content"], str):
                # Simple text message
                parts.append(types.Part(text=msg["content"]))
            elif isinstance(msg["content"], list):
                # Multi-part message (text + images)
                for part in msg["content"]:
                    if isinstance(part, str):
                        parts.append(types.Part(text=part))
                    elif isinstance(part, dict):
                        if part["type"] == "text":
                            parts.append(types.Part(text=part["text"]))
                        elif part["type"] == "image":
                            # All images should be stored as bytes in the message history
                            if "image_bytes" in part and "mime_type" in part:
                                parts.append(types.Part.from_bytes(
                                    data=part["image_bytes"],
                                    mime_type=part["mime_type"]
                                ))
                            elif "image_uri" in part:
                                # Google Cloud Storage URI
                                parts.append(types.Part.from_uri(
                                    file_uri=part["image_uri"],
                                    mime_type=part.get("mime_type", "image/jpeg")
                                ))
            
            contents.append(types.Content(
                role="user",
                parts=parts
            ))
            
        elif msg["role"] == "assistant":
            # Handle assistant messages with potential generated images
            parts = []
            
            if isinstance(msg["content"], str):
                # Simple text response
                parts.append(types.Part(text=msg["content"]))
            elif isinstance(msg["content"], dict):
                # Response with text and possibly generated images
                if "text" in msg["content"]:
                    parts.append(types.Part(text=msg["content"]["text"]))
                if "generated_images" in msg["content"]:
                    # Add generated images to the conversation
                    for img_data in msg["content"]["generated_images"]:
                        parts.append(types.Part.from_bytes(
                            data=img_data["image_bytes"],
                            mime_type=img_data.get("mime_type", "image/jpeg")
                        ))
            
            contents.append(types.Content(
                role="model",  # Google uses "model" instead of "assistant"
                parts=parts
            ))
        # Skip system messages as they're handled separately in Google API
    
    # Build config
    config_params = {
        "system_instruction": system_prompt
    }
    
    # Enable image generation if requested and model supports it
    if enable_image_generation:
        config_params["response_modalities"] = ["TEXT", "IMAGE"]
    
    # Generate response using Google's API
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**config_params)
    )
    
    # Extract text and images from response
    response_text = ""
    generated_images = []
    
    # Handle the response - check if it has candidates
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
                    elif hasattr(part, 'inline_data') and part.inline_data:
                        # Generated image
                        generated_images.append({
                            "image_bytes": part.inline_data.data,
                            "mime_type": getattr(part.inline_data, 'mime_type', 'image/jpeg')
                        })
    
    # Fallback to simple text extraction if no candidates structure
    if not response_text and hasattr(response, 'text'):
        response_text = response.text
    
    return response_text, generated_images

def extract_tools_json(text):
    """Extract JSON from inside <tools></tools> tags"""
    pattern = r'<tools>(.*?)</tools>'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        tools_content = match.group(1).strip()
        try:
            # Parse the JSON
            return json.loads(tools_content)
        except json.JSONDecodeError as e:
            print(f"Error parsing tools JSON: {e}")
            return None
    return None

class ChatChain:
    def __init__(self, model, client, system_prompt, enable_image_generation=False):
        self.model = model
        self.client = client
        self.system_prompt = system_prompt
        self.enable_image_generation = enable_image_generation
        # Maintain standard messages for compatibility
        self.messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        # Store generated images separately for easy access
        self.generated_images_history = []
    
    def _process_image_for_storage(self, img) -> Dict[str, Any]:
        """
        Process an image into a format that can be stored in message history.
        Always converts to bytes format for consistency.
        
        Args:
            img: Image in various formats (PIL Image, file path, URL, dict with image data, or raw bytes)
        
        Returns:
            Dict with image_bytes and mime_type
        """
        # Headers for URL requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        if isinstance(img, Image.Image):
            # PIL Image - convert to bytes
            image_bytes, mime_type = pil_image_to_bytes(img)
            return {
                "type": "image",
                "image_bytes": image_bytes,
                "mime_type": mime_type
            }
        elif isinstance(img, str):
            # Check if it's a URL or file path
            if img.startswith('http://') or img.startswith('https://'):
                # URL - download and convert to bytes
                response = requests.get(img, headers=headers)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                # Create PIL image from downloaded content
                image_bytes_io = BytesIO(response.content)
                pil_img = Image.open(image_bytes_io)
                
                # Convert to bytes with appropriate format
                image_bytes, mime_type = pil_image_to_bytes(pil_img)
                return {
                    "type": "image",
                    "image_bytes": image_bytes,
                    "mime_type": mime_type
                }
            else:
                # File path - load and convert to bytes
                pil_img = Image.open(img)
                # Determine format from file extension
                format = "JPEG"
                if img.lower().endswith('.png'):
                    format = "PNG"
                elif img.lower().endswith('.webp'):
                    format = "WEBP"
                elif img.lower().endswith('.gif'):
                    format = "GIF"
                
                image_bytes, mime_type = pil_image_to_bytes(pil_img, format)
                return {
                    "type": "image",
                    "image_bytes": image_bytes,
                    "mime_type": mime_type
                }
        elif isinstance(img, dict):
            # Already in dict format
            if "image_uri" in img:
                # GCS URI - keep as is
                return {
                    "type": "image",
                    "image_uri": img["image_uri"],
                    "mime_type": img.get("mime_type", "image/jpeg")
                }
            elif "image_bytes" in img:
                # Already in bytes format
                return {
                    "type": "image",
                    "image_bytes": img["image_bytes"],
                    "mime_type": img.get("mime_type", "image/jpeg")
                }
        elif isinstance(img, bytes):
            # Raw bytes
            return {
                "type": "image",
                "image_bytes": img,
                "mime_type": "image/jpeg"
            }
        
        raise ValueError(f"Unsupported image type: {type(img)}")
    
    def chat(self, message: Union[str, List], images: Optional[List] = None, generate_image=None):
        """
        Send a message to the chat, optionally with images.
        
        Args:
            message: Text message or list of message parts
            images: Optional list of images. Each can be:
                - PIL Image object
                - Path to image file (str)
                - URL to image (str starting with http:// or https://)
                - Dict with image data: {"image_bytes": bytes, "mime_type": str}
                - Dict with GCS URI: {"image_uri": str, "mime_type": str}
                - Raw bytes
            generate_image: Whether to enable image generation for this response.
                           If None, uses the default from initialization.
        
        Returns:
            str: Assistant's response text
        """
        # Determine if image generation should be enabled for this call
        enable_gen = generate_image if generate_image is not None else self.enable_image_generation
        
        # Construct the user message content
        if images:
            content = []
            
            # Add text first
            if isinstance(message, str):
                content.append({"type": "text", "text": message})
            elif isinstance(message, list):
                for item in message:
                    if isinstance(item, str):
                        content.append({"type": "text", "text": item})
                    else:
                        content.append(item)
            
            # Process and add images
            for img in images:
                processed_img = self._process_image_for_storage(img)
                content.append(processed_img)
            
            # Add user message with content list
            self.messages.append({
                "role": "user",
                "content": content
            })
        else:
            # Simple text message
            self.messages.append({
                "role": "user",
                "content": message
            })
        
        # Use the generic call_llm function
        response_text, generated_images = call_llm(
            self.model, 
            self.client, 
            self.system_prompt, 
            self.messages,
            enable_image_generation=enable_gen
        )
        
        # Check for tool calls in the response
        tool_calls = extract_tools_json(response_text)
        tool_generated_images = []
        
        if tool_calls:
            # Create function registry with available functions
            function_registry = {
                'generate_style': generate_style
            }
            
            try:
                # Execute the tool call
                tool_result = execute_function_with_args(tool_calls, function_registry)
                
                # Check if the result is a PIL Image (from generate_style)
                if isinstance(tool_result, Image.Image):
                    # Convert PIL image to the format used for generated images
                    image_bytes, mime_type = pil_image_to_bytes(tool_result)
                    tool_generated_images.append({
                        "image_bytes": image_bytes,
                        "mime_type": mime_type
                    })
                    print(f"Tool execution successful: Generated image from {list(tool_calls.keys())[0]}")
                else:
                    print(f"Tool execution result: {tool_result}")
            
            except Exception as e:
                print(f"Error executing tool call: {e}")
        
        # Combine LLM generated images with tool generated images
        all_generated_images = generated_images + tool_generated_images
        
        # Store the response
        if all_generated_images:
            # Store response with generated images
            self.messages.append({
                "role": "assistant",
                "content": {
                    "text": response_text,
                    "generated_images": all_generated_images
                }
            })
            # Add to generated images history
            self.generated_images_history.append({
                "message_index": len(self.messages) - 1,
                "images": all_generated_images
            })
        else:
            # Simple text response
            self.messages.append({
                "role": "assistant", 
                "content": response_text
            })
        
        # Remove tool tags from the response text for cleaner output
        response_text = re.sub(r'<tools>.*?</tools>', '', response_text, flags=re.DOTALL)
        return response_text
    
    def get_messages(self):
        """Return messages in OpenAI-like format for compatibility"""
        return self.messages
    
    def get_last_raw_response(self):
        """Get the last assistant response text"""
        if len(self.messages) > 1 and self.messages[-1]["role"] == "assistant":
            content = self.messages[-1]["content"]
            if isinstance(content, str):
                return content
            elif isinstance(content, dict) and "text" in content:
                return content["text"]
        return None
    
    def get_last_generated_images(self) -> List[Image.Image]:
        """
        Get the last generated images as PIL Image objects.
        
        Returns:
            List of PIL Image objects from the last response that generated images
        """
        if not self.generated_images_history:
            return []
        
        last_gen = self.generated_images_history[-1]
        pil_images = []
        for img_data in last_gen["images"]:
            pil_img = bytes_to_pil_image(img_data["image_bytes"])
            pil_images.append(pil_img)
        
        return pil_images
    
    def get_all_generated_images(self) -> List[Tuple[int, List[Image.Image]]]:
        """
        Get all generated images from the conversation.
        
        Returns:
            List of tuples (message_index, list_of_pil_images)
        """
        all_images = []
        for gen_record in self.generated_images_history:
            pil_images = []
            for img_data in gen_record["images"]:
                pil_img = bytes_to_pil_image(img_data["image_bytes"])
                pil_images.append(pil_img)
            all_images.append((gen_record["message_index"], pil_images))
        
        return all_images
    
    def save_last_generated_images(self, base_path: str = "generated_image", format: str = "PNG"):
        """
        Save the last generated images to files.
        
        Args:
            base_path: Base path for saving images (without extension)
            format: Image format (PNG, JPEG, etc.)
        
        Returns:
            List of saved file paths
        """
        images = self.get_last_generated_images()
        saved_paths = []
        
        for i, img in enumerate(images):
            if len(images) == 1:
                path = f"{base_path}.{format.lower()}"
            else:
                path = f"{base_path}_{i+1}.{format.lower()}"
            
            img.save(path, format=format)
            saved_paths.append(path)
            print(f"Saved image to: {path}")
        
        return saved_paths
    
    def get_tool_calls(self):
        """Return tool calls from the last response as a Python dict"""
        last_raw_response = self.get_last_raw_response()
        if last_raw_response:
            return extract_tools_json(last_raw_response)
        return None
    
    def get_last_response(self):
        """Get the last response, handling tool calls if present"""
        last_response = chat.get_messages()[-1]
        last_response = re.sub(r'<tools>.*?</tools>', '', last_response, flags=re.DOTALL)
        return last_response

tool_desc = get_function_info_as_string(generate_style)

# Example usage with system prompt
system_prompt = f"""
<instructions>
You are Lindy a personal style assistant from Lindex. You suggest styles for your client given the data in their wordrobe:
{{
  "success": true,
  "count": 44,
  "data": [
    {{
      "_id": "68c3e50400f82f6d925ca6a7",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3006801_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [
        "Party",
        "Casual"
      ],
      "createdAt": "2025-09-12T09:16:52.040Z",
      "updatedAt": "2025-09-17T06:56:38.556Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6a9",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3004292_6940_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [
        "Business"
      ],
      "createdAt": "2025-09-12T09:16:52.051Z",
      "updatedAt": "2025-09-17T10:53:31.511Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6ab",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3001152_8652_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [
        "Party"
      ],
      "createdAt": "2025-09-12T09:16:52.063Z",
      "updatedAt": "2025-09-17T11:37:38.746Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6ad",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3001152_7697_PS_F/gra-topp-i-ullblandning?w=1200&h=1600&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.076Z",
      "updatedAt": "2025-09-12T09:16:52.076Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6af",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/8362682_8704_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.087Z",
      "updatedAt": "2025-09-12T09:16:52.087Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6b1",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007769_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.099Z",
      "updatedAt": "2025-09-12T09:16:52.099Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6b3",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007653_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.111Z",
      "updatedAt": "2025-09-12T09:16:52.111Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6b5",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007370_9618_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.127Z",
      "updatedAt": "2025-09-12T09:16:52.127Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6b7",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007715_70_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.140Z",
      "updatedAt": "2025-09-12T09:16:52.140Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6b9",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007598_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.151Z",
      "updatedAt": "2025-09-12T09:16:52.151Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6bb",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3004091_8671_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.163Z",
      "updatedAt": "2025-09-12T09:16:52.163Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6bd",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007389_9609_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.176Z",
      "updatedAt": "2025-09-12T09:16:52.176Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6bf",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3002147_9608_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.186Z",
      "updatedAt": "2025-09-12T09:16:52.186Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6c1",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3008624_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.197Z",
      "updatedAt": "2025-09-12T09:16:52.197Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6c3",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/8624794_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.210Z",
      "updatedAt": "2025-09-12T09:16:52.210Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6c5",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007926_9608_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.226Z",
      "updatedAt": "2025-09-12T09:16:52.226Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6c7",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007668_9612_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "top\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.237Z",
      "updatedAt": "2025-09-12T09:16:52.237Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6c9",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007855_9608_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "bottoms\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.248Z",
      "updatedAt": "2025-09-12T09:16:52.248Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6cb",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007539_9612_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "bottoms\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.261Z",
      "updatedAt": "2025-09-12T09:16:52.261Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6cd",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007688_8600_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "bottoms\r",
      "tags": [
        "Business"
      ],
      "createdAt": "2025-09-12T09:16:52.274Z",
      "updatedAt": "2025-09-17T10:53:43.778Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6cf",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007685_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "bottoms\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.285Z",
      "updatedAt": "2025-09-12T09:16:52.285Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6d1",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3001715_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "bottoms\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.298Z",
      "updatedAt": "2025-09-12T09:16:52.298Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6d3",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3004972_7862_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "bottoms\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.310Z",
      "updatedAt": "2025-09-12T09:16:52.310Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6d5",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3008183_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "bottoms\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.322Z",
      "updatedAt": "2025-09-12T09:16:52.322Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6d7",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3000038_9609_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "dress\r",
      "tags": [
        "tag1"
      ],
      "createdAt": "2025-09-12T09:16:52.334Z",
      "updatedAt": "2025-09-17T08:33:15.397Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6d9",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3008422_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "dress\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.347Z",
      "updatedAt": "2025-09-12T09:16:52.347Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6db",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3006780_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "dress\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.359Z",
      "updatedAt": "2025-09-12T09:16:52.359Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6dd",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3008435_2659_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "dress\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.374Z",
      "updatedAt": "2025-09-12T09:16:52.374Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6df",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007732_8117_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "jacket\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.387Z",
      "updatedAt": "2025-09-12T09:16:52.387Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6e1",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3006623_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "jacket\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.399Z",
      "updatedAt": "2025-09-12T09:16:52.399Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6e3",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007344_7199_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "jacket\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.410Z",
      "updatedAt": "2025-09-12T09:16:52.410Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6e5",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007917_9609_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "jacket\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.423Z",
      "updatedAt": "2025-09-12T09:16:52.423Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6e7",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/8464557_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "bag\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.436Z",
      "updatedAt": "2025-09-12T09:16:52.436Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6e9",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3007236_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "bag\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.448Z",
      "updatedAt": "2025-09-12T09:16:52.448Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6eb",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3002899_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "bag\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.460Z",
      "updatedAt": "2025-09-12T09:16:52.460Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6ed",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3009387_8494_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "accessory\r",
      "tags": [
        "Business"
      ],
      "createdAt": "2025-09-12T09:16:52.471Z",
      "updatedAt": "2025-09-17T10:54:03.183Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6ef",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3006898_10_PS_B?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "accessory\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.483Z",
      "updatedAt": "2025-09-12T09:16:52.483Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6f1",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3009599_10_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "accessory\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.494Z",
      "updatedAt": "2025-09-12T09:16:52.494Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6f3",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3010226_10_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "accessory\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.507Z",
      "updatedAt": "2025-09-12T09:16:52.507Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6f5",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3009018_10_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "accessory\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.519Z",
      "updatedAt": "2025-09-12T09:16:52.519Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6f7",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3006558_9072_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "accessory\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.531Z",
      "updatedAt": "2025-09-12T09:16:52.531Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6f9",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3008947_9806_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "accessory\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.543Z",
      "updatedAt": "2025-09-12T09:16:52.543Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6fb",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/7202897_80_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "accessory\r",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.555Z",
      "updatedAt": "2025-09-12T09:16:52.555Z",
      "__v": 0
    }},
    {{
      "_id": "68c3e50400f82f6d925ca6fd",
      "userId": 1,
      "imageUrl": "https://i8.amplience.net/i/Lindex/3006955_250_PS_F?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c",
      "type": "accessory",
      "tags": [],
      "createdAt": "2025-09-12T09:16:52.568Z",
      "updatedAt": "2025-09-12T09:16:52.568Z",
      "__v": 0
    }}
  ],
  "fromCache": true
}}
You always dress you client in the clothes you recommend by generating an image using the style_generatin tool. Here is your clients base image url: https://i8.amplience.net/i/Lindex/3003049_1753_PS_MF?w=1600&h=2133&fmt=auto&qlt=70&fmt.jp2.qlt=50&sm=c
IMPORTANT: You MUST ALWAYS use tools in a SINGLE JSON object at the end of your output inside <tools></tools>, include only the tools you want to call.
</instructions>
<tools>
{tool_desc}
</tools> 
"""