# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import base64
import streamlit as st
import fitz
import torch
from io import BytesIO
from PIL import Image
import requests
from transformers import MllamaForConditionalGeneration, AutoProcessor
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAI
from pdf2image import convert_from_path
import shutil


@st.cache_resource
def initialize_vlm():
    """Initialize and load the Vision-Language Model (VLM) for image description from a specified model ID."""
    # model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # vlm_model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    vlm_model = OpenAI(base_url="https://095kiew15yzv2e-8000.proxy.runpod.net/v1/", api_key="volker123")
    return vlm_model

def get_b64_image_from_content(image_content):
    """Convert image content to base64 encoded string."""
    img = Image.open(BytesIO(image_content))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def is_graph(image_content):
    """Determine if an image is a graph, plot, chart, or table."""
    res = describe_image(image_content)
    return any(keyword in res.lower() for keyword in ["graph", "plot", "chart", "table"])

def process_graph(image_content, llm):
    """Process a graph image and generate a description."""
    deplot_description = process_graph_deplot(image_content)
    response = llm.complete("Your responsibility is to explain charts. You are an expert in describing the responses of linearized tables into plain English text for LLMs to use. Explain the following linearized table. " + deplot_description)
    return response.text

def describe_image(image_content):
    """Generate a description of an image using the multimodal LLM."""
    vlm_model = initialize_vlm()
    image = Image.open(BytesIO(image_content))
    base64_image = get_b64_image_from_content(image_content)
    messages = [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "What is in this image?",
            },
            {
            "type": "image_url",
            "image_url": {
                "url":  f"data:image/jpeg;base64,{base64_image}"
            }, 
            },
        ],
        }
    ]
    response = vlm_model.chat.completions.create(messages=messages, model="unsloth/Llama-3.2-11B-Vision-Instruct")
    print(response.choices[0].message.content)
    return response.choices[0].message.content

 
def process_graph_deplot(image_content):
    """Generate a description of an image using the multimodal LLM."""
    vlm_model = initialize_vlm()
    image = Image.open(BytesIO(image_content))
    base64_image = get_b64_image_from_content(image_content)
    messages = [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "What is in this image?",
            },
            {
            "type": "image_url",
            "image_url": {
                "url":  f"data:image/jpeg;base64,{base64_image}"
            }, 
            },
        ],
        }
    ]
    response = vlm_model.chat.completions.create(messages=messages, model="unsloth/Llama-3.2-11B-Vision-Instruct")
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def extract_text_around_item(text_blocks, bbox, page_height, threshold_percentage=0.1):
    """Extract text above and below a given bounding box on a page."""
    before_text, after_text = "", ""
    vertical_threshold_distance = page_height * threshold_percentage
    horizontal_threshold_distance = bbox.width * threshold_percentage

    for block in text_blocks:
        block_bbox = fitz.Rect(block[:4])
        vertical_distance = min(abs(block_bbox.y1 - bbox.y0), abs(block_bbox.y0 - bbox.y1))
        horizontal_overlap = max(0, min(block_bbox.x1, bbox.x1) - max(block_bbox.x0, bbox.x0))

        if vertical_distance <= vertical_threshold_distance and horizontal_overlap >= -horizontal_threshold_distance:
            if block_bbox.y1 < bbox.y0 and not before_text:
                before_text = block[4]
            elif block_bbox.y0 > bbox.y1 and not after_text:
                after_text = block[4]
                break

    return before_text, after_text

def process_text_blocks(text_blocks, char_count_threshold=500):
    """Group text blocks based on a character count threshold."""
    current_group = []
    grouped_blocks = []
    current_char_count = 0

    for block in text_blocks:
        if block[-1] == 0:  # Check if the block is of text type
            block_text = block[4]
            block_char_count = len(block_text)

            if current_char_count + block_char_count <= char_count_threshold:
                current_group.append(block)
                current_char_count += block_char_count
            else:
                if current_group:
                    grouped_content = "\n".join([b[4] for b in current_group])
                    grouped_blocks.append((current_group[0], grouped_content))
                current_group = [block]
                current_char_count = block_char_count

    # Append the last group
    if current_group:
        grouped_content = "\n".join([b[4] for b in current_group])
        grouped_blocks.append((current_group[0], grouped_content))

    return grouped_blocks

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary directory."""
    temp_dir = os.path.join(os.getcwd(), "vectorstore", "ppt_references", "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    
    return temp_file_path

def pdf_to_fotos(file_path):
    output_dir = os.path.basename(file_path)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pages = convert_from_path(f'{file_path}', 500)
    for count, page in enumerate(pages):
        page.save(f'{output_dir}/{count}.jpg', 'JPEG')
    return output_dir


def image_description(file_path):
    vlm_model = OpenAI(base_url="https://095kiew15yzv2e-8000.proxy.runpod.net/v1/", api_key="volker123")
   # image = Image.open("/Users/xufanlu/ALEX/rag/test.png")
    with open(file_path, "rb") as image_file:
        image_content = image_file.read()
    print(f"line 308 {file_path}")

    base64_image = get_b64_image_from_content(image_content)
    messages = [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Could you please: 1. Recognize and extract the text content directly. 2.Identify and describe any images separately. 3.Detect any formulas and provide them using LaTeX format. Give back answer in this format: **Text Content**:; **Images**:;**Formulas**:;",
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                }, 
                },
            ],
            }
        ]
    response = vlm_model.chat.completions.create(messages=messages, model="unsloth/Llama-3.2-11B-Vision-Instruct")
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def get_description_whole_pdf(folder_name):
    for item in os.listdir(folder_name):
        item_path = os.path.join(folder_name, item)
        if os.path.isfile(item_path) and item.lower().endswith(".jpg"):
            # Call image_description() for each file
            description = image_description(item_path)

            print(f"Description for {item}: {description}")
            description_file_path = os.path.join(folder_name, f"{item}.txt")
            with open(f'{description_file_path}', "w") as description_file:
                description_file.write(description)
            
            print(f"Description for {item} saved to {description_file_path}")

def move_files(folder_name ):
    new_folder_name=f'{folder_name}_txt'
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)

    for item in os.listdir(folder_name):

        item_path = os.path.join(folder_name, item)
        if os.path.isfile(item_path) and item.lower().endswith(".txt"):
            shutil.move(item_path, os.path.join(new_folder_name, item))
            print(f"Moved {item} to {new_folder_name}")

        

def pre_processing_slides(file_path):
    '''
    file_path: the path for your local pdf file 
    '''
    outdir=pdf_to_fotos(file_path)
    get_description_whole_pdf(outdir)
    move_files(outdir)