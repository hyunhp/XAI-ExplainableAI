'''
TO DO: Model Caption Performace Enhancement
frequently inference
-- there is a man that is standing in the dark with a cell phone
-- a close up of a colorful circle of blood on a blue background
'''
from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import time
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_image_caption(target_image, device):
    """
    Generates a short caption for the provided image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: A string representing the caption for the image.
    """
    tik = time.time()
    image_array = np.uint8(target_image)

    # Create an image from the NumPy array
    image = Image.fromarray(image_array)
    image = image.convert('RGB')
    model_name = "Salesforce/blip-image-captioning-large"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    inputs = processor(image, return_tensors='pt').to(device)
    output = model.generate(**inputs, max_new_tokens=20)

    caption = processor.decode(output[0], skip_special_tokens=True)
    tok = time.time()
    print(f'TIME ELAPSED : {tok-tik}')
    print(f'CAPTION INFORMATION : {caption}')
    print(f'--------------------------------')
    return caption

def diagnose_lcel(default_prompt, llm, original_info, inference_info, model_classification, query):
    prompt = ChatPromptTemplate.from_template(default_prompt)
    output_parser = StrOutputParser()
    llm.streaming = True
    chain = prompt | llm | output_parser

    answer = chain.invoke({
                            "original_caption": original_info,
                            "model_caption" : inference_info, 
                            "model_classified" : model_classification, 
                            "question" : query,
                            })
    print(f'LLM INFERENCED THE DIAGNOSE........')
    print(f'-----------------------------------')
    return answer
