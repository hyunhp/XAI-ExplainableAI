# Skin Diagnosis Chatbot

## Overview

This repository contains the code for an AI chatbot designed for skin disease diagnosis. The chatbot utilizes a combination of computer vision and natural language processing to assist users in diagnosing skin conditions based on uploaded images and user queries.

## Demo

## Features

- Image-based Diagnosis: Users can upload images of skin conditions for diagnosis.
- Natural Language Interaction: Users can ask questions and receive responses from the chatbot.
- Model Integration: The chatbot integrates pre-trained models for image classification and language generation.

## Prerequisites

1. Install the required Python packages using pip

```bash
pip install -r pip_requirements.txt
```

2. Or install the required Python pacakes using conda

```bash
conda create --name <env> --file conda_requirements.txt
```


## Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/hyunhp/XAI-ExplainableAI.git
    cd Model/Streamlit
    ```

2. **Set up your environment variables:**

    Create a `.env` file in the root directory and set the following variables.
    Pretrained model is saved under the root directory, "pretrained".

    ```bash
    pretrained_model_path=/path/to/pretrained/resnet18_pretrained.pth
    openai_api_key=your_openai_api_key
    DEFAULT_PROMPT=your_default_prompt
    GPT4_PROMPT=your_vlm_prompt
    ```

3. **Run the Streamlit app:**

    ```bash
    streamlit run streamlit.py
    ```

4. **Access the chatbot in your browser:**

    Open your browser and navigate to http://localhost:8501

## Usage

1. **Login to the chatbot with your credentials.**
2. **Upload skin disease images for diagnosis.**
3. **Ask questions related to skin conditions.**
4. **Receive responses and diagnostic information from the chatbot.**

## Additional Notes

- This code is designed for educational and research purposes.
- Ensure proper authentication and security measures before deploying in production.

## Author

**Hyunho Park**

## Updated At

2024-06-04