# Skin Diagnosis Chatbot

## Overview

This repository contains the code for an AI chatbot designed for skin disease diagnosis. The chatbot utilizes a combination of computer vision and natural language processing to assist users in diagnosing skin conditions based on uploaded images and user queries.

## Features

- Image-based Diagnosis: Users can upload images of skin conditions for diagnosis.
- Natural Language Interaction: Users can ask questions and receive responses from the chatbot.
- Model Integration: The chatbot integrates pre-trained models for image classification and language generation.

## Prerequisites

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Getting Started

1. **Clone the repository:**

    ```bash
    git clone https://github.com/hyunhp/XAI-ExplainableAI.git
    cd Model/Streamlit
    ```

2. **Set up your environment variables:**

    Create a `.env` file in the root directory and set the following variables:

    ```bash
    local_dir=/path/to/label_data/
    pretrained_model_path=/path/to/pretrained/model.pth
    openai_api_key=your_openai_api_key
    default_prompt=your_default_prompt
    ```

3. **Run the Streamlit app:**

    ```bash
    streamlit run main_app.py
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

[Hyunho Park]

