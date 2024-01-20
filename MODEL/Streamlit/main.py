'''
TO DO: MUTLI MODAL RESERACH
to do: library cleansing
UPDATED AT : 2024-01-20
CREATED AT : 2024-01-17
'''

import numpy as np
from io import StringIO
import cv2
from PIL import Image 
import os
import json
import pickle
import torch
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
from langchain.callbacks.base import BaseCallbackHandler
from datetime import datetime, timedelta, timezone
from utils.inference import inference, Get_label_mapping, Set_pretrained_model, Get_MeanStd_value
import sys
from dotenv import load_dotenv, find_dotenv
from utils.image_caption import get_image_caption, diagnose_lcel
from langchain_openai import ChatOpenAI

class StreamHandler(BaseCallbackHandler):
    '''Streamling on the Langchain Chain method'''
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def clear_cache():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)

dotenv_file = find_dotenv()
load_dotenv(dotenv_file)

# Change the local path to your label data folder
local_dir = os.environ['local_dir']
num_classes, label_mapping, label_dic = Get_label_mapping(local_dir=local_dir)

# Change the pretrained model saved path
pretrained_model_path = os.environ['pretrained_model_path']
model, model_architecture = Set_pretrained_model(num_classes=num_classes, pretrained_model_path=pretrained_model_path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = Get_MeanStd_value(local_dir=local_dir)

# llm setting
openai_api_key = os.environ['openai_api_key']
default_prompt = os.environ['default_prompt']

openai_llm = ChatOpenAI(
                temperature=0.7,
                openai_api_key=openai_api_key,
                model="gpt-3.5-turbo",
                callbacks=[StreamHandler(st.empty())],
                )

st.markdown("""
    <style>
        header{visibility:hidden;}
    </style>
""", unsafe_allow_html=True
)

# 로그인 권한 확인
## 권한 없을 시, 초기 로그인
if "authenticator" not in st.session_state or st.session_state['logout']:
    names = ["admin", "user"]
    usernames = ["admin", "user"]
    file_path = Path(__file__).parent/"pkl/hashed_pw.pkl"
    with file_path.open("rb") as file:
        hashed_passwords = pickle.load(file)
        
    login_credentials = {"usernames":{}}
          
    for uname,name,pwd in zip(usernames,names,hashed_passwords):
        user_dict = {"name": name, "password": pwd}
        login_credentials["usernames"].update({uname: user_dict})

    authenticator = stauth.Authenticate(login_credentials,'some_cookie_name', 'some_signature_key', cookie_expiry_days=30)
    name, authentication_status, username = authenticator.login("Lesion Skin Dieases Diagnose Chatbot", "main")

    if authentication_status == False:
        st.error("Username/password is incorrect")
        st.markdown("""
            <style>
                section[data-testid="stSidebar"][aria-expanded="true"]{
                    display: none;
                }
            </style>
            """, unsafe_allow_html=True)

    elif authentication_status == None:
        clear_cache()
        st.warning("📢 Please login with authorized ID and Password")
        st.markdown("""
            <style>
                section[data-testid="stSidebar"][aria-expanded="true"]{
                    display: none;
                }
            </style>
            """, unsafe_allow_html=True)
    else:
        st.session_state["authenticator"] = authenticator


## Pass authentication _> Access to chat bot service
if "authenticator" in st.session_state and st.session_state["authentication_status"] is True:
    username = st.session_state["username"]
    authenticator = st.session_state["authenticator"]

    ## SIDEBAR    
    with st.sidebar:
        col1 =st.sidebar.columns(1)[0]
        with col1:
            st.write('''
                    ※ Please follow the below instructions to use chatbot service.
                    1. Please upload the skin dieases image to diagnose.
                    2. Please ask what kinds of information what you want to ask for.
                    ''')
            
        col2 = st.sidebar.columns(1)[0]
        with col2: 
            img_file = st.file_uploader('If you want to check dieases, please upload the image.', type=['png', 'jpg', 'jpeg'])  

        st.write("Logout Buttom")
        authenticator.logout('Logout', 'main')

    if img_file is not None:
        img = Image.open(img_file)

    # Main screen
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", 
                                        "content": 
                                            f"Dear '{username}', Welcome Lesion Skin Dieases Diagnose Chatbot!!\n\n"
                                            f"If you want to diagnose skin dieases, Please Upload Skin Dieases Image on the Sidebar!!",
                                        "OriginalImage" : None,
                                        "InferenceImage": None,
                                        }]
            
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:  
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                if msg['InferenceImage'] is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(msg['OriginalImage'])
                    with col2:
                        st.image(msg['InferenceImage'])
                        
    user_message = st.chat_input()
    
    if user_message:
        with st.chat_message("user"):
            with st.spinner('Wait for it...'):
                st.markdown(user_message)    
        st.session_state["messages"].append({"role": "user", "content": user_message})  
        
        with st.chat_message("assistant"):
            openai_llm.callbacks=[StreamHandler(st.empty())]
            with st.spinner('Wait for it...'):
                if img_file is None:                
                    response = 'Please insert image what you want to diagnose.'
                    st.markdown(f'{response}')
                    resized_img = None
                    output_img  = None
                    additional_info= ''
                else:
                    resized_img, output_img, predicted_label, target_answer = inference(target_image=img_file, 
                                                                                        model = model, 
                                                                                        model_architecture = model_architecture, 
                                                                                        device = device, 
                                                                                        transform = transform, 
                                                                                        label_mapping = label_mapping,
                                                                                        label_dic = label_dic,)
                    original_caption = get_image_caption(resized_img, device)
                    cam_caption = get_image_caption(output_img, device)
                    
                    response = diagnose_lcel(default_prompt= default_prompt, 
                                            llm = openai_llm,
                                            original_info = original_caption, 
                                            inference_info= cam_caption, 
                                            model_classification = target_answer, 
                                            query = user_message)
                    
                    additional_info = f'\n\nUploaded image (left) is labeled as "{target_answer.upper()}" and expected (right) is "{predicted_label.upper()}"'
                    additional_info+= f'\n\nModel "Mostly focused" on the "Red area", and "Leastly" on the "Blue area".'
                    st.markdown(additional_info)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(resized_img)
                    with col2:
                        st.image(output_img)
                        

                    # st.write(f'{additional_info}')
                    # response += additional_info
                                
                st.session_state["messages"].append({"role": "assistant", 
                                                    "content": response + additional_info, 
                                                    "OriginalImage" : resized_img,
                                                    "InferenceImage": output_img
                                                    })
            
