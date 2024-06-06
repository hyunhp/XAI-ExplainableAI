from PIL import Image 
import os
import pickle
import torch
from torchvision import transforms
from pathlib import Path
import streamlit as st
import streamlit_authenticator as stauth
from utils.inference import Set_pretrained_model, cam_inference, vlm_inference
from dotenv import load_dotenv, find_dotenv
from utils.random_state import set_seed
from utils.streamlit_cache import clear_cache

# Set inference statement
set_seed(seed=42)

# Loda .env file
dotenv_file = find_dotenv()
load_dotenv(dotenv_file)

# Change the pretrained model (resnet18 on HAM10000) saved path
pretrained_model_path = os.environ['pretrained_model_path']
num_classes = 7
label_mapping = {'bkl': 0, 'nv': 1, 'df': 2, 'mel': 3, 'vasc': 4, 'bcc': 5, 'akiec': 6}
model, model_architecture = Set_pretrained_model(num_classes=num_classes, pretrained_model_path=pretrained_model_path)

# Set torch enviroment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor()
    , transforms.Resize((224, 224), antialias=True) # PRE TRAINED MODEL MEAND AND STANDARD DEVIATION
    , transforms.Normalize(mean=[0.7635212557080773, 0.5461279508434921, 0.5705303582621197], 
                           std =[0.08962782189107416, 0.11830749629626626, 0.13295368820124384])
    ])

# OpenAI setting
openai_api_key = os.environ['openai_api_key']
gpt4_prompt    = os.environ['GPT4_PROMPT']
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {openai_api_key}"}

st.markdown("""
    <style>
        header{visibility:hidden;}
    </style>
""", unsafe_allow_html=True
)

# Check login authorization
if "authenticator" not in st.session_state or st.session_state['logout']:
    names = ["admin", "doctor", 'user']
    usernames = ["admin", "doctor", 'user']
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
        st.write('')
        with st.chat_message("user"):
            with st.spinner('Wait for it...'):
                st.markdown(user_message)    
        st.session_state["messages"].append({"role": "user", "content": user_message})  
        
        with st.chat_message("assistant"):
            with st.spinner('Wait for it...'):
                if img_file is None:                
                    response = 'Please insert image what you want to diagnose.'
                    st.markdown(f'{response}')
                    resized_img = None
                    output_img  = None
                    additional_info= ''
                else:
                    resized_img, output_img, predicted_label, max_activation_coord, base64_image = cam_inference(target_image=img_file, 
                                                                                                                 model = model, 
                                                                                                                 model_architecture = model_architecture, 
                                                                                                                 device = device, 
                                                                                                                 transform = transform, 
                                                                                                                 label_mapping = label_mapping,
                                                                                                                 )
                    messageCoordPoint = user_message + f"\nPaying attention to the region around coordinates {max_activation_coord}."
                    input_variables = ['inference_label', 'max_activation_coord', 'question']
                    input_data = [predicted_label, max_activation_coord, messageCoordPoint]
                    input_dict = {k:v for k, v in zip(input_variables, input_data)}
                    input_text = gpt4_prompt.format(**input_dict)
                    
                    response = vlm_inference(input_text=input_text, base64_image=base64_image, headers=headers)
                    response += f'\n\nModel "Mostly focused" on the "Red area", and "Leastly" on the "Blue area".'
                    st.markdown(response)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(resized_img)
                    with col2:
                        st.image(output_img)

                st.session_state["messages"].append({"role": "assistant", 
                                                    "content": response, 
                                                    "OriginalImage" : resized_img,
                                                    "InferenceImage": output_img
                                                    })
            
