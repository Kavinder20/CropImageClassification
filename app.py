# python3 -m streamlit run app.py
import streamlit as st
from PIL import Image
import os
import urllib.request
import urllib
import cv2
import pandas as pd 
import plotly.express as px
import base64
import numpy as np
import time
import tensorflow as tf 
from deep_translator import GoogleTranslator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
import PIL as image_lib
import tensorflow as tf
# from keras.layers.core import Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, InputLayer, Input
from tensorflow.keras.models import load_model

custom_css = """
<style>
body {
    color: #000000; /* Text color */
    background-color: #FFFFFF; /* Background color */
}
</style>
"""


# class_labels = {
#     0: 'Good Growth (G)',
#     1: 'Drought (DR)',
#     2: 'Nutrient Deficient (ND)',
#     3: 'Weed (WD)',
#     4: 'Other (pest, disease or wind damage)'
# }

class_labels = {
    0: 'DR',
    1: 'G',
    2: 'ND',
    3: 'WD',
    4: 'other'
}

model = load_model("temp_model.h5", safe_mode=True)
height, width = 180,180

def process_image(uploaded_files):
    results = []
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        img_resized = img.resize((height, width)) 
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_label = class_labels[predicted_class_index]
        
        results.append(predicted_class_label)
    return results
#     import random
#     labels = ['G', 'DR', 'ND', 'WD', 'other']
#     time.sleep(5)
#     return [random.choice(labels) for i in range(len(uploaded_files))]


crop_damage_info = {
    'G': {
        'class_id': 'G',
        'class_name': 'Good Growth',
        'description': 'This indicates that the crop is growing well without any visible signs of stress, damage, or disease. The plants are healthy, with adequate water, nutrients, and sunlight.',
        'indicators': 'Uniform color, consistent growth patterns, and no signs of wilting or discoloration.',
        'management_tips': 'Continue with current agricultural practices, monitoring for any early signs of potential problems.'
    },
    'DR': {
        'class_id': 'DR',
        'class_name': 'Drought',
        'description': 'This class indicates that the crops are suffering from a lack of sufficient water. Drought stress can severely impact crop yield and quality.',
        'indicators': 'Wilting, yellowing of leaves, reduced growth rate, and in severe cases, leaf drop or plant death.',
        'management_tips': 'Implement irrigation strategies, mulch to conserve soil moisture, and consider drought-resistant crop varieties.'
    },
    'ND': {
        'class_id': 'ND',
        'class_name': 'Nutrient Deficient',
        'description': 'This indicates that the crops are lacking essential nutrients required for healthy growth. Nutrient deficiencies can affect various physiological processes in plants.',
        'indicators': 'Yellowing or discoloration of leaves, stunted growth, poor fruit or grain development, and specific patterns of deficiency (e.g., interveinal chlorosis for iron deficiency).',
        'management_tips': 'Conduct soil tests to identify lacking nutrients, apply appropriate fertilizers, and use soil amendments to improve nutrient availability.'
    },
    'WD': {
        'class_id': 'WD',
        'class_name': 'Weed',
        'description': 'This class indicates that the crops are being affected by weeds, which compete with crops for resources such as light, water, and nutrients.',
        'indicators': 'Presence of unwanted plants growing among crops, reduced growth of crops due to competition, and possible physical damage to crops from larger or more aggressive weed species.',
        'management_tips': 'Implement effective weed management practices, including mechanical removal, herbicides, and cover crops to suppress weed growth.'
    },
    'other': {
        'class_id': 'other',
        'class_name': 'Other',
        'description': 'This class includes various other types of damage that can affect crops, such as pests, diseases, or physical damage from wind.',
        'indicators': 'Pests: Chewed leaves, holes, visible insects or larvae, and signs of infestation like webbing or frass. Diseases: Spots on leaves, mold or mildew, wilting, and other pathogen-related symptoms. Wind Damage: Broken stems, bent plants, and physical damage caused by strong winds.',
        'management_tips': 'Pests: Use integrated pest management (IPM) strategies, including biological control, chemical pesticides, and cultural practices. Diseases: Apply fungicides or bactericides, use disease-resistant varieties, and practice crop rotation. Wind Damage: Implement windbreaks or shelterbelts, stake or support vulnerable plants, and choose resilient crop varieties.'
    }
}


global results

if 'results_temp' not in st.session_state:
    st.session_state.results_temp = None

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# Initialize session state
if 'get_res_button_click' not in st.session_state:
    st.session_state.get_res_button_click = False

language_codes = {
    'English': 'en',
    'Japanese': 'ja',
    'French': 'fr',
    'Spanish': 'es',
    'German': 'de',
    'Chinese (Simplified)': 'zh-CN',
    'Russian': 'ru',
    'Arabic': 'ar',
    'Hindi': 'hi'
}
lang_opts = ["German" , 'English', 'Japanese', 'French', 'Spanish', 'Chinese (Simplified)', 'Russian', 'Arabic', 'Hindi']

def translate_description(input, selected_languages='English'):
    selected_languages = language_codes[selected_languages]
    translated = GoogleTranslator(source='auto', target=selected_languages).translate(input)   # 
    return translated

st.set_page_config(layout="wide", page_icon="✅",initial_sidebar_state="expanded",menu_items={'Get Help': 'https://www.extremelycoolapp.com/help','Report a bug': "https://www.extremelycoolapp.com/bug", 'About': "# This is a header. This is an *extremely* cool app!" })
# my_logo = Image.open('LEGO_logo.svg.png')
my_logo = Image.open('farmerlogo.jpeg')
st.sidebar.image(my_logo)

# selected_languages = st.sidebar.multiselect("Caption Languages", ["English", "French"], default=["English","French"])

uploaded_image = st.sidebar.file_uploader("Choose a picture", type=["jpg", "jpeg", "png"], key='sidebar', accept_multiple_files = True)
uploaded_files = uploaded_image
# include_captions = st.sidebar.checkbox("Include Captions")
include_captions=None

options       = ['Fast Loading']#, 'Show Both'] #, 'High Accuracy'
model_options = st.sidebar.radio("Model Selection", options)

colsb1, colsb2 = st.sidebar.columns([0.35,0.6])

get_res_button_click = colsb1.button("Get Results",key='sb1')
clear_cache_btn      = colsb2.button("Clear Cache",key='sb2')

if get_res_button_click:
    st.session_state.get_res_button_click = True
    
my_logo = Image.open('university.png')
# my_logo = Image.open('farmerlogo.jpeg')
st.sidebar.image(my_logo, use_column_width=True)


df = px.data.iris()
# @st.experimental_memo
# @st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

artistic_title_with_box = """
    <div style="text-align: center; margin-bottom:20px; padding: 0px; background-color: #ffffff; border-radius: 15px; ">
        <h1 style="color: #4A90E2 ; text-shadow: 2px 2px 4px #000000; padding: 5px; margin:0px; font-size: 45px;">Your Local Guide For Farm Information</h1>
    </div>
"""
st.markdown(artistic_title_with_box, unsafe_allow_html=True)

col3a, col3b, col3c = st.columns([1,8,1], gap='small')
with col3b:
    if len(uploaded_image) != 0:
        # image = Image.open(uploaded_image[0])
        # st.image(image,  use_column_width=True,) #caption="Uploaded Image",
        pass
    else:
        image2 = Image.open("farmbuddy.png")
        st.image(image2,  use_column_width=True, output_format="GIF") 


def download_csv():
    # Example DataFrame
    data = {'Name': ['John', 'Jane', 'Doe'],
            'Age': [25, 30, 22],
            'City': ['New York', 'San Francisco', 'Chicago']}
    df = pd.DataFrame(data)
    try:
        results_temp = st.session_state.results_temp
        results = results_temp.copy()

        df_res = pd.DataFrame([results])
        df_res['image'] = uploaded_image.name
        csv_data = df_res.to_csv(index=False)
    except:
        csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download File</a>'
    col3b4.markdown(href, unsafe_allow_html=True)

like_icon     = "👍"
dislike_icon  = "👎"
download_icon = "📥"
with col3b:
    col3b1,col3b2,col3b3,col3b4,col3b5 = st.columns([1,1,2,1,3], gap='small')
    if col3b1.button(f"{like_icon} Like"):
        pass
    if col3b2.button(f"{dislike_icon} Dislike"):
        pass
    # if col3b3.button(f"{download_icon} Download"):
    #     download_csv()
    selected_language_trans = col3b5.selectbox("Translation Language", lang_opts,key='trans_all')

if clear_cache_btn:
    st.session_state.clear()
    st.session_state.uploaded_image = None
    st.session_state.results_temp = None
    clear_cache_btn = False

get_res_button_click = st.session_state.get_res_button_click 
print('btn', get_res_button_click)

if get_res_button_click:
    if st.session_state.uploaded_image != uploaded_image:
        results_temp = process_image(uploaded_files)
        st.session_state.results_temp = results_temp
        st.session_state.uploaded_image = uploaded_image
        results = results_temp.copy()
    else:
        results_temp = st.session_state.results_temp
        results = results_temp.copy()

if get_res_button_click:
    for i in range(len(uploaded_files)):
        image = Image.open(uploaded_files[i])
        st.image(image, caption=uploaded_files[i].name)
        image_array = np.array(image)
        label = results[i]
        class_info = crop_damage_info[label]

        # Display label and class information
        st.markdown(f"**Label:** {class_info['class_name']}")
        st.markdown(f"**Description:** {class_info['description']}")
        st.markdown(f"**Description ({selected_language_trans}):**  {translate_description(class_info['description'], selected_language_trans)}")

        st.markdown(f"**Indicators:** {class_info['indicators']}")
        st.markdown(f"**Indicators ({selected_language_trans}):** {translate_description(class_info['indicators'], selected_language_trans)}")
        st.markdown(f"**Management Tips:** {class_info['management_tips']}")
        st.markdown(f"**Management Tips ({selected_language_trans}):** {translate_description(class_info['management_tips'], selected_language_trans)}")


hide_streamlit_style = """
<style>
    .sub_div {
        position: absolute;
        bottom: 0px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after {
        content:'Developed by Sandy @University of Germany'; 
        visibility: visible;
        display: block;
        position: center;
        padding: 10px;
        top: 5px;
        # left: 600px;
    }
    div.abs{
        position: fixed;
        bottom: 0px;
        right: 10px;
        padding: 0px;
    }
</style>
"""

imageaads = Image.open("LEGO_logo.svg.png")
imageaads = Image.open("farmerlogo.jpeg")
col_bot1,col_bot2 = st.columns([0.8,0.4])
# col_bot2.image(imageaads,  use_column_width=True,output_format="GIF")
st.sidebar.markdown(hide_streamlit_style, unsafe_allow_html=True,)

style_image1 = """
width: auto;
max-width: 850px;
height: auto;
max-height: 750px;
display: block;
justify-content: center;
border-radius: 20%;
"""
link = "https://people.com/thmb/TzDJt_cDuFa_EShaPF1WzqC8cy0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc():focal(216x0:218x2)/michael-jordan-435-3-4fc019926b644905a27a3fc98180cc41.jpg"
