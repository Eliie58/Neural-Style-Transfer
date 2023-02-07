'''
Web app module.
'''

import os
from io import BytesIO
import numpy as np
import requests

import streamlit as st

st.set_page_config(
    page_title="Neural Style Transfer 🎨.py",
    layout="wide"
)

API_ENDPOINT = os.getenv('API_ENDPOINT', 'http://localhost:8000/nst')
STYLES_DIR = os.getenv('STYLES_DIR', 'web/styles')


def get_styles():
    '''
    Get available styles from Styles Directory.
    '''
    return [style.replace(".jpg", "") for style in os.listdir(STYLES_DIR)]


def main():
    '''
    Render web app content.
    '''

    st.title('Neural Style Transfer 🎨')
    with st.sidebar:
        st.write('Upload your image, and choose the new style')
        option = st.selectbox(
            'What style do you want to apply to your image?',
            get_styles())

        uploaded_file = st.file_uploader("Upload an Image")
        if uploaded_file is not None:
            st.write('Your image is : ')
            st.image(uploaded_file)

    if option is not None and uploaded_file is not None:
        with open(f'{STYLES_DIR}/{option}.jpg', 'rb') as style_file:
            files = {'image_file': (uploaded_file.name,
                                    uploaded_file,
                                    "multipart/form-data"),
                     'style_file': style_file}

            response = requests.post(API_ENDPOINT,
                                     files=files,
                                     stream=True)

            st.write(f'Neural Style Transfer into {option}')

        consume_stream(response)


def consume_stream(stream):
    '''
    Consume api stream response.
    '''
    data = b''
    progress = st.progress(0)
    image = st.empty()
    progress_percent = 0
    for np_bytes in stream.iter_content(chunk_size=1024):
        if np_bytes == b'--DELIMITER--':
            np_rr = bytes_to_array(data)
            progress_percent += 1
            progress.progress(progress_percent)
            image.image(np_rr)
            data = b''
        else:
            data += np_bytes


def bytes_to_array(bytes_arr: bytes) -> np.ndarray:
    '''
    Transform bytes to numpy array.
    '''
    np_bytes = BytesIO(bytes_arr)
    return np.load(np_bytes, allow_pickle=True)


if __name__ == "__main__":
    main()
