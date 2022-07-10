import streamlit as st
import time
import numpy as np
from ModelScript import ModelScript

LABEL_DICT={
    0:'Seat',
    1:'Enough/Satisfied',
    2:'Mosque',
    3:'Temple',
    4:'Friend',
    5:'Me',
    6:'Church',
    7:'You',
    8:'Love'
}

st.title("Sign Language Recognition")
method=st.selectbox('Capture or Upload an Image',('Upload Image','Capture Image'))
if method=='Upload Image':
    image_file=st.file_uploader('Upload an Image',type=['jpg','png','jpeg'])
else:
    image_file=st.camera_input("Capture an Image")

if image_file:
    progress_bar=st.progress(0)
    for i in range(100):
        time.sleep(0.001)
        progress_bar.progress(i+1)
    st.info('Image uploaded successfully')
    st.image(image_file.getvalue())

if image_file is not None:
    model=ModelScript.load_model()
    with st.spinner("Please wait..."):
        image=  ModelScript.preprocessing(image_file.getvalue())
        prediction=ModelScript.predict(image,model)
        print(f"Prediction:{prediction}")
        st.balloons()
    st.success(f"Prediction:{LABEL_DICT[prediction]}")
else:
    st.sidebar.warning("Please upload or capture an image")