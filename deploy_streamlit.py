import streamlit as st
import os

def listar_imagenes(dataset_path):
    return [file for file in os.listdir(dataset_path) if file.endswith('.jpg') or file.endswith('.png')]

dataset_path = 'datathon/images/'

st.title("Select the clothes for your outfit")

if 'show_selector' not in st.session_state:
    st.session_state['show_selector'] = False

if 'selected_images' not in st.session_state:
    st.session_state['selected_images'] = []

if st.button('Select Images'):
    st.session_state['show_selector'] = not st.session_state['show_selector']

if st.session_state['show_selector']:
    imagenes = listar_imagenes(dataset_path)
    cols = st.columns(3) 
    for index, image in enumerate(imagenes):
        col = cols[index % 3] 
        img_path = os.path.join(dataset_path, image)
        if col.button(f"Select {index}", key=image):
            if image not in st.session_state['selected_images']:
                st.session_state['selected_images'].append(image)
        col.image(img_path, width=100) 

st.subheader("Your Current Outfit")
for image in st.session_state['selected_images']:
    st.image(os.path.join(dataset_path, image), width=100)

num_recommendations = st.number_input("How many items do you want to be recommended?", min_value=1, max_value=10)

if st.button("Recommend"):
    st.write("You have selected:")
    for image in st.session_state['selected_images']:
        st.write(image)

    # script for recommend!
