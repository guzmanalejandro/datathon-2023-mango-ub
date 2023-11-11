import streamlit as st

import pandas as pd

outfit_data = pd.read_csv('datathon/dataset/outfit_data.csv')
product_data = pd.read_csv('datathon/dataset/product_data.csv')
files_by_cat = product_data[['des_product_category', 'des_filename']].groupby('des_product_category').agg(lambda x: list(x))['des_filename']


dataset_path = 'datathon/images/'

st.title("Select the clothes for your outfit")

option = st.selectbox("Select a category", list(files_by_cat.keys()))
def listar_imagenes(dataset_path):
    return files_by_cat[option][:20]

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
        img_path = image
        if col.button(f"Select {index}", key=image):
            # lst.session_state['show_selector'] = False
            if image not in st.session_state['selected_images']:
                st.session_state['selected_images'].append(image)
        col.image(img_path, width=100) 

st.subheader("Your Current Outfit")
for image in st.session_state['selected_images']:
    st.image(image, width=100)

num_recommendations = st.number_input("How many items do you want to be recommended?", min_value=1, max_value=10)

if st.button("Recommend"):
    st.write("You have selected:")
    for image in st.session_state['selected_images']:
        st.write(image)

    # script for recommend!
