import streamlit as st

import pandas as pd
import recommender


# Navbar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Recommendations", "Data analysis"])

def home_page():
    import streamlit as st

    # Configura el título y la introducción
    st.title("Welcome to the Outfit Recommendation App!")
    st.markdown("""
        <style>
            .main {
                background-color: #F5F5F5;
                color: #444;
            }
        </style>
    """, unsafe_allow_html=True)

    st.write("""
    # Our Team
    We are a dynamic team of four computer engineers, passionate about fashion and technology.
    
    ## What We Do
    During a recent datathon, we developed an innovative model to recommend products that perfectly match your current outfit. Leveraging advanced algorithms and a keen eye for style, our solution is designed to enhance your wardrobe effortlessly.
    
    ### Our Mission
    Our mission is to blend fashion sense with cutting-edge technology, providing you with an exceptional outfit planning experience.
    """)
    
    

    st.markdown("---")
    st.markdown("© 2023 Outfit Recommendation App - All Rights Reserved")


def recommendations_page():
    import streamlit as st
    import pandas as pd

    selected_items = []

    def extract_model_code(image_path):
        path = image_path.split('/')[-1].split('.')[0]
        path = path.replace('_', '-')
        return path[5:]
    
    def construct_image_path(model_code):
        path = "2019_" + model_code.replace('-', '_') + ".jpg"
        full_path = f"datathon/images/{path}"
        return full_path
    
    # Reading data
    outfit_data = pd.read_csv('datathon/dataset/outfit_data.csv')
    product_data = pd.read_csv('datathon/dataset/product_data.csv')
    files_by_cat = product_data[['des_product_category', 'des_filename']].groupby('des_product_category').agg(list)['des_filename']

    # Merging outfit data with product data to get detailed attributes of products in outfits
    outfit_product_data = outfit_data.merge(product_data, on='cod_modelo_color', how='left')

    # Define a function to create a refined category
    def create_refined_category(row):
        return f"{row['des_product_aggregated_family']} - {row['des_product_family']}"

    # Apply this function to create a new 'refined_category' column
    outfit_product_data['refined_category'] = outfit_product_data.apply(create_refined_category, axis=1)
    
    # Apply this function to create a new 'category_type' column in the outfit_product_data
    outfit_product_data['category_type'] = outfit_product_data.apply(recommender.determine_category_type, axis=1)

    dataset_path = 'datathon/images/'
    st.title("Select the clothes for your outfit")

    # Selection of category
    option = st.selectbox("Select a category", list(files_by_cat.keys()))

    def list_images(dataset_path):
        return files_by_cat[option][:20]

    # Session state for image selector
    if 'show_selector' not in st.session_state:
        st.session_state['show_selector'] = False

    if 'selected_images' not in st.session_state:
        st.session_state['selected_images'] = []

    # Button to toggle image selector
    if st.button('Select Images'):
        st.session_state['show_selector'] = not st.session_state['show_selector']

    # Display image selector
    if st.session_state['show_selector']:
        images = list_images(dataset_path)
        cols = st.columns(3)
        for index, image in enumerate(images):
            col = cols[index % 3]
            img_path = image
            if col.button(f"Select {index}", key=image):
                if image not in st.session_state['selected_images']:
                    st.session_state['selected_images'].append(image)
            col.image(img_path, width=100)

    # Display selected outfit
    st.subheader("Your Current Outfit")
    for image in st.session_state['selected_images']:
        st.image(image, width=100)

    # Input for number of recommendations
    num_recommendations = st.number_input("How many items do you want to be recommended?", min_value=1, max_value=10)

    # Recommendation button
    if st.button("Recommend"):
        # Validation for selected items
        if not st.session_state['selected_images']:
            st.warning("Please select at least one item.")
        else:
            for image in st.session_state['selected_images']:
                item = extract_model_code(image)
                selected_items.append(item)

            # Implement the logic to recommend items based on the rules
            outfit_recommended = recommender.process_outfit_recommendation(selected_items, num_recommendations)

            recommended_items = outfit_recommended['cod_modelo_color'].tolist()

            st.subheader("Your Final Outfit")
            # Display recommended items
            for item in recommended_items:
                st.image(construct_image_path(item), width=100)


def data_analysis():
    st.title("Data analysis")
    st.write("Welcome to the Outfit Recommendation App!")

if page == "Home":
    home_page()
elif page == "Recommendations":
    recommendations_page()
elif page == "Data analysis":
    data_analysis()

