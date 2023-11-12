from utils import *
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the provided CSV files
outfit_data = pd.read_csv('datathon/dataset/outfit_data.csv')
product_data = pd.read_csv('datathon/dataset/product_data.csv')

# Function to display product images
def display_product_images(outfit_data):
    # Display images for each product in the outfit
    fig, axes = plt.subplots(1, len(outfit_data), figsize=(15, 5))
    if len(outfit_data) == 1:
        axes = [axes]
    for ax, (index, row) in zip(axes, outfit_data.iterrows()):
        img = mpimg.imread(row['des_filename'])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{row['des_product_type']}")
    plt.show()

# Function to identify and exclude conflicting categories
def exclude_conflicting_categories(outfit_items):
    has_tshirt = any(item['des_product_family'] == 'T-shirt' for _, item in outfit_items.iterrows())
    has_shirt = any(item['des_product_family'] == 'Shirt' for _, item in outfit_items.iterrows())

    excluded_categories = []
    if has_tshirt:
        excluded_categories.append('Shirts - Shirt')
    if has_shirt:
        excluded_categories.append('T-shirts - T-shirt')

    return excluded_categories

# Function to calculate the best matching category
def find_best_matching_category(cod_modelo_color_list, missing_categories, outfit_product_data):
    outfit_items = outfit_product_data[outfit_product_data['cod_modelo_color'].isin(cod_modelo_color_list)]

    # List of columns to include in the TF-IDF descriptions
    description_columns = [
        'cod_color_code', 'des_color_specification_esp', 'des_agrup_color_eng', 
        'des_sex', 'des_age', 'des_line', 'des_fabric', 
        'des_product_category', 'des_product_aggregated_family', 
        'des_product_family', 'des_product_type'
    ]

    # Creating enriched descriptions for the outfit
    outfit_descriptions = outfit_items[description_columns].astype(str).agg(' '.join, axis=1)

    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    outfit_vector = tfidf.fit_transform(outfit_descriptions)

    best_category = None
    highest_similarity = -1

    # Iterate over missing categories to find the best match
    for category in missing_categories:
        category_products = outfit_product_data[outfit_product_data['refined_category'] == category]
        category_descriptions = category_products[description_columns].astype(str).agg(' '.join, axis=1)

        if category_descriptions.empty:
            continue

        category_vector = tfidf.transform(category_descriptions)
        cosine_sim = cosine_similarity(outfit_vector, category_vector)
        average_similarity = np.mean(cosine_sim)

        if average_similarity > highest_similarity:
            highest_similarity = average_similarity
            best_category = category

    return best_category

# Function to recommend a product to complete an outfit
def recommend_product_for_outfit(cod_modelo_color_list, category_to_add, outfit_product_data):
    outfit_items = outfit_product_data[outfit_product_data['cod_modelo_color'].isin(cod_modelo_color_list)]
    category_products = outfit_product_data[outfit_product_data['refined_category'] == category_to_add]

    # If there are no products in the desired category, return None
    if category_products.empty:
        print(f"No products available in the '{category_to_add}' category.")
        return None

    # List of columns to include in the TF-IDF descriptions
    description_columns = [
        'cod_color_code', 'des_color_specification_esp', 'des_agrup_color_eng', 
        'des_sex', 'des_age', 'des_line', 'des_fabric', 
        'des_product_category', 'des_product_aggregated_family', 
        'des_product_family', 'des_product_type'
    ]

    # Creating enriched descriptions by concatenating the values from the selected columns
    outfit_descriptions = outfit_items[description_columns].astype(str).agg(' '.join, axis=1)
    category_descriptions = category_products[description_columns].astype(str).agg(' '.join, axis=1)

    # Creating a TF-IDF Vectorizer to analyze product descriptions
    tfidf = TfidfVectorizer(stop_words='english')

    # Fitting the vectorizer to the descriptions
    tfidf_matrix = tfidf.fit_transform(category_descriptions)
    outfit_vector = tfidf.transform(outfit_descriptions)

    # Calculating cosine similarity between the outfit and the category products
    cosine_sim = cosine_similarity(outfit_vector, tfidf_matrix)

    # Finding the best matching product
    best_match_idx = np.argmax(np.sum(cosine_sim, axis=0))
    recommended_product = category_products.iloc[best_match_idx]

    # Return the recommended product as a DataFrame
    return pd.DataFrame([recommended_product])

# Main function to run the process with iterative recommendations
def process_outfit_recommendation(cod_modelo_color_list, num_recommendations=1):
    # Load the product data
    outfit_data = pd.read_csv('datathon/dataset/outfit_data.csv')
    product_data = pd.read_csv('datathon/dataset/product_data.csv')

    # Merging outfit data with product data to get detailed attributes of products in outfits
    outfit_product_data = outfit_data.merge(product_data, on='cod_modelo_color', how='left')

    # Apply this function to create a new 'refined_category' column
    outfit_product_data['refined_category'] = outfit_product_data.apply(create_refined_category, axis=1)

    # Apply this function to create a new 'category_type' column in the outfit_product_data
    outfit_product_data['category_type'] = outfit_product_data.apply(determine_category_type, axis=1)

    # Filter the product data to only include the provided cod_modelo_color items
    outfit_items = outfit_product_data[outfit_product_data['cod_modelo_color'].isin(cod_modelo_color_list)]

    # Check if there are any items in the provided list
    if outfit_items.empty:
        print("No items found for the provided cod_modelo_color list.")
        return

    # Delete the repeated items that share the same des_product_type
    outfit_items = outfit_items.drop_duplicates(subset=['des_product_type'])

    # Displaying the original outfits
    print("Provided Outfit Composition:")
    display_product_images(outfit_items)

    recommended_products = pd.DataFrame()

    for _ in range(num_recommendations):
        # Identify missing category types
        missing_types = identify_missing_category_types(outfit_items)

        # If there are no missing types, skip the recommendation
        if not missing_types:
            print("Outfit already contains products from all types.")
            break

        # Get categories to exclude
        excluded_categories = exclude_conflicting_categories(outfit_items)

        # Adjust missing categories based on missing types
        refined_categories = set()
        for category in missing_types:
            if category == 'Top':
                refined_categories.update(outfit_product_data[outfit_product_data['category_type'] == 'Top']['refined_category'])
            elif category == 'Bottom':
                refined_categories.update(outfit_product_data[outfit_product_data['category_type'] == 'Bottom']['refined_category'])
            elif category == 'Accessories':
                refined_categories.update(outfit_product_data[outfit_product_data['category_type'] == 'Accessories']['refined_category'])
            elif category == 'Footwear':
                refined_categories.update(outfit_product_data[outfit_product_data['category_type'] == 'Footwear']['refined_category'])
            elif category == 'Outerwear':
                refined_categories.update(outfit_product_data[outfit_product_data['category_type'] == 'Outerwear']['refined_category'])

        refined_categories -= set(excluded_categories)

        # Find the best matching category using TF-IDF
        best_matching_category = find_best_matching_category(cod_modelo_color_list, refined_categories, outfit_product_data)
        print(f"\nIdentified Best Matching Category: {best_matching_category}") 
        if best_matching_category:
            print(f"\nIdentified Best Matching Category: {best_matching_category}")

            # Recommend a product in the selected missing category
            recommended_product = recommend_product_for_outfit(cod_modelo_color_list, best_matching_category, outfit_product_data)

            # If a product is recommended, add it to the outfit and the recommended products list
            if recommended_product is not None:
                print(f"\nRecommended Addition in '{best_matching_category}' category:")
                display_product_images(recommended_product)

                # Updating outfit_items and recommended_products for next iteration
                outfit_items = pd.concat([outfit_items, recommended_product])
                recommended_products = pd.concat([recommended_products, recommended_product])
            else:
                print("No suitable product found for recommendation.")
                break
        else:
            print("No suitable category found for recommendation.")
            break

    if not recommended_products.empty:
        print("\nFinal Recommended Products:")
        display_product_images(recommended_products)
        print("\nFinal Outfit Composition After Adding All Recommended Products:")
        display_product_images(outfit_items)
        return outfit_items
    else:
        print("No products were recommended.")

# Function to determine the category type (Top, Bottom, etc.) based on the product family
def determine_category_type(row):
    top_categories = ['Tops', 'Shirts', 'T-shirts', 'Sweaters and Cardigans', 'Poloshirts']
    bottom_categories = ['Bottoms', 'Trousers & leggings', 'Jeans', 'Skirts and shorts', 'Leggings and joggers']
    footwear_categories = ['Footwear']
    accessories_categories = ['Accesories, Swim and Intimate', 'Accessories', 'Jewellery', 'Bags', 'Glasses', 'Wallets & cases', 'Belts and Ties', 'Hats, scarves and gloves']
    outerwear_categories = ['Outerwear', 'Jackets and Blazers', 'Coats and Parkas', 'Trenchcoats', 'Puffer coats', 'Leather jackets', 'Parkas']

    if row['des_product_category'] in top_categories or row['des_product_aggregated_family'] in top_categories or row['des_product_family'] in top_categories:
        return 'Top'
    elif row['des_product_category'] in bottom_categories or row['des_product_aggregated_family'] in bottom_categories or row['des_product_family'] in bottom_categories:
        return 'Bottom'
    elif row['des_product_category'] in footwear_categories or row['des_product_aggregated_family'] in footwear_categories:
        return 'Footwear'
    elif any(cat in row['des_product_category'] for cat in accessories_categories) or \
         any(cat in row['des_product_aggregated_family'] for cat in accessories_categories) or \
         any(cat in row['des_product_family'] for cat in accessories_categories):
        return 'Accessories'
    elif any(cat in row['des_product_category'] for cat in outerwear_categories) or \
         any(cat in row['des_product_aggregated_family'] for cat in outerwear_categories) or \
         any(cat in row['des_product_family'] for cat in outerwear_categories):
        return 'Outerwear'
    else:
        return 'Other'
    
def identify_missing_category_types(outfit_items):
    existing_types = set(outfit_items.apply(determine_category_type, axis=1))
    all_types = {'Top', 'Bottom', 'Footwear', 'Accessories', 'Outerwear'}
    return all_types - existing_types

# Define a function to create a refined category
def create_refined_category(row):
    return f"{row['des_product_aggregated_family']} - {row['des_product_family']}"