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
    excluded_categories = []

    # Exclude categories already present in the outfit
    for _, item in outfit_items.iterrows():
        category_type = determine_category_type(item)
        if category_type in ['Top', 'Bottom', 'Outerwear']:
            excluded_categories.append(item['des_product_family'])
        elif category_type == 'Accessories' and item['des_product_family'] != 'Footwear':
            excluded_categories.append(item['des_product_family'])

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

    print(f"\nMissing Categories: {missing_categories}")

    # Creating enriched descriptions for the outfit
    outfit_descriptions = outfit_items[description_columns].astype(str).agg(' '.join, axis=1)

    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    outfit_vector = tfidf.fit_transform(outfit_descriptions)

    priority_categories = ['Top', 'Bottom', 'Footwear']

    best_category = None
    highest_similarity = -1

    for category in missing_categories:
        categorized_products = outfit_product_data[outfit_product_data['category_type'] == category.split(' - ')[0]]

        if categorized_products.empty or category.split(' - ')[0] not in priority_categories:
            continue

        category_descriptions = categorized_products[description_columns].astype(str).agg(' '.join, axis=1)
        category_vector = tfidf.transform(category_descriptions)
        cosine_sim = cosine_similarity(outfit_vector, category_vector)
        average_similarity = np.mean(cosine_sim)

        if average_similarity > highest_similarity:
            highest_similarity = average_similarity
            best_category = category

    if best_category is None:
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
    # Merging outfit data with product data to get detailed attributes of products in outfits
    outfit_product_data = outfit_data.merge(product_data, on='cod_modelo_color', how='left')

    # Apply this function to create a new 'refined_category' column
    outfit_product_data['refined_category'] = outfit_product_data.apply(create_refined_category, axis=1)

    outfit_product_data['category_type'] = outfit_product_data.apply(categorize_product, axis=1)

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

        # Adjust the refined categories to ensure unique recommendations
        refined_categories = set()
        for category in missing_types:
            category_specific_products = outfit_product_data[outfit_product_data['category_type'] == category]
            for _, product in category_specific_products.iterrows():
                if product['des_product_family'] not in excluded_categories:
                    refined_category = create_refined_category(product)
                    refined_categories.add(refined_category)

        print(f"\nRefined Categories: {refined_categories}")
        # Removing excluded categories
        refined_categories -= set(excluded_categories)

        # Find the best matching category using TF-IDF
        best_matching_category = find_best_matching_category(cod_modelo_color_list, refined_categories, outfit_product_data)

        print(f"\nBest Matching Category: {best_matching_category}")
        if best_matching_category:

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
    # Define categories based on the hierarchy
    top_categories = ['Tops']
    bottom_categories = ['Bottoms']
    outerwear_categories = ['Outerwear']
    accessory_aggregated_families = ['Accessories']
    footwear_family = ['Footwear']

    if row['des_product_category'] in top_categories:
        return 'Top'
    elif row['des_product_category'] in bottom_categories:
        return 'Bottom'
    elif row['des_product_category'] in outerwear_categories:
        return 'Outerwear'
    elif row['des_product_aggregated_family'] in accessory_aggregated_families:
        if row['des_product_family'] in footwear_family:
            return 'Footwear'
        else:
            return 'Accessories'
    else:
        return 'Other'
    
def identify_missing_category_types(outfit_items):
    existing_types = set(outfit_items.apply(determine_category_type, axis=1))
    all_types = {'Top', 'Bottom', 'Footwear', 'Accessories', 'Outerwear'}

    # Handle Accessories and Footwear to allow multiple items but avoid duplicates
    accessory_items = outfit_items[outfit_items['category_type'] == 'Accessories']
    if not accessory_items.empty:
        existing_types.remove('Accessories')
        accessory_subcategories = set(accessory_items['des_product_family'])
        if 'Footwear' in accessory_subcategories:
            existing_types.add('Footwear')

    return all_types - existing_types

# Define a function to create a refined category
def create_refined_category(row):
    return f"{row['des_product_aggregated_family']} - {row['des_product_family']}"

def categorize_product(row):
    if row['des_product_category'] == 'Tops' or row['des_product_aggregated_family'] in ['Shirts', 'T-shirts']:
        return 'Top'
    elif row['des_product_category'] == 'Bottoms' or row['des_product_aggregated_family'] in ['Trousers & leggings', 'Skirts and shorts', 'Jeans']:
        return 'Bottom'
    elif row['des_product_category'] == 'Accesories, Swim and Intimate' and row['des_product_family'] == 'Footwear':
        return 'Footwear'
    return 'Other'