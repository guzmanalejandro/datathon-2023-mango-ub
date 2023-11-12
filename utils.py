import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Plot all outfit images
def plot_all_outfit_images(outfit_data, product_data):
    unique_outfits = outfit_data['cod_outfit'].unique()
    i = 0

    for outfit_id in unique_outfits:
        if i < 20:
            i += 1
            # Filter the products that belong to the specified outfit
            outfit_products = outfit_data[outfit_data['cod_outfit'] == outfit_id]['cod_modelo_color']
            linked_products = product_data[product_data['cod_modelo_color'].isin(outfit_products)]

            # Plotting the images
            print(f"Outfit ID: {outfit_id}")
            fig, axes = plt.subplots(1, len(linked_products), figsize=(20, 10))
            if len(linked_products) == 1:  # Handle case for single product
                axes = [axes]
            for ax, (_, product) in zip(axes, linked_products.iterrows()):
                img_path = product['des_filename']
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(product['cod_modelo_color'])
                    ax.axis('off')
                except FileNotFoundError:
                    ax.set_title("Image not found")
                    ax.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            break

# Plot all outfit images for a given product
def plot_outfits_for_similar_products(product_data, outfit_data, product_id):
    # Extract the left part of the product_id (before the hyphen)
    product_base_id = product_id.split('-')[0]

    # Find all products that start with the same base ID
    similar_products = product_data[product_data['cod_modelo_color'].str.startswith(product_base_id)]

    # Process each product
    for _, product in similar_products.iterrows():
        current_product_id = product['cod_modelo_color']
        print(f"Product ID: {current_product_id}")
        img_path = product['des_filename']
        try:
            img = Image.open(img_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        except FileNotFoundError:
            print("Image not found")

        # Get the outfits that contain the current product
        outfit_ids = outfit_data[outfit_data['cod_modelo_color'] == current_product_id]['cod_outfit'].unique()
        print(f"\nNumber of outfits containing product {current_product_id}: {len(outfit_ids)}")
        for outfit_id in outfit_ids:
            # Filter the products that belong to the specified outfit
            outfit_products = outfit_data[outfit_data['cod_outfit'] == outfit_id]['cod_modelo_color']
            linked_products = product_data[product_data['cod_modelo_color'].isin(outfit_products)]

            # Plotting the images
            print(f"\nOutfit ID: {outfit_id}")
            fig, axes = plt.subplots(1, len(linked_products), figsize=(20, 10))
            if len(linked_products) == 1:  # Handle case for single product
                axes = [axes]
            for ax, (_, product) in zip(axes, linked_products.iterrows()):
                img_path = product['des_filename']
                try:
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(product['cod_modelo_color'])
                    ax.axis('off')
                except FileNotFoundError:
                    ax.set_title("Image not found")
                    ax.axis('off')
            plt.tight_layout()
            plt.show()

def get_single_outfit(outfit_data, product_data, outfit_id):
    # Merge the outfit_data and product_data dataframes on cod_modelo_color
    outfit_products_data = outfit_data[outfit_data['cod_outfit'] == outfit_id].merge(product_data, on='cod_modelo_color')

    # Get the image paths for each product in the outfit
    image_paths = outfit_products_data['des_filename'].tolist()

    # Plot the images
    fig, axes = plt.subplots(1, len(image_paths), figsize=(20, 10))
    if len(image_paths) == 1:  # Handle case for single product
        axes = [axes]

    for ax, img_path in zip(axes, image_paths):
        try:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
        except FileNotFoundError:
            ax.set_title("Image not found")
            ax.axis('off')

    plt.tight_layout()
    plt.show()
