#!/usr/bin/env python3
"""
Direct Background Removal with CSV Integration
Extract product tiles from source image and apply background removal
"""

import cv2
import pandas as pd
import numpy as np
import os
from rembg import remove
from PIL import Image
import io

def extract_and_remove_backgrounds():
    print('🚀 DIRECT BACKGROUND REMOVAL WITH CSV INTEGRATION')
    print('Extracting product tiles and applying background removal')
    print('=' * 70)

    # Load the source image
    screenshot_path = '/Users/davemooney/_dev/Flink/IMG_7805.PNG'
    image = cv2.imread(screenshot_path)

    if image is None:
        print(f'❌ ERROR: Could not load image from {screenshot_path}')
        return None

    print(f'📂 Loaded source image: {screenshot_path}')
    print(f'   📏 Image size: {image.shape[1]}x{image.shape[0]}')

    # Known tile coordinates for IMG_7805 (4 products in 2x2 grid)
    tiles = [
        {'x': 48, 'y': 767, 'w': 573, 'h': 573, 'desc': 'Bananen 5 Stk.'},
        {'x': 669, 'y': 767, 'w': 573, 'h': 573, 'desc': 'Bananen Chiquita 5 Stk.'},
        {'x': 48, 'y': 1700, 'w': 573, 'h': 573, 'desc': 'Apfel Snack Gala 1kg'},
        {'x': 669, 'y': 1700, 'w': 573, 'h': 573, 'desc': 'Apfel Pink Lady 1kg'}
    ]

    # Create directories
    original_tiles_dir = 'original_product_tiles'
    bg_removed_dir = 'background_removed_products'
    os.makedirs(original_tiles_dir, exist_ok=True)
    os.makedirs(bg_removed_dir, exist_ok=True)

    print('\n🔄 STEP 1: Extracting product tiles...')

    def remove_pink_button(tile_image):
        """Remove pink add-to-cart button from tile"""
        # Convert to different color spaces for better detection
        hsv = cv2.cvtColor(tile_image, cv2.COLOR_BGR2HSV)

        # Define pink color range (more precise range)
        lower_pink = np.array([140, 50, 50])
        upper_pink = np.array([170, 255, 255])

        # Create mask for pink areas
        pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)

        # Find contours of pink areas
        contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create output image with alpha channel
        output = cv2.cvtColor(tile_image, cv2.COLOR_BGR2BGRA)

        # Remove pink button areas by making them transparent
        for contour in contours:
            # Check if contour is roughly circular (button-like)
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                # Fill contour area with transparency
                cv2.fillPoly(output, [contour], (0, 0, 0, 0))

        return output

    enhanced_data = []

    for i, tile_info in enumerate(tiles, 1):
        print(f'\n📦 Processing Product {i}: {tile_info["desc"]}')

        # Extract tile from source image
        x, y, w, h = tile_info['x'], tile_info['y'], tile_info['w'], tile_info['h']
        tile = image[y:y+h, x:x+w].copy()

        print(f'   📏 Tile size: {w}x{h}')

        # Save original tile
        original_filename = f'IMG_7805_product_{i}_original.png'
        original_path = os.path.join(original_tiles_dir, original_filename)
        cv2.imwrite(original_path, tile)
        print(f'   💾 Original tile saved: {original_path}')

        # Remove pink button
        print('   🎯 Removing pink add-to-cart button...')
        clean_tile = remove_pink_button(tile)

        # Convert to PIL Image for rembg
        print('   🎨 Applying background removal...')
        try:
            # Convert OpenCV image to PIL
            if clean_tile.shape[2] == 4:  # BGRA
                clean_tile_rgb = cv2.cvtColor(clean_tile, cv2.COLOR_BGRA2RGBA)
            else:  # BGR
                clean_tile_rgb = cv2.cvtColor(clean_tile, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(clean_tile_rgb)

            # Apply background removal using rembg
            bg_removed_data = remove(pil_image)

            # Save background-removed image
            bg_removed_filename = f'IMG_7805_product_{i}_no_bg.png'
            bg_removed_path = os.path.join(bg_removed_dir, bg_removed_filename)
            bg_removed_data.save(bg_removed_path)

            print(f'   ✅ Background removed: {bg_removed_path}')

            # Calculate costs using the same logic as before
            def calculate_costs(description, price_str=''):
                # Use estimated prices for demo
                prices = {'1,49 €': 1.49, '2,79 €': 2.79, '4,29 €': 4.29, '3,99 €': 3.99}

                # Determine price based on description
                if 'Bananen 5 Stk' in description and 'Chiquita' not in description:
                    price = 1.49
                    price_str = '1,49 €'
                elif 'Chiquita' in description:
                    price = 2.79
                    price_str = '2,79 €'
                elif 'Gala' in description:
                    price = 4.29
                    price_str = '4,29 €'
                elif 'Pink Lady' in description:
                    price = 3.99
                    price_str = '3,99 €'
                else:
                    return '', '', '', ''

                # Smart detection logic
                desc_lower = description.lower()
                is_weight_based = any(indicator in desc_lower for indicator in ['kg', 'g ', 'gram'])
                is_piece_based = any(indicator in desc_lower for indicator in ['stk', 'stück'])

                cost_per_kg = ''
                cost_per_piece = ''

                if is_weight_based and not is_piece_based:
                    # Weight-based product
                    cost_per_kg = f'{price:.2f} €/kg'
                elif is_piece_based and not is_weight_based:
                    # Piece-based product
                    quantity = 5 if '5' in description else 1
                    cost_per_piece = f'{price / quantity:.2f} €/Stk'

                return price_str, cost_per_kg, cost_per_piece

            price_str, cost_per_kg, cost_per_piece = calculate_costs(tile_info['desc'])

            # Create CSV entry
            enhanced_entry = {
                'A_FileName': 'IMG_7805.PNG',
                'B_ProductImage_Filename': original_filename,
                'C_Product_Description': tile_info['desc'],
                'D_Product_Price': price_str,
                'E_Product_Brand': 'Chiquita' if 'Chiquita' in tile_info['desc'] else 'Pink Lady' if 'Pink Lady' in tile_info['desc'] else '',
                'F_Product_Weight': '1kg' if 'kg' in tile_info['desc'] else '',
                'G_Product_Quantity': '5 Stück' if '5 Stk' in tile_info['desc'] else '1',
                'H_Cost_per_Kg': cost_per_kg,
                'I_Cost_per_Piece': cost_per_piece,
                'J_Product_Category': 'Obst',
                'K_Product_SubCategory': 'Bananen' if 'Bananen' in tile_info['desc'] else 'Äpfel',
                'L_BackgroundRemoved_ImagePath': bg_removed_path
            }
            enhanced_data.append(enhanced_entry)

        except Exception as e:
            print(f'   ❌ Background removal failed: {e}')

    print('\n🔄 STEP 2: Generating enhanced CSV...')

    if enhanced_data:
        df = pd.DataFrame(enhanced_data)

        # Save enhanced CSV
        csv_path = 'IMG_7805_WITH_BACKGROUND_REMOVAL.csv'
        df.to_csv(csv_path, index=False)

        print('\n🎯 BACKGROUND REMOVAL COMPLETE!')
        print('=' * 60)
        print(f'📁 Enhanced CSV: {csv_path}')
        print(f'📁 Original tiles: {original_tiles_dir}/')
        print(f'📁 Background-removed images: {bg_removed_dir}/')
        print()
        print('📄 ENHANCED CSV CONTENT:')
        print('=' * 40)
        print(df.to_string(index=False))

        print('\n🎉 SUCCESS SUMMARY:')
        print('=' * 30)
        print('✅ Product tiles extracted with pink button removal')
        print('✅ Background removal applied using rembg')
        print('✅ Mutually exclusive cost calculations preserved')
        print('✅ CSV updated with background-removed image paths')

        return {
            'csv_path': csv_path,
            'original_tiles_dir': original_tiles_dir,
            'bg_removed_dir': bg_removed_dir,
            'products_processed': len(enhanced_data)
        }
    else:
        print('❌ No products processed successfully')
        return None

if __name__ == "__main__":
    result = extract_and_remove_backgrounds()
    if result:
        print(f'\n📂 Open the background-removed images folder:')
        print(f'   {result["bg_removed_dir"]}')