#!/usr/bin/env python3
"""
Batch Process 4 Images from Flink Folder
Processes multiple images through complete pipeline and creates consolidated summary
"""

import cv2
import pandas as pd
import numpy as np
import os
from datetime import datetime
from rembg import remove
from PIL import Image
import io
import glob

def detect_product_grid(image):
    """
    Detect product grid layout in the image
    Returns list of tile coordinates
    """
    height, width = image.shape[:2]

    # Standard 2x2 grid for grocery app screenshots (based on IMG_7805)
    # These are approximate positions that should work for most Flink screenshots

    # Calculate dynamic positions based on image size
    margin_x = int(width * 0.037)  # ~48px for 1290px width
    margin_y = int(height * 0.274)  # ~767px for 2796px height

    tile_width = int(width * 0.444)   # ~573px for 1290px width
    tile_height = int(height * 0.205) # ~573px for 2796px height

    gap_x = int(width * 0.479)  # ~621px spacing (48+573)
    gap_y = int(height * 0.608) # ~1700px for second row

    tiles = []

    # 2x2 grid
    for row in range(2):
        for col in range(2):
            x = margin_x + (col * gap_x)
            y = margin_y + (row * (gap_y - margin_y))

            # Ensure coordinates are within image bounds
            if x + tile_width <= width and y + tile_height <= height:
                tiles.append({
                    'x': x,
                    'y': y,
                    'w': tile_width,
                    'h': tile_height
                })

    return tiles

def remove_pink_button(tile_image):
    """Remove pink add-to-cart button from tile"""
    # Convert to different color spaces for better detection
    hsv = cv2.cvtColor(tile_image, cv2.COLOR_BGR2HSV)

    # Define pink color range
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
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area threshold
            cv2.fillPoly(output, [contour], (0, 0, 0, 0))

    return output

def calculate_costs(description, price_str):
    """Calculate costs using mutually exclusive logic"""
    if not price_str:
        return '', ''

    # Clean price
    price_clean = price_str.replace('€', '').replace(',', '.').strip()
    try:
        price_float = float(price_clean)
    except:
        return '', ''

    # Smart detection logic
    desc_lower = description.lower()
    is_weight_based = any(indicator in desc_lower for indicator in ['kg', 'g ', 'gram'])
    is_piece_based = any(indicator in desc_lower for indicator in ['stk', 'stück', ' x ', 'pieces'])

    cost_per_kg = ''
    cost_per_piece = ''

    if is_weight_based and not is_piece_based:
        # Weight-based product
        import re
        weight_match = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g)', desc_lower)
        if weight_match:
            weight_value = float(weight_match.group(1))
            weight_unit = weight_match.group(2)

            if weight_unit == 'g':
                weight_value = weight_value / 1000

            cost_per_kg = f'{price_float / weight_value:.2f} €/kg'

    elif is_piece_based and not is_weight_based:
        # Piece-based product
        import re
        piece_match = re.search(r'(\d+)\s*stk', desc_lower)
        if piece_match:
            piece_count = int(piece_match.group(1))
            cost_per_piece = f'{price_float / piece_count:.2f} €/Stk'
        else:
            # Default to 1 piece if no count specified
            cost_per_piece = f'{price_float:.2f} €/Stk'

    return cost_per_kg, cost_per_piece

def extract_product_info_with_consensus(image, tile_coords, image_name):
    """
    Extract product information using the real consensus system (Step 4B)
    This integrates with the LocalConsensusAnalyzer for proper product analysis
    """
    try:
        # Import the consensus analyzer
        import sys
        import os
        sys.path.append('src')
        from local_consensus_analyzer import LocalConsensusAnalyzer

        # Initialize consensus analyzer
        analyzer = LocalConsensusAnalyzer()

        print(f'   🧠 Using 3-model consensus system for product analysis...')

        products = []

        for i, tile_info in enumerate(tile_coords, 1):
            print(f'   📦 Analyzing tile {i} with consensus system...')

            # Extract tile image
            x, y, w, h = tile_info['x'], tile_info['y'], tile_info['w'], tile_info['h']
            tile_image = image[y:y+h, x:x+w].copy()

            # Save temporary tile for analysis
            temp_tile_path = f'temp_tile_{image_name}_{i}.png'
            cv2.imwrite(temp_tile_path, tile_image)

            try:
                # Run consensus analysis on this tile
                consensus_result = analyzer.analyze_product_tile(temp_tile_path)

                print(f'      ✅ Consensus result: {consensus_result.get("description", "Unknown")}')

                # Extract consensus data
                product_data = {
                    "desc": consensus_result.get("description", f"Product {i}"),
                    "price": consensus_result.get("price", "0,00 €"),
                    "brand": consensus_result.get("brand", ""),
                    "category": consensus_result.get("category", "Food"),
                    "subcategory": consensus_result.get("subcategory", ""),
                    "weight": consensus_result.get("weight", ""),
                    "quantity": consensus_result.get("quantity", ""),
                    "cost_per_kg": consensus_result.get("cost_per_kg", ""),
                    "cost_per_piece": consensus_result.get("cost_per_piece", "")
                }

                products.append(product_data)

                # Clean up temp file
                if os.path.exists(temp_tile_path):
                    os.remove(temp_tile_path)

            except Exception as tile_error:
                print(f'      ❌ Consensus analysis failed for tile {i}: {tile_error}')
                # Fallback to basic data
                products.append({
                    "desc": f"Product {i}",
                    "price": "0,00 €",
                    "brand": "",
                    "category": "Food",
                    "subcategory": "",
                    "weight": "",
                    "quantity": "",
                    "cost_per_kg": "",
                    "cost_per_piece": ""
                })

        return products

    except Exception as e:
        print(f'   ❌ Consensus system unavailable ({e}), using fallback...')
        # Fallback to basic extraction
        return [
            {"desc": f"Product {i}", "price": "0,00 €", "brand": "", "category": "Food", "subcategory": "", "weight": "", "quantity": "", "cost_per_kg": "", "cost_per_piece": ""}
            for i in range(1, len(tile_coords) + 1)
        ]

def process_single_image(image_path, output_base_dir):
    """Process a single image through the complete pipeline"""
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f'\n🖼️  PROCESSING: {image_name}')
    print('=' * 50)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f'❌ Could not load {image_path}')
        return None

    print(f'   📏 Image size: {image.shape[1]}x{image.shape[0]}')

    # Create output directories for this image
    image_output_dir = os.path.join(output_base_dir, image_name)
    original_tiles_dir = os.path.join(image_output_dir, 'original_tiles')
    bg_removed_dir = os.path.join(image_output_dir, 'background_removed')

    os.makedirs(original_tiles_dir, exist_ok=True)
    os.makedirs(bg_removed_dir, exist_ok=True)

    # Detect product grid
    tiles = detect_product_grid(image)
    print(f'   🎯 Detected {len(tiles)} product tiles')

    if not tiles:
        print('   ❌ No product tiles detected')
        return None

    # Extract product info using real consensus system (Step 4B)
    base_products = extract_product_info_with_consensus(image, tiles, image_name)

    enhanced_data = []

    for i, (tile_info, product_info) in enumerate(zip(tiles, base_products), 1):
        print(f'   📦 Processing tile {i}...')

        # Extract tile
        x, y, w, h = tile_info['x'], tile_info['y'], tile_info['w'], tile_info['h']
        tile = image[y:y+h, x:x+w].copy()

        # Save original tile
        original_filename = f'{image_name}_product_{i}_original.png'
        original_path = os.path.join(original_tiles_dir, original_filename)
        cv2.imwrite(original_path, tile)

        # Remove pink button
        clean_tile = remove_pink_button(tile)

        # Apply background removal
        try:
            # Convert to PIL format
            if clean_tile.shape[2] == 4:  # BGRA
                clean_tile_rgb = cv2.cvtColor(clean_tile, cv2.COLOR_BGRA2RGBA)
            else:  # BGR
                clean_tile_rgb = cv2.cvtColor(clean_tile, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(clean_tile_rgb)
            bg_removed_data = remove(pil_image)

            # Save background-removed image
            bg_removed_filename = f'{image_name}_product_{i}_no_bg.png'
            bg_removed_path = os.path.join(bg_removed_dir, bg_removed_filename)
            bg_removed_data.save(bg_removed_path)

            print(f'      ✅ Background removed: {bg_removed_filename}')

        except Exception as e:
            print(f'      ❌ Background removal failed: {e}')
            bg_removed_path = ''

        # Create CSV entry with consensus data
        enhanced_entry = {
            'A_FileName': f'{image_name}.PNG',
            'B_ProductImage_Filename': original_filename,
            'C_Product_Description': product_info['desc'],
            'D_Product_Price': product_info['price'],
            'E_Product_Brand': product_info['brand'],
            'F_Product_Weight': product_info.get('weight', ''),
            'G_Product_Quantity': product_info.get('quantity', ''),
            'H_Cost_per_Kg': product_info.get('cost_per_kg', ''),
            'I_Cost_per_Piece': product_info.get('cost_per_piece', ''),
            'J_Product_Category': product_info['category'],
            'K_Product_SubCategory': product_info['subcategory'],
            'L_BackgroundRemoved_ImagePath': bg_removed_path
        }
        enhanced_data.append(enhanced_entry)

    # Generate individual CSV for this image
    if enhanced_data:
        df = pd.DataFrame(enhanced_data)
        csv_path = os.path.join(image_output_dir, f'{image_name}_products.csv')
        df.to_csv(csv_path, index=False)
        print(f'   📄 Individual CSV: {csv_path}')

        return {
            'image_name': image_name,
            'csv_path': csv_path,
            'output_dir': image_output_dir,
            'products_count': len(enhanced_data),
            'enhanced_data': enhanced_data
        }

    return None

def batch_process_images():
    """Process 4 selected images and create consolidated summary"""
    print('🚀 BATCH PROCESSING 4 FLINK IMAGES')
    print('Complete pipeline: extraction + consensus + background removal')
    print('=' * 80)

    # Selected images
    selected_images = [
        '/Users/davemooney/_dev/Flink/IMG_8138.PNG',
        '/Users/davemooney/_dev/Flink/IMG_8104.PNG',
        '/Users/davemooney/_dev/Flink/IMG_8110.PNG',
        '/Users/davemooney/_dev/Flink/IMG_8448.PNG'
    ]

    # Create main output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = f'batch_processing_results_{timestamp}'
    os.makedirs(output_base_dir, exist_ok=True)

    print(f'📁 Output directory: {output_base_dir}')

    # Process each image
    all_results = []
    consolidated_data = []

    for i, image_path in enumerate(selected_images, 1):
        print(f'\n🔄 PROCESSING IMAGE {i}/4')

        if not os.path.exists(image_path):
            print(f'❌ Image not found: {image_path}')
            continue

        result = process_single_image(image_path, output_base_dir)
        if result:
            all_results.append(result)
            consolidated_data.extend(result['enhanced_data'])

    print(f'\n🔄 CREATING CONSOLIDATED SUMMARY...')

    # Create consolidated summary CSV
    if consolidated_data:
        consolidated_df = pd.DataFrame(consolidated_data)

        # Add additional summary columns
        consolidated_df['M_ProcessingTimestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        consolidated_df['N_SourceImagePath'] = consolidated_df['A_FileName'].apply(
            lambda x: next((img for img in selected_images if os.path.basename(img) == x), '')
        )

        # Save consolidated CSV
        consolidated_csv_path = os.path.join(output_base_dir, 'CONSOLIDATED_ALL_PRODUCTS.csv')
        consolidated_df.to_csv(consolidated_csv_path, index=False)

        # Create summary statistics
        summary_stats = {
            'total_images_processed': len(all_results),
            'total_products_extracted': len(consolidated_data),
            'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'output_directory': output_base_dir,
            'individual_csvs': [r['csv_path'] for r in all_results],
            'consolidated_csv': consolidated_csv_path
        }

        # Save summary stats
        summary_df = pd.DataFrame([summary_stats])
        summary_stats_path = os.path.join(output_base_dir, 'PROCESSING_SUMMARY.csv')
        summary_df.to_csv(summary_stats_path, index=False)

        print('\n🎯 BATCH PROCESSING COMPLETE!')
        print('=' * 70)
        print(f'📊 RESULTS SUMMARY:')
        print(f'   • Images processed: {len(all_results)}')
        print(f'   • Total products: {len(consolidated_data)}')
        print(f'   • Output directory: {output_base_dir}')
        print(f'   • Consolidated CSV: {consolidated_csv_path}')
        print(f'   • Summary file: {summary_stats_path}')

        # Display consolidated results
        print('\n📄 CONSOLIDATED CSV PREVIEW:')
        print('=' * 40)
        print(consolidated_df.head().to_string(index=False, max_cols=8))

        return {
            'success': True,
            'output_directory': output_base_dir,
            'consolidated_csv': consolidated_csv_path,
            'summary_stats': summary_stats,
            'results': all_results
        }

    else:
        print('❌ No products were successfully processed')
        return {'success': False}

if __name__ == "__main__":
    result = batch_process_images()

    if result['success']:
        print(f'\n📂 OPEN OUTPUT FOLDER:')
        print(f'   {result["output_directory"]}')

        # Open output directory
        os.system(f'open "{result["output_directory"]}"')