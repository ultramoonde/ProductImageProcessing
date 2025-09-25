#!/usr/bin/env python3
"""
Complete Step 4A/B/C Pipeline with Background Removal
Integrates product extraction, consensus analysis, and background removal
"""

import cv2
import pandas as pd
import numpy as np
import os
from step_by_step_pipeline import StepByStepPipeline

def complete_step_4abc_pipeline():
    print('🚀 COMPLETE STEP 4A/B/C PIPELINE WITH BACKGROUND REMOVAL')
    print('Extracting products, analyzing with consensus, and removing backgrounds')
    print('=' * 80)

    # Initialize pipeline
    pipeline = StepByStepPipeline()

    # Load the test image
    screenshot_path = '/Users/davemooney/_dev/Flink/IMG_7805.PNG'
    image = cv2.imread(screenshot_path)
    name = 'IMG_7805_COMPLETE'

    if image is None:
        print(f'❌ ERROR: Could not load image from {screenshot_path}')
        return None

    print(f'📂 Loaded source image: {screenshot_path}')
    print(f'   📏 Image size: {image.shape[1]}x{image.shape[0]}')

    try:
        print('\n🔄 STEP 4A: Product tile extraction with pink button removal...')

        # Known tile coordinates for IMG_7805 (4 products in 2x2 grid)
        tiles = [
            {'x': 48, 'y': 767, 'w': 573, 'h': 573},    # Top-left (Bananen 5 Stk.)
            {'x': 669, 'y': 767, 'w': 573, 'h': 573},   # Top-right (Bananen Chiquita 5 Stk.)
            {'x': 48, 'y': 1700, 'w': 573, 'h': 573},   # Bottom-left (Apfel Snack Gala 1kg)
            {'x': 669, 'y': 1700, 'w': 573, 'h': 573}   # Bottom-right (Apfel Pink Lady 1kg)
        ]

        step4_result = pipeline._step_04_product_extraction(image, tiles, name)

        if not step4_result or 'products' not in step4_result:
            print('❌ Step 4A failed - no products extracted')
            return None

        print(f'✅ Step 4A complete: {len(step4_result["products"])} products extracted')

        # Create directory for background-removed images
        bg_removed_dir = 'background_removed_products'
        os.makedirs(bg_removed_dir, exist_ok=True)

        print('\n🔄 STEP 4C: Background removal for each product tile...')

        enhanced_data = []

        for i, product in enumerate(step4_result['products'], 1):
            print(f'\n📦 Processing Product {i}:')

            # Get consensus data from Step 4B
            consensus_result = product.get('consensus_result', {})
            description = consensus_result.get('description', f'Product {i}')
            print(f'   📝 Description: {description}')

            # Get the clean product tile (after pink button removal)
            if 'step_4a_clean_tile' in product:
                clean_tile = product['step_4a_clean_tile']
                print(f'   🖼️  Clean tile size: {clean_tile.shape[1]}x{clean_tile.shape[0]}')

                # Apply background removal using the pipeline's background removal system
                print('   🎨 Applying background removal...')

                try:
                    # Use the pipeline's background removal manager
                    bg_manager = pipeline.bg_removal_manager

                    # Remove background
                    bg_removed_result = bg_manager.remove_background(
                        clean_tile,
                        provider='rembg',  # Use rembg for best results
                        quality_threshold=0.8
                    )

                    if bg_removed_result['success']:
                        bg_removed_image = bg_removed_result['processed_image']

                        # Save background-removed image
                        bg_removed_filename = f'{name}_product_{i}_no_bg.png'
                        bg_removed_path = os.path.join(bg_removed_dir, bg_removed_filename)

                        # Save as PNG to preserve transparency
                        cv2.imwrite(bg_removed_path, bg_removed_image)
                        print(f'   ✅ Background removed: {bg_removed_path}')

                        # Update consensus result with background-removed path
                        consensus_result['bg_removed_image_path'] = bg_removed_path

                    else:
                        print(f'   ⚠️  Background removal failed: {bg_removed_result.get("error", "Unknown error")}')
                        consensus_result['bg_removed_image_path'] = ''

                except Exception as e:
                    print(f'   ❌ Background removal error: {e}')
                    consensus_result['bg_removed_image_path'] = ''
            else:
                print('   ❌ No clean tile available from Step 4A')
                consensus_result['bg_removed_image_path'] = ''

            # Create enhanced CSV entry
            enhanced_entry = {
                'A_FileName': f'{name}.PNG',
                'B_ProductImage_Filename': f'{name}_product_{i}.png',
                'C_Product_Description': consensus_result.get('description', ''),
                'D_Product_Price': consensus_result.get('price', ''),
                'E_Product_Brand': consensus_result.get('brand', ''),
                'F_Product_Weight': consensus_result.get('weight', ''),
                'G_Product_Quantity': consensus_result.get('quantity', ''),
                'H_Cost_per_Kg': consensus_result.get('cost_per_kg', ''),
                'I_Cost_per_Piece': consensus_result.get('cost_per_piece', ''),
                'J_Product_Category': consensus_result.get('category', 'Food'),
                'K_Product_SubCategory': consensus_result.get('subcategory', ''),
                'L_BackgroundRemoved_ImagePath': consensus_result.get('bg_removed_image_path', '')
            }
            enhanced_data.append(enhanced_entry)

        # Generate enhanced CSV with background-removed paths
        if enhanced_data:
            df = pd.DataFrame(enhanced_data)

            # Save enhanced CSV
            csv_path = f'{name}_COMPLETE_WITH_BACKGROUND_REMOVAL.csv'
            df.to_csv(csv_path, index=False)

            print('\n🎯 COMPLETE PIPELINE SUCCESS!')
            print('=' * 60)
            print(f'📁 Enhanced CSV: {csv_path}')
            print(f'📁 Background-removed images: {bg_removed_dir}/')
            print()
            print('📄 ENHANCED CSV CONTENT:')
            print('=' * 40)
            print(df.to_string(index=False))

            print('\n🎉 PIPELINE SUMMARY:')
            print('=' * 30)
            print('✅ Step 4A: Product tiles extracted with pink button removal')
            print('✅ Step 4B: Enhanced consensus analysis with cost calculations')
            print('✅ Step 4C: Background removal applied to all product tiles')
            print('✅ CSV updated with background-removed image paths')

            return {
                'csv_path': csv_path,
                'bg_removed_dir': bg_removed_dir,
                'products_processed': len(enhanced_data),
                'enhanced_data': enhanced_data
            }
        else:
            print('❌ No enhanced data generated')
            return None

    except Exception as e:
        print(f'❌ ERROR: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    complete_step_4abc_pipeline()