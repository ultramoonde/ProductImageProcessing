#!/usr/bin/env python3
"""
Generate Enhanced CSV with Step 4B Consensus Data
Shows the final CSV output with all missing fields populated by the enhanced consensus system
"""

import cv2
import pandas as pd
from step_by_step_pipeline import StepByStepPipeline

def generate_enhanced_csv():
    print('🚀 GENERATING ENHANCED CSV WITH STEP 4B CONSENSUS DATA')
    print('Including all missing fields: Brand, Cost per kg, Cost per piece')
    print('=' * 70)

    # Initialize pipeline
    pipeline = StepByStepPipeline()

    # Load the test image
    screenshot_path = '/Users/davemooney/_dev/Flink/IMG_7805.PNG'
    image = cv2.imread(screenshot_path)
    name = 'IMG_7805_ENHANCED'

    print('📊 Running pipeline steps 4-9 to generate enhanced CSV...')

    try:
        # Step 4: Product extraction with enhanced consensus (this is where our improvements are)
        print('🔄 Step 4: Product extraction with enhanced consensus analysis...')

        # Mock tiles from step 3 (we know IMG_7805 has 4 products)
        tiles = [
            {'x': 48, 'y': 767, 'w': 573, 'h': 573},
            {'x': 669, 'y': 767, 'w': 573, 'h': 573},
            {'x': 48, 'y': 1700, 'w': 573, 'h': 573},
            {'x': 669, 'y': 1700, 'w': 573, 'h': 573}
        ]

        step4_result = pipeline._step_04_product_extraction(image, tiles, name)

        # Skip steps 5-8 (not needed for CSV generation)
        print('🔄 Steps 5-8: Skipping intermediate steps...')

        # Step 9: CSV generation (this should use Step 4B consensus data)
        print('🔄 Step 9: Generating enhanced CSV with consensus data...')

        # Create enhanced data from Step 4B consensus results
        enhanced_data = []

        if step4_result and 'products' in step4_result:
            for i, product in enumerate(step4_result['products'], 1):
                if 'consensus_result' in product:
                    consensus = product['consensus_result']

                    # Create enhanced product entry with all fields populated
                    enhanced_entry = {
                        'filename': f'{name}.PNG',
                        'product_image': f'{name}_product_{i}.png',
                        'description': consensus.get('description', ''),
                        'price': consensus.get('price', ''),
                        'brand': consensus.get('brand', ''),
                        'weight': consensus.get('weight', ''),
                        'quantity': consensus.get('quantity', ''),
                        'cost_per_kg': consensus.get('cost_per_kg', ''),
                        'cost_per_piece': consensus.get('cost_per_piece', ''),
                        'category': consensus.get('category', 'Food'),
                        'subcategory': consensus.get('subcategory', '')
                    }
                    enhanced_data.append(enhanced_entry)

        # Generate CSV with enhanced data
        if enhanced_data:
            # Create DataFrame with all columns
            df = pd.DataFrame(enhanced_data)

            # Reorder columns to match expected format
            column_order = [
                'filename', 'product_image', 'description', 'price', 'brand',
                'weight', 'quantity', 'cost_per_kg', 'cost_per_piece',
                'category', 'subcategory'
            ]

            # Ensure all columns exist
            for col in column_order:
                if col not in df.columns:
                    df[col] = ''

            df = df[column_order]

            # Rename columns to match the original format
            df.columns = [
                'A_FileName', 'B_ProductImage_Filename', 'C_Product_Description',
                'D_Product_Price', 'E_Product_Brand', 'F_Product_Weight',
                'G_Product_Quantity', 'H_Cost_per_Kg', 'I_Cost_per_Piece',
                'J_Product_Category', 'K_Product_SubCategory'
            ]

            # Save enhanced CSV
            csv_path = f'step_by_step_demo/steps/step_09/{name}_09_enhanced_final_products.csv'
            df.to_csv(csv_path, index=False)

            print()
            print('🎯 ENHANCED CSV GENERATED SUCCESSFULLY!')
            print('=' * 60)
            print(f'📁 File: {csv_path}')
            print()
            print('📄 ENHANCED CSV CONTENT:')
            print('=' * 40)
            print(df.to_string(index=False))

            return csv_path
        else:
            print('❌ No enhanced data generated')
            return None

    except Exception as e:
        print(f'❌ ERROR: {e}')
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    generate_enhanced_csv()