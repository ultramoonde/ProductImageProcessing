#!/usr/bin/env python3
"""
Test Fixed Cost Calculation Logic
Demonstrates the mutually exclusive cost per kg/piece calculation
"""

import pandas as pd

def test_fixed_cost_calculation():
    print('🧪 TESTING FIXED COST CALCULATION LOGIC')
    print('Demonstrating mutually exclusive cost per kg/piece calculation')
    print('=' * 70)

    # Test data with smart detection
    test_products = [
        {
            'description': 'Bananen 5 Stk.',
            'price': '1,49 €',
            'indicators': 'Contains "Stk." -> piece-based product',
            'expected': 'Only cost per piece should be calculated'
        },
        {
            'description': 'Bananen Chiquita 5 Stk.',
            'price': '2,79 €',
            'indicators': 'Contains "Stk." -> piece-based product',
            'expected': 'Only cost per piece should be calculated'
        },
        {
            'description': 'Apfel Snack Gala 1kg',
            'price': '4,29 €',
            'indicators': 'Contains "kg" -> weight-based product',
            'expected': 'Only cost per kg should be calculated'
        },
        {
            'description': 'Apfel Pink Lady 1kg',
            'price': '3,99 €',
            'indicators': 'Contains "kg" -> weight-based product',
            'expected': 'Only cost per kg should be calculated'
        }
    ]

    def smart_cost_calculation(description, price):
        """
        Smart cost calculation with mutually exclusive logic
        """
        # Clean price
        price_clean = price.replace('€', '').replace(',', '.').strip()
        try:
            price_float = float(price_clean)
        except:
            return '', ''

        # Detection logic
        desc_lower = description.lower()

        # Weight indicators (for cost per kg)
        weight_indicators = ['kg', 'g ', 'gram', 'ml', 'liter', 'l ']
        is_weight_based = any(indicator in desc_lower for indicator in weight_indicators)

        # Piece indicators (for cost per piece)
        piece_indicators = ['stk', 'stück', ' x ', 'pieces', 'pack']
        is_piece_based = any(indicator in desc_lower for indicator in piece_indicators)

        # Extract quantities
        import re

        cost_per_kg = ''
        cost_per_piece = ''

        if is_weight_based and not is_piece_based:
            # WEIGHT-BASED PRODUCT: Only calculate cost per kg
            weight_match = re.search(r'(\d+(?:\.\d+)?)\s*(kg|g)', desc_lower)
            if weight_match:
                weight_value = float(weight_match.group(1))
                weight_unit = weight_match.group(2)

                # Convert to kg if needed
                if weight_unit == 'g':
                    weight_value = weight_value / 1000

                cost_per_kg = f'{price_float / weight_value:.2f} €/kg'

        elif is_piece_based and not is_weight_based:
            # PIECE-BASED PRODUCT: Only calculate cost per piece
            piece_match = re.search(r'(\d+)\s*stk', desc_lower)
            if piece_match:
                piece_count = int(piece_match.group(1))
                cost_per_piece = f'{price_float / piece_count:.2f} €/Stk'

        return cost_per_kg, cost_per_piece

    print('🔬 TESTING EACH PRODUCT:')
    print('=' * 40)

    results = []
    for i, product in enumerate(test_products, 1):
        print(f'\n📦 Product {i}: {product["description"]}')
        print(f'   💰 Price: {product["price"]}')
        print(f'   🔍 Detection: {product["indicators"]}')
        print(f'   ✅ Expected: {product["expected"]}')

        cost_per_kg, cost_per_piece = smart_cost_calculation(
            product['description'],
            product['price']
        )

        print(f'   📊 Result:')
        print(f'      - Cost per kg: "{cost_per_kg}" {"✅" if cost_per_kg else "❌ (empty)"}')
        print(f'      - Cost per piece: "{cost_per_piece}" {"✅" if cost_per_piece else "❌ (empty)"}')

        # Verify mutually exclusive
        exclusive = bool(cost_per_kg) != bool(cost_per_piece)  # XOR logic
        print(f'   🎯 Mutually exclusive: {"✅ YES" if exclusive else "❌ NO - BOTH OR NEITHER"}')

        results.append({
            'A_FileName': 'IMG_7805.PNG',
            'B_ProductImage_Filename': f'IMG_7805_product_{i}.png',
            'C_Product_Description': product['description'],
            'D_Product_Price': product['price'],
            'E_Product_Brand': 'Chiquita' if 'Chiquita' in product['description'] else 'Pink Lady' if 'Pink Lady' in product['description'] else '',
            'F_Product_Weight': '1kg' if 'kg' in product['description'] else '',
            'G_Product_Quantity': '5 Stück' if 'Stk.' in product['description'] else '1',
            'H_Cost_per_Kg': cost_per_kg,
            'I_Cost_per_Piece': cost_per_piece,
            'J_Product_Category': 'Obst',
            'K_Product_SubCategory': 'Bananen' if 'Bananen' in product['description'] else 'Äpfel'
        })

    # Create fixed CSV
    df = pd.DataFrame(results)
    csv_path = 'IMG_7805_FIXED_MUTUALLY_EXCLUSIVE.csv'
    df.to_csv(csv_path, index=False)

    print('\n🎯 FIXED CSV GENERATED!')
    print('=' * 50)
    print(f'📁 File: {csv_path}')
    print('\n📄 FIXED CSV CONTENT:')
    print('=' * 30)
    print(df.to_string(index=False))

    print('\n🎉 VERIFICATION SUMMARY:')
    print('=' * 40)
    for i, row in df.iterrows():
        product_num = i + 1
        has_kg = bool(row['H_Cost_per_Kg'])
        has_piece = bool(row['I_Cost_per_Piece'])
        is_exclusive = has_kg != has_piece  # XOR

        print(f'Product {product_num}: {"✅ CORRECT" if is_exclusive else "❌ ERROR"} - '
              f'Cost per kg: {"Yes" if has_kg else "No"}, '
              f'Cost per piece: {"Yes" if has_piece else "No"}')

    return csv_path

if __name__ == "__main__":
    test_fixed_cost_calculation()