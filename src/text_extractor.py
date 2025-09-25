import easyocr
import cv2
import numpy as np
import re
from typing import Dict, List, Optional, Tuple

class TextExtractor:
    def __init__(self, languages: List[str] = ['en', 'de']):
        """
        Initialize EasyOCR reader with specified languages.
        Default includes English and German for European food delivery apps.
        """
        self.reader = easyocr.Reader(languages, gpu=False)
    
    def extract_text_from_tile(self, tile: np.ndarray) -> Dict[str, str]:
        """
        Extract product information from a single tile.
        Returns dict with product_name, price, and price_per_unit.
        """
        # Preprocess image for better OCR
        processed_tile = self._preprocess_for_ocr(tile)
        
        # Extract all text from the tile
        results = self.reader.readtext(processed_tile)
        
        # Parse the extracted text - enhanced with more fields
        extracted_data = {
            'product_name': '',
            'price': '',
            'price_per_unit': '',
            'manufacturer': '',
            'weight': '',
            'cost_per_unit': '',
            'discount': '',
            'additional_info': ''
        }
        
        if not results:
            return extracted_data
        
        # Extract text and confidence scores
        texts = [(text, confidence, bbox) for bbox, text, confidence in results if confidence > 0.3]
        
        # Sort by vertical position (top to bottom)
        texts.sort(key=lambda x: x[2][0][1])  # Sort by top-left y coordinate
        
        # Parse extracted text
        extracted_data = self._parse_product_text(texts)
        
        return extracted_data
    
    def extract_raw_text_from_tile(self, tile: np.ndarray) -> str:
        """
        Extract raw text from a tile as a simple string.
        Used for basic OCR processing and debugging.
        """
        # Preprocess image for better OCR
        processed_tile = self._preprocess_for_ocr(tile)
        
        # Extract all text from the tile
        results = self.reader.readtext(processed_tile)
        
        if not results:
            return ""
        
        # Extract just the text with confidence > 0.3
        texts = [text for bbox, text, confidence in results if confidence > 0.3]
        
        # Join all text with spaces
        return ' '.join(texts).strip()
    
    def extract_category_from_header(self, header_image: np.ndarray) -> Tuple[str, str]:
        """
        Extract category and subcategory from header region.
        Returns (category, subcategory) tuple.
        """
        # Preprocess header for OCR
        processed_header = self._preprocess_for_ocr(header_image)
        
        # Extract text from header
        results = self.reader.readtext(processed_header)
        
        category = ""
        subcategory = ""
        
        if results:
            # Get all text with high confidence
            header_texts = [text for _, text, confidence in results if confidence > 0.5]
            
            if header_texts:
                # Assume the most prominent text is the category/subcategory
                # This might need adjustment based on actual app layout
                if len(header_texts) >= 2:
                    category = header_texts[0].strip()
                    subcategory = header_texts[1].strip()
                elif len(header_texts) == 1:
                    category = header_texts[0].strip()
                    subcategory = ""
        
        return category, subcategory
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for OCR.
        """
        # Handle RGBA images
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Convert RGBA to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Increase image size for better OCR (upscale by 2x)
        height, width = gray.shape
        upscaled = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(upscaled)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply slight sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def _parse_product_text(self, texts: List[Tuple[str, float, List]]) -> Dict[str, str]:
        """
        Parse extracted text to identify comprehensive product information.
        """
        extracted_data = {
            'product_name': '',
            'price': '',
            'price_per_unit': '',
            'manufacturer': '',
            'weight': '',
            'cost_per_unit': '',
            'discount': '',
            'additional_info': ''
        }
        
        if not texts:
            return extracted_data
        
        all_text = ' '.join([text for text, _, _ in texts])
        
        # Extract prices (€X.XX format)
        price_pattern = r'€\s*(\d+[.,]\d{2})'
        prices = re.findall(price_pattern, all_text)
        
        # Extract price per unit (€X.XX/kg, €X.XX/100g, etc.)
        price_per_unit_pattern = r'(€\s*\d+[.,]\d{2}\s*/\s*(?:kg|g|100g|l|ml|stück|piece|pc|Stk))'
        price_per_unit_matches = re.findall(price_per_unit_pattern, all_text, re.IGNORECASE)
        
        # Extract discounts (-XX%)
        discount_pattern = r'(-\d+%)'
        discount_matches = re.findall(discount_pattern, all_text)
        
        # Extract weights/quantities (XXXg, XXXml, XXkg, etc.)
        weight_pattern = r'(\d+(?:[.,]\d+)?\s*(?:kg|g|ml|l|stück|piece|pc|Stk))'
        weight_matches = re.findall(weight_pattern, all_text, re.IGNORECASE)
        
        # Known manufacturers/brands (add more as needed)
        known_brands = [
            'REWE', 'YoPRO', 'HEIMATGUT', 'Vivera', 'Oatly', 'Feine Welt',
            'ja!', 'EDEKA', 'Netto', 'Lidl', 'Aldi', 'Danone', 'Nestlé', 
            'Unilever', 'Coca-Cola', 'Pepsi', 'Dr. Oetker', 'Knorr',
            'Bio', 'Organic', 'GRAN RESERVA', 'BLACK ANGUS'
        ]
        
        # Extract manufacturer
        manufacturer = ''
        for brand in known_brands:
            if brand.upper() in all_text.upper():
                manufacturer = brand
                break
        
        # Assign extracted data
        if prices:
            extracted_data['price'] = f"€{prices[0]}"
        
        if price_per_unit_matches:
            extracted_data['price_per_unit'] = price_per_unit_matches[0]
            # Also extract cost per unit value
            cost_match = re.search(r'€\s*(\d+[.,]\d{2})', price_per_unit_matches[0])
            if cost_match:
                extracted_data['cost_per_unit'] = f"€{cost_match.group(1)}"
        
        if discount_matches:
            extracted_data['discount'] = discount_matches[0]
        
        if weight_matches:
            # Take the most likely weight (usually the largest number)
            weights = []
            for weight in weight_matches:
                # Extract numeric part
                numeric = re.search(r'(\d+(?:[.,]\d+)?)', weight)
                if numeric:
                    weights.append((float(numeric.group(1).replace(',', '.')), weight))
            if weights:
                weights.sort(key=lambda x: x[0], reverse=True)
                extracted_data['weight'] = weights[0][1]
        
        if manufacturer:
            extracted_data['manufacturer'] = manufacturer
        
        # Extract product name (clean up text)
        text_for_name = all_text
        
        # Remove prices, discounts, weights
        patterns_to_remove = [
            r'€\s*\d+[.,]\d{2}(?:\s*/\s*(?:kg|g|100g|l|ml|stück|piece|pc|Stk))?',
            r'-\d+%',
            r'\d+(?:[.,]\d+)?\s*(?:kg|g|ml|l|stück|piece|pc|Stk)',
            r'[0-9]+[.,][0-9]+\s*€',
            r'\d+\s*€'
        ]
        
        for pattern in patterns_to_remove:
            text_for_name = re.sub(pattern, '', text_for_name, flags=re.IGNORECASE)
        
        # Clean up product name
        product_name = text_for_name.strip()
        product_name = re.sub(r'\s+', ' ', product_name)  # Remove extra whitespace
        product_name = re.sub(r'^[^\w]*|[^\w]*$', '', product_name)  # Remove leading/trailing non-word chars
        
        if product_name and len(product_name) > 3:
            extracted_data['product_name'] = product_name[:100]  # Limit length
        
        # Additional info (everything else)
        additional_info_parts = []
        if prices and len(prices) > 1:
            additional_info_parts.extend([f"€{p}" for p in prices[1:]])
        
        extracted_data['additional_info'] = ' '.join(additional_info_parts)
        
        return extracted_data
    
    def validate_extracted_data(self, data: Dict[str, str]) -> bool:
        """
        Validate that extracted data contains reasonable information.
        """
        # Check if we have at least a product name
        if not data.get('product_name', '').strip():
            return False
        
        # Check price format if present
        price = data.get('price', '')
        if price and not re.match(r'€\d+[.,]\d{2}', price):
            return False
        
        return True
    
    def debug_ocr_results(self, image: np.ndarray, output_path: str = None) -> List[Tuple[str, float]]:
        """
        Debug OCR results by showing all detected text and confidence scores.
        """
        processed = self._preprocess_for_ocr(image)
        results = self.reader.readtext(processed)
        
        debug_info = []
        for bbox, text, confidence in results:
            debug_info.append((text, confidence))
        
        if output_path:
            # Save debug image with text boxes
            debug_image = image.copy()
            if len(debug_image.shape) == 3 and debug_image.shape[2] == 4:
                debug_image = cv2.cvtColor(debug_image, cv2.COLOR_RGBA2BGR)
            
            for bbox, text, confidence in results:
                if confidence > 0.3:
                    pts = np.array(bbox, dtype=np.int32)
                    cv2.polylines(debug_image, [pts], True, (0, 255, 0), 2)
                    cv2.putText(debug_image, f"{text} ({confidence:.2f})", 
                              tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imwrite(output_path, debug_image)
        
        return debug_info