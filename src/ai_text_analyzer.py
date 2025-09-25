#!/usr/bin/env python3
"""
AI-powered text analysis for food product information extraction.
Supports both OpenAI GPT and local LLM (Ollama) for intelligent text parsing.
"""

import json
import requests
import base64
from typing import Dict, Optional
from pathlib import Path
import numpy as np
import cv2
import easyocr

class AITextAnalyzer:
    def __init__(self, use_openai: bool = False, api_key: Optional[str] = None, local_model: str = "llama3.2", use_claude: bool = True):
        """
        Initialize AI text analyzer.
        
        Args:
            use_openai: If True, uses OpenAI GPT
            api_key: OpenAI API key (required if use_openai=True) 
            local_model: Local model name for Ollama (default: llama3.2)
            use_claude: If True, prioritizes Claude API for vision analysis
        """
        self.use_openai = use_openai
        self.api_key = api_key
        self.local_model = local_model
        self.use_claude = use_claude
        
        # Initialize OCR reader for text extraction
        self.reader = easyocr.Reader(['en', 'de'])
        
        # Initialize Claude API if available
        self.claude_api_key = None
        if use_claude:
            try:
                from vision_api_config import get_vision_api_key
                claude_key, model = get_vision_api_key()
                if model == "claude" and claude_key:
                    self.claude_api_key = claude_key
                    print("✅ Claude API initialized for intelligent vision analysis")
                else:
                    print("⚠️ No Claude API key found, falling back to other methods")
            except ImportError:
                print("⚠️ vision_api_config not found, Claude API unavailable")
        
        if use_openai and not api_key:
            raise ValueError("OpenAI API key required when use_openai=True")
    
    def analyze_product_tile_and_text(self, tile_image: np.ndarray, text_region_image: np.ndarray) -> Dict[str, str]:
        """
        Analyze both product tile image and text region using AI vision for comprehensive analysis.
        
        Args:
            tile_image: The product tile image (573x573px)
            text_region_image: The text region below the tile (660x240px)
            
        Returns:
            Dictionary with comprehensive product information including nutritional data
        """
        
        extracted_text = ""
        try:
            # First extract text using OCR for reference
            ocr_results = self.reader.readtext(text_region_image)
            extracted_text = ' | '.join([text for _, text, _ in ocr_results])
            
            print(f"📝 OCR extracted: {extracted_text}")
            
            # Prioritize Claude API if available
            if self.claude_api_key and self.use_claude:
                print("🧠 Using Claude API for intelligent vision analysis...")
                result = self._analyze_with_claude_vision(tile_image, text_region_image, extracted_text)
                if result and any(result.values()):
                    return result
                print("⚠️ Claude analysis failed, trying fallbacks...")
            
            # Send both tile image and text to AI Vision for comprehensive analysis
            if self.use_openai:
                return self._analyze_with_vision_openai(tile_image, text_region_image, extracted_text)
            else:
                return self._analyze_with_vision_ollama(tile_image, text_region_image, extracted_text)
                
        except Exception as e:
            print(f"⚠️ Vision analysis failed: {e}")
            # Fallback to intelligent OCR-based analysis
            if extracted_text and extracted_text.strip():
                print("🔧 Attempting intelligent OCR analysis...")
                return self._intelligent_ocr_analysis(tile_image, text_region_image, extracted_text)
            return self._empty_result()
    
    def analyze_product_text_region(self, text_region_image: np.ndarray) -> Dict[str, str]:
        """
        Legacy method - Extract text from image using OCR, then send to AI for intelligent analysis.
        Use analyze_product_tile_and_text() for enhanced vision analysis.
        """
        
        extracted_text = ""
        try:
            # First extract text using OCR
            ocr_results = self.reader.readtext(text_region_image)
            extracted_text = ' | '.join([text for _, text, _ in ocr_results])
            
            if not extracted_text.strip():
                print("⚠️ No text found in region")
                return self._empty_result()
            
            print(f"📝 OCR extracted: {extracted_text}")
            
            # Send extracted text to AI for intelligent parsing
            if self.use_openai:
                return self._analyze_text_with_openai(extracted_text)
            else:
                return self._analyze_text_with_ollama(extracted_text)
                
        except Exception as e:
            print(f"⚠️ Text region analysis failed: {e}")
            # Fallback to regex parsing
            if extracted_text and extracted_text.strip():
                print("🔧 Attempting regex parsing of OCR text...")
                return self._parse_ocr_text_regex(extracted_text)
            return self._empty_result()
    
    def _create_text_analysis_prompt(self, extracted_text: str) -> str:
        """Create the AI prompt for text-only product information extraction."""
        return f"""
You are analyzing OCR-extracted text from a German food delivery app (like Flink/Rewe) product listing.

The text typically contains:
1. Price (e.g., "2,19 €")
2. Brand + Product Name + Weight/Quantity (e.g., "Rewe To Go Obst Süsse Mango 130g")
3. Price per unit (e.g., "16,85 € / 1kg")

OCR extracted text: "{extracted_text}"

Extract the following information and return ONLY valid JSON:

{{
    "price": "€2.19",
    "brand": "Rewe", 
    "product_name": "To Go Obst Süsse Mango",
    "weight": "130g",
    "quantity": "",
    "price_per_unit": "16,85 € / 1kg"
}}

CRITICAL Rules:
- Fix OCR errors: 1309→130g, 1259→125g, IStk→1 Stk., Ikg→1kg
- BRAND extraction rules:
  * Use descriptive properties as brand when no explicit brand: "Braeburn Apfel" → brand="Braeburn", product="Apfel"
  * Apple varieties: Braeburn, Pink Lady, Granny Smith, Gala = brand names
  * Fruit varieties: Chiquita (bananas), specific cultivar names
  * If generic product with no descriptors, use "Generic" as last resort
  * Never leave brand completely empty - always extract some identifier
- Weight = mass units (g, kg, ml, l) - use for products sold by weight
- Quantity = count units (Stk, Stück, x, pieces) - use for products sold by piece
- Price per unit = price per weight OR price per quantity
- Common brands: Rewe, Bio, Chiquita, Dole, Del Monte, Edeka, Alnatura, Corny, Raw Bite
- Return ONLY the JSON object, no explanation or markdown

JSON:"""
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string."""
        # Encode image as PNG
        _, buffer = cv2.imencode('.png', image)
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def _analyze_text_with_openai(self, extracted_text: str) -> Dict[str, str]:
        """Analyze extracted text using OpenAI GPT."""
        prompt = self._create_text_analysis_prompt(extracted_text)
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        print(f"⚠️ OpenAI JSON parse error: {content}")
                        return self._empty_result()
            else:
                print(f"⚠️ OpenAI API error: {response.status_code} - {response.text}")
                return self._empty_result()
                
        except Exception as e:
            print(f"⚠️ OpenAI analysis failed: {e}")
            return self._empty_result()

    def _analyze_with_openai_legacy(self, image_base64: str, prompt: str) -> Dict[str, str]:
        """Analyze text region using OpenAI GPT-4 Vision."""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        print(f"⚠️ OpenAI JSON parse error: {content}")
                        return self._empty_result()
            else:
                print(f"⚠️ OpenAI API error: {response.status_code}")
                return self._empty_result()
                
        except Exception as e:
            print(f"⚠️ OpenAI analysis failed: {e}")
            return self._empty_result()
    
    def _analyze_text_with_ollama(self, extracted_text: str) -> Dict[str, str]:
        """Analyze extracted text using local Ollama model."""
        prompt = self._create_text_analysis_prompt(extracted_text)
        
        try:
            payload = {
                "model": self.local_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '').strip()
                
                # Parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        print(f"⚠️ Ollama JSON parse error: {content}")
                        return self._empty_result()
            else:
                print(f"⚠️ Ollama API error: {response.status_code}")
                return self._empty_result()
                
        except Exception as e:
            print(f"⚠️ Ollama analysis failed: {e}")
            return self._empty_result()

    def _analyze_with_ollama_legacy(self, image_base64: str, prompt: str) -> Dict[str, str]:
        """Analyze text region using local Ollama model."""
        try:
            payload = {
                "model": self.local_model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '')
                
                # Parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        print(f"⚠️ Ollama JSON parse error: {content}")
                        return self._empty_result()
            else:
                print(f"⚠️ Ollama API error: {response.status_code}")
                return self._empty_result()
                
        except Exception as e:
            print(f"⚠️ Ollama analysis failed: {e}")
            return self._empty_result()
    
    def analyze_category_header(self, screenshot_image: np.ndarray) -> Dict[str, str]:
        """
        Extract category and subcategory from the top header region of the screenshot.
        
        Args:
            screenshot_image: Full screenshot image
            
        Returns:
            Dictionary with category and subcategory information
        """
        try:
            # Extract header region: 268px height starting at y=280, full width
            img_height, img_width = screenshot_image.shape[:2]
            header_region = screenshot_image[280:548, 0:img_width]  # y=280 to y=548 (268px height)
            
            if header_region.size == 0:
                return {'category': '', 'subcategory': ''}
            
            # Extract text using OCR
            ocr_results = self.reader.readtext(header_region)
            extracted_text = ' | '.join([text for _, text, _ in ocr_results])
            
            if not extracted_text.strip():
                return {'category': '', 'subcategory': ''}
                
            print(f"🏷️ Category OCR extracted: {extracted_text}")
            
            # Send to AI for category parsing
            if self.use_openai:
                return self._analyze_categories_with_openai(extracted_text)
            else:
                return self._analyze_categories_with_ollama(extracted_text)
                
        except Exception as e:
            print(f"⚠️ Category header analysis failed: {e}")
            return {'category': '', 'subcategory': ''}
    
    def _create_category_analysis_prompt(self, extracted_text: str) -> str:
        """Create AI prompt for category extraction."""
        return f"""
You are analyzing OCR-extracted text from the category header of a German food delivery app.

The header typically shows:
- Active category (highlighted/selected)
- Active subcategory (underlined or highlighted)

OCR extracted text: "{extracted_text}"

Examples:
- "Schokolade & Kekse" (category), "Müsli- & Proteinriegel" (subcategory)
- "Fruchtgummi" (category), "Gummibärchen" (subcategory)

Extract the most prominent/active category and subcategory. Return ONLY valid JSON:

{{
    "category": "Schokolade & Kekse",
    "subcategory": "Müsli- & Proteinriegel"
}}

Rules:
- category = main product category (broader classification)
- subcategory = specific product type within category
- If unclear, use most prominent text elements
- Return ONLY the JSON object, no explanation

JSON:"""

    def _analyze_categories_with_openai(self, extracted_text: str) -> Dict[str, str]:
        """Analyze category text using OpenAI GPT."""
        prompt = self._create_category_analysis_prompt(extracted_text)
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\\{.*\\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        print(f"⚠️ Category OpenAI JSON parse error: {content}")
                        return {'category': '', 'subcategory': ''}
            else:
                print(f"⚠️ Category OpenAI API error: {response.status_code}")
                return {'category': '', 'subcategory': ''}
                
        except Exception as e:
            print(f"⚠️ Category OpenAI analysis failed: {e}")
            return {'category': '', 'subcategory': ''}

    def _analyze_categories_with_ollama(self, extracted_text: str) -> Dict[str, str]:
        """Analyze category text using local Ollama model."""
        prompt = self._create_category_analysis_prompt(extracted_text)
        
        try:
            payload = {
                "model": self.local_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '').strip()
                
                # Parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from response
                    import re
                    json_match = re.search(r'\\{.*\\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    else:
                        print(f"⚠️ Category Ollama JSON parse error: {content}")
                        return {'category': '', 'subcategory': ''}
            else:
                print(f"⚠️ Category Ollama API error: {response.status_code}")
                return {'category': '', 'subcategory': ''}
                
        except Exception as e:
            print(f"⚠️ Category Ollama analysis failed: {e}")
            return {'category': '', 'subcategory': ''}
    
    def adjust_subcategory_for_product(self, category: str, subcategory: str, product_name: str) -> str:
        """
        Intelligently adjust subcategory based on actual product detected.
        
        Args:
            category: Main category (e.g. "Obst")
            subcategory: Detected subcategory (e.g. "Birnen") 
            product_name: Actual product name (e.g. "Heidelbeeren")
            
        Returns:
            Corrected subcategory
        """
        if not product_name or not category:
            return subcategory
        
        product_lower = product_name.lower()
        
        # Fruit category adjustments
        if category.lower() == "obst":
            if "heidelbeeren" in product_lower or "brombeeren" in product_lower or "himbeeren" in product_lower or "erdbeeren" in product_lower:
                return "Beeren"
            elif "bananen" in product_lower:
                return "Bananen"
            elif "apfel" in product_lower or "äpfel" in product_lower:
                return "Äpfel & Birnen"
            elif "birne" in product_lower:
                return "Äpfel & Birnen" 
            elif "ananas" in product_lower or "avocado" in product_lower or "mango" in product_lower:
                return "Exoten"
            elif "trauben" in product_lower:
                return "Trauben & Steinobst"
        
        # Keep original if no specific match
        return subcategory

    def _create_vision_analysis_prompt(self, extracted_text: str) -> str:
        """Create comprehensive AI vision prompt for product analysis."""
        return f"""
You are analyzing a German food delivery app product using BOTH the product tile image AND text region.

OCR extracted text: "{extracted_text}"

Please analyze the images carefully and provide comprehensive product information.

CRITICAL CORRECTIONS:
- Fix OCR errors: IStk→1 Stk., Ikg→1kg, 1309→130g, 1259→125g
- Look at BOTH tile image AND text to avoid mistakes
- Distinguish fresh vs processed products (e.g. "Rewe ToGo Ananas" vs fresh "Ananas")
- Use descriptive properties as brand when no explicit brand
- Apple varieties: Braeburn, Pink Lady, Granny Smith, Gala = brand names
- Never leave fields empty - use "Generic" only as last resort

NUTRITIONAL ANALYSIS:
Based on the product shown, provide estimated nutritional information per 100g:
- Calories (kcal)
- Fat (g)
- Carbohydrates (g) 
- Fiber (g)
- Sugar (g)
- Protein (g)

PRODUCT ENRICHMENT:
- Short description (1-2 sentences about the product)
- Health notes (any beneficial or concerning aspects)
- Product type (fresh, processed, organic, etc.)

Return ONLY valid JSON:

{{
    "price": "€2.19",
    "brand": "Rewe",
    "product_name": "ToGo Ananas",
    "weight": "130g",
    "quantity": "",
    "price_per_unit": "16,85 € / 1kg",
    "calories_per_100g": "50",
    "fat_per_100g": "0.1",
    "carbs_per_100g": "13",
    "fiber_per_100g": "1.4",
    "sugar_per_100g": "10",
    "protein_per_100g": "0.5",
    "description": "Fresh pineapple, sweet tropical fruit rich in vitamin C and bromelain enzyme.",
    "health_notes": "High vitamin C, contains bromelain for digestion. Natural sugars.",
    "product_type": "Fresh"
}}

JSON:"""

    def _analyze_with_vision_openai(self, tile_image: np.ndarray, text_region_image: np.ndarray, extracted_text: str) -> Dict[str, str]:
        """Analyze product using OpenAI GPT Vision with both tile and text images."""
        prompt = self._create_vision_analysis_prompt(extracted_text)
        
        try:
            # Convert both images to base64
            tile_base64 = self._image_to_base64(tile_image)
            text_base64 = self._image_to_base64(text_region_image)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{tile_base64}",
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/png;base64,{text_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown response (```json...```)
                    import re
                    # Remove markdown formatting
                    content_clean = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
                    content_clean = re.sub(r'\s*```$', '', content_clean, flags=re.MULTILINE)
                    
                    try:
                        return json.loads(content_clean)
                    except json.JSONDecodeError:
                        # Fallback: extract JSON object
                        json_match = re.search(r'\\{.*\\}', content, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group())
                        else:
                            print(f"⚠️ Vision OpenAI JSON parse error: {content}")
                            return self._empty_result()
            else:
                print(f"⚠️ Vision OpenAI API error: {response.status_code} - {response.text}")
                return self._empty_result()
                
        except Exception as e:
            print(f"⚠️ Vision OpenAI analysis failed: {e}")
            return self._empty_result()

    def _analyze_with_vision_ollama(self, tile_image: np.ndarray, text_region_image: np.ndarray, extracted_text: str) -> Dict[str, str]:
        """Analyze product using local Ollama vision model."""
        prompt = self._create_vision_analysis_prompt(extracted_text)
        
        try:
            # Convert both images to base64
            tile_base64 = self._image_to_base64(tile_image)
            text_base64 = self._image_to_base64(text_region_image)
            
            payload = {
                "model": self.local_model,
                "prompt": prompt,
                "images": [tile_base64, text_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '')
                
                # Parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown response (```json...```)
                    import re
                    # Remove markdown formatting
                    content_clean = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
                    content_clean = re.sub(r'\s*```$', '', content_clean, flags=re.MULTILINE)
                    
                    try:
                        return json.loads(content_clean)
                    except json.JSONDecodeError:
                        # Fallback: extract JSON object
                        json_match = re.search(r'\\{.*\\}', content, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group())
                        else:
                            print(f"⚠️ Vision Ollama JSON parse error: {content}")
                            return self._empty_result()
            else:
                print(f"⚠️ Vision Ollama API error: {response.status_code}")
                return self._empty_result()
                
        except Exception as e:
            print(f"⚠️ Vision Ollama analysis failed: {e}")
            return self._empty_result()

    def _analyze_with_claude_vision(self, tile_image: np.ndarray, text_region_image: np.ndarray, extracted_text: str) -> Dict[str, str]:
        """
        Analyze product using Claude API for intelligent vision analysis.
        """
        try:
            # Convert both images to base64
            tile_base64 = self._image_to_base64(tile_image)
            text_base64 = self._image_to_base64(text_region_image)
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.claude_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            prompt = self._create_flink_analysis_prompt(extracted_text)
            
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": tile_base64
                                }
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png", 
                                    "data": text_base64
                                }
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['content'][0]['text'].strip()
                
                # Parse JSON response
                try:
                    parsed_result = json.loads(content)
                    print(f"✅ Claude vision analysis successful: {parsed_result.get('product_name', 'Unknown')}")
                    return parsed_result
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown response
                    import re
                    # Remove markdown formatting
                    content_clean = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
                    content_clean = re.sub(r'\s*```$', '', content_clean, flags=re.MULTILINE)
                    
                    try:
                        parsed_result = json.loads(content_clean)
                        print(f"✅ Claude vision analysis successful (after cleanup): {parsed_result.get('product_name', 'Unknown')}")
                        return parsed_result
                    except json.JSONDecodeError:
                        print(f"⚠️ Claude JSON parse error: {content}")
                        # Fallback: try intelligent text parsing
                        return self._intelligent_text_parsing_from_claude(content, extracted_text)
            else:
                print(f"⚠️ Claude API error: {response.status_code} - {response.text}")
                return self._empty_result()
                
        except Exception as e:
            print(f"⚠️ Claude vision analysis failed: {e}")
            return self._empty_result()

    def _create_flink_analysis_prompt(self, extracted_text: str) -> str:
        """Create specialized prompt for Flink food delivery app product analysis."""
        return f"""
You are analyzing a product from a German food delivery app (Flink). You have two images:
1. Product tile image (573x573px) showing the actual food product
2. Text region image (660x240px) showing product information below the tile

OCR extracted text: "{extracted_text}"

Analyze both images and extract detailed product information. Return ONLY valid JSON:

{{
    "price": "2.79€",
    "brand": "Chiquita", 
    "product_name": "Bananen",
    "weight": "5 Stk",
    "quantity": "5",
    "unit": "Stk",
    "price_per_unit": "0.56€/Stk",
    "manufacturer": "Chiquita Brands International",
    "category": "Obst",
    "subcategory": "Bananen",
    "additional_info": "Fresh bananas from premium brand"
}}

CRITICAL ANALYSIS RULES:
1. Look at BOTH the product tile image AND text region image
2. Extract price, brand, product name, weight/quantity, and price per unit
3. Correct OCR errors: IStk→Stk, 1309→130g, Ikg→kg, etc.
4. German brands: Rewe, Bio, Chiquita, Müller, Edeka, Alnatura, etc.
5. Use product varieties as brand when no explicit brand (e.g., "Braeburn" for apples)
6. Extract manufacturer information if visible
7. Determine appropriate category and subcategory
8. Clean product names: remove brand, weight, price info
9. Calculate price per unit if shown
10. Return detailed, accurate JSON based on visual analysis

Focus on extracting meaningful, structured product data for a food delivery database.

JSON:"""

    def _intelligent_text_parsing_from_claude(self, claude_response: str, extracted_text: str) -> Dict[str, str]:
        """Parse Claude's non-JSON response intelligently."""
        result = self._empty_result()
        
        try:
            # Extract key information from Claude's text response
            import re
            
            # Try to extract price
            price_match = re.search(r'(\d+[.,]\d+)\s*€', claude_response)
            if price_match:
                result['price'] = price_match.group(1).replace(',', '.') + '€'
            
            # Try to extract product name (look for quoted product names)
            product_match = re.search(r'product[_\s]*name["\s]*:\s*["\']([^"\']+)["\']', claude_response, re.IGNORECASE)
            if not product_match:
                product_match = re.search(r'["\']([^"\']{3,30})["\'].*product', claude_response, re.IGNORECASE)
            if product_match:
                result['product_name'] = product_match.group(1).strip()
            
            # Try to extract brand
            brand_match = re.search(r'brand["\s]*:\s*["\']([^"\']+)["\']', claude_response, re.IGNORECASE)
            if brand_match:
                result['brand'] = brand_match.group(1).strip()
            
            print(f"🧠 Extracted from Claude response: {result.get('product_name', 'Unknown')}")
            
        except Exception as e:
            print(f"⚠️ Claude text parsing failed: {e}")
        
        # Fallback to OCR parsing if Claude parsing didn't work well
        if not result.get('product_name'):
            return self._intelligent_ocr_analysis(None, None, extracted_text)
            
        return result

    def _intelligent_ocr_analysis(self, tile_image, text_region_image, extracted_text: str) -> Dict[str, str]:
        """
        Intelligent analysis of OCR text without using regex patterns.
        Uses smart text processing to extract product information.
        """
        result = self._empty_result()
        
        try:
            print(f"🔧 Intelligent OCR analysis: {extracted_text}")
            
            # Split text by common separators
            text_parts = []
            for separator in ['|', '•', '·', '\n', ' - ', ' – ']:
                if separator in extracted_text:
                    text_parts = [part.strip() for part in extracted_text.split(separator) if part.strip()]
                    break
            
            if not text_parts:
                # Fallback: split by spaces and reassemble
                words = extracted_text.split()
                text_parts = [extracted_text]  # Keep full text as single part
            
            # Extract price (usually first or last part)
            import re
            for part in text_parts:
                price_match = re.search(r'(\d+[.,]\d+)\s*€', part)
                if price_match:
                    result['price'] = price_match.group(1).replace(',', '.') + '€'
                    break
            
            # Extract product information from the longest meaningful part
            product_part = max(text_parts, key=len) if text_parts else extracted_text
            
            # German grocery brands to look for
            german_brands = ['Rewe', 'Bio', 'Chiquita', 'Müller', 'Edeka', 'Alnatura', 'Harry', 'Florette', 'Coppenrath']
            
            brand_found = None
            for brand in german_brands:
                if brand.lower() in product_part.lower():
                    brand_found = brand
                    result['brand'] = brand
                    break
            
            # Clean product name
            clean_name = product_part
            if brand_found:
                clean_name = clean_name.replace(brand_found, '').strip()
            
            # Remove price information from name
            clean_name = re.sub(r'\d+[.,]\d+\s*€', '', clean_name).strip()
            # Remove weight/quantity information 
            weight_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(kg|g|ml|l|Stk\.?)', clean_name, re.IGNORECASE)
            if weight_match:
                result['weight'] = weight_match.group(1) + weight_match.group(2)
                result['quantity'] = weight_match.group(1)
                result['unit'] = weight_match.group(2)
                clean_name = re.sub(r'\d+(?:[.,]\d+)?\s*(?:kg|g|ml|l|Stk\.?)', '', clean_name, flags=re.IGNORECASE).strip()
            
            # Remove common descriptor words
            clean_name = re.sub(r'\b(Beste|Wahl|To|Go|Obst|Fresh|Bio)\b', '', clean_name, flags=re.IGNORECASE).strip()
            clean_name = re.sub(r'\s+', ' ', clean_name).strip()  # Remove multiple spaces
            
            if clean_name:
                result['product_name'] = clean_name
            
            # Try to extract price per unit
            for part in text_parts:
                price_per_unit_match = re.search(r'(\d+[.,]\d+)\s*€\s*/\s*(\w+)', part)
                if price_per_unit_match:
                    result['price_per_unit'] = f"{price_per_unit_match.group(1).replace(',', '.')}€/{price_per_unit_match.group(2)}"
                    break
            
            # Clean empty values
            result = {k: v for k, v in result.items() if v and v.strip()}
            
            if result.get('product_name'):
                print(f"✅ Intelligent OCR successful: {result['product_name']}")
            else:
                print("❌ Intelligent OCR failed to extract product name")
                
        except Exception as e:
            print(f"❌ Intelligent OCR error: {e}")
        
        return result

    def _empty_result(self) -> Dict[str, str]:
        """Return empty result structure."""
        return {
            'price': '',
            'brand': '',
            'product_name': '',
            'weight': '',
            'quantity': '',
            'price_per_unit': '',
            'calories_per_100g': '',
            'fat_per_100g': '',
            'carbs_per_100g': '',
            'fiber_per_100g': '',
            'sugar_per_100g': '',
            'protein_per_100g': '',
            'description': '',
            'health_notes': '',
            'product_type': ''
        }
    
    def _parse_ocr_text_regex(self, extracted_text: str) -> Dict[str, str]:
        """
        Parse OCR text using regex patterns when AI analysis fails.
        Handles the structured format: "Price | Product Name | Price per unit"
        Example: "2,79 € | Bananen Chiquita 5 Stk. | 0,56 € | IStk."
        """
        import re
        
        print(f"🔧 Regex parsing: {extracted_text}")
        
        result = self._empty_result()
        
        try:
            # Split by pipe separator 
            parts = [part.strip() for part in extracted_text.split('|')]
            
            if len(parts) >= 2:
                # First part is usually the price
                price_match = re.search(r'(\d+[.,]\d+)\s*€', parts[0])
                if price_match:
                    result['price'] = price_match.group(1).replace(',', '.') + '€'
                
                # Second part is usually the product name (may include brand)
                product_part = parts[1].strip()
                
                # Extract brand (common German grocery brands)
                brands = ['Rewe', 'Chiquita', 'Bio', 'Müller', 'Harry', 'Florette', 'Coppenrath', 'REWE']
                brand_found = None
                for brand in brands:
                    if brand.lower() in product_part.lower():
                        brand_found = brand
                        break
                
                if brand_found:
                    result['brand'] = brand_found
                    # Remove brand from product name
                    product_name = product_part.replace(brand_found, '').strip()
                    # Remove common words
                    product_name = re.sub(r'\b(Beste|Wahl|To|Go|Obst)\b', '', product_name).strip()
                else:
                    product_name = product_part
                
                # Extract weight/quantity from product name
                weight_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(kg|g|ml|l|Stk\.?)', product_name, re.IGNORECASE)
                if weight_match:
                    result['weight'] = weight_match.group(1) + weight_match.group(2)
                    result['quantity'] = weight_match.group(1)
                    result['unit'] = weight_match.group(2)
                    # Clean product name
                    product_name = re.sub(r'\d+(?:[.,]\d+)?\s*(?:kg|g|ml|l|Stk\.?)', '', product_name, flags=re.IGNORECASE).strip()
                
                result['product_name'] = product_name
                
                # Third part might be price per unit
                if len(parts) >= 3:
                    price_per_unit_match = re.search(r'(\d+[.,]\d+)\s*€', parts[2])
                    if price_per_unit_match:
                        unit_part = parts[3] if len(parts) >= 4 else ''
                        result['price_per_unit'] = f"{price_per_unit_match.group(1).replace(',', '.')}€/{unit_part.strip()}"
            
            # Clean empty values
            result = {k: v for k, v in result.items() if v}
            
            if result:
                print(f"✅ Regex parsing successful: {result.get('product_name', 'Unknown')}")
            else:
                print("❌ Regex parsing failed")
                
        except Exception as e:
            print(f"❌ Regex parsing error: {e}")
        
        return result

def test_ai_analyzer():
    """Test function for the AI analyzer."""
    # Test with a sample image
    analyzer = AITextAnalyzer(use_openai=False)  # Use local Ollama by default
    
    # You can switch to OpenAI by setting:
    # analyzer = AITextAnalyzer(use_openai=True, api_key="your-openai-api-key")
    
    print("🤖 AI Text Analyzer initialized")
    print(f"🔧 Using: {'OpenAI GPT' if analyzer.use_openai else 'Local Ollama'}")
    
if __name__ == "__main__":
    test_ai_analyzer()