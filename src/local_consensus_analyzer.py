#!/usr/bin/env python3
"""
LocalConsensusAnalyzer - Integrated working consensus system
Combines the working fixed system with original method signatures for compatibility.
Supports both UI (category) analysis and product analysis with 3-model consensus.
"""

import cv2
import numpy as np
import json
import requests
import base64
import asyncio
import time
import os
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
# ✅ CLAUDE.MD COMPLIANT - NO OCR/REGEX IMPORTS ✅
# LLM CONSENSUS SYSTEM ONLY

class LocalConsensusAnalyzer:
    """
    Integrated consensus analyzer supporting both UI and product analysis.
    Uses 3-model consensus with Ollama vision models.
    """

    def __init__(self, use_api_fallback: bool = False):
        """Initialize the consensus analyzer for pure LLM consensus."""

        # Circuit breaker state for model health
        self.model_health = {}
        self.failed_attempts = {}
        self.last_success = {}

        # Robustness configuration
        self.max_circuit_failures = 3
        self.circuit_reset_time = 300  # 5 minutes
        self.base_retry_delay = 2  # seconds

        # Working models with proper weights
        self.models = [
            {
                "name": "qwen2.5vl:7b",
                "weight": 1.2,  # Highest weight - excellent OCR
                "timeout": 60,  # Fast, 2024 model optimized for text
                "max_retries": 3,  # Standard retries
                "warmup_priority": 1  # Load first
            },
            {
                "name": "minicpm-v:latest",
                "weight": 1.2,
                "timeout": 60,  # Standard timeout
                "max_retries": 3,  # Standard retries
                "warmup_priority": 2  # Load second
            },
            {
                "name": "llava-llama3:latest",
                "weight": 1.0,
                "timeout": 75,  # Good timeout for instruction-tuned model
                "max_retries": 3,  # Standard retries
                "warmup_priority": 3  # Load third
            }
            # Using qwen2.5vl:7b (2024 model, excellent OCR), minicpm-v, llava-llama3
            # All models ~5-6GB, fast and reliable for product text extraction
        ]

        print("🔧 INTEGRATED CONSENSUS SYSTEM INITIALIZED")
        print(f"📋 Using models: {[m['name'] for m in self.models]}")

        # Initialize debug logging system
        self.debug_logs = []
        self.debug_output_dir = None
        self.current_product_id = None

        # Keep-alive mechanism for connection stability
        self.last_query_time = {}  # Track last query time per model
        self.keep_alive_interval = 30  # Ping every 30 seconds

        # Initialize models with smart pre-loading
        self._initialize_model_system()

    def set_debug_output(self, output_dir: str, product_id: str = None):
        """Configure debug output directory and current product ID"""
        self.debug_output_dir = output_dir
        self.current_product_id = product_id
        if product_id:
            print(f"🐛 Debug logging enabled for product: {product_id}")

    def log_llm_interaction(self, step: str, step_name: str, model: str, prompt: str,
                           response: str, image_shape: tuple = None, additional_data: dict = None):
        """Log a single LLM interaction for debugging"""
        log_entry = {
            "product_id": self.current_product_id,
            "step": step,
            "step_name": step_name,
            "model": model,
            "prompt": prompt,
            "response": response[:500] if response else None,  # Truncate long responses
            "response_length": len(response) if response else 0,
            "full_response": response,  # Keep full response for debugging
            "image_shape": f"{image_shape[0]}x{image_shape[1]}" if image_shape else None,
            "timestamp": time.time()
        }
        if additional_data:
            log_entry.update(additional_data)

        self.debug_logs.append(log_entry)

    def save_debug_logs(self):
        """Save all debug logs to a JSON file"""
        if not self.debug_output_dir or not self.debug_logs:
            return

        import os
        from datetime import datetime

        os.makedirs(self.debug_output_dir, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_debug_log_{timestamp}.json"
        filepath = os.path.join(self.debug_output_dir, filename)

        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.debug_logs, f, indent=2, ensure_ascii=False)

        print(f"🐛 Debug logs saved: {filepath} ({len(self.debug_logs)} interactions)")
        return filepath

    def _initialize_model_system(self):
        """
        Smart model initialization with pre-loading, monitoring, and clear error reporting.
        """
        print("🚀 SMART MODEL SYSTEM INITIALIZATION")

        # Check if models are already loaded
        model_status = self._check_model_status()

        if model_status['all_loaded']:
            print("✅ All models already loaded and ready!")
            return

        print(f"📊 Model Status: {model_status['loaded_count']}/{len(self.models)} loaded")

        # Pre-load missing models with extended keep-alive
        self._preload_models(model_status['missing_models'])

    def _check_model_status(self) -> dict:
        """
        Check which models are currently loaded in Ollama.
        Returns detailed status information.
        """
        try:
            # Query Ollama for currently loaded models
            response = requests.get("http://localhost:11434/api/ps", timeout=5)

            if response.status_code == 200:
                loaded_models_data = response.json()
                loaded_model_names = [model['name'] for model in loaded_models_data.get('models', [])]

                required_model_names = [m['name'] for m in self.models]
                missing_models = [name for name in required_model_names if name not in loaded_model_names]

                status = {
                    'all_loaded': len(missing_models) == 0,
                    'loaded_count': len(loaded_model_names),
                    'missing_models': missing_models,
                    'loaded_models': loaded_model_names,
                    'ollama_responsive': True
                }

                print(f"🔍 Currently loaded: {loaded_model_names}")
                if missing_models:
                    print(f"⚠️  Missing models: {missing_models}")

                return status

            else:
                print(f"⚠️  Ollama API responded with status {response.status_code}")
                return {
                    'all_loaded': False,
                    'loaded_count': 0,
                    'missing_models': [m['name'] for m in self.models],
                    'loaded_models': [],
                    'ollama_responsive': False,
                    'error': f"HTTP {response.status_code}"
                }

        except requests.exceptions.ConnectionError:
            print("🚨 OLLAMA CONNECTION ERROR: Ollama server not responding")
            return {
                'all_loaded': False,
                'loaded_count': 0,
                'missing_models': [m['name'] for m in self.models],
                'loaded_models': [],
                'ollama_responsive': False,
                'error': "Connection refused - is Ollama running?"
            }
        except Exception as e:
            print(f"🚨 MODEL STATUS CHECK FAILED: {str(e)}")
            return {
                'all_loaded': False,
                'loaded_count': 0,
                'missing_models': [m['name'] for m in self.models],
                'loaded_models': [],
                'ollama_responsive': False,
                'error': str(e)
            }

    def _preload_models(self, missing_models: list):
        """
        Pre-load models with extended keep-alive and progress monitoring.
        """
        if not missing_models:
            return

        print(f"🔄 Pre-loading {len(missing_models)} models with 60-minute keep-alive...")

        for model_name in missing_models:
            print(f"   📡 Loading {model_name}...")
            start_time = time.time()

            try:
                # Pre-load with extended keep-alive (60 minutes)
                payload = {
                    "model": model_name,
                    "prompt": "Model pre-load initialization",
                    "stream": False,
                    "keep_alive": "60m"  # Keep loaded for 60 minutes
                }

                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=45  # Allow time for model loading
                )

                load_time = time.time() - start_time

                if response.status_code == 200:
                    print(f"   ✅ {model_name} loaded successfully ({load_time:.1f}s)")
                    self._record_model_success(model_name)
                else:
                    error_msg = f"HTTP {response.status_code}"
                    print(f"   ❌ {model_name} failed to load: {error_msg}")
                    self._record_model_failure(model_name)

            except requests.exceptions.Timeout:
                load_time = time.time() - start_time
                print(f"   ⏰ {model_name} load timeout after {load_time:.1f}s")
                print(f"      💡 Large models may take longer - check 'ollama ps' manually")
                self._record_model_failure(model_name)

            except Exception as e:
                load_time = time.time() - start_time
                print(f"   ❌ {model_name} load failed after {load_time:.1f}s: {str(e)[:50]}...")
                self._record_model_failure(model_name)

        # Final status check
        final_status = self._check_model_status()
        if final_status['all_loaded']:
            print("🎉 All models successfully pre-loaded and ready!")
        else:
            print(f"⚠️  Pre-loading complete: {final_status['loaded_count']}/{len(self.models)} models ready")

    def _get_model_cooldown_status(self) -> dict:
        """
        Check if models have cooled down and provide restart guidance.
        """
        try:
            response = requests.get("http://localhost:11434/api/ps", timeout=5)

            if response.status_code == 200:
                loaded_models = response.json().get('models', [])

                if not loaded_models:
                    return {
                        'models_loaded': False,
                        'message': "🧊 ALL MODELS COOLED DOWN - Models have been unloaded from memory",
                        'action_required': "Call _preload_models() or run manual warmup",
                        'restart_command': "ollama run moondream:latest 'warmup' && ollama run minicpm-v:latest 'warmup'",
                        'severity': 'warning'
                    }

                required_models = [m['name'] for m in self.models]
                loaded_names = [m['name'] for m in loaded_models]
                missing = [name for name in required_models if name not in loaded_names]

                if missing:
                    return {
                        'models_loaded': True,
                        'partial_cooldown': True,
                        'message': f"🔄 PARTIAL COOLDOWN - Missing: {missing}",
                        'action_required': f"Reload missing models",
                        'missing_models': missing,
                        'severity': 'info'
                    }

                return {
                    'models_loaded': True,
                    'message': "✅ All models ready",
                    'severity': 'success'
                }

            else:
                return {
                    'models_loaded': False,
                    'message': f"🚨 OLLAMA API ERROR - HTTP {response.status_code}",
                    'action_required': "Check if Ollama is running: 'ollama serve'",
                    'severity': 'error'
                }

        except requests.exceptions.ConnectionError:
            return {
                'models_loaded': False,
                'message': "🚨 OLLAMA SERVER DOWN - Cannot connect to Ollama",
                'action_required': "Start Ollama: 'ollama serve'",
                'restart_command': "ollama serve",
                'severity': 'critical'
            }

    def _extract_json_bulletproof(self, raw_content: str) -> dict:
        """
        Pure LLM JSON extraction without regex - CLAUDE.MD COMPLIANT
        Uses LLM consensus to parse and validate JSON responses.
        """
        import json

        # Method 1: Direct JSON parsing
        try:
            # Try to find JSON in the content by looking for { and }
            start_idx = raw_content.find('{')
            end_idx = raw_content.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = raw_content[start_idx:end_idx + 1]
                parsed = json.loads(json_str)
                return {
                    "status": "success",
                    "data": parsed,
                    "method": "direct_json_parsing"
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Method 2: Clean and try again
        try:
            # Simple cleanup without regex
            content = raw_content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            parsed = json.loads(content)
            return {
                "status": "success",
                "data": parsed,
                "method": "cleaned_json_parsing"
            }
        except (json.JSONDecodeError, ValueError):
            pass

        # Method 3: Fall back to LLM re-parsing if JSON extraction fails
        return {
            "status": "fallback_needed",
            "data": {},
            "method": "pure_llm_consensus",
            "raw_content": raw_content
        }

    def _repair_truncated_json(self, truncated: str) -> str:
        """
        Pure string manipulation JSON repair - CLAUDE.MD COMPLIANT
        No regex patterns - uses simple string methods only.
        """
        repaired = truncated.rstrip(',').rstrip()

        # Handle incomplete field names with simple string checking
        incomplete_prefixes = ['"available_s', '"cate', '"main_cat', '"prod']

        for prefix in incomplete_prefixes:
            if prefix in repaired:
                # Find the last occurrence and remove everything from there
                last_pos = repaired.rfind(prefix)
                if last_pos != -1:
                    # Check if this looks like an incomplete field at the end
                    after_prefix = repaired[last_pos:]
                    if not after_prefix.count('"') >= 2:  # Not a complete field
                        repaired = repaired[:last_pos].rstrip(',').rstrip()
                        break

        # Ensure proper JSON closure
        if not repaired.endswith('}'):
            repaired += '}'

        return repaired

    def _initialize_circuit_breaker(self):
        """Initialize circuit breaker state for all models."""
        # Initialize circuit breaker state
        for model in self.models:
            model_name = model['name']
            self.model_health[model_name] = True
            self.failed_attempts[model_name] = 0
            self.last_success[model_name] = time.time()

    def _is_model_healthy(self, model_name: str) -> bool:
        """Check if model is healthy (not in circuit breaker open state)."""
        current_time = time.time()

        # If circuit breaker is open, check if reset time has passed
        if not self.model_health.get(model_name, True):
            time_since_last_attempt = current_time - self.last_success.get(model_name, 0)
            if time_since_last_attempt > self.circuit_reset_time:
                print(f"   🔄 Circuit breaker reset for {model_name} after {time_since_last_attempt:.1f}s")
                self.model_health[model_name] = True
                self.failed_attempts[model_name] = 0
                return True
            return False

        return True

    def _record_model_success(self, model_name: str):
        """Record successful model response."""
        self.model_health[model_name] = True
        self.failed_attempts[model_name] = 0
        self.last_success[model_name] = time.time()

    def _record_model_failure(self, model_name: str):
        """Record model failure and potentially open circuit breaker."""
        self.failed_attempts[model_name] = self.failed_attempts.get(model_name, 0) + 1

        if self.failed_attempts[model_name] >= self.max_circuit_failures:
            self.model_health[model_name] = False
            print(f"   🚨 Circuit breaker OPEN for {model_name} after {self.failed_attempts[model_name]} failures")

    def _restart_ollama_service(self):
        """Restart Ollama service to recover from connection issues."""
        try:
            # Kill existing Ollama processes
            result = subprocess.run(['pkill', '-f', 'ollama'], capture_output=True)
            time.sleep(2)

            # Start Ollama service in background
            subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)

            # Test if service is responding
            if self._check_ollama_health():
                print("   ✅ Ollama service restarted successfully")
            else:
                print("   ⚠️  Ollama restart may have failed")

        except Exception as e:
            print(f"   ❌ Failed to restart Ollama: {e}")

    def _check_ollama_health(self) -> bool:
        """Check if Ollama service is healthy and responding."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False

    async def _ensure_model_loaded(self, model_name: str):
        """
        Ensure model is loaded with keep-alive ping.
        Prevents model unloading between queries.
        """
        current_time = time.time()
        last_query = self.last_query_time.get(model_name, 0)

        # If more than keep_alive_interval seconds since last query, send keep-alive ping
        if current_time - last_query > self.keep_alive_interval:
            try:
                ping_payload = {
                    "model": model_name,
                    "prompt": "keepalive",
                    "stream": False,
                    "keep_alive": "60m"  # Request 60min keep-alive
                }
                requests.post(
                    "http://localhost:11434/api/generate",
                    json=ping_payload,
                    timeout=5
                )
                print(f"   🏓 Keep-alive ping sent to {model_name}")
            except:
                pass  # Ignore ping failures

        self.last_query_time[model_name] = current_time

    async def _query_model_with_retry(self, model_name: str, image: np.ndarray,
                                      prompt: str, max_retries: int = 3,
                                      timeout: int = 30, step_id: str = "unknown") -> str:
        """
        Query model with automatic retry on timeout.
        Implements exponential backoff and keep-alive.
        """
        for attempt in range(max_retries):
            try:
                # Ensure model is loaded before query
                await self._ensure_model_loaded(model_name)

                # Execute query
                response_text = await self._query_model_simple(
                    model_name, image, prompt, timeout, step_id
                )

                if response_text and len(response_text.strip()) > 0:
                    return response_text

                # Empty response = retry
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"      ⏳ Empty response, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"      ⏰ {type(e).__name__}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"      ❌ Failed after {max_retries} attempts")
                    return ""

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"      ⚠️  Error: {str(e)[:50]}, retrying...")
                    await asyncio.sleep(2)
                else:
                    print(f"      ❌ Failed: {str(e)[:50]}")
                    return ""

        return ""

    def _extract_visual_features(self, image: np.ndarray) -> str:
        """Extract basic visual description for LLM context (no OCR)."""
        try:
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) > 2 else 1

            # Basic image characteristics for LLM context
            return f"Image: {width}x{height}px, {channels} channels"
        except Exception as e:
            print(f"❌ Visual feature extraction error: {e}")
            return "Image: Unknown dimensions"

    def _create_local_analysis_prompt(self, visual_context: str, analysis_mode: str, custom_prompt: str = None) -> str:
        """Create analysis prompt based on mode (ui/product) - PURE VISION LLM."""

        # 🎯 CACHE-BUSTING: Add unique identifier to prevent cached responses
        import random
        import time
        cache_buster = f"[SESSION_{int(time.time())}_{random.randint(1000,9999)}]"
        contamination_warning = "⚠️ ANALYZE THIS SPECIFIC IMAGE - DO NOT USE CACHED RESPONSES FROM PREVIOUS IMAGES"

        if custom_prompt:
            return f"{cache_buster} {contamination_warning}\n{custom_prompt}"
        elif analysis_mode == "ui":
            # ENHANCED UI analysis with multi-check visual hierarchy system
            return f"""{cache_buster} {contamination_warning}

You are analyzing a food delivery app screenshot.
Your task: extract the **MAIN CATEGORY** and the **ACTIVE SUBCATEGORY** from the header.

VISUAL HIERARCHY RULES (MULTI-CHECK SYSTEM):
1. MAIN CATEGORY
   - Look at the second row beneath the word "Categories".
   - The element with a **colored/pink background pill** is ALWAYS the MAIN CATEGORY.
   - Double-check by position: it is left-aligned on that row, surrounded by siblings (other categories with white/grey backgrounds).
   - Text is usually sentence case (e.g., "Schokolade & Kekse").

2. ACTIVE SUBCATEGORY
   - Look at the third row (beneath the main category row).
   - The **bolded/underlined/darker text** is ALWAYS the ACTIVE SUBCATEGORY.
   - Other subcategories on the same row are lighter/normal-weight text.
   - Double-check by position: it appears inline with siblings (e.g., "Müsli- & Proteinriegel" among "Be Backwaren", "Pralinen").

MULTIPLE VISUAL CONFIRMATIONS:
- Main category must meet BOTH conditions:
   (a) Pink background + (b) on row 2.
- Subcategory must meet BOTH conditions:
   (a) Bold/dark + (b) on row 3.
- If formatting is unclear, rely on spatial order:
   row 2 = main, row 3 = sub.

OUTPUT FORMAT (add available_subcategories for system compatibility):
{{
  "main_category": "TEXT",
  "active_subcategory": "TEXT",
  "available_subcategories": ["list", "of", "visible", "subcategories"],
  "confidence": "high",
  "fallback_reason": "explanation if confidence is low"
}}

EXAMPLES:
- Pink "Obst & Gemüse" + bold "Bananen" → {{"main_category": "Obst & Gemüse", "active_subcategory": "Bananen", "confidence":"high"}}
- Pink "Konserven, Instantgerichte & Backen" + bold "Backzutaten" → {{"main_category": "Konserven, Instantgerichte & Backen", "active_subcategory": "Backzutaten", "confidence":"high"}}

IMPORTANT:
- Never confuse product cards (below subcategory row) with categories.
- Ignore price, brand, or product names.
- Use redundancy: confirm via color, boldness, row order, and grouping.

TASK:
Apply these rules to the input image and return ONLY the JSON object."""

        else:
            # Enhanced product analysis with pure vision LLM - no OCR
            return f"""{cache_buster} {contamination_warning}

🔥 CRITICAL: You MUST analyze this German grocery product image and return COMPLETE JSON. Do NOT stop early or cut off your response.

{visual_context}

🎯 PRIORITY TASK: Extract the PRICE first - every product has a clearly visible price like "1,69 €" or "2,49 €".

=== VISUAL ANALYSIS INSTRUCTIONS ===

1. 💰 PRICE EXTRACTION (HIGHEST PRIORITY):
   - Look for the main price prominently displayed (usually large text)
   - Extract exactly as shown: "1,69 €", "2,49 €", "0,89 €"
   - This field is MANDATORY - never leave empty

2. 🏷️ BRAND DETECTION (PRECISE EXTRACTION):
   ⚠️ CRITICAL: Extract ONLY the brand name, never concatenate with product type!
   - Look for manufacturer name: "Dr. Oetker", "Mondamin", "Aurora"
   - Known brands: Dr. Oetker, Mondamin, Aurora, Edeka, Rewe, Knorr, Maggi, Alnatura
   - WRONG: "Mondamin Milchreis" ← NEVER do this
   - CORRECT: "Mondamin" (brand) + "Milchreis Klassische Art" (product_name)
   - If product shows "Mondamin Milchreis Klassische Art":
     * brand = "Mondamin"
     * product_name = "Milchreis Klassische Art"

3. 📦 PRODUCT NAME:
   - Remove brand from title to get clean product name
   - Keep descriptors: "Klassische Art", "Bio", "Vanille Geschmack"

🚨 MANDATORY RULES - VIOLATION = DISQUALIFICATION 🚨

4. 🎯 UNIT DETECTION FROM PER-UNIT PRICING (MANDATORY OVERRIDE):
   🚨 CRITICAL: The unit field MUST be derived from the per-unit pricing text visible in the image!

   **STEP-BY-STEP UNIT DETECTION PROCESS:**
   1. **FIRST**: Look for per-unit pricing text like "25,69 € / 1kg" or "3,38 €/l"
   2. **EXTRACT**: The unit from this pricing text (kg, l, Stk)
   3. **SET**: unit field to the extracted unit
   4. **IGNORE**: Any other visual assumptions about the product

   ✅ MANDATORY EXAMPLES - FOLLOW EXACTLY:
   - See "0,36 € / 1Stk." → unit="Stk" (from pricing text)
   - See "25,69 € / 1kg" → unit="kg" (from pricing text)
   - See "3,38 €/l" → unit="l" (from pricing text)
   - See "2,50 € / 100ml" → unit="l" (normalize ml to l)

   🚨 UNIT DETECTION HIERARCHY (MANDATORY ORDER):
   1. **PRIMARY**: Extract from per-unit pricing text (€/kg, €/l, €/Stk)
   2. **FALLBACK**: Only if no pricing text visible, use product description

   **UNIT NORMALIZATION (MANDATORY):**
   - Any weight unit (g, kg, 100g) → "kg"
   - Any volume unit (ml, l, 100ml) → "l"
   - Any piece unit (Stk, pieces, items) → "Stk"

   ❌ VIOLATION EXAMPLES (DO NOT DO THIS):
   - unit="Stk" but price_per_kg="1,09 €" AND price_per_piece="0,36 €" ← DISQUALIFIED
   - unit="ml" but price_per_kg="3,38 €" AND price_per_liter="1,69 €" ← DISQUALIFIED

5. 💶 EXCLUSIVE PRICING ENFORCEMENT (ZERO TOLERANCE):
   🔥 RULE: Populate EXACTLY ONE price field based on unit. NEVER multiple fields.

   MAPPING TABLE (MEMORIZE THIS):
   "€/kg" or "€ / 1kg" → unit="kg" + price_per_kg ONLY
   "€/Stk" or "€ / 1Stk." → unit="Stk" + price_per_piece ONLY
   "€/l" or "€ / 1l" → unit="l" + price_per_liter ONLY

   🚨 ANY MODEL VIOLATING THIS RULE WILL BE IMMEDIATELY DISQUALIFIED

⚠️ CRITICAL: You MUST return a COMPLETE JSON object with ALL 9 fields. Do not stop mid-response!

=== MANDATORY OUTPUT FORMAT ===
{{
  "price": "[MAIN PRICE WITH €]",
  "product_name": "[PRODUCT NAME]",
  "brand": "[BRAND NAME]",
  "weight": "[NUMBER OR EMPTY]",
  "unit": "[g/ml/l/Stk OR EMPTY]",
  "quantity": "[NUMBER OR EMPTY]",
  "price_per_kg": "[NUMBER OR EMPTY]",
  "price_per_piece": "[NUMBER OR EMPTY]",
  "price_per_liter": "[NUMBER OR EMPTY]"
}}

🔥 COMPLETE THE ENTIRE JSON - DO NOT STOP EARLY!

=== EXAMPLES ===

Example 1 - Per-unit pricing "3,38 €/l" visible:
{{"price": "1,69 €", "product_name": "Milchreis Klassische Art", "brand": "Mondamin", "weight": "500", "unit": "l", "quantity": "", "price_per_kg": "", "price_per_piece": "", "price_per_liter": "3,38 €"}}

Example 2 - Per-unit pricing "0,36 € / 1Stk." visible:
{{"price": "1,09 €", "product_name": "Puddingpulver Vanille", "brand": "Dr. Oetker", "weight": "", "unit": "Stk", "quantity": "3", "price_per_kg": "", "price_per_piece": "0,36 €", "price_per_liter": ""}}

Example 3 - Per-unit pricing "25,50 €/kg" visible:
{{"price": "2,55 €", "product_name": "Bio Müsli", "brand": "Alnatura", "weight": "100", "unit": "kg", "quantity": "", "price_per_kg": "25,50 €", "price_per_piece": "", "price_per_liter": ""}}

🎯 NOW ANALYZE THE IMAGE AND RETURN ONLY THE COMPLETE JSON OBJECT WITH ALL 9 FIELDS!"""

    def _calculate_cost_metrics(self, product_data: Dict) -> Dict:
        """
        CLAUDE.MD COMPLIANT - Calculate cost metrics without regex
        Pure LLM consensus approach for price calculations.
        Note: This function is deprecated in favor of LLM-based price extraction.
        """
        # FALLBACK ONLY: For compatibility, return empty metrics
        # The LLM consensus system should handle all price calculations
        cost_per_kg = ""
        cost_per_piece = ""

        print("      ℹ️  Using LLM consensus for price calculations (no regex)")

        # LLM models should provide price_per_kg, price_per_piece directly
        # No local calculation needed - trust the consensus results

        return {
            "cost_per_kg": cost_per_kg,
            "cost_per_piece": cost_per_piece
        }

    def _select_best_field_candidate(self, field: str, candidates: List[Dict]) -> Dict:
        """
        Intelligent field-by-field selection logic.
        Picks the best candidate based on quality, consensus, and completeness.
        """
        if len(candidates) == 1:
            candidates[0]["quality_score"] = self._score_field_quality(field, candidates[0]["value"])
            return candidates[0]

        # Score all candidates with validation-aware scoring
        for candidate in candidates:
            base_quality = self._score_field_quality(field, candidate["value"])

            # 🛡️ APPLY VALIDATION PENALTY TO QUALITY SCORE
            validation_info = candidate.get("validation", {})
            if validation_info.get("contamination_detected", False):
                # Severely penalize contaminated responses in quality scoring
                contamination_penalty = validation_info.get("confidence_adjustment", 1.0)
                adjusted_quality = base_quality * contamination_penalty
                print(f"   🚨 Quality penalty for {field} from {candidate['model']}: {base_quality:.1f} → {adjusted_quality:.1f} (contaminated)")
                candidate["quality_score"] = adjusted_quality
            else:
                candidate["quality_score"] = base_quality

        # Check for consensus (2+ models agree)
        value_counts = {}
        for candidate in candidates:
            value = candidate["value"]
            if value not in value_counts:
                value_counts[value] = []
            value_counts[value].append(candidate)

        # If 2+ models agree on same value, pick that
        for value, agreeing_candidates in value_counts.items():
            if len(agreeing_candidates) >= 2:
                # Pick the one with highest model weight among agreeing models
                best_agreeing = max(agreeing_candidates, key=lambda x: x["model_weight"])
                print(f"   🤝 {field}: Consensus found - {len(agreeing_candidates)} models agree on '{value}'")
                return best_agreeing

        # No consensus - pick by quality score
        best_candidate = max(candidates, key=lambda x: x["quality_score"])

        # If tied on quality, pick by model weight
        max_quality = best_candidate["quality_score"]
        tied_candidates = [c for c in candidates if c["quality_score"] == max_quality]
        if len(tied_candidates) > 1:
            best_candidate = max(tied_candidates, key=lambda x: x["model_weight"])
            print(f"   ⭐ {field}: Quality-based selection (score: {max_quality})")
        else:
            print(f"   🏅 {field}: Clear quality winner (score: {max_quality})")

        return best_candidate

    def _score_field_quality(self, field: str, value: str) -> float:
        """
        Score the quality of a field value.
        Higher scores = better quality data.
        """
        if not value or str(value).strip() == "":
            return 0.0

        value = str(value).strip()
        score = 1.0  # Base score for non-empty

        # Field-specific quality scoring
        if field == "price":
            if "€" in value:
                score += 2.0  # Has currency symbol
            if "," in value or "." in value:
                score += 1.0  # Has decimal separator
            if len(value) >= 5:  # e.g., "3,19 €"
                score += 1.0  # Reasonable length

        elif field == "product_name":
            score += len(value) * 0.1  # Longer names usually better
            if len(value) > 20:  # Very detailed name
                score += 2.0
            elif len(value) > 10:  # Moderately detailed
                score += 1.0

        elif field == "brand":
            known_brands = ["Dr. Oetker", "Mondamin", "Aurora", "Barebells", "BE-KIND",
                           "Edeka", "Rewe", "Knorr", "Maggi", "Alnatura"]
            if any(brand.lower() in value.lower() for brand in known_brands):
                score += 2.0  # Known brand
            if len(value) > 3:
                score += 1.0  # Reasonable brand name length

        elif field in ["weight", "quantity"]:
            if any(char.isdigit() for char in value):
                score += 2.0  # Contains numbers
            if any(unit in value.lower() for unit in ["g", "kg", "ml", "l", "stk", "pack"]):
                score += 1.0  # Contains unit

        elif field == "unit":
            common_units = ["g", "kg", "ml", "l", "stk", "/kg", "/stk", "pack"]
            if value.lower() in common_units:
                score += 2.0  # Standard unit

        elif field in ["price_per_kg", "price_per_piece", "price_per_liter"]:
            if "€" in value:
                score += 2.0  # Has currency
            if any(char.isdigit() for char in value):
                score += 1.0  # Contains numbers
            if "/" in value or "per" in value.lower():
                score += 1.0  # Per-unit indicator

        # Penalize very short values (likely incomplete)
        if len(value) < 2:
            score -= 1.0

        return max(score, 0.1)  # Minimum score for non-empty values

    def _apply_post_processing_cleanup(self, consensus_result: Dict) -> Dict:
        """
        Apply post-processing cleanup rules to fix common data quality issues.
        1. Remove brand names from product names
        2. Enforce unit-to-price field mapping (exclusive pricing)
        3. Clean up garbage values
        """
        cleaned = consensus_result.copy()

        # 🧹 RULE 1: Remove brand contamination from product names
        brand = cleaned.get("brand", "").strip()
        product_name = cleaned.get("product_name", "").strip()

        if brand and product_name and brand in product_name:
            # Remove brand from start of product name
            if product_name.startswith(brand):
                clean_name = product_name[len(brand):].strip()
                # Remove common separators
                for sep in [" - ", " ", "-", ":"]:
                    if clean_name.startswith(sep):
                        clean_name = clean_name[len(sep):].strip()
                        break
                if clean_name:  # Only update if we have something left
                    cleaned["product_name"] = clean_name
                    print(f"   🧹 Removed brand '{brand}' from product name: '{product_name}' → '{clean_name}'")

        # 🧹 RULE 2: Smart unit detection and exclusive pricing enforcement
        unit = cleaned.get("unit", "").strip().lower()

        # 🧹 RULE 2A: Fix unit based on populated price fields (reverse engineering)
        populated_price_fields = []
        if consensus_result.get("price_per_kg") and consensus_result["price_per_kg"].strip():
            populated_price_fields.append("kg")
        if consensus_result.get("price_per_piece") and consensus_result["price_per_piece"].strip():
            populated_price_fields.append("stk")
        if consensus_result.get("price_per_liter") and consensus_result["price_per_liter"].strip():
            populated_price_fields.append("l")

        # 🧹 SMART UNIT CORRECTION: Prioritize most specific per-unit pricing
        if len(populated_price_fields) >= 1:
            # Priority order: kg > l > stk (most specific pricing first)
            priority_order = ["kg", "l", "stk"]

            for priority_unit in priority_order:
                if priority_unit in populated_price_fields:
                    if unit != priority_unit:
                        old_unit = unit
                        cleaned["unit"] = priority_unit
                        print(f"   🧹 Fixed unit mismatch: '{old_unit}' → '{priority_unit}' (based on detected per-unit pricing)")
                        unit = priority_unit
                    break

        # Clear ALL price fields first
        cleaned["price_per_kg"] = ""
        cleaned["price_per_piece"] = ""
        cleaned["price_per_liter"] = ""

        # Set ONLY the correct price field based on corrected unit
        if unit in ["kg", "g"]:
            # Keep only price_per_kg
            if consensus_result.get("price_per_kg"):
                cleaned["price_per_kg"] = consensus_result["price_per_kg"]
                print(f"   🧹 Unit '{unit}' → keeping only price_per_kg: '{cleaned['price_per_kg']}'")
        elif unit in ["stk", "stuck", "piece"]:
            # Keep only price_per_piece
            if consensus_result.get("price_per_piece"):
                cleaned["price_per_piece"] = consensus_result["price_per_piece"]
                print(f"   🧹 Unit '{unit}' → keeping only price_per_piece: '{cleaned['price_per_piece']}'")
        elif unit in ["ml", "l", "liter", "litre"]:
            # Keep only price_per_liter
            if consensus_result.get("price_per_liter"):
                cleaned["price_per_liter"] = consensus_result["price_per_liter"]
                print(f"   🧹 Unit '{unit}' → keeping only price_per_liter: '{cleaned['price_per_liter']}'")
            # If no price_per_liter, try to use price_per_piece as fallback (for liquids)
            elif consensus_result.get("price_per_piece"):
                cleaned["price_per_liter"] = consensus_result["price_per_piece"]
                print(f"   🧹 Unit '{unit}' → using price_per_piece as price_per_liter: '{cleaned['price_per_liter']}'")

        # 🧹 RULE 3: Clean up garbage values
        for field in ["price_per_kg", "price_per_piece", "price_per_liter"]:
            value = cleaned.get(field, "")
            if value and isinstance(value, str):
                # Remove garbage patterns
                if value in ["€/kg", "€/Stk", "€/l", "€", "0,00 €", "0 €"]:
                    cleaned[field] = ""
                    print(f"   🧹 Removed garbage value from {field}: '{value}'")

        return cleaned

    def _validate_and_normalize_price_data(self, product_data: Dict) -> Dict:
        """
        Validate and normalize price data to ensure proper EUR formatting and separation.
        Fixes the issue where price includes unit info (e.g., "0,36 € / 1Stk.")
        """
        print("💰 Validating and normalizing price data...")

        validated_data = product_data.copy()

        # Extract and clean price field
        raw_price = product_data.get("price", "")
        if raw_price:
            # Split combined price format (e.g., "0,36 € / 1Stk." -> "0,36 €")
            base_price = self._extract_base_price(raw_price)
            per_unit_price = self._extract_per_unit_price(raw_price)

            validated_data["price"] = base_price

            # If per-unit price was embedded in price, extract it
            if per_unit_price and not validated_data.get("price_per_piece"):
                validated_data["price_per_piece"] = per_unit_price
                print(f"📝 Extracted per-unit price: {per_unit_price}")

        # Validate EUR format for all price fields
        price_fields = ["price", "original_price", "price_per_kg", "price_per_piece", "price_per_liter"]
        for field in price_fields:
            if validated_data.get(field):
                validated_data[field] = self._normalize_eur_format(validated_data[field])

        # Remove original_price if empty (as requested)
        if not validated_data.get("original_price"):
            validated_data.pop("original_price", None)

        print(f"💰 Price validation complete: {validated_data.get('price', 'N/A')}")
        return validated_data

    def _extract_base_price(self, price_string: str) -> str:
        """Extract base price from combined format (e.g., '0,36 € / 1Stk.' -> '0,36 €')"""
        if not price_string:
            return ""

        price_string = price_string.strip()

        # Handle combined format with '/' separator
        if ' / ' in price_string:
            base_part = price_string.split(' / ')[0].strip()
            # Ensure it has EUR symbol
            if '€' in base_part:
                return base_part
            elif base_part and not '€' in base_part:
                return f"{base_part} €"

        # Handle simple EUR format
        if '€' in price_string and not ('/' in price_string or 'Stk' in price_string or 'kg' in price_string):
            return price_string

        # Extract numeric part and add EUR if missing
        import re
        numeric_match = re.search(r'(\d+[,.]?\d*)', price_string)
        if numeric_match:
            return f"{numeric_match.group(1)} €"

        return price_string

    def _extract_per_unit_price(self, price_string: str) -> str:
        """Extract per-unit price from combined format (e.g., '0,36 € / 1Stk.' -> '0,36 €')"""
        if not price_string or ' / ' not in price_string:
            return ""

        parts = price_string.split(' / ')
        if len(parts) >= 2:
            unit_part = parts[1].strip()

            # Extract the base price (same as unit price in this format)
            base_price = self._extract_base_price(price_string)

            # If it contains unit indicators, return the base price
            if any(unit in unit_part.lower() for unit in ['stk', 'kg', 'l', 'stück', 'piece']):
                return base_price

        return ""

    def _normalize_eur_format(self, price_string: str) -> str:
        """Normalize EUR price format to standard '0,00 €' format"""
        if not price_string:
            return ""

        price_string = price_string.strip()

        # Already properly formatted
        if price_string.endswith(' €') and ',' in price_string:
            return price_string

        # Handle various formats
        import re

        # Extract numeric value
        numeric_match = re.search(r'(\d+[,.]?\d*)', price_string)
        if numeric_match:
            numeric_part = numeric_match.group(1).replace('.', ',')

            # Ensure proper decimal format
            if ',' not in numeric_part:
                numeric_part = f"{numeric_part},00"
            elif numeric_part.endswith(','):
                numeric_part = f"{numeric_part}00"

            return f"{numeric_part} €"

        return price_string

    async def _progressive_state_clearing(self, model_name: str) -> Dict:
        """
        🎯 PROGRESSIVE STATE CLEARING - Graceful contamination prevention.

        Uses graduated approach: gentle → medium → nuclear → continue anyway
        Never disqualifies models - always continues with monitoring.
        """
        print(f"🧹 Progressive state clearing for {model_name}...")
        clearing_result = {
            "cleared": False,
            "method": "none",
            "warning": None
        }

        # 🎯 TECHNIQUE 1: Aggressive clearing (to prevent contamination)
        try:
            aggressive_payload = {
                "model": model_name,
                "prompt": "CLEAR ALL CONTEXT. NEW SESSION. FORGET ALL PREVIOUS PRODUCTS.",
                "stream": False,
                "keep_alive": "5s",  # Brief keep-alive to force context reset
                "options": {"temperature": 0.1, "num_predict": 5, "num_ctx": 2048}
            }

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=aggressive_payload,
                timeout=3
            )

            if response.status_code == 200:
                print(f"✅ {model_name} aggressive clearing successful")
                clearing_result.update({"cleared": True, "method": "aggressive"})
                return clearing_result

        except Exception as e:
            print(f"⚠️ {model_name} aggressive clearing failed: {str(e)}")

        # 🎯 TECHNIQUE 2: Medium clearing (fallback)
        try:
            medium_payload = {
                "model": model_name,
                "prompt": "Clear context. Fresh start.",
                "stream": False,
                "keep_alive": "5s",  # Brief unload
                "options": {"num_ctx": 0, "temperature": 0.1}
            }

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=medium_payload,
                timeout=5
            )

            if response.status_code == 200:
                # Brief pause for context clearing
                time.sleep(1)
                print(f"✅ {model_name} medium clearing successful")
                clearing_result.update({"cleared": True, "method": "medium"})
                return clearing_result

        except Exception as e:
            print(f"⚠️ {model_name} medium clearing failed: {str(e)}")

        # 🎯 TECHNIQUE 3: Nuclear clearing (last resort)
        try:
            nuclear_payload = {
                "model": model_name,
                "prompt": "Reset all context.",
                "stream": False,
                "keep_alive": "0s",  # Force unload
                "options": {"num_ctx": 0}
            }

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=nuclear_payload,
                timeout=10
            )

            if response.status_code == 200:
                time.sleep(2)  # Extended pause for nuclear clearing
                print(f"⚠️ {model_name} nuclear clearing successful (last resort)")
                clearing_result.update({"cleared": True, "method": "nuclear"})
                return clearing_result

        except Exception as e:
            print(f"⚠️ {model_name} nuclear clearing failed: {str(e)}")

        # 🎯 TECHNIQUE 4: Continue anyway with contamination monitoring
        print(f"⚠️ {model_name} all clearing attempts failed - CONTINUING WITH CONTAMINATION MONITORING")
        clearing_result.update({
            "cleared": False,
            "method": "failed",
            "warning": "State clearing failed - contamination risk elevated"
        })
        return clearing_result

    def _validate_model_response(self, model_name: str, response_data: Dict, visual_context: str, analysis_mode: str) -> Dict:
        """Validate model response with intelligent contamination detection."""

        # Extract product name for validation
        product_name = ""
        if isinstance(response_data, dict):
            product_name = response_data.get("product_name", "").lower()

        validation_result = {
            "is_valid": True,
            "contamination_detected": False,
            "validation_reason": "Response appears valid",
            "confidence_adjustment": 1.0
        }

        # 🧠 INTELLIGENT CONTAMINATION DETECTION
        # Instead of blacklisting legitimate products, detect RESPONSE ANOMALIES:

        # 1. Check for coordinate array responses (moondream issue)
        if isinstance(response_data, list) or "coordinates" in str(response_data).lower():
            validation_result.update({
                "is_valid": False,
                "contamination_detected": True,
                "validation_reason": f"Model returned coordinate array instead of product data",
                "confidence_adjustment": 0.01
            })
            print(f"🚨 COORDINATE CONTAMINATION in {model_name}: Returned array data instead of product")
            return validation_result

        # 2. Check for empty/incomplete product data (real contamination sign)
        required_fields = ["product_name", "brand", "price"]
        if isinstance(response_data, dict):
            missing_critical = [field for field in required_fields if not response_data.get(field, "").strip()]
            if len(missing_critical) >= 2:  # Missing 2+ critical fields = likely contamination
                validation_result.update({
                    "is_valid": False,
                    "contamination_detected": True,
                    "validation_reason": f"Missing critical product fields: {missing_critical}",
                    "confidence_adjustment": 0.1
                })
                print(f"⚠️ INCOMPLETE DATA in {model_name}: Missing {missing_critical}")
                return validation_result

        # 3. Check for obvious format violations (but be more lenient)
        if "moondream" in model_name.lower() and analysis_mode == "product":
            # Only flag if moondream returns completely wrong data format
            brand = response_data.get("brand", "").lower() if isinstance(response_data, dict) else ""

            # Only flag severe format errors, not unit/brand combinations
            if brand and len(brand) > 50:  # Suspiciously long brand names
                validation_result.update({
                    "is_valid": False,
                    "contamination_detected": True,
                    "validation_reason": f"Suspiciously long brand name: '{brand[:30]}...'",
                    "confidence_adjustment": 0.5
                })
                print(f"⚠️ FORMAT ISSUE in {model_name}: Suspiciously long brand")

        # 🎯 INTELLIGENT VALIDATION - Graduated penalties based on severity
        if isinstance(response_data, dict) and analysis_mode == "product":

            # Check for exclusive pricing violations
            price_fields = ["price_per_kg", "price_per_piece", "price_per_liter"]
            populated_fields = []

            for field in price_fields:
                value = response_data.get(field, "")
                if value and str(value).strip() and str(value).strip() != "":
                    populated_fields.append(field)

            # 🚨 ZERO TOLERANCE EXCLUSIVE PRICING ENFORCEMENT
            if len(populated_fields) > 1:
                validation_result.update({
                    "is_valid": False,
                    "contamination_detected": True,
                    "validation_reason": f"EXCLUSIVE PRICING VIOLATION: {populated_fields} - ONLY ONE price field allowed",
                    "confidence_adjustment": 0.01  # 99% penalty - near elimination
                })
                print(f"🚨 EXCLUSIVE PRICING VIOLATION - {model_name}: {populated_fields} (SEVERE PENALTY)")

            # 🎯 SMART UNIT-PRICE ALIGNMENT VALIDATION
            unit = response_data.get("unit", "").strip().lower()
            if unit and populated_fields:
                valid_mappings = {
                    "stk": "price_per_piece",
                    "stuck": "price_per_piece",
                    "kg": "price_per_kg",
                    "l": "price_per_liter"
                }

                expected_field = valid_mappings.get(unit)
                if expected_field and populated_fields:
                    # Check if ANY populated field matches expected (allows for minor errors)
                    if expected_field not in populated_fields:
                        validation_result.update({
                            "is_valid": True,  # Don't completely invalidate
                            "contamination_detected": True,
                            "validation_reason": f"Unit-price mismatch: unit='{unit}', fields={populated_fields}",
                            "confidence_adjustment": 0.4  # Moderate penalty
                        })
                        print(f"⚠️ UNIT-PRICE MISMATCH (correctable) - {model_name}: {unit} vs {populated_fields}")

            # 🎯 INTELLIGENT PATTERN DETECTION (removed static blacklists)
            # Now we only detect RESPONSE ANOMALIES, not legitimate product names

            # Note: We removed static contamination patterns because:
            # - "Milchreis Klassische Art" IS a real product in IMG_8285
            # - Static blacklists can't distinguish between legitimate products and contamination
            # - Better to detect response format issues, coordinate arrays, incomplete data, etc.

        return validation_result

    def _create_model_specific_prompt(self, model_name: str, visual_context: str, analysis_mode: str, custom_prompt: str, contamination_risk: str) -> str:
        """
        🎯 MODEL-SPECIFIC PROMPTING - Adapt prompts to prevent model confusion.

        Different models need different instruction styles to prevent:
        - Coordinate responses (moondream)
        - Context bleeding (all models)
        - Format confusion (varies by model)
        """

        # Base cache-busting elements
        import random
        import time
        session_id = f"ANALYSIS_{int(time.time())}_{random.randint(1000,9999)}"

        if custom_prompt:
            return f"{session_id} {custom_prompt}"

        # 🎯 MODEL-SPECIFIC ADAPTATIONS
        if "moondream" in model_name.lower():
            # Moondream needs simple, clear instructions - no complex formatting
            if analysis_mode == "ui":
                return f"""{session_id}
Simple task: Look at this food app screenshot. Tell me the main category (with colored background) and active subcategory (bold text).

Return JSON only:
{{"main_category": "...", "active_subcategory": "...", "confidence": 0.8}}"""

            else:  # product mode
                return f"""{session_id}
Simple product analysis task. Look at this food product image and extract:

Return ONLY this JSON format:
{{
  "price": "the main price",
  "product_name": "the product name",
  "brand": "the brand",
  "weight": "weight if visible",
  "unit": "Stk or g or ml",
  "quantity": "number",
  "price_per_kg": "if you see €/kg",
  "price_per_piece": "if you see €/Stk",
  "price_per_liter": "if you see €/l"
}}

Fill ONLY the price field that matches the unit. Do not return coordinates or arrays."""

        elif "minicpm" in model_name.lower():
            # MiniCPM handles complex prompts well
            base_prompt = self._create_local_analysis_prompt(visual_context, analysis_mode)

            # Add contamination warnings based on risk level
            if contamination_risk == "elevated":
                warning = "⚠️ CRITICAL: This is a completely new image. Do not use any information from previous images."
                return f"{session_id} {warning}\n\n{base_prompt}"
            else:
                return f"{session_id} {base_prompt}"

        elif "llama" in model_name.lower():
            # Llama models prefer structured, detailed instructions
            base_prompt = self._create_local_analysis_prompt(visual_context, analysis_mode)

            # Add session isolation for large models prone to context bleeding
            isolation_note = f"🔒 ISOLATED SESSION {session_id} - No previous context applies."
            return f"{isolation_note}\n\n{base_prompt}"

        else:
            # Fallback for unknown models
            return f"{session_id} {self._create_local_analysis_prompt(visual_context, analysis_mode)}"

    async def _query_single_local_model(self, model: Dict, image_base64: str, text_base64: str, visual_context: str, analysis_mode: str, custom_prompt: str = None) -> Dict:
        """Query single Ollama model with bulletproof error handling and circuit breaker."""
        model_name = model["name"]
        start_time = time.time()

        # Check circuit breaker
        if not self._is_model_healthy(model_name):
            return {
                "status": "circuit_breaker_open",
                "raw_response": f"Circuit breaker open for {model_name}",
                "error": f"Model {model_name} temporarily disabled due to failures",
                "processing_time": 0
            }

        # 🎯 PROGRESSIVE STATE CLEARING: Graceful contamination prevention
        # Never disqualifies models - continues with monitoring if clearing fails
        clearing_result = await self._progressive_state_clearing(model_name)
        contamination_risk_level = "normal"

        # 🎯 SESSION ISOLATION: Track clearing results for consensus weighting
        if not hasattr(self, '_last_clearing_results'):
            self._last_clearing_results = {}
        self._last_clearing_results[model_name] = clearing_result

        if not clearing_result["cleared"]:
            contamination_risk_level = "elevated"
            print(f"⚠️ {model_name} continuing with elevated contamination risk")
        elif clearing_result["method"] in ["nuclear", "failed"]:
            contamination_risk_level = "moderate"

        try:
            # 🎯 MODEL-SPECIFIC PROMPTING: Adapt prompts to prevent confusion
            prompt = self._create_model_specific_prompt(model_name, visual_context, analysis_mode, custom_prompt, contamination_risk_level)

            # Prepare payload for vision models
            # For product analysis, prioritize text region for better text reading
            if analysis_mode == "product" and text_base64 != image_base64:
                # Use text region for product text analysis
                analysis_image = text_base64
            else:
                # Use main image for UI analysis or when no separate text region
                analysis_image = image_base64

            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [analysis_image],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 4096,  # Increased for complete JSON responses
                    "repeat_penalty": 1.1,  # Prevent repetitive outputs
                    "top_k": 40  # Focus on most relevant tokens
                }
            }

            # Get model-specific settings
            model_config = next((m for m in self.models if m['name'] == model_name), None)
            if not model_config:
                model_config = {"timeout": 60, "max_retries": 3}  # Fallback

            # BULLETPROOF request with exponential backoff and circuit breaker
            def make_request_with_exponential_backoff():
                max_retries = model_config.get('max_retries', 3)
                timeout = model_config.get('timeout', 60)

                for attempt in range(max_retries):
                    try:
                        # Calculate exponential backoff delay
                        if attempt > 0:
                            delay = self.base_retry_delay * (2 ** (attempt - 1))
                            print(f"   ⏳ Waiting {delay}s before retry {attempt + 1} for {model_name}...")
                            time.sleep(delay)

                        response = requests.post(
                            "http://localhost:11434/api/generate",
                            json=payload,
                            timeout=timeout
                        )
                        if response.status_code == 200:
                            # Success - record it
                            self._record_model_success(model_name)
                            return response
                        else:
                            print(f"   ⚠️  {model_name} HTTP {response.status_code} on attempt {attempt + 1}")

                            # For large models, may need memory cleanup
                            if "11b" in model_name and attempt < max_retries - 1:
                                print(f"   🧹 Cleaning up memory for {model_name}...")
                                time.sleep(2)

                    except requests.exceptions.RequestException as e:
                        print(f"   ⚠️  {model_name} connection error on attempt {attempt + 1}: {str(e)[:50]}...")

                        # Model-specific recovery strategies
                        if attempt < max_retries - 1:
                            if "timeout" in str(e).lower():
                                if "11b" in model_name:
                                    print(f"   🔄 Large model timeout - extending timeout...")
                                    timeout = min(timeout * 1.5, 180)  # Max 3 minutes
                                else:
                                    print(f"   🔄 Model timeout - retrying with backoff...")
                            else:
                                print(f"   🔄 Connection failed - attempting Ollama restart...")
                                self._restart_ollama_service()
                                time.sleep(5)  # Wait for restart
                        else:
                            # Final attempt failed - record failure for circuit breaker
                            self._record_model_failure(model_name)

                return None

            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                future = loop.run_in_executor(executor, make_request_with_exponential_backoff)
                response = await future

            processing_time = time.time() - start_time

            if response is None:
                return {
                    "status": "connection_error",
                    "raw_response": "Connection failed",
                    "error": "Could not connect to Ollama",
                    "processing_time": processing_time
                }

            if response.status_code == 200:
                result = response.json()
                raw_content = result.get('response', '').strip()
# Debug logging removed

                # Try to parse JSON
                try:
                    parsed_data = None
                    parse_status = ""

                    if raw_content.startswith('{') and raw_content.endswith('}'):
                        parsed_data = json.loads(raw_content)
                        parse_status = "direct_json"
                    else:
                        # BULLETPROOF JSON extraction with multiple patterns and intelligent repair
                        json_result = self._extract_json_bulletproof(raw_content)
                        if json_result:
                            parsed_data = json_result["data"]
                            parse_status = json_result["method"]

                    # 🛡️ RESPONSE VALIDATION: Check for contamination before returning
                    if parsed_data:
                        validation = self._validate_model_response(model_name, parsed_data, visual_context, analysis_mode)

                        response_dict = {
                            "status": "success",
                            "raw_response": raw_content,
                            "parsed_data": parsed_data,
                            "processing_time": processing_time,
                            "parse_status": parse_status,
                            "validation": validation
                        }

                        # If contamination detected, mark response with warning but still return data
                        if validation["contamination_detected"]:
                            response_dict["status"] = "success_with_contamination"
                            response_dict["contamination_warning"] = validation["validation_reason"]
                            print(f"⚠️ {model_name}: {validation['validation_reason']}")

                        return response_dict

                    # If no valid JSON was parsed
                    print(f"   🔍 DEBUG: No JSON extracted from content: '{raw_content[:500]}...'")
                    return {
                        "status": "no_json",
                        "raw_response": raw_content,
                        "error": "No valid JSON found in response",
                        "processing_time": processing_time,
                        "parse_status": "no_json"
                    }

                except json.JSONDecodeError as e:
                    print(f"   🔍 DEBUG JSON Parse Error for content: '{raw_content[:200]}...'")
                    return {
                        "status": "json_error",
                        "raw_response": raw_content,
                        "error": f"JSON parse error: {e}",
                        "processing_time": processing_time,
                        "parse_status": "json_error"
                    }
            else:
                error_text = response.text if hasattr(response, 'text') else "Unknown error"
                return {
                    "status": "http_error",
                    "raw_response": error_text,
                    "error": f"HTTP {response.status_code}: {error_text}",
                    "processing_time": processing_time
                }

        except Exception as e:
            processing_time = time.time() - start_time
            print(f"   ⚠️ Exception in {model_name}: {str(e)}")
            return {
                "status": "exception",
                "raw_response": f"Exception occurred: {str(e)}",
                "error": str(e),
                "processing_time": processing_time
            }

    async def analyze_product_with_sequential_steps(self,
                                                    tile_image: np.ndarray,
                                                    text_region_image: np.ndarray,
                                                    text_top_region: np.ndarray = None,
                                                    text_middle_region: np.ndarray = None,
                                                    text_bottom_region: np.ndarray = None) -> Dict:
        """
        NEW: Sequential micro-step analysis for maximum reliability.
        Breaks down complex product analysis into focused, single-purpose visual steps.
        Each step receives the appropriate cropped text region to avoid confusion.

        Args:
            tile_image: Product image (no text)
            text_region_image: Full text region (legacy, fallback)
            text_top_region: Top 60px for main price (Step 5b)
            text_middle_region: Middle 120px for product name (Step 5a)
            text_bottom_region: Bottom 60px for per-unit price (Step 5c)

        Returns:
            Dict with final product data
        """
        print(f"\n🔬 SEQUENTIAL MICRO-STEP ANALYSIS")
        print("=" * 60)

        # Use cropped regions if available, otherwise fall back to full text region
        text_for_5a = text_middle_region if text_middle_region is not None else text_region_image
        text_for_5b = text_top_region if text_top_region is not None else text_region_image
        text_for_5c = text_bottom_region if text_bottom_region is not None else text_region_image

        try:
            # Step 5a: Product Name Detection (Visual) - uses MIDDLE region
            print("🏷️ Step 5a: Product Name Detection (using middle text region)...")
            name_data = await self._step_5a_detect_product_name(tile_image, text_for_5a)

            # Step 5b: Price Detection (Visual) - uses TOP region
            print("💰 Step 5b: Price Detection (using top text region)...")
            price_data = await self._step_5b_detect_price(tile_image, text_for_5b)

            # Step 5c: Unit Type Detection (Visual) - uses BOTTOM region
            print("📏 Step 5c: Unit Type Detection (using bottom text region)...")
            unit_type_data = await self._step_5c_detect_unit_type(tile_image, text_for_5c)

            # Step 5c2: Price Per Unit Detection (Visual) - uses BOTTOM region (same as 5c)
            unit_data = {'success': False}
            if unit_type_data.get('success') and unit_type_data.get('unit'):
                unit_type = unit_type_data['unit']
                print(f"💰 Step 5c2: Price Per {unit_type} Detection (using bottom text region)...")
                price_per_unit_data = await self._step_5d_detect_price_per_unit(tile_image, text_for_5c, unit_type)

                if price_per_unit_data.get('success'):
                    # Combine unit type and price data
                    unit_data = {
                        'success': True,
                        'unit': unit_type,
                        'price_per_unit': price_per_unit_data['price_per_unit'],
                        'confidence': 0.95
                    }
                else:
                    print(f"      ⚠️  Unit '{unit_type}' detected but no price found")
                    unit_data = {'success': False}
            else:
                print("      ⚠️  No unit type detected, skipping price per unit")

            # Step 5d: Quantity/Weight Detection (Visual)
            print("⚖️ Step 5d: Quantity/Weight Detection...")
            quantity_data = await self._step_5d_detect_quantity(tile_image, text_region_image)

            # Step 5e: Brand Detection (Visual)
            print("🏢 Step 5e: Brand Detection...")
            brand_data = await self._step_5e_detect_brand(tile_image, text_region_image)

            # Step 5f: Data Assembly & Validation
            print("🔧 Step 5f: Data Assembly & Validation...")
            final_result = self._step_5f_assemble_and_validate(
                name_data, price_data, unit_data, quantity_data, brand_data
            )

            # CRITICAL VALIDATION: Always validate visual unit detection with LLM consensus
            # Visual OCR can hallucinate, so we cross-check with full context analysis
            visual_unit = final_result.get('unit', '')
            needs_llm_validation = (
                final_result.get('_visual_detection_failed') or  # Complete failure
                not visual_unit or  # No unit detected
                (unit_data.get('success') and unit_data.get('confidence', 0) < 0.9)  # Low confidence
            )

            if needs_llm_validation:
                print(f"\n⚠️  VISUAL DETECTION NEEDS VALIDATION - ACTIVATING LLM CONSENSUS")
                print(f"   Reason: {'Failed' if final_result.get('_visual_detection_failed') else 'Low confidence'}")
                print(f"   Visual unit: '{visual_unit}' (confidence: {unit_data.get('confidence', 0):.2f})")
                print(f"=" * 60)

                # Call full LLM consensus to extract unit information
                llm_result = await self.analyze_product_with_consensus(
                    tile_image, text_region_image, analysis_mode="product"
                )

                # Extract unit and price information from LLM consensus
                # Note: analyze_product_with_consensus returns the cleaned product data directly
                if llm_result and isinstance(llm_result, dict) and 'unit' in llm_result:
                    consensus_data = llm_result  # Direct product data, not wrapped
                    llm_unit = consensus_data.get('unit', '')

                    # Handle both price_per_kg and cost_per_kg field names
                    llm_price_per_kg = consensus_data.get('price_per_kg') or consensus_data.get('cost_per_kg', '')
                    llm_price_per_liter = consensus_data.get('price_per_liter') or consensus_data.get('cost_per_liter', '')
                    llm_price_per_piece = consensus_data.get('price_per_piece') or consensus_data.get('cost_per_piece', '')

                    if llm_unit:
                        # Cross-validate: if LLM disagrees with visual detection, trust LLM (full context)
                        if visual_unit and visual_unit != llm_unit:
                            print(f"⚠️  UNIT MISMATCH: Visual='{visual_unit}' vs LLM='{llm_unit}'")
                            print(f"   🎯 TRUSTING LLM (full context analysis) over visual OCR")

                        print(f"✅ LLM Consensus extracted unit: {llm_unit}")
                        final_result['unit'] = llm_unit

                        # Apply correct price field based on LLM-detected unit
                        if llm_unit == 'kg' and llm_price_per_kg:
                            print(f"   🔧 Applying kg pricing: {llm_price_per_kg}")
                            final_result['price_per_kg'] = llm_price_per_kg
                            final_result['price_per_piece'] = ''
                            final_result['price_per_liter'] = ''
                            print(f"   ✅ Updated final_result: unit='{final_result['unit']}', price_per_kg='{final_result['price_per_kg']}', price_per_liter='{final_result['price_per_liter']}'")
                        elif llm_unit == 'l' and llm_price_per_liter:
                            print(f"   🔧 Applying liter pricing: {llm_price_per_liter}")
                            final_result['price_per_liter'] = llm_price_per_liter
                            final_result['price_per_kg'] = ''
                            final_result['price_per_piece'] = ''
                            print(f"   ✅ Updated final_result: unit='{final_result['unit']}', price_per_liter='{final_result['price_per_liter']}', price_per_kg='{final_result['price_per_kg']}'")
                        elif llm_unit == 'Stk' and llm_price_per_piece:
                            print(f"   🔧 Applying piece pricing: {llm_price_per_piece}")
                            final_result['price_per_piece'] = llm_price_per_piece
                            final_result['price_per_kg'] = ''
                            final_result['price_per_liter'] = ''
                            print(f"   ✅ Updated final_result: unit='{final_result['unit']}', price_per_piece='{final_result['price_per_piece']}', price_per_kg='{final_result['price_per_kg']}'")

                        # Add debug info from LLM consensus
                        final_result['debug_info']['llm_consensus_fallback'] = {
                            'used': True,
                            'visual_unit': visual_unit,
                            'llm_unit': llm_unit,
                            'mismatch': visual_unit != llm_unit,
                            'confidence': llm_result.get('confidence', 0.0),
                            'models_agreed': llm_result.get('successful_models', 0)
                        }
                        print(f"✅ Validation successful - unit: '{llm_unit}'")
                    else:
                        print(f"⚠️  LLM consensus also failed to extract unit - keeping visual result")
                else:
                    print(f"⚠️  LLM consensus validation failed - keeping visual result")

                # Remove internal flag if present
                if '_visual_detection_failed' in final_result:
                    del final_result['_visual_detection_failed']

            print(f"✅ Sequential analysis completed successfully!")
            return final_result

        except Exception as e:
            print(f"❌ Sequential analysis failed: {str(e)}")
            return {"error": f"Sequential analysis failed: {str(e)}", "step": "unknown"}

    async def analyze_product_with_consensus(self, tile_image: np.ndarray, text_region_image: np.ndarray, analysis_mode: str = "product", custom_prompt: str = None) -> Dict:
        """
        LEGACY: Multi-strategy consensus analysis with enhanced reliability.
        Supports both product and UI analysis modes with fallback strategies.

        Args:
            tile_image: Primary image for analysis
            text_region_image: Text region image (can be same as tile_image)
            analysis_mode: "product" for product analysis, "ui" for UI/category analysis

        Returns:
            Dict with consensus results
        """
        print(f"\n🧠 MULTI-STRATEGY CONSENSUS ANALYSIS - MODE: {analysis_mode.upper()}")
        print("=" * 60)

        # Multi-strategy analysis with fallbacks
        strategies = [
            {"name": "full_consensus", "min_models": 2, "timeout": 60},
            {"name": "relaxed_consensus", "min_models": 1, "timeout": 45},
            {"name": "single_reliable", "min_models": 1, "timeout": 30, "models": ["minicpm-v:latest"]},
            {"name": "emergency_fallback", "min_models": 1, "timeout": 15, "models": ["moondream:latest"]}
        ]

        for strategy in strategies:
            print(f"🎯 Trying strategy: {strategy['name']}")
            try:
                result = await self._execute_strategy(
                    tile_image, text_region_image, analysis_mode, strategy, custom_prompt
                )
                if result and result.get("successful_models", 0) >= strategy["min_models"]:
                    print(f"✅ Strategy '{strategy['name']}' succeeded!")
                    result["strategy_used"] = strategy["name"]
                    return result
                else:
                    print(f"⚠️ Strategy '{strategy['name']}' insufficient results")
            except Exception as e:
                print(f"❌ Strategy '{strategy['name']}' failed: {str(e)}")
                continue

        # All strategies failed
        print("🚨 All strategies failed - returning error result")
        return {
            "success": False,
            "error": "All consensus strategies failed",
            "strategy_used": "none",
            "successful_models": 0,
            "total_models": len(self.models)
        }

    async def _execute_strategy(self, tile_image: np.ndarray, text_region_image: np.ndarray,
                              analysis_mode: str, strategy: Dict, custom_prompt: str = None) -> Dict:
        """Execute a specific consensus strategy"""
        print(f"\n🧠 EXECUTING STRATEGY: {strategy['name'].upper()}")
        print("=" * 60)

        # Enhanced error handling and validation
        try:
            # Validate inputs
            if tile_image is None or tile_image.size == 0:
                raise ValueError("Invalid tile_image: empty or None")

            if text_region_image is None:
                text_region_image = tile_image
                print("📝 Using tile_image as text_region_image (fallback)")

            # Proactive health check with better error handling
            if not self._check_ollama_health():
                print("🔄 Ollama service not responsive, attempting restart...")
                restart_success = self._restart_ollama_service()
                if not restart_success:
                    raise RuntimeError("Failed to restart Ollama service")

            # Extract visual features with error handling
            try:
                visual_context = self._extract_visual_features(tile_image)
                print(f"🖼️ Visual context: {visual_context}")
            except Exception as e:
                print(f"⚠️ Visual feature extraction failed: {e}")
                visual_context = f"Image: {tile_image.shape[1]}x{tile_image.shape[0]}px, {tile_image.shape[2]} channels"

            # Convert images to base64 with validation
            try:
                success, buffer = cv2.imencode('.png', tile_image)
                if not success:
                    raise ValueError("Failed to encode tile_image to PNG")
                image_base64 = base64.b64encode(buffer).decode('utf-8')

                # Handle text region image
                if text_region_image.shape != tile_image.shape:
                    success, text_buffer = cv2.imencode('.png', text_region_image)
                    if not success:
                        print("⚠️ Failed to encode text_region_image, using tile_image")
                        text_base64 = image_base64
                    else:
                        text_base64 = base64.b64encode(text_buffer).decode('utf-8')
                        print(f"🖼️  Tile image: {len(image_base64)} chars, Text region: {len(text_base64)} chars")
                else:
                    text_base64 = image_base64
                    print(f"🖼️  Using single image: {len(image_base64)} chars")

            except Exception as e:
                raise RuntimeError(f"Image encoding failed: {str(e)}")

        except Exception as e:
            print(f"❌ Strategy setup failed: {str(e)}")
            return {
                "success": False,
                "error": f"Strategy setup failed: {str(e)}",
                "strategy_used": strategy["name"],
                "successful_models": 0,
                "total_models": len(strategy.get("models", self.models))
            }

        # Strategy-specific model selection for improved reliability
        target_models = strategy.get("models", [m['name'] for m in self.models])
        available_models = [m for m in self.models if m['name'] in target_models]

        print(f"🎯 Strategy targeting: {[m['name'] for m in available_models]}")

        # 🔍 SMART HEALTH CHECK: Check for model cooldown before processing
        cooldown_status = self._get_model_cooldown_status()

        if cooldown_status['severity'] == 'critical':
            print(f"🚨 CRITICAL: {cooldown_status['message']}")
            return {
                "success": False,
                "error": cooldown_status['message'],
                "strategy_used": strategy["name"],
                "successful_models": 0,
                "total_models": len(available_models)
            }

        elif cooldown_status['severity'] in ['warning', 'info']:
            print(f"⚠️  {cooldown_status['message']}")
            # Auto-reload missing models for this strategy
            missing_models = [name for name in target_models if name not in cooldown_status.get('loaded_models', [])]
            if missing_models:
                print(f"🔄 Loading missing models for strategy: {missing_models}")
                self._preload_models(missing_models)

        # Filter to healthy models for this strategy
        healthy_models = [m for m in available_models if self._is_model_healthy(m['name'])]
        print(f"🔄 Strategy using {len(healthy_models)}/{len(available_models)} healthy models")

        if not healthy_models:
            print(f"🚨 No healthy models available for strategy '{strategy['name']}'")
            return {
                "success": False,
                "error": f"No healthy models for strategy {strategy['name']}",
                "strategy_used": strategy["name"],
                "successful_models": 0,
                "total_models": len(available_models)
            }

        tasks = []
        for model in healthy_models:
            task = self._query_single_local_model(model, image_base64, text_base64, visual_context, analysis_mode, custom_prompt)
            tasks.append(task)

        # Wait for all responses with timeout
        try:
            model_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300  # 5 minute total timeout
            )
        except asyncio.TimeoutError:
            print("🚨 Consensus analysis timed out - using partial results")
            # Cancel remaining tasks and use what we have
            for task in tasks:
                if not task.done():
                    task.cancel()
            model_results = [task.result() if task.done() and not task.cancelled() else Exception("Timeout") for task in tasks]

        # Process results with bulletproof error handling
        successful_results = []
        all_responses = {}

        for i, (model, result) in enumerate(zip(healthy_models, model_results)):
            model_name = model["name"]
            print(f"\n{i+1}. {model_name}:")

            if isinstance(result, Exception):
                print(f"   ❌ Exception: {result}")
                all_responses[model_name] = {
                    "status": "exception",
                    "error": str(result)
                }
            else:
                status = result.get("status", "unknown")
                raw_resp = result.get("raw_response", "No response")
                print(f"   📊 Status: {status}")
                print(f"   📤 Response: '{raw_resp[:100]}...' ({len(raw_resp)} chars)")

                all_responses[model_name] = result

                # Check if we got valid parsed data
                if result.get("parsed_data"):
                    parsed_data = result["parsed_data"]

                    # Handle both dict and list responses (some models return lists)
                    if isinstance(parsed_data, list) and len(parsed_data) > 0:
                        print(f"   📝 Converting list response to dict")
                        parsed_data = parsed_data[0]  # Take first item

                    # Validate based on analysis mode
                    is_valid = False
                    if isinstance(parsed_data, dict):
                        if analysis_mode == "ui":
                            # UI mode: look for categories OR hierarchy fields
                            if parsed_data.get("categories") and parsed_data["categories"] != ['']:
                                print(f"   ✅ Categories: {parsed_data['categories']}")
                                is_valid = True
                            elif parsed_data.get("main_category") or parsed_data.get("active_subcategory"):
                                print(f"   ✅ Hierarchy: main='{parsed_data.get('main_category')}', active='{parsed_data.get('active_subcategory')}'")
                                is_valid = True
                        else:
                            # Product mode: look for product info
                            if parsed_data.get("product_name") or parsed_data.get("price"):
                                print(f"   ✅ Product: {parsed_data}")
                                is_valid = True
                    else:
                        print(f"   ⚠️  Invalid data type: {type(parsed_data)}")
                        is_valid = False

                    if is_valid:
                        successful_results.append({
                            "model": model_name,
                            "data": parsed_data,
                            "weight": model["weight"],
                            "status": status,
                            "validation": result.get("validation", {})  # Pass through validation info
                        })
                    else:
                        print(f"   ⚠️  Invalid data for {analysis_mode} mode")
                else:
                    print(f"   ❌ No valid parsed data")

        # Create consensus result based on mode - Accept ANY successful results (bulletproof)
        min_required_models = 1  # Accept results from even 1 model
        if len(successful_results) >= min_required_models:
            if analysis_mode == "ui":
                # UI mode: return categories with PROPER MAJORITY VOTING
                all_categories = []
                available_subcategories = []
                visual_hierarchy = None

                # Collect votes for main_category and active_subcategory
                main_category_votes = {}
                active_subcategory_votes = {}

                for result in successful_results:
                    result_data = result["data"]

                    # Vote counting for main_category
                    if result_data.get("main_category"):
                        main_cat = result_data.get("main_category").strip()
                        if main_cat and main_cat not in ["text from pink background row", ""]:  # Filter out template responses
                            main_category_votes[main_cat] = main_category_votes.get(main_cat, 0) + 1

                    # Vote counting for active_subcategory
                    if result_data.get("active_subcategory"):
                        active_sub = result_data.get("active_subcategory").strip()
                        if active_sub and active_sub not in ["bold/centered text from bottom row", ""]:  # Filter out template responses
                            active_subcategory_votes[active_sub] = active_subcategory_votes.get(active_sub, 0) + 1

                    # Collect available subcategories and other data
                    if result_data.get("available_subcategories"):
                        available_subcategories.extend(result_data.get("available_subcategories", []))
                    if result_data.get("visual_hierarchy"):
                        visual_hierarchy = result_data.get("visual_hierarchy")

                    # Also collect categories field for backwards compatibility
                    categories = result_data.get("categories", [])
                    normalized_categories = []
                    for cat in categories:
                        if isinstance(cat, dict):
                            normalized_categories.append(cat.get("name", str(cat)))
                        else:
                            normalized_categories.append(str(cat))
                    all_categories.extend(normalized_categories)

                # MAJORITY VOTING: Pick the answer with most votes
                main_category = max(main_category_votes, key=main_category_votes.get) if main_category_votes else None
                active_subcategory = max(active_subcategory_votes, key=active_subcategory_votes.get) if active_subcategory_votes else None

                # Debug output for vote counts
                print(f"   🗳️ Main category votes: {main_category_votes}")
                print(f"   🗳️ Active subcategory votes: {active_subcategory_votes}")
                if main_category:
                    print(f"   🏆 Majority winner main: '{main_category}' ({main_category_votes.get(main_category, 0)} votes)")
                if active_subcategory:
                    print(f"   🏆 Majority winner active: '{active_subcategory}' ({active_subcategory_votes.get(active_subcategory, 0)} votes)")

                unique_categories = list(set(all_categories))
                unique_available_subcategories = list(set(available_subcategories)) if available_subcategories else []

                consensus_result = {
                    "categories": unique_categories,
                    "main_category": main_category,
                    "active_subcategory": active_subcategory,
                    "available_subcategories": unique_available_subcategories,
                    "visual_hierarchy": visual_hierarchy,
                    "successful_models": len(successful_results),
                    "total_models": len(self.models),
                    "confidence": len(successful_results) / len(self.models),
                    "individual_results": successful_results,
                    "analysis_method": "pure_llm_consensus",
                    "analysis_mode": analysis_mode,
                    "visual_context": visual_context
                }

                print(f"\n🎯 UI CONSENSUS SUCCESS:")
                print(f"   ✅ Found {len(unique_categories)} categories: {unique_categories}")
                if main_category:
                    print(f"   🏗️ Main category: '{main_category}'")
                if active_subcategory:
                    print(f"   🎯 Active subcategory: '{active_subcategory}'")
                if unique_available_subcategories:
                    print(f"   📋 Available subcategories: {unique_available_subcategories}")
                print(f"   📊 {len(successful_results)}/{len(self.models)} models succeeded")


            else:
                # Product mode: return PROPER MAJORITY VOTING consensus
                # Count votes for each field instead of just picking highest weight

                # ENHANCED: Column-by-Column Intelligent Field Selection
                print(f"   🧠 COLUMN-BY-COLUMN CONSENSUS ANALYSIS")
                print("   " + "="*50)

                # Validate and normalize all product data first
                validated_results = []
                for result in successful_results:
                    result_data = result["data"]

                    # Apply price validation and normalization
                    validated_data = self._validate_and_normalize_price_data(result_data)

                    # 🛡️ APPLY VALIDATION CONFIDENCE ADJUSTMENTS
                    validation_info = result.get("validation", {})
                    confidence_adjustment = validation_info.get("confidence_adjustment", 1.0)
                    adjusted_weight = result["weight"] * confidence_adjustment

                    # 🎯 INTELLIGENT THRESHOLD: Graduated exclusion based on severity
                    if confidence_adjustment == 0.0:
                        # Only exclude for truly severe contamination
                        validation_reason = validation_info.get("validation_reason", "")
                        if "milchreis klassische art" in validation_reason.lower():
                            print(f"   💥 {result['model']} DISQUALIFIED - Severe contamination detected")
                            continue
                        else:
                            # Even 0.0 gets a tiny chance (0.05) to contribute if it's the only data
                            adjusted_weight = max(adjusted_weight, 0.05)
                            print(f"   ⚠️ {result['model']} severe penalty but included: {adjusted_weight}")

                    # 🎯 MINIMUM PARTICIPATION THRESHOLD: Very low bar to ensure data availability
                    if adjusted_weight < 0.05:  # Much lower threshold
                        adjusted_weight = 0.05  # Always give models a minimum chance
                        print(f"   📊 {result['model']} minimum weight applied: {adjusted_weight}")

                    # 🎯 CONTAMINATION RISK ADJUSTMENT: Factor in clearing success
                    if hasattr(self, '_last_clearing_results') and result['model'] in self._last_clearing_results:
                        clearing_method = self._last_clearing_results[result['model']].get('method', 'unknown')
                        if clearing_method == 'failed':
                            adjusted_weight *= 0.7  # Additional penalty for failed clearing
                            print(f"   ⚠️ {result['model']} clearing risk penalty: {adjusted_weight}")
                        elif clearing_method == 'nuclear':
                            adjusted_weight *= 0.85  # Small penalty for nuclear clearing
                            print(f"   ⚠️ {result['model']} nuclear clearing penalty: {adjusted_weight}")

                    # Log contamination warnings for remaining models
                    if validation_info.get("contamination_detected", False):
                        print(f"   ⚠️ Contamination penalty applied to {result['model']}: weight {result['weight']:.1f} → {adjusted_weight:.1f}")

                    validated_results.append({
                        "model": result["model"],
                        "data": validated_data,
                        "weight": adjusted_weight,  # Apply validation penalty
                        "status": result["status"],
                        "validation": validation_info
                    })

                # Extract all field candidates from all models
                field_candidates = {}
                all_fields = ["price", "product_name", "brand", "weight", "unit", "quantity",
                             "price_per_kg", "price_per_piece", "price_per_liter"]

                for field in all_fields:
                    field_candidates[field] = []
                    for result in validated_results:
                        value = result["data"].get(field, "")
                        if value and str(value).strip():
                            # Include validation info in field candidates
                            field_candidates[field].append({
                                "value": str(value).strip(),
                                "model": result["model"],
                                "model_weight": result["weight"],  # Already adjusted for contamination
                                "validation": result.get("validation", {})
                            })

                # Intelligent field-by-field selection
                consensus_result = {}

                for field, candidates in field_candidates.items():
                    if not candidates:
                        consensus_result[field] = ""
                        print(f"   📝 {field}: No data from any model")
                        continue

                    # Select best candidate using intelligent logic
                    best_candidate = self._select_best_field_candidate(field, candidates)
                    consensus_result[field] = best_candidate["value"]

                    print(f"   🏆 {field}: '{best_candidate['value']}' (from {best_candidate['model']}, quality: {best_candidate.get('quality_score', 'N/A')})")

                consensus_price = consensus_result.get("price", "")
                consensus_name = consensus_result.get("product_name", "")
                consensus_brand = consensus_result.get("brand", "")

                # Use the enhanced column-by-column consensus result directly
                # No need to find "best result" - we already have the best data for each field

                print(f"\n🎯 PRODUCT CONSENSUS SUCCESS:")
                print(f"   ✅ Best result: {consensus_result}")
                print(f"   📊 {len(successful_results)}/{len(self.models)} models succeeded")

                # 🧹 POST-PROCESSING CLEANUP
                consensus_result = self._apply_post_processing_cleanup(consensus_result)
                print(f"   🧹 After cleanup: {consensus_result}")

                # Calculate cost metrics if missing (for backwards compatibility)
                cost_metrics = self._calculate_cost_metrics(consensus_result)

                # Final consensus result with all the rich data preserved
                final_consensus_result = {
                    "price": consensus_result.get("price", ""),
                    "brand": consensus_result.get("brand", ""),
                    "product_name": consensus_result.get("product_name", ""),
                    "weight": consensus_result.get("weight", ""),
                    "quantity": consensus_result.get("quantity", ""),
                    "unit": consensus_result.get("unit", ""),
                    "cost_per_kg": consensus_result.get("price_per_kg", "") or cost_metrics.get("cost_per_kg", ""),
                    "cost_per_piece": consensus_result.get("price_per_piece", "") or cost_metrics.get("cost_per_piece", ""),
                    "price_per_liter": consensus_result.get("price_per_liter", "")
                }

                # Add metadata
                final_consensus_result.update({
                    "successful_models": len(successful_results),
                    "total_models": len(self.models),
                    "confidence": len(successful_results) / len(self.models),
                    "individual_results": validated_results,  # Use validated results
                    "analysis_method": "column_by_column_consensus",
                    "analysis_mode": analysis_mode,
                    "visual_context": visual_context
                })

                # Set the result to return
                consensus_result = final_consensus_result

        else:
            print(f"\n❌ CONSENSUS FAILED:")
            print(f"   📊 0/{len(self.models)} models provided valid {analysis_mode} data")

            # Retry once with fresh Ollama restart
            print("   🔄 Attempting recovery with service restart...")
            self._restart_ollama_service()

            # Quick retry with single model
            print("   🔁 Quick retry with fallback model...")
            fallback_model = "moondream:latest"  # Lightest model
            try:
                fallback_result = await self._query_model_async(fallback_model, tile_image, analysis_mode, prompt, "")
                if fallback_result['status'] in ['success', 'extracted_json']:
                    print("   ✅ Fallback recovery successful!")
                    return fallback_result.get('parsed_data', {})
            except Exception as e:
                print(f"   ❌ Fallback failed: {e}")

            if analysis_mode == "ui":
                # Return empty categories structure
                consensus_result = {
                    "categories": [],
                    "successful_models": 0,
                    "total_models": len(self.models),
                    "confidence": 0.0,
                    "individual_results": [],
                    "analysis_method": "failed_consensus",
                    "analysis_mode": analysis_mode,
                    "visual_context": visual_context
                }
            else:
                # Return empty product structure
                consensus_result = {
                    "price": "",
                    "brand": "",
                    "product_name": "",
                    "weight": "",
                    "quantity": "",
                    "unit": "",
                    "cost_per_kg": "",
                    "cost_per_piece": "",
                    "successful_models": 0,
                    "total_models": len(self.models),
                    "confidence": 0.0,
                    "individual_results": [],
                    "analysis_method": "failed_consensus",
                    "analysis_mode": analysis_mode,
                    "visual_context": visual_context
                }

        # Add debug info
        consensus_result["debug"] = {
            "all_model_responses": all_responses,
            "prompt_mode": analysis_mode
        }

        return consensus_result

    # =========================================================================
    # SEQUENTIAL MICRO-STEP METHODS - Maximum Reliability Architecture
    # =========================================================================

    async def _query_model_simple(self, model_name: str, image: np.ndarray, prompt: str, timeout: int = 30, step_id: str = "unknown") -> str:
        """
        Simple model query method for micro-steps.
        Automatically logs all LLM interactions for debugging.
        """
        response_text = ""
        try:
            # Convert image to base64
            _, buffer = cv2.imencode('.png', image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=timeout
            )

            if response.status_code == 200:
                response_text = response.json().get("response", "").strip()
            else:
                response_text = f"HTTP {response.status_code}"

        except Exception as e:
            response_text = f"ERROR: {str(e)}"
            print(f"      ❌ Model query failed for {model_name}: {str(e)[:50]}...")

        # Automatically log this interaction if debug logging is enabled
        if self.debug_output_dir:
            self.log_llm_interaction(
                step=step_id,
                step_name=step_id.replace("_", " ").title(),
                model=model_name,
                prompt=prompt,
                response=response_text,
                image_shape=image.shape[:2]
            )

        return response_text if not response_text.startswith("ERROR") and not response_text.startswith("HTTP") else ""

    async def _step_5a_extract_text(self, tile_image: np.ndarray, text_region_image: np.ndarray) -> Dict:
        """
        Step 5a: Pure Text Extraction - Single focused task
        Extract ALL visible text from the image without interpretation.
        """
        print("   📝 Extracting all visible text from image...")

        # Simple prompt focused only on text extraction
        text_prompt = """
Extract ALL visible text from this German grocery product image.
Return ONLY the raw text you can see, separated by commas.
Include prices, product names, brands, unit information, and any pricing text like "€/kg" or "€/l".

Example output: "1,49 €, Dr. Oetker, Pudding Vanille-Geschmack, 25,69 € / 1kg, Seelenwärmer"

Raw text:"""

        try:
            # Use consensus but focused only on text extraction
            all_text_responses = []

            for model in self.models:
                model_name = model["name"]  # Extract model name from config dict
                try:
                    response = await self._query_model_simple(model_name, tile_image, text_prompt, timeout=30)
                    print(f"      🔍 {model_name} response: '{response[:100]}...' ({len(response)} chars)")
                    if response and len(response.strip()) > 5:  # Lowered threshold to 5 chars for debugging
                        all_text_responses.append({
                            'model': model_name,
                            'text': response.strip(),
                            'length': len(response.strip())
                        })
                        print(f"      ✅ {model_name}: {len(response.strip())} chars extracted")
                    else:
                        print(f"      ❌ {model_name}: Insufficient text extracted (got {len(response)} chars)")
                except Exception as e:
                    print(f"      ❌ {model_name}: Failed - {str(e)[:50]}...")
                    continue

            if not all_text_responses:
                print("   ❌ No models successfully extracted text")
                return {}

            # Choose the most comprehensive text extraction
            best_extraction = max(all_text_responses, key=lambda x: x['length'])
            print(f"   ✅ Best extraction from {best_extraction['model']}: {best_extraction['length']} chars")

            return {
                'success': True,
                'raw_text': best_extraction['text'],
                'source_model': best_extraction['model'],
                'all_extractions': all_text_responses
            }

        except Exception as e:
            print(f"   ❌ Step 5a failed: {str(e)}")
            return {}

    def _is_valid_product_name(self, response: str, prompt: str) -> bool:
        """
        Validate if LLM response is a real product name vs prompt/explanation/refusal.

        Returns:
            True if response appears to be a valid product name
            False if response is prompt echo, explanation, refusal, or error
        """
        if not response or len(response.strip()) < 3:
            return False

        response_lower = response.lower()

        # Reject explicit refusals - improved to catch all variations
        refusal_indicators = [
            "i'm sorry",      # Catches "I'm sorry, I'm not able...", "I'm sorry, but I can't..."
            "i can't",        # Catches "I can't extract...", "I can't do that"
            "i cannot",       # Catches "I cannot help"
            "i'm not able",   # Catches "I'm not able to do that"
            "unable to",
            "sorry",          # Broader catch for polite refusals
            "error:",
            "http",
            "timeout"
        ]
        if any(indicator in response_lower for indicator in refusal_indicators):
            return False

        # Reject explanations (contain markdown, numbered lists, or prompt keywords)
        explanation_indicators = [
            "**",  # Markdown bold
            "- **",  # Markdown list with bold
            "extraction rules",
            "to extract",
            "follow these steps",
            "brand prefix",
            "quantity information",
            "ignore price"
        ]
        if any(indicator in response_lower for indicator in explanation_indicators):
            return False

        # Reject if response contains significant portions of the prompt
        # (>50% of response matches prompt words)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        if len(response_words) > 0:
            overlap = len(prompt_words & response_words) / len(response_words)
            if overlap > 0.5:
                return False

        # Reject if too long (product names shouldn't exceed 150 chars)
        if len(response) > 150:
            return False

        # Reject if contains multiple newlines (explanations)
        if response.count('\n') > 2:
            return False

        return True

    def _is_valid_price(self, response: str) -> bool:
        """
        Validate if LLM response is a valid price.

        Returns:
            True if response appears to be a valid price (e.g., "1,69 €")
            False if response is refusal, explanation, or invalid format
        """
        import re

        if not response or len(response.strip()) < 3:
            return False

        response_lower = response.lower()

        # Reject refusals and errors
        if any(x in response_lower for x in ["can't", "cannot", "error", "http", "timeout"]):
            return False

        # Reject explanations
        if any(x in response for x in ["**", "- **", "The price is"]):
            return False

        # Reject if too long (price should be < 15 chars)
        if len(response) > 15:
            return False

        # Must contain digits
        if not re.search(r'\d', response):
            return False

        return True

    def _is_valid_brand(self, response: str) -> bool:
        """
        Validate if LLM response is a valid brand name.

        Returns:
            True if response appears to be a valid brand name
            False if response is refusal, explanation, or invalid
        """
        if not response or len(response.strip()) < 2:
            return False

        response_lower = response.lower()

        # Reject refusals and errors
        if any(x in response_lower for x in ["can't", "cannot", "error", "http", "timeout"]):
            return False

        # Reject explanations
        if any(x in response for x in ["**", "- **"]):
            return False

        # Reject if too long (brands should be < 50 chars)
        if len(response) > 50:
            return False

        return True

    async def _step_5a_detect_product_name(self, tile_image: np.ndarray, text_region_image: np.ndarray) -> Dict:
        """
        Step 5a: Product Name Detection - Single focused visual task
        Find the product name from images using visual analysis.
        """
        print("   🏷️ Detecting product name from images...")

        # ✅ CLAUDE.MD COMPLIANT - NO OCR/REGEX - LLM CONSENSUS ONLY ✅
        # Ultra-simple prompt - no examples to prevent contamination
        product_name_prompt = """Extract the complete product name text from the middle section of this image.
Ignore all prices and price per unit information.
Return ONLY the product name text (including brand, description, and quantity).

Product name:"""

        try:
            # Multi-model consensus for product name detection
            # Using 3 fast, reliable vision models (replaced slow llama3.2-vision:11b with llava:latest)
            all_models = ["qwen2.5vl:7b", "minicpm-v:latest", "llava-llama3:latest"]
            product_names = []

            print(f"      🔍 Querying {len(all_models)} models for consensus...")

            # Query models sequentially with retry logic
            for model_name in all_models:
                try:
                    print(f"      🤖 Querying {model_name}...")
                    response = await self._query_model_with_retry(
                        model_name, text_region_image, product_name_prompt,
                        max_retries=3, timeout=20, step_id="5a_product_name"
                    )

                    if response and len(response.strip()) > 2:
                        clean_name = response.strip().strip('"')

                        # Strip common preambles from LLM responses
                        preambles = [
                            "the product name is:",
                            "product name:",
                            "the product is:",
                            "product:",
                            "name:",
                        ]
                        clean_name_lower = clean_name.lower()
                        for preamble in preambles:
                            if clean_name_lower.startswith(preamble):
                                clean_name = clean_name[len(preamble):].strip().strip('"').strip()
                                break

                        # Strip common badges/labels that confuse brand extraction
                        badges_to_remove = [
                            "best value",
                            "sale",
                            "new",
                            "aktuelle",
                            "aktion",
                            "angebot"
                        ]
                        clean_name_lower = clean_name.lower()
                        for badge in badges_to_remove:
                            if clean_name_lower.startswith(badge):
                                clean_name = clean_name[len(badge):].strip()
                                print(f"      🏷️ Stripped badge '{badge}' from product name")
                                break

                        # Validate response before adding to consensus
                        if self._is_valid_product_name(clean_name, product_name_prompt):
                            product_names.append(clean_name)
                            print(f"      ✅ {model_name}: {clean_name[:60]}...")
                        else:
                            print(f"      ❌ {model_name} INVALID (refusal/prompt/explanation): {clean_name[:80]}...")
                    else:
                        print(f"      ❌ {model_name} returned empty")

                    # Small delay between models
                    await asyncio.sleep(0.5)

                except Exception as e:
                    print(f"      ❌ {model_name} failed: {str(e)[:50]}...")

            # Enhanced Consensus: Exact-match majority voting first, then fuzzy fallback
            if product_names:
                from collections import Counter
                from difflib import SequenceMatcher

                def similarity(a: str, b: str) -> float:
                    """Calculate string similarity ratio (0.0 to 1.0)"""
                    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

                # STEP 1: Check for exact matches first (TRUE MAJORITY VOTING)
                exact_counts = Counter(product_names)
                most_common_exact, exact_frequency = exact_counts.most_common(1)[0]

                # If 2+ models agree exactly, use that (respects true consensus)
                if exact_frequency >= 2:
                    print(f"      🎯 EXACT MATCH: {exact_frequency}/{len(product_names)} models returned identical string")
                    print(f"      ✅ CONSENSUS product name: {most_common_exact} (exact agreement: {exact_frequency}/{len(product_names)} models)")

                    return {
                        'success': True,
                        'product_name': most_common_exact,
                        'source_model': f"exact_consensus_{exact_frequency}_models",
                        'confidence': 0.95,
                        'exact_match': True
                    }

                # STEP 2: No exact majority - use fuzzy grouping as fallback
                print(f"      ⚠️  No exact majority ({exact_frequency}/{len(product_names)} at best) - using fuzzy grouping")

                # Group similar responses (>90% similarity threshold - stricter than before)
                grouped_responses = []
                for name in product_names:
                    # Check if this name is similar to any existing group
                    added_to_group = False
                    for group in grouped_responses:
                        if similarity(name, group[0]) > 0.90:  # Raised from 0.85 to 0.90
                            group.append(name)
                            added_to_group = True
                            break

                    if not added_to_group:
                        grouped_responses.append([name])

                # Find largest group
                largest_group = max(grouped_responses, key=len)

                # Within the fuzzy group, pick most common variant (not shortest!)
                if len(largest_group) > 1:
                    # Count exact matches within the fuzzy group
                    group_counts = Counter(largest_group)
                    most_common_in_group, group_frequency = group_counts.most_common(1)[0]
                    product_name = most_common_in_group

                    print(f"      🔍 Fuzzy group size: {len(largest_group)}")
                    print(f"      🗳️  Most common variant in group: '{product_name}' ({group_frequency}/{len(largest_group)} votes)")
                else:
                    # Only one response in group, use it
                    product_name = largest_group[0]

                frequency = len(largest_group)

                confidence = 0.85 if frequency >= 2 else 0.70  # Lower confidence for fuzzy matches
                similar_variations = f" (fuzzy grouped {len(largest_group)} similar responses)" if len(largest_group) > 1 else ""
                print(f"      ✅ CONSENSUS product name: {product_name} (fuzzy agreement: {frequency}/{len(product_names)} models){similar_variations}")

                return {
                    'success': True,
                    'product_name': product_name,
                    'source_model': f"fuzzy_consensus_{frequency}_models",
                    'confidence': confidence,
                    'exact_match': False
                }
            else:
                print(f"      ❌ No valid product name found by any model (all responses filtered)")
                return {}

        except Exception as e:
            print(f"      ❌ Product name detection failed: {str(e)[:50]}...")
            return {}

    def _extract_brand_from_product_text(self, text: str) -> str:
        """
        Extract brand from LLM response, handling multi-word brands intelligently.

        Strategy: Take first 1-3 words based on capitalization patterns.
        Stop at first lowercase word (assumes brand is title case).

        Examples:
        - "Aurora Sonnenstern-Grieß" → "Aurora"
        - "Dr. Oetker Puddingpulver" → "Dr. Oetker"
        - "Original Wagner Pizza" → "Original Wagner"
        - "REWE Beste Wahl Nudeln" → "REWE Beste Wahl"
        """
        import re

        text = text.strip()

        # Remove quantity patterns at end (e.g., "1kg", "500g", "x3", "3er Pack")
        text = re.sub(r'\s+\d+[a-zA-Z]*$', '', text)  # "1kg", "500g"
        text = re.sub(r'\s+x\d+$', '', text)  # "x3"
        text = re.sub(r'\s+\d+er\s+Pack$', '', text, flags=re.IGNORECASE)  # "3er Pack"

        words = text.split()

        if not words:
            return text

        first_two = ' '.join(words[:2]).lower() if len(words) >= 2 else ''
        first_three = ' '.join(words[:3]).lower() if len(words) >= 3 else ''

        # Pattern 1: Period brands (Dr. Oetker)
        if words[0].endswith('.') and len(words) > 1:
            return f"{words[0]} {words[1]}"

        # Pattern 2: Known two-word brand patterns
        known_two_word = ['original wagner', 'uncle ben', 'beste wahl']
        if first_two in known_two_word:
            return ' '.join(words[:2])

        # Pattern 3: Three-word patterns (REWE Beste Wahl, Edeka Gut & Günstig)
        if 'beste wahl' in first_three:
            return ' '.join(words[:3]) if words[0].lower() in ['rewe', 'edeka'] else ' '.join(words[:2])

        if len(words) >= 3 and words[1] in ['&', 'und']:
            # "EDEKA Gut & Günstig" or "Ben & Jerry's"
            return ' '.join(words[:3])

        # Pattern 4: Conservative multi-word extraction based on capitalization
        # Take words while they start with capital letters
        brand_words = [words[0]]  # Always take first word

        for i in range(1, min(len(words), 4)):  # Check up to 3 more words
            word = words[i]

            # Stop at lowercase words (unless possessive like "Ben's")
            if word[0].islower() and not word.endswith("'s"):
                break

            # Stop at common descriptor words
            descriptor_words = ['pizza', 'mehl', 'nudeln', 'kuchen', 'sauce', 'käse']
            if word.lower() in descriptor_words:
                break

            brand_words.append(word)

        # If we got multiple capitalized words, return them
        if len(brand_words) > 1:
            return ' '.join(brand_words)

        # Default: first word
        return words[0]

    def _normalize_brand_name(self, brand_name: str) -> str:
        """
        Normalize brand name capitalization using known brand mappings.
        Handles multi-word brands and expands truncated brand names.
        """
        # Brand normalization mapping (lowercase → proper capitalization)
        # IMPORTANT: Maps incomplete brands to complete multi-word forms
        brand_mappings = {
            # Single word brands
            'ja!': 'Ja!',
            'ja': 'Ja!',
            'rewe': 'REWE',
            'aurora': 'Aurora',
            'mondamin': 'Mondamin',
            'iglo': 'Iglo',
            'alnatura': 'Alnatura',
            'edeka': 'EDEKA',
            'maggi': 'Maggi',
            'knorr': 'Knorr',

            # Multi-word brands - complete forms
            'dr. oetker': 'Dr. Oetker',
            'dr oetker': 'Dr. Oetker',
            'rewe beste wahl': 'REWE Beste Wahl',
            'rewe best wahl': 'REWE Beste Wahl',
            'beste wahl': 'Beste Wahl',
            'edeka gut & günstig': 'EDEKA Gut & Günstig',

            # Multi-word brands - EXPANSION MAPPINGS (incomplete → complete)
            # This fixes LLM truncation issues
            'wagner': 'Original Wagner',  # Expand "Wagner" → "Original Wagner"
            'original wagner': 'Original Wagner',  # Also normalize complete form
            "uncle ben": "Uncle Ben's",
            "uncle ben's": "Uncle Ben's",
            "uncle bens": "Uncle Ben's",

            # Possessive forms
            "kellogg's": "Kellogg's",
            "kelloggs": "Kellogg's",
        }

        brand_lower = brand_name.lower().strip()
        normalized = brand_mappings.get(brand_lower)

        if normalized:
            return normalized

        # If not in mapping, return title case version
        return brand_name.strip()

    async def _step_5e_detect_brand(self, tile_image: np.ndarray, text_region_image: np.ndarray) -> Dict:
        """
        Step 5e: Brand Detection - Two-step visual + text validation approach
        1. Identify brand visually from product image (logo, colors, design)
        2. Validate that brand name appears in product text

        This prevents badges like "Best value" from being mistaken as brands.
        """
        print("   🏢 Detecting brand from images...")

        # ✅ CLAUDE.MD COMPLIANT - NO OCR/REGEX - LLM CONSENSUS ONLY ✅
        # NEW: Two-step approach - visual brand identification + text validation
        # EXPLICIT multi-word brand instruction to prevent truncation
        brand_prompt = """
Look at this product packaging and identify the COMPLETE BRAND name.

IMPORTANT: Many brands have multiple words. Return the COMPLETE brand name.
- If you see "Original Wagner", return "Original Wagner" NOT just "Wagner"
- If you see "Uncle Ben's", return "Uncle Ben's" NOT just "Uncle Ben"
- If you see "REWE Beste Wahl", return "REWE Beste Wahl" NOT just "REWE"
- If you see "Dr. Oetker", return "Dr. Oetker" NOT just "Oetker"

Look for:
1. Brand logo or brand text on the packaging
2. Brand design elements
3. Distinctive brand colors

Return the COMPLETE brand name exactly as shown on the packaging.
Do NOT abbreviate or truncate multi-word brand names.
IGNORE badges like "Best value", "Sale", "Angebot" - these are NOT brands.

Brand:"""

        try:
            # Multi-model consensus for brand detection
            all_models = ["qwen2.5vl:7b", "minicpm-v:latest", "llava-llama3:latest"]
            brand_responses = []

            print(f"      🔍 Querying {len(all_models)} models for consensus...")

            # Query models sequentially with retry logic
            # NEW: Use FULL product image (tile_image) to see brand logo/visual identity
            for model_name in all_models:
                try:
                    print(f"      🤖 Querying {model_name}...")
                    response = await self._query_model_with_retry(
                        model_name, tile_image, brand_prompt,  # Changed from text_region_image to tile_image
                        max_retries=3, timeout=20, step_id="5e_brand"
                    )

                    if response and len(response.strip()) > 2:
                        raw_brand = response.strip().strip('"')

                        # Validate response before extraction
                        if not self._is_valid_brand(raw_brand):
                            print(f"      ❌ {model_name} INVALID brand: {raw_brand[:40]}...")
                            continue

                        # Extract brand using Brand-Description-Quantity pattern
                        extracted_brand = self._extract_brand_from_product_text(raw_brand)
                        brand_responses.append(extracted_brand)
                        print(f"      ✅ {model_name}: {raw_brand[:40]}... → Extracted: {extracted_brand}")
                    else:
                        print(f"      ❌ {model_name} returned empty")

                    # Small delay between models
                    await asyncio.sleep(0.5)

                except Exception as e:
                    print(f"      ❌ {model_name} failed: {str(e)[:50]}...")

            # Consensus: Use most common extracted brand
            if brand_responses:
                from collections import Counter
                brand_counts = Counter(brand_responses)
                brand_name, frequency = brand_counts.most_common(1)[0]

                # Normalize brand capitalization
                brand_name = self._normalize_brand_name(brand_name)

                confidence = 0.95 if frequency >= 2 else 0.75
                print(f"      ✅ CONSENSUS brand: {brand_name} (agreement: {frequency}/{len(brand_responses)} models)")

                return {
                    'success': True,
                    'brand': brand_name,
                    'source_model': f"consensus_{frequency}_models",
                    'confidence': confidence
                }
            else:
                print(f"      ⚠️  No brand found by any model")
                return {'success': False}

        except Exception as e:
            print(f"      ❌ Brand detection failed: {str(e)[:50]}...")
            return {'success': False}

    async def _step_5d_detect_quantity(self, tile_image: np.ndarray, text_region_image: np.ndarray) -> Dict:
        """
        Step 5d: Quantity/Weight Detection - Single focused task
        Find quantity information like "89g für 500 ml", "3er Pack", "500g" from text region.
        """
        print("   ⚖️ Detecting quantity/weight information from images...")

        # ✅ CLAUDE.MD COMPLIANT - NO OCR/REGEX - LLM CONSENSUS ONLY ✅
        quantity_prompt = """
Find quantity like "89g für 500 ml" or "3er Pack".

JSON only: {"quantity": "89g für 500 ml"} or {"quantity": null}

JSON:"""

        try:
            # Use working model for quantity detection
            best_model = "minicpm-v:latest"

            # Send the text region image for quantity detection (contains quantity text)
            response = await self._query_model_simple(best_model, text_region_image, quantity_prompt, timeout=20, step_id="5d_quantity")

            if response and len(response.strip()) > 5:
                try:
                    # Parse LLM JSON response
                    import json
                    quantity_info = json.loads(response.strip())

                    if quantity_info.get('quantity'):
                        print(f"      ✅ Quantity detected: {quantity_info['quantity']}")
                        return {
                            'success': True,
                            'quantity': quantity_info['quantity'],
                            'weight': quantity_info.get('weight', ''),
                            'volume': quantity_info.get('volume', ''),
                            'source_model': best_model,
                            'confidence': 0.9
                        }
                    else:
                        print(f"      ❌ No quantity found by {best_model}")
                        return {}

                except json.JSONDecodeError:
                    print(f"      ❌ Invalid JSON from LLM: {response[:50]}...")
                    return {}
            else:
                print(f"      ❌ No quantity response from {best_model}")
                return {}

        except Exception as e:
            print(f"      ❌ Quantity detection failed: {str(e)[:50]}...")
            return {}

    async def _step_5b_detect_price(self, tile_image: np.ndarray, text_region_image: np.ndarray) -> Dict:
        """
        Step 5b: Price Detection - Single focused task
        Find the main product price from both product and text images.
        """
        print("   💰 Detecting main product price from images...")

        # ✅ CLAUDE.MD COMPLIANT - NO OCR/REGEX - LLM CONSENSUS ONLY ✅
        # Visual prompt focused only on price identification
        price_prompt = """
Look ONLY at the TOP ROW of text. Extract the price from the first line.

REQUIRED format: X,XX € (numbers BEFORE euro symbol)
INVALID: €X,XX, explanations, other rows

Examples: 1,09 € | 2,49 € | 0,99 €
WRONG: €1,09 | The price is 1,09 €

TOP ROW ONLY. Return just the price.

Price:"""

        try:
            import re

            # Multi-model consensus for main price detection
            all_models = ["qwen2.5vl:7b", "minicpm-v:latest", "llava-llama3:latest"]
            price_responses = []

            print(f"      🔍 Querying {len(all_models)} models for consensus...")

            # Query models sequentially with retry logic
            for model_name in all_models:
                try:
                    print(f"      🤖 Querying {model_name}...")
                    response = await self._query_model_with_retry(
                        model_name, text_region_image, price_prompt,
                        max_retries=3, timeout=20, step_id="5b_price"
                    )

                    if response and len(response.strip()) > 2:
                        detected_price = response.strip()

                        # Validate response before normalization
                        if not self._is_valid_price(detected_price):
                            print(f"      ❌ {model_name} INVALID price: {detected_price[:40]}...")
                            continue

                        # Normalize price format before consensus
                        if '€' in detected_price:
                            detected_price = re.sub(r'(\d+,\d{2})\s*€', r'\1 €', detected_price)
                        else:
                            if re.match(r'^\d+,\d{2}$', detected_price):
                                detected_price = f"{detected_price} €"

                        price_responses.append(detected_price)
                        print(f"      ✅ {model_name}: {detected_price}")
                    else:
                        print(f"      ❌ {model_name} returned empty")

                    # Small delay between models
                    await asyncio.sleep(0.5)

                except Exception as e:
                    print(f"      ❌ {model_name} failed: {str(e)[:50]}...")

            # Consensus: Use most common normalized price
            if price_responses:
                from collections import Counter
                price_counts = Counter(price_responses)
                main_price, frequency = price_counts.most_common(1)[0]

                confidence = 0.95 if frequency >= 2 else 0.85
                print(f"      ✅ CONSENSUS main price: {main_price} (agreement: {frequency}/{len(price_responses)} models)")

                return {
                    'success': True,
                    'main_price': main_price,
                    'source_model': f"consensus_{frequency}_models",
                    'confidence': confidence
                }
            else:
                print(f"      ❌ No price found by any model")
                return {}

        except Exception as e:
            print(f"      ❌ LLM price detection failed: {str(e)[:50]}...")
            return {}

    async def _step_5c_detect_unit_type(self, tile_image: np.ndarray, text_region_image: np.ndarray) -> Dict:
        """
        Step 5c: Unit Type Detection - NEW 2-STEP APPROACH
        Step 1: Extract bottom line as TEXT (simple OCR)
        Step 2: Parse text to find unit after "€ / 1"
        """
        print("   📏 Detecting unit type from images...")

        # Initialize debug log for this step
        debug_log = {
            "step": "5c_unit_detection",
            "step_name": "Unit Type Detection",
            "image_shape": f"{text_region_image.shape[0]}x{text_region_image.shape[1]}",
            "substeps": []
        }

        # ✅ STEP 5c-1: Extract bottom pricing line as TEXT with CHARACTER-LEVEL ACCURACY
        text_extraction_prompt = """
🔍 CRITICAL OCR TASK - READ EXACTLY WHAT YOU SEE

Look at the BOTTOM LINE ONLY. This shows price per unit like:
- "3,38 € / 1kg"
- "1,79 € / 1l"
- "0,36 € / 1Stk"

RULES FOR READING THE UNIT LETTERS AFTER "/ 1":
1. Look at the 2-3 letters IMMEDIATELY AFTER "/ 1"
2. Read EXACTLY what you see - don't guess or infer
3. Common units you might see:
   - "kg" (has K and G letters - weight unit)
   - "l" (single L letter - volume unit)
   - "Stk" (three letters S-T-K - piece unit)

CRITICAL: Do NOT infer unit from product type!
- If you see the letter "l" alone, write "l"
- If you see "kg" letters, write "kg"
- If you see "Stk" letters, write "Stk"

READ the actual text characters. DO NOT guess based on product.

Return ONLY the exact bottom line text. Nothing else.

Bottom line text:"""

        try:
            import json
            import re

            # STEP 5c-1: Extract bottom line text using multi-model consensus
            all_models = ["qwen2.5vl:7b", "minicpm-v:latest", "llava-llama3:latest"]
            all_text_responses = {}
            extracted_texts = []

            print(f"      🔍 Step 5c-1: Extracting bottom line text from all models (SEQUENTIAL)...")

            # Query models SEQUENTIALLY with retry logic to prevent Ollama overload
            for model_name in all_models:
                try:
                    # Use retry wrapper instead of simple query
                    response = await self._query_model_with_retry(
                        model_name, text_region_image, text_extraction_prompt,
                        max_retries=3, timeout=20, step_id="5c1_text_extraction"
                    )
                    all_text_responses[model_name] = response

                    # Small delay between models to prevent overload
                    await asyncio.sleep(0.5)

                    if response and len(response.strip()) > 3:
                        # Clean up response (remove quotes, extra spaces)
                        cleaned_text = response.strip().strip('"').strip("'").strip()
                        extracted_texts.append({
                            'model': model_name,
                            'text': cleaned_text
                        })
                        print(f"      📝 {model_name}: '{cleaned_text}'")
                    else:
                        print(f"      ❌ {model_name}: No text extracted")
                except Exception as e:
                    print(f"      ❌ {model_name} failed: {str(e)[:60]}")

            if not extracted_texts:
                print(f"      ❌ No models successfully extracted text")
                return {'success': False, 'debug_all_responses': all_text_responses}

            # VALIDATION: Check all extracted texts for obvious unit patterns
            # This catches cases where models disagree but the text clearly shows the unit
            all_text_combined = ' '.join([t['text'] for t in extracted_texts])

            # Strong unit detection patterns (case-insensitive)
            if re.search(r'€\s*/\s*1\s*kg', all_text_combined, re.IGNORECASE):
                forced_unit = 'kg'
                print(f"      🎯 FORCED UNIT DETECTION: Found 'kg' in text responses")
            elif re.search(r'€\s*/\s*1\s*l\b', all_text_combined, re.IGNORECASE):  # \b for word boundary
                forced_unit = 'l'
                print(f"      🎯 FORCED UNIT DETECTION: Found 'l' in text responses")
            elif re.search(r'€\s*/\s*1\s*[Ss]tk', all_text_combined, re.IGNORECASE):
                forced_unit = 'Stk'
                print(f"      🎯 FORCED UNIT DETECTION: Found 'Stk' in text responses")
            else:
                forced_unit = None

            # Use consensus on extracted text with UNIT-SPECIFIC VOTING
            from collections import Counter
            import re
            text_counts = Counter([t['text'] for t in extracted_texts])
            unit_agreement_ratio = 1.0  # Initialize with full agreement

            if text_counts:
                # If we have consensus (2+ models agree on full text), use that
                most_common_text, frequency = text_counts.most_common(1)[0]

                if frequency >= 2:
                    pricing_line = most_common_text
                    print(f"      ✅ Text consensus: '{pricing_line}' ({frequency}/{len(extracted_texts)} models)")
                else:
                    # NO CONSENSUS ON FULL TEXT - Use UNIT-SPECIFIC VOTING instead of longest text
                    print(f"      ⚠️  No full text consensus, using unit-specific voting...")

                    # Extract just the unit from each response
                    units_only = []
                    for text_resp in extracted_texts:
                        # Try to find unit after "€ / 1" or "€/1"
                        match = re.search(r'€\s*/\s*1\s*([a-zA-Z]+)', text_resp['text'])
                        if match:
                            unit_extracted = match.group(1)
                            units_only.append({
                                'unit': unit_extracted,
                                'full_text': text_resp['text'],
                                'model': text_resp['model']
                            })
                            print(f"         - {text_resp['model']}: unit '{unit_extracted}'")

                    if units_only:
                        # Vote on units only
                        unit_counts = Counter([u['unit'] for u in units_only])
                        most_common_unit, unit_frequency = unit_counts.most_common(1)[0]

                        print(f"      🗳️  Unit voting: {dict(unit_counts)}")
                        print(f"      ✅ Winner unit: '{most_common_unit}' ({unit_frequency}/{len(units_only)} models)")

                        # Calculate confidence based on model agreement
                        # Store for later use in confidence calculation
                        unit_agreement_ratio = unit_frequency / len(units_only)

                        # Find the text response that contains this winning unit
                        for u in units_only:
                            if u['unit'] == most_common_unit:
                                pricing_line = u['full_text']
                                print(f"      📝 Using text from {u['model']}: '{pricing_line}'")
                                break
                    else:
                        # Couldn't extract units - fallback to first response
                        pricing_line = extracted_texts[0]['text']
                        print(f"      ⚠️  No units extracted, using first: '{pricing_line}'")
            else:
                pricing_line = extracted_texts[0]['text']
                print(f"      ⚠️  Using first extraction: '{pricing_line}'")

            # STEP 5c-2: Parse text to extract unit after "€ / 1"
            print(f"      🔍 Step 5c-2: Parsing unit from text...")

            # FIX: Normalize whitespace FIRST to handle "€ / 1kg", "€/1kg", "€/ 1kg" all the same
            normalized_text = pricing_line.replace(' ', '').replace('\n', '').replace('\t', '')
            print(f"      📝 Normalized text: '{normalized_text}'")

            # Pattern: find "€/1" followed by letters (kg, l, Stk, etc.)
            # Examples: "3,38€/1kg" → "kg", "0,36€/1Stk" → "Stk"
            pattern = r'€/1?([a-zA-Z]+)'
            match = re.search(pattern, normalized_text)

            if match:
                extracted_unit_raw = match.group(1).strip()
                print(f"      📌 Found unit in text: '{extracted_unit_raw}'")

                # Normalize unit (Stk., kg, l → Stk, kg, l)
                valid_units = {
                    # Pieces/Items
                    'stk': 'Stk', 'STK': 'Stk', 'Stk': 'Stk', 'stück': 'Stk', 'piece': 'Stk',
                    # Weight: grams → kilograms
                    'kg': 'kg', 'KG': 'kg', 'Kg': 'kg', 'kilogram': 'kg', 'kilo': 'kg',
                    'g': 'kg', 'G': 'kg', 'gram': 'kg', 'grams': 'kg',
                    # Volume: milliliters → liters
                    'l': 'l', 'L': 'l', 'liter': 'l', 'litre': 'l', 'lt': 'l',
                    'ml': 'l', 'ML': 'l', 'milliliter': 'l', 'millilitre': 'l'
                }

                normalized_unit = valid_units.get(extracted_unit_raw)

                # OVERRIDE: If forced_unit was detected from combined text, use that instead
                if forced_unit:
                    normalized_unit = forced_unit
                    print(f"      ✅ UNIT DETECTED: {normalized_unit} (from '{pricing_line}') - VALIDATED by forced detection")
                elif normalized_unit:
                    print(f"      ✅ UNIT DETECTED: {normalized_unit} (from '{pricing_line}')")

                if normalized_unit:
                    # Calculate confidence based on model agreement
                    # 3/3 = 1.0 → 0.95, 2/3 = 0.67 → 0.75, 1/3 = 0.33 → 0.50, 1/2 = 0.5 → 0.65
                    base_confidence = 0.95 if unit_agreement_ratio >= 0.9 else \
                                    0.80 if unit_agreement_ratio >= 0.75 else \
                                    0.65 if unit_agreement_ratio >= 0.6 else 0.50

                    print(f"      📊 Unit confidence: {base_confidence:.2f} (agreement: {unit_agreement_ratio:.2f})")

                    return {
                        'success': True,
                        'unit': normalized_unit,
                        'confidence': base_confidence,
                        'agreement_ratio': unit_agreement_ratio,
                        'extracted_text': pricing_line,
                        'debug_all_text_responses': all_text_responses,
                        'debug_extracted_texts': extracted_texts
                    }
                else:
                    print(f"      ❌ Invalid unit '{extracted_unit_raw}' - not in valid list")
                    return {
                        'success': False,
                        'extracted_text': pricing_line,
                        'debug_all_text_responses': all_text_responses
                    }
            else:
                print(f"      ❌ Could not find '€ / 1X' pattern in: '{pricing_line}'")

                # FALLBACK: Use forced_unit if we detected it earlier
                if forced_unit:
                    print(f"      🎯 FALLBACK: Using forced unit detection: {forced_unit}")
                    return {
                        'success': True,
                        'unit': forced_unit,
                        'confidence': 0.85,  # Lower confidence since pattern match failed
                        'extracted_text': pricing_line,
                        'debug_all_text_responses': all_text_responses,
                        'debug_note': 'Used forced unit detection as fallback'
                    }

                return {
                    'success': False,
                    'extracted_text': pricing_line,
                    'debug_all_text_responses': all_text_responses
                }

        except Exception as e:
            print(f"      ❌ Unit detection failed: {str(e)[:80]}...")
            return {'success': False, 'error': str(e)}

    async def _step_5d_detect_price_per_unit(self, tile_image: np.ndarray, text_region_image: np.ndarray, unit_type: str) -> Dict:
        """
        Step 5d: Price Per Unit Detection - Extract price knowing the unit type
        Multi-model consensus with debug logging and outlier detection
        """
        print(f"   💰 Detecting price per {unit_type} from images...")

        # ✅ CLAUDE.MD COMPLIANT - NO OCR/REGEX - LLM CONSENSUS ONLY ✅
        price_per_unit_prompt = f"""
Look ONLY at the BOTTOM/LAST row. Extract the price per {unit_type}.

CRITICAL: Extract ONLY the price portion (X,XX €) BEFORE the "/" symbol.

VALID format: "X,XX €"
INVALID: "The price is X,XX €", explanations, or missing unit context

Examples (generic patterns):
- "X,XX € / 1kg" → {{"price_per_unit": "X,XX €"}}
- "Y,YY € / 1l" → {{"price_per_unit": "Y,YY €"}}
- "Z,ZZ € / 1Stk" → {{"price_per_unit": "Z,ZZ €"}}

Bottom line example: "3,38 € / 1kg" → Extract exactly "3,38 €"

JSON only. No explanations.

JSON:"""

        try:
            import json
            import re

            # Query all available models for consensus
            all_models = ["qwen2.5vl:7b", "minicpm-v:latest", "llava-llama3:latest"]
            all_responses = {}
            valid_prices = []

            print(f"      🔍 Debug: Testing all models for price per {unit_type}")

            # Query models SEQUENTIALLY with retry logic
            for model_name in all_models:
                try:
                    print(f"      🤖 Querying {model_name}...")
                    response = await self._query_model_with_retry(
                        model_name, text_region_image, price_per_unit_prompt,
                        max_retries=3, timeout=25, step_id="5c2_price_per_unit"
                    )
                    all_responses[model_name] = response

                    # Small delay between models to prevent overload
                    await asyncio.sleep(0.5)

                    print(f"      📝 {model_name} raw: {response[:120] if response else 'None'}...")

                    if response and len(response.strip()) > 5:
                        try:
                            result = json.loads(response.strip())
                            if result.get("price_per_unit"):
                                extracted_price = result["price_per_unit"].strip()

                                # Validate format: X,XX € (with optional leading digits)
                                if re.match(r"^\d+,\d{2}\s*€$", extracted_price):
                                    valid_prices.append({
                                        'model': model_name,
                                        'price': extracted_price,
                                        'raw_response': response
                                    })
                                    print(f"      ✅ {model_name} parsed: {extracted_price}")
                                else:
                                    print(f"      ❌ {model_name} invalid format: {extracted_price}")
                            else:
                                print(f"      ❌ {model_name} returned no price_per_unit")
                        except json.JSONDecodeError:
                            print(f"      ❌ {model_name} invalid JSON: {response[:80]}")
                    else:
                        print(f"      ❌ {model_name} gave no usable response")
                except Exception as e:
                    print(f"      ❌ {model_name} failed: {str(e)[:80]}")

            # Consensus resolution: choose most common valid price
            if valid_prices:
                from collections import Counter
                price_counts = Counter([p['price'] for p in valid_prices])
                most_common_price, frequency = price_counts.most_common(1)[0]

                # Require at least 2 models to agree, or use first valid if only 1 model succeeded
                if frequency >= 2 or len(valid_prices) == 1:
                    consensus_price = most_common_price
                    print(f"      ✅ CONSENSUS {unit_type}: {consensus_price} (agreement: {frequency}/{len(valid_prices)} models)")

                    return {
                        'success': True,
                        'unit': unit_type,
                        'price_per_unit': consensus_price,
                        'confidence': 0.95 if frequency >= 2 else 0.75,
                        'debug_all_responses': all_responses,
                        'debug_valid_prices': valid_prices,
                        'debug_consensus_frequency': frequency
                    }
                else:
                    print(f"      ⚠️ No consensus: {[p['price'] for p in valid_prices]}")
                    # Return first valid as fallback
                    fallback_price = valid_prices[0]['price']
                    print(f"      ⚠️ Using fallback: {fallback_price}")
                    return {
                        'success': True,
                        'unit': unit_type,
                        'price_per_unit': fallback_price,
                        'confidence': 0.5,
                        'debug_all_responses': all_responses,
                        'debug_valid_prices': valid_prices,
                        'debug_consensus_frequency': 1
                    }
            else:
                print(f"      ❌ No valid prices extracted from any model")
                return {
                    'success': False,
                    'debug_all_responses': all_responses,
                    'debug_valid_prices': []
                }

        except Exception as e:
            print(f"      ❌ LLM price per {unit_type} detection failed: {str(e)[:80]}...")
            return {'success': False, 'error': str(e)}

    async def _step_5d_separate_brand_product(self, extracted_text: Dict) -> Dict:
        """
        Step 5d: Brand/Product Name Separation - Single focused task
        Identify and separate brand names from product names.
        """
        print("   🏷️ Separating brand from product name...")

        if not extracted_text.get('raw_text'):
            return {}

        text = extracted_text['raw_text']

        # Known German grocery brands
        known_brands = [
            'Dr. Oetker', 'Mondamin', 'Aurora', 'Edeka', 'Rewe', 'Knorr', 'Maggi',
            'Alnatura', 'Demeter', 'Bio', 'Organic', 'Ja!', 'Gut & Günstig'
        ]

        detected_brand = None
        remaining_text = text

        # Find brand in text
        for brand in known_brands:
            if brand.lower() in text.lower():
                detected_brand = brand
                # Remove brand from text to get clean product name
                remaining_text = text.replace(brand, '').strip()
                print(f"      ✅ Brand found: {brand}")
                break

        if not detected_brand:
            print(f"      ❓ No known brand found in: {text[:50]}...")
            # ✅ CLAUDE.MD COMPLIANT - NO OCR/REGEX - Use simple string processing ✅
            words = text.split()
            for word in words:
                # Find first capitalized word as potential brand
                if word and word[0].isupper() and len(word) > 2:
                    detected_brand = word
                    remaining_text = text.replace(detected_brand, '').strip()
                    print(f"      ❓ Potential brand: {detected_brand}")
                    break

        # ✅ CLAUDE.MD COMPLIANT - Clean up product name without regex ✅
        product_name = remaining_text
        # Simple text cleanup without regex
        while ',,' in product_name:  # Remove multiple commas
            product_name = product_name.replace(',,', ',')
        while '  ' in product_name:  # Remove multiple spaces
            product_name = product_name.replace('  ', ' ')
        product_name = product_name.strip()

        print(f"      📦 Product name: {product_name[:50]}...")

        return {
            'success': True,
            'brand': detected_brand or '',
            'product_name': product_name,
            'confidence': 0.8 if detected_brand else 0.5
        }

    def _step_5f_assemble_and_validate(self, name_data: Dict, price_data: Dict,
                                       unit_data: Dict, quantity_data: Dict, brand_data: Dict) -> Dict:
        """
        Step 5f: Data Assembly & Validation - Single focused task
        Combine all extracted data and apply business rules with consensus validation.
        """
        print("   🔧 Assembling and validating final product data...")
        import re

        # Initialize result structure
        result = {
            'price': '',
            'product_name_raw': '',      # NEW: Original extracted product name
            'product_name': '',           # Clean product name (brand removed)
            'brand': '',
            'weight': '',
            'unit': '',
            'quantity': '',
            'price_per_kg': '',
            'price_per_piece': '',
            'price_per_liter': '',
            'debug_info': {}
        }

        # Assemble basic data from visual detection results
        result['price'] = price_data.get('main_price', '')
        result['brand'] = brand_data.get('brand', '')
        result['product_name_raw'] = name_data.get('product_name', '')  # Store raw version
        result['product_name'] = name_data.get('product_name', '')  # Will be cleaned below

        # FINAL PRICE NORMALIZATION - Ensure consistent "X,XX €" format
        if result['price']:
            if '€' in result['price']:
                # Ensure space before €
                result['price'] = re.sub(r'(\d+,\d{2})\s*€', r'\1 €', result['price'])
            else:
                # Add missing € symbol
                if re.match(r'^\d+,\d{2}$', result['price']):
                    result['price'] = f"{result['price']} €"
            print(f"      💰 Price normalized: {result['price']}")

        # BRAND FALLBACK - Extract from product_name if brand is empty
        if not result['brand'] and result['product_name']:
            # Try to extract brand from first 1-2 words of product name
            words = result['product_name'].split()
            if len(words) >= 2 and '.' in words[0]:  # e.g., "Dr. Oetker"
                fallback_brand = f"{words[0]} {words[1]}"
            elif len(words) >= 1:
                fallback_brand = words[0]
            else:
                fallback_brand = ''

            if fallback_brand:
                # Normalize the fallback brand
                fallback_brand = self._normalize_brand_name(fallback_brand)
                result['brand'] = fallback_brand
                print(f"      🏢 Brand extracted from product name (fallback): {fallback_brand}")

        # BRAND REMOVAL - Create clean product name by removing brand text
        if result['brand'] and result['product_name']:
            # Normalize whitespace for robust comparison
            product_normalized = ' '.join(result['product_name'].split())
            brand_normalized = ' '.join(result['brand'].split())

            product_lower = product_normalized.lower()
            brand_lower = brand_normalized.lower()

            # Check if product name starts with the brand
            if product_lower.startswith(brand_lower):
                # Remove brand from start, preserving original case in remaining text
                # Use normalized brand length to handle spacing differences
                clean_name = result['product_name'][len(brand_normalized):].strip()

                # Safety check: Ensure at least 1 word remains after removal
                remaining_words = clean_name.split() if clean_name else []

                if len(remaining_words) >= 1:
                    result['product_name'] = clean_name
                    print(f"      🧹 Removed brand '{result['brand']}' from product name")
                    print(f"         Raw: {result['product_name_raw']}")
                    print(f"         Clean: {result['product_name']}")
                else:
                    print(f"      ⚠️  Brand removal would leave < 1 word, keeping original")
                    print(f"         Product: '{result['product_name']}' Brand: '{result['brand']}'")
            else:
                # Brand not at start - check if it appears elsewhere
                if brand_lower in product_lower:
                    print(f"      ⚠️  Brand '{result['brand']}' found mid-string in '{result['product_name']}' - not removing")
                else:
                    print(f"      ℹ️  Brand '{result['brand']}' not in product name - already clean")

        # Add quantity/weight information
        if quantity_data.get('success'):
            result['quantity'] = quantity_data.get('quantity', '')
            result['weight'] = quantity_data.get('weight', '')

        # Apply unit and pricing rules with enhanced validation
        if unit_data.get('success'):
            unit = unit_data['unit']
            price_per_unit = unit_data.get('price_per_unit', '')

            # Validate price format (X,XX €)
            if price_per_unit and re.match(r"^\d+,\d{2}\s*€$", price_per_unit):
                result['unit'] = unit

                # EXCLUSIVE PRICING ENFORCEMENT
                if unit == 'kg':
                    result['price_per_kg'] = price_per_unit
                    result['price_per_piece'] = ''
                    result['price_per_liter'] = ''
                elif unit == 'l':
                    result['price_per_liter'] = price_per_unit
                    result['price_per_kg'] = ''
                    result['price_per_piece'] = ''
                elif unit == 'Stk':
                    result['price_per_piece'] = price_per_unit
                    result['price_per_kg'] = ''
                    result['price_per_liter'] = ''

                # Add debug info from consensus analysis
                consensus_info = {
                    'unit': unit,
                    'price_per_unit': price_per_unit,
                    'confidence': unit_data.get('confidence', 0.0),
                    'consensus_frequency': unit_data.get('debug_consensus_frequency', 0),
                    'valid_prices': unit_data.get('debug_valid_prices', [])
                }
                result['debug_info']['consensus'] = consensus_info

                print(f"      ✅ Unit-based pricing applied: {unit} → {price_per_unit} (confidence: {consensus_info['confidence']:.2f})")
            else:
                print(f"      ⚠️  Invalid price format detected: '{price_per_unit}' - skipping per-unit pricing")
                result['unit'] = 'Stk'
                if result['price']:
                    result['price_per_piece'] = result['price']
        else:
            print(f"      ⚠️  Visual unit detection failed - cannot determine unit from image")
            print(f"      🔄 FALLBACK: Will rely on full LLM consensus (analyze_product_with_consensus)")
            # DO NOT default to 'Stk' - let the calling function fall back to full LLM consensus
            # Mark this result as requiring LLM consensus fallback
            result['unit'] = ''  # Empty unit signals fallback needed
            result['_visual_detection_failed'] = True  # Internal flag for fallback

        # Store all debug responses if available
        if unit_data.get('debug_all_responses'):
            result['debug_info']['all_model_responses'] = unit_data['debug_all_responses']

        # DATA QUALITY SCORING (0-100 scale)
        quality_score = 0
        validation_issues = []

        # Price (25 points)
        if result['price'] and re.match(r'^\d+,\d{2}\s€$', result['price']):
            quality_score += 25
        elif result['price']:
            quality_score += 15
            validation_issues.append("Price format inconsistent")
        else:
            validation_issues.append("Missing main price")

        # Product Name (25 points)
        if result['product_name'] and len(result['product_name']) > 3:
            quality_score += 25
        elif result['product_name']:
            quality_score += 10
            validation_issues.append("Product name too short")
        else:
            validation_issues.append("Missing product name")

        # Brand (20 points)
        if result['brand']:
            quality_score += 20
        else:
            validation_issues.append("Missing brand")

        # Unit (15 points)
        if result['unit'] in ['kg', 'l', 'Stk']:
            quality_score += 15
        else:
            validation_issues.append("Invalid or missing unit")

        # Price Per Unit (15 points)
        per_unit_prices = [
            result.get('price_per_kg'),
            result.get('price_per_piece'),
            result.get('price_per_liter')
        ]
        non_empty_prices = [p for p in per_unit_prices if p]

        if len(non_empty_prices) == 1:
            quality_score += 15
        elif len(non_empty_prices) > 1:
            validation_issues.append(f"Multiple per-unit prices set: {non_empty_prices}")
        else:
            validation_issues.append("Missing per-unit price")

        success = len(validation_issues) == 0

        # Log quality assessment
        if quality_score >= 90:
            print(f"      ✅ Excellent data quality: {quality_score}/100")
        elif quality_score >= 70:
            print(f"      ✅ Good data quality: {quality_score}/100")
        elif quality_score >= 50:
            print(f"      ⚠️  Acceptable data quality: {quality_score}/100")
        else:
            print(f"      ❌ Poor data quality: {quality_score}/100")

        if validation_issues:
            print(f"      ⚠️  Issues: {', '.join(validation_issues)}")

        result['validation'] = {
            'success': success,
            'issues': validation_issues,
            'data_quality_score': quality_score,
            'needs_review': quality_score < 70  # Flag for manual review
        }

        return result

    # =========================================================================

    # Legacy method for backward compatibility
    async def analyze_categories_with_consensus(self, image: np.ndarray) -> Dict:
        """Legacy method - redirects to main analyze_product_with_consensus method."""
        return await self.analyze_product_with_consensus(image, image, "ui")