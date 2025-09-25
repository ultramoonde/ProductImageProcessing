#!/usr/bin/env python3
"""
Intelligent Background Removal Manager
Handles provider selection, fallbacks, and quality-based optimization
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import threading

from background_removal_providers import (
    BackgroundRemovalProvider, 
    BackgroundRemovalResult,
    create_provider,
    get_available_providers,
    AVAILABLE_PROVIDERS
)
from background_quality_assessor import BackgroundQualityAssessor, QualityScore


@dataclass
class ProcessingStrategy:
    """Configuration for processing strategy"""
    name: str
    primary_provider: str
    fallback_chain: List[str]
    quality_threshold: float
    max_cost_per_image: float
    max_attempts: int = 3
    parallel_processing: bool = False


class BackgroundRemovalManager:
    """
    Intelligent manager for background removal with automatic fallbacks and quality assessment
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger('BackgroundManager')
        
        # Initialize components
        self.providers: Dict[str, BackgroundRemovalProvider] = {}
        self.quality_assessor = BackgroundQualityAssessor()
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'provider_usage': {},
            'total_cost': 0.0,
            'average_quality': 0.0,
            'processing_time': 0.0
        }
        
        # Thread lock for statistics
        self._stats_lock = threading.Lock()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize providers
        self._initialize_providers()
        
        # Set default strategy
        self.current_strategy = self._create_strategy_from_config()
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'providers': {
                'primary': 'rembg',
                'fallback_chain': ['rembg', 'photoroom', 'removal_ai', 'remove_bg']
            },
            'quality_thresholds': {
                'minimum_acceptable': 0.60,
                'good': 0.75,
                'excellent': 0.90
            },
            'cost_optimization': {
                'max_cost_per_image': 0.20,
                'prefer_free': True
            },
            'performance': {
                'timeout_seconds': 30,
                'parallel_processing': False,
                'max_retries': 2
            },
            'strategies': {
                'cost_first': {
                    'primary_provider': 'rembg',
                    'fallback_chain': ['rembg', 'photoroom', 'removal_ai'],
                    'quality_threshold': 0.60,
                    'max_cost_per_image': 0.15
                },
                'quality_first': {
                    'primary_provider': 'remove_bg',
                    'fallback_chain': ['remove_bg', 'removal_ai', 'photoroom', 'rembg'],
                    'quality_threshold': 0.85,
                    'max_cost_per_image': 0.25
                },
                'speed_first': {
                    'primary_provider': 'rembg',
                    'fallback_chain': ['rembg', 'photoroom'],
                    'quality_threshold': 0.65,
                    'max_cost_per_image': 0.10
                },
                'balanced': {
                    'primary_provider': 'rembg',
                    'fallback_chain': ['rembg', 'photoroom', 'removal_ai'],
                    'quality_threshold': 0.70,
                    'max_cost_per_image': 0.15
                }
            }
        }
        
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                try:
                    if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                        with open(config_path, 'r') as f:
                            file_config = yaml.safe_load(f)
                    else:
                        with open(config_path, 'r') as f:
                            file_config = json.load(f)
                    
                    # Merge with defaults
                    self._deep_merge(default_config, file_config)
                    self.logger.info(f"Loaded configuration from {config_path}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load config from {config_path}: {str(e)}")
                    self.logger.info("Using default configuration")
            else:
                self.logger.warning(f"Config file not found: {config_path}, using defaults")
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _initialize_providers(self) -> None:
        """Initialize all available providers"""
        for provider_name in get_available_providers():
            try:
                provider = create_provider(provider_name)
                self.providers[provider_name] = provider
                
                if provider.is_available():
                    self.logger.info(f"✅ Provider initialized: {provider_name}")
                else:
                    self.logger.warning(f"⚠️  Provider unavailable: {provider_name}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize provider {provider_name}: {str(e)}")
    
    def _create_strategy_from_config(self) -> ProcessingStrategy:
        """Create processing strategy from configuration"""
        providers_config = self.config['providers']
        quality_config = self.config['quality_thresholds']
        cost_config = self.config['cost_optimization']
        perf_config = self.config['performance']
        
        return ProcessingStrategy(
            name='default',
            primary_provider=providers_config['primary'],
            fallback_chain=providers_config['fallback_chain'],
            quality_threshold=quality_config['minimum_acceptable'],
            max_cost_per_image=cost_config['max_cost_per_image'],
            max_attempts=perf_config.get('max_retries', 2) + 1,
            parallel_processing=perf_config.get('parallel_processing', False)
        )
    
    def set_strategy(self, strategy_name: str) -> bool:
        """Set processing strategy by name"""
        if strategy_name not in self.config['strategies']:
            self.logger.error(f"Unknown strategy: {strategy_name}")
            return False
        
        strategy_config = self.config['strategies'][strategy_name]
        
        self.current_strategy = ProcessingStrategy(
            name=strategy_name,
            primary_provider=strategy_config['primary_provider'],
            fallback_chain=strategy_config['fallback_chain'],
            quality_threshold=strategy_config['quality_threshold'],
            max_cost_per_image=strategy_config['max_cost_per_image'],
            max_attempts=len(strategy_config['fallback_chain'])
        )
        
        self.logger.info(f"Strategy set to: {strategy_name}")
        return True
    
    def process_with_fallback(self, input_path: str, output_path: str, 
                            quality_threshold: Optional[float] = None,
                            max_cost: Optional[float] = None) -> BackgroundRemovalResult:
        """
        Process image with automatic fallback to better providers if quality is insufficient
        
        Args:
            input_path: Path to input image
            output_path: Path for output image
            quality_threshold: Override default quality threshold
            max_cost: Override default max cost per image
            
        Returns:
            BackgroundRemovalResult: Best result achieved
        """
        start_time = time.time()
        
        # Use provided thresholds or fall back to strategy defaults
        quality_threshold = quality_threshold or self.current_strategy.quality_threshold
        max_cost = max_cost or self.current_strategy.max_cost_per_image
        
        best_result = None
        best_quality_score = 0.0
        
        # Try providers in fallback chain order
        for attempt, provider_name in enumerate(self.current_strategy.fallback_chain, 1):
            if provider_name not in self.providers:
                self.logger.warning(f"Provider not available: {provider_name}")
                continue
            
            provider = self.providers[provider_name]
            
            # Check cost constraint
            if provider.get_cost_per_image() > max_cost:
                self.logger.info(f"Skipping {provider_name} (cost ${provider.get_cost_per_image():.2f} > limit ${max_cost:.2f})")
                continue
            
            # Check if provider is available
            if not provider.is_available():
                self.logger.warning(f"Provider unavailable: {provider_name}")
                continue
            
            self.logger.info(f"Attempt {attempt}/{self.current_strategy.max_attempts}: Trying {provider_name}")
            
            # Try background removal
            result = provider.remove_background(input_path, output_path)
            
            if not result.success:
                self.logger.warning(f"Provider {provider_name} failed: {result.error_message}")
                continue
            
            # Assess quality
            quality_score = self.quality_assessor.assess_quality(input_path, result.output_path)
            result.quality_score = quality_score.overall_score
            
            self.logger.info(f"{provider_name} quality score: {quality_score.overall_score:.3f}")
            
            # Update statistics
            self._update_stats(provider_name, result, quality_score)
            
            # Check if quality meets threshold
            if quality_score.overall_score >= quality_threshold:
                self.logger.info(f"✅ Quality threshold met with {provider_name}")
                result.metadata = result.metadata or {}
                result.metadata['quality_assessment'] = asdict(quality_score)
                result.metadata['attempts'] = attempt
                return result
            
            # Keep track of best result so far
            if quality_score.overall_score > best_quality_score:
                best_result = result
                best_quality_score = quality_score.overall_score
                best_result.metadata = best_result.metadata or {}
                best_result.metadata['quality_assessment'] = asdict(quality_score)
        
        # If we get here, no provider met the quality threshold
        if best_result:
            self.logger.warning(f"No provider met quality threshold {quality_threshold:.3f}. Best: {best_quality_score:.3f}")
            best_result.metadata = best_result.metadata or {}
            best_result.metadata['quality_warning'] = f"Best quality {best_quality_score:.3f} below threshold {quality_threshold:.3f}"
            best_result.metadata['total_attempts'] = len(self.current_strategy.fallback_chain)
            return best_result
        else:
            # All providers failed
            processing_time = time.time() - start_time
            return BackgroundRemovalResult(
                success=False,
                error_message="All providers failed",
                processing_time=processing_time,
                metadata={'total_attempts': len(self.current_strategy.fallback_chain)}
            )
    
    def process_batch(self, input_output_pairs: List[tuple], 
                     max_workers: int = 4) -> List[BackgroundRemovalResult]:
        """
        Process multiple images in batch, optionally with parallel processing
        
        Args:
            input_output_pairs: List of (input_path, output_path) tuples
            max_workers: Number of parallel workers (if parallel processing enabled)
            
        Returns:
            List[BackgroundRemovalResult]: Results for each image
        """
        if not input_output_pairs:
            return []
        
        if self.current_strategy.parallel_processing and len(input_output_pairs) > 1:
            return self._process_batch_parallel(input_output_pairs, max_workers)
        else:
            return self._process_batch_sequential(input_output_pairs)
    
    def _process_batch_sequential(self, input_output_pairs: List[tuple]) -> List[BackgroundRemovalResult]:
        """Process images sequentially"""
        results = []
        
        for i, (input_path, output_path) in enumerate(input_output_pairs, 1):
            self.logger.info(f"Processing image {i}/{len(input_output_pairs)}: {Path(input_path).name}")
            result = self.process_with_fallback(input_path, output_path)
            results.append(result)
        
        return results
    
    def _process_batch_parallel(self, input_output_pairs: List[tuple], max_workers: int) -> List[BackgroundRemovalResult]:
        """Process images in parallel"""
        def process_single(pair):
            input_path, output_path = pair
            return self.process_with_fallback(input_path, output_path)
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(process_single, pair): pair[0] 
                for pair in input_output_pairs
            }
            
            for future in future_to_path:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    input_path = future_to_path[future]
                    self.logger.error(f"Processing failed for {input_path}: {str(e)}")
                    results.append(BackgroundRemovalResult(
                        success=False,
                        error_message=str(e)
                    ))
        
        return results
    
    def _update_stats(self, provider_name: str, result: BackgroundRemovalResult, quality_score: QualityScore):
        """Update processing statistics"""
        with self._stats_lock:
            self.stats['total_processed'] += 1
            
            if result.success:
                self.stats['successful'] += 1
            else:
                self.stats['failed'] += 1
            
            # Provider usage
            if provider_name not in self.stats['provider_usage']:
                self.stats['provider_usage'][provider_name] = {'count': 0, 'cost': 0.0, 'avg_quality': 0.0}
            
            provider_stats = self.stats['provider_usage'][provider_name]
            provider_stats['count'] += 1
            provider_stats['cost'] += result.cost
            
            # Update average quality
            old_avg = provider_stats['avg_quality']
            new_count = provider_stats['count']
            provider_stats['avg_quality'] = (old_avg * (new_count - 1) + quality_score.overall_score) / new_count
            
            # Total cost
            self.stats['total_cost'] += result.cost
            
            # Average quality (overall)
            old_avg_quality = self.stats['average_quality']
            total_processed = self.stats['total_processed']
            self.stats['average_quality'] = (old_avg_quality * (total_processed - 1) + quality_score.overall_score) / total_processed
            
            # Processing time
            self.stats['processing_time'] += result.processing_time
    
    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific provider"""
        if provider_name not in self.providers:
            return None
        
        provider = self.providers[provider_name]
        info = provider.get_provider_info()
        
        # Add usage statistics if available
        if provider_name in self.stats['provider_usage']:
            info['usage_stats'] = self.stats['provider_usage'][provider_name]
        
        return info
    
    def get_all_providers_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all providers"""
        return {name: self.get_provider_info(name) for name in self.providers.keys()}
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        with self._stats_lock:
            return {
                **self.stats.copy(),
                'current_strategy': asdict(self.current_strategy),
                'providers_available': len([p for p in self.providers.values() if p.is_available()]),
                'providers_total': len(self.providers)
            }
    
    def estimate_cost(self, image_count: int, strategy_name: Optional[str] = None) -> Dict[str, float]:
        """Estimate processing cost for a batch of images"""
        strategy = self.current_strategy
        if strategy_name and strategy_name in self.config['strategies']:
            strategy_config = self.config['strategies'][strategy_name]
            primary_provider = strategy_config['primary_provider']
        else:
            primary_provider = strategy.primary_provider
        
        if primary_provider not in self.providers:
            return {'error': f'Unknown provider: {primary_provider}'}
        
        provider = self.providers[primary_provider]
        cost_per_image = provider.get_cost_per_image()
        
        return {
            'primary_provider': primary_provider,
            'cost_per_image': cost_per_image,
            'total_cost_estimate': cost_per_image * image_count,
            'strategy': strategy_name or strategy.name
        }
    
    def reset_stats(self):
        """Reset processing statistics"""
        with self._stats_lock:
            self.stats = {
                'total_processed': 0,
                'successful': 0,
                'failed': 0,
                'provider_usage': {},
                'total_cost': 0.0,
                'average_quality': 0.0,
                'processing_time': 0.0
            }
        self.logger.info("Statistics reset")
    
    def save_config(self, config_path: str):
        """Save current configuration to file"""
        config_path = Path(config_path)
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
    
    def get_optimal_provider(self, criteria: str = 'balanced') -> Optional[str]:
        """Get optimal provider based on criteria"""
        available_providers = [name for name, provider in self.providers.items() if provider.is_available()]
        
        if not available_providers:
            return None
        
        if criteria == 'cost':
            return min(available_providers, key=lambda name: self.providers[name].get_cost_per_image())
        elif criteria == 'speed':
            return max(available_providers, key=lambda name: self.providers[name].get_speed_rating())
        elif criteria == 'quality':
            return max(available_providers, key=lambda name: self.providers[name].get_quality_rating())
        else:  # balanced
            # Weighted score: cost (lower better), speed and quality (higher better)
            def score(name):
                provider = self.providers[name]
                cost_score = 1.0 / (provider.get_cost_per_image() + 0.01)  # Avoid division by zero
                speed_score = provider.get_speed_rating() / 10.0
                quality_score = provider.get_quality_rating() / 10.0
                return cost_score * 0.4 + speed_score * 0.3 + quality_score * 0.3
            
            return max(available_providers, key=score)