#!/usr/bin/env python3
"""
Background Removal Quality Assessment Engine
Automatically evaluates the quality of background removal results
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import logging


@dataclass
class QualityScore:
    """Comprehensive quality score for background removal"""
    overall_score: float  # 0.0 - 1.0
    edge_sharpness: float  # 0.0 - 1.0
    transparency_accuracy: float  # 0.0 - 1.0
    color_preservation: float  # 0.0 - 1.0
    artifact_score: float  # 0.0 - 1.0 (higher = fewer artifacts)
    background_removal_completeness: float  # 0.0 - 1.0
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class BackgroundQualityAssessor:
    """
    Automated quality assessment for background removal results
    """
    
    def __init__(self):
        self.logger = logging.getLogger('QualityAssessor')
        
        # Quality thresholds
        self.thresholds = {
            'excellent': 0.90,
            'good': 0.75,
            'acceptable': 0.60,
            'poor': 0.40
        }
    
    def assess_quality(self, original_path: str, processed_path: str, 
                      reference_path: Optional[str] = None) -> QualityScore:
        """
        Comprehensive quality assessment of background removal
        
        Args:
            original_path: Path to original image
            processed_path: Path to background-removed image
            reference_path: Optional path to reference/ground truth image
            
        Returns:
            QualityScore: Detailed quality metrics
        """
        try:
            # Load images
            original = self._load_image(original_path)
            processed = self._load_image(processed_path)
            
            if original is None or processed is None:
                return QualityScore(
                    overall_score=0.0,
                    edge_sharpness=0.0,
                    transparency_accuracy=0.0,
                    color_preservation=0.0,
                    artifact_score=0.0,
                    background_removal_completeness=0.0,
                    details={'error': 'Could not load images'}
                )
            
            # Ensure processed image has alpha channel
            if processed.shape[2] != 4:
                self.logger.warning("Processed image doesn't have alpha channel, converting...")
                processed = self._ensure_alpha_channel(processed)
            
            # Individual quality metrics
            edge_score = self._assess_edge_sharpness(original, processed)
            transparency_score = self._assess_transparency_accuracy(processed)
            color_score = self._assess_color_preservation(original, processed)
            artifact_score = self._assess_artifacts(processed)
            background_score = self._assess_background_removal_completeness(original, processed)
            
            # Calculate overall score (weighted average)
            weights = {
                'edge_sharpness': 0.25,
                'transparency_accuracy': 0.20,
                'color_preservation': 0.20,
                'artifact_score': 0.15,
                'background_removal_completeness': 0.20
            }
            
            overall_score = (
                edge_score * weights['edge_sharpness'] +
                transparency_score * weights['transparency_accuracy'] +
                color_score * weights['color_preservation'] +
                artifact_score * weights['artifact_score'] +
                background_score * weights['background_removal_completeness']
            )
            
            # Additional analysis details
            details = {
                'weights_used': weights,
                'image_dimensions': {
                    'original': original.shape,
                    'processed': processed.shape
                },
                'alpha_channel_stats': self._analyze_alpha_channel(processed),
                'quality_category': self._categorize_quality(overall_score)
            }
            
            return QualityScore(
                overall_score=overall_score,
                edge_sharpness=edge_score,
                transparency_accuracy=transparency_score,
                color_preservation=color_score,
                artifact_score=artifact_score,
                background_removal_completeness=background_score,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {str(e)}")
            return QualityScore(
                overall_score=0.0,
                edge_sharpness=0.0,
                transparency_accuracy=0.0,
                color_preservation=0.0,
                artifact_score=0.0,
                background_removal_completeness=0.0,
                details={'error': str(e)}
            )
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image with proper format handling"""
        try:
            path = Path(image_path)
            if not path.exists():
                return None
                
            # Load with OpenCV (handles most formats)
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                # Try with PIL as fallback
                pil_img = Image.open(path)
                img = np.array(pil_img)
                
                # Convert RGB to BGR if needed (OpenCV format)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            
            return img
            
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {str(e)}")
            return None
    
    def _ensure_alpha_channel(self, image: np.ndarray) -> np.ndarray:
        """Ensure image has alpha channel"""
        if len(image.shape) == 2:  # Grayscale
            # Convert to BGRA
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            alpha = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
            return np.concatenate([bgr, alpha], axis=2)
        elif image.shape[2] == 3:  # BGR
            # Add full alpha channel
            alpha = np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype) * 255
            return np.concatenate([image, alpha], axis=2)
        else:  # Already has alpha
            return image
    
    def _assess_edge_sharpness(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Assess how well edges are preserved"""
        try:
            # Convert to grayscale for edge detection
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
            
            # For processed image, use RGB channels only (ignore alpha)
            if processed.shape[2] == 4:
                proc_rgb = processed[:, :, :3]
                proc_gray = cv2.cvtColor(proc_rgb, cv2.COLOR_BGR2GRAY)
            else:
                proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            
            # Resize if dimensions don't match
            if orig_gray.shape != proc_gray.shape:
                proc_gray = cv2.resize(proc_gray, (orig_gray.shape[1], orig_gray.shape[0]))
            
            # Detect edges using Canny
            orig_edges = cv2.Canny(orig_gray, 50, 150)
            proc_edges = cv2.Canny(proc_gray, 50, 150)
            
            # Calculate edge preservation score
            orig_edge_pixels = np.sum(orig_edges > 0)
            proc_edge_pixels = np.sum(proc_edges > 0)
            
            if orig_edge_pixels == 0:
                return 1.0  # No edges to preserve
            
            # Compare edge similarity
            edge_similarity = np.sum(np.logical_and(orig_edges > 0, proc_edges > 0)) / orig_edge_pixels
            
            return min(1.0, edge_similarity)
            
        except Exception as e:
            self.logger.error(f"Edge sharpness assessment failed: {str(e)}")
            return 0.5  # Default moderate score
    
    def _assess_transparency_accuracy(self, processed: np.ndarray) -> float:
        """Assess quality of transparency/alpha channel"""
        try:
            if processed.shape[2] != 4:
                return 0.0  # No alpha channel
            
            alpha = processed[:, :, 3]
            
            # Check for binary alpha (good) vs gray alpha (potentially problematic)
            unique_alpha_values = len(np.unique(alpha))
            
            # Good transparency should have mostly binary values (0 or 255)
            binary_pixels = np.sum((alpha == 0) | (alpha == 255))
            total_pixels = alpha.size
            
            binary_ratio = binary_pixels / total_pixels
            
            # Penalize too many intermediate alpha values (could indicate artifacts)
            if unique_alpha_values > 20:  # Too many gradations
                return binary_ratio * 0.7
            
            return binary_ratio
            
        except Exception as e:
            self.logger.error(f"Transparency assessment failed: {str(e)}")
            return 0.5
    
    def _assess_color_preservation(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Assess how well original colors are preserved"""
        try:
            # Convert both to same format for comparison
            if len(original.shape) == 3 and original.shape[2] == 3:
                orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            else:
                orig_rgb = original[:, :, :3]
            
            if processed.shape[2] == 4:
                # Only compare non-transparent pixels
                proc_rgb = processed[:, :, :3]
                alpha = processed[:, :, 3]
                mask = alpha > 128  # Non-transparent pixels
            else:
                proc_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                mask = np.ones(processed.shape[:2], dtype=bool)
            
            # Resize if needed
            if orig_rgb.shape != proc_rgb.shape:
                proc_rgb = cv2.resize(proc_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]))
                if mask.shape != orig_rgb.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (orig_rgb.shape[1], orig_rgb.shape[0])) > 0
            
            # Compare colors only in non-transparent regions
            if np.sum(mask) == 0:
                return 0.0  # No visible pixels
            
            orig_colors = orig_rgb[mask]
            proc_colors = proc_rgb[mask]
            
            # Calculate color similarity using MSE
            mse = np.mean((orig_colors.astype(float) - proc_colors.astype(float)) ** 2)
            
            # Convert MSE to similarity score (0-1)
            max_mse = 255 ** 2  # Maximum possible MSE
            similarity = 1.0 - (mse / max_mse)
            
            return max(0.0, similarity)
            
        except Exception as e:
            self.logger.error(f"Color preservation assessment failed: {str(e)}")
            return 0.5
    
    def _assess_artifacts(self, processed: np.ndarray) -> float:
        """Assess presence of processing artifacts"""
        try:
            if processed.shape[2] != 4:
                return 0.5  # Can't assess without alpha
            
            alpha = processed[:, :, 3]
            
            # Look for common artifacts
            artifact_score = 1.0
            
            # 1. Check for noise in alpha channel
            alpha_noise = self._calculate_noise(alpha)
            artifact_score *= (1.0 - min(1.0, alpha_noise / 50.0))  # Penalize high noise
            
            # 2. Check for rough edges (should be smooth)
            edge_roughness = self._calculate_edge_roughness(alpha)
            artifact_score *= (1.0 - min(1.0, edge_roughness / 100.0))
            
            # 3. Check for isolated transparent pixels (holes in subject)
            isolated_pixels = self._count_isolated_pixels(alpha)
            total_pixels = alpha.size
            isolated_ratio = isolated_pixels / total_pixels
            artifact_score *= (1.0 - min(1.0, isolated_ratio * 10))
            
            return max(0.0, artifact_score)
            
        except Exception as e:
            self.logger.error(f"Artifact assessment failed: {str(e)}")
            return 0.5
    
    def _assess_background_removal_completeness(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Assess how completely the background was removed"""
        try:
            if processed.shape[2] != 4:
                return 0.0  # No transparency
            
            alpha = processed[:, :, 3]
            
            # Calculate the ratio of transparent vs opaque pixels
            transparent_pixels = np.sum(alpha < 128)
            total_pixels = alpha.size
            
            # For product images, we expect significant background removal
            # but not too much (shouldn't remove the product itself)
            transparency_ratio = transparent_pixels / total_pixels
            
            # Optimal range: 40-80% transparent (product in center, background removed)
            if 0.3 <= transparency_ratio <= 0.8:
                return 1.0
            elif transparency_ratio < 0.1:
                return 0.2  # Barely any background removed
            elif transparency_ratio > 0.95:
                return 0.1  # Too much removed (likely the product too)
            else:
                # Gradually decrease score as we move away from optimal range
                if transparency_ratio < 0.3:
                    return transparency_ratio / 0.3
                else:  # transparency_ratio > 0.8
                    return (1.0 - transparency_ratio) / 0.2
            
        except Exception as e:
            self.logger.error(f"Background removal assessment failed: {str(e)}")
            return 0.5
    
    def _calculate_noise(self, image: np.ndarray) -> float:
        """Calculate noise level in image"""
        try:
            # Use Laplacian variance as noise metric
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            return laplacian.var()
        except:
            return 0.0
    
    def _calculate_edge_roughness(self, alpha: np.ndarray) -> float:
        """Calculate edge roughness in alpha channel"""
        try:
            # Find edges in alpha channel
            edges = cv2.Canny(alpha, 50, 150)
            
            # Calculate edge density (more edges = rougher)
            edge_pixels = np.sum(edges > 0)
            total_pixels = alpha.size
            
            return (edge_pixels / total_pixels) * 1000  # Scale for readability
        except:
            return 0.0
    
    def _count_isolated_pixels(self, alpha: np.ndarray, threshold: int = 128) -> int:
        """Count isolated transparent pixels (potential artifacts)"""
        try:
            # Create binary mask
            mask = alpha > threshold
            
            # Use morphological operations to find isolated pixels
            kernel = np.ones((3, 3), np.uint8)
            opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            
            # Count difference (isolated pixels)
            isolated = mask.astype(int) - opened.astype(int)
            return np.sum(isolated > 0)
        except:
            return 0
    
    def _analyze_alpha_channel(self, processed: np.ndarray) -> Dict[str, Any]:
        """Analyze alpha channel statistics"""
        try:
            if processed.shape[2] != 4:
                return {'has_alpha': False}
            
            alpha = processed[:, :, 3]
            
            return {
                'has_alpha': True,
                'unique_values': len(np.unique(alpha)),
                'mean_alpha': float(np.mean(alpha)),
                'transparent_ratio': float(np.sum(alpha < 128) / alpha.size),
                'opaque_ratio': float(np.sum(alpha > 200) / alpha.size),
                'semi_transparent_ratio': float(np.sum((alpha >= 128) & (alpha <= 200)) / alpha.size)
            }
        except:
            return {'has_alpha': False, 'error': 'Failed to analyze alpha channel'}
    
    def _categorize_quality(self, score: float) -> str:
        """Categorize quality score into human-readable category"""
        for category, threshold in sorted(self.thresholds.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return category
        return 'poor'
    
    def is_acceptable_quality(self, quality_score: QualityScore, min_threshold: float = 0.6) -> bool:
        """Check if quality meets minimum acceptable standards"""
        return quality_score.overall_score >= min_threshold
    
    def get_quality_report(self, quality_score: QualityScore) -> str:
        """Generate human-readable quality report"""
        category = quality_score.details.get('quality_category', 'unknown')
        
        report = f"Quality Assessment Report\n"
        report += f"========================\n"
        report += f"Overall Score: {quality_score.overall_score:.3f} ({category})\n\n"
        report += f"Detailed Metrics:\n"
        report += f"  Edge Sharpness: {quality_score.edge_sharpness:.3f}\n"
        report += f"  Transparency: {quality_score.transparency_accuracy:.3f}\n"
        report += f"  Color Preservation: {quality_score.color_preservation:.3f}\n"
        report += f"  Artifact Score: {quality_score.artifact_score:.3f}\n"
        report += f"  Background Removal: {quality_score.background_removal_completeness:.3f}\n"
        
        if 'alpha_channel_stats' in quality_score.details:
            stats = quality_score.details['alpha_channel_stats']
            if stats.get('has_alpha'):
                report += f"\nAlpha Channel Analysis:\n"
                report += f"  Transparent Pixels: {stats.get('transparent_ratio', 0):.1%}\n"
                report += f"  Opaque Pixels: {stats.get('opaque_ratio', 0):.1%}\n"
                report += f"  Unique Alpha Values: {stats.get('unique_values', 0)}\n"
        
        return report