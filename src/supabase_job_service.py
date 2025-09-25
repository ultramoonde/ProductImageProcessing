"""
Supabase Job Service
Integrates image processing with existing Supabase project for persistent job management
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️ Supabase not available. Install: pip install supabase")

# Import configuration
try:
    from config import get_supabase_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

@dataclass
class ProcessingJob:
    """Data class for processing job information"""
    id: str
    job_name: str
    source_type: str
    source_folder: str
    status: str
    total_files: int = 0
    files_pending: int = 0
    files_processing: int = 0
    files_completed: int = 0
    files_failed: int = 0
    processing_config: Dict = None
    created_at: str = None
    created_by: str = None

class SupabaseJobService:
    """Service for managing image processing jobs using Supabase"""
    
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """Initialize Supabase job service"""
        
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase client not available. Install: pip install supabase")
        
        # Get credentials - prioritize parameters, then config, then environment
        if supabase_url and supabase_key:
            self.supabase_url = supabase_url
            self.supabase_key = supabase_key
        elif CONFIG_AVAILABLE:
            try:
                self.supabase_url, self.supabase_key = get_supabase_config()
            except ValueError:
                # Fall back to direct environment lookup
                self.supabase_url = os.getenv('SUPABASE_URL')
                self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        else:
            self.supabase_url = os.getenv('SUPABASE_URL')
            self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and key required. Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables or create .env file")
        
        # Initialize Supabase client
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        print("✅ Supabase job service initialized")
    
    def create_job(self, job_name: str, source_type: str, source_folder: str, 
                   processing_config: Dict = None, created_by: str = None) -> str:
        """Create a new processing job and return job ID"""
        
        job_data = {
            'job_name': job_name,
            'source_type': source_type,
            'source_folder': source_folder,
            'status': 'pending',
            'processing_config': processing_config or {},
            'created_by': created_by
        }
        
        try:
            result = self.supabase.table('processing_jobs').insert(job_data).execute()
            job_id = result.data[0]['id']
            print(f"✅ Created job: {job_name} (ID: {job_id})")
            return job_id
            
        except Exception as e:
            print(f"❌ Failed to create job: {e}")
            raise
    
    def register_files_for_job(self, job_id: str, file_paths: List[str]) -> int:
        """Register multiple files for processing under a job"""
        
        file_records = []
        for file_path in file_paths:
            try:
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                file_records.append({
                    'job_id': job_id,
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'file_size_bytes': file_size,
                    'status': 'pending'
                })
            except Exception as e:
                print(f"Warning: Failed to register {file_path}: {e}")
        
        if not file_records:
            return 0
        
        try:
            # Insert files
            result = self.supabase.table('processing_files').insert(file_records).execute()
            
            # Update job totals
            total_files = len(file_records)
            self.supabase.table('processing_jobs').update({
                'total_files': total_files,
                'files_pending': total_files
            }).eq('id', job_id).execute()
            
            print(f"📝 Registered {total_files} files for job {job_id}")
            return total_files
            
        except Exception as e:
            print(f"❌ Failed to register files: {e}")
            raise
    
    def claim_next_file(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Atomically claim next pending file for processing"""
        
        try:
            # Find pending file
            result = self.supabase.table('processing_files').select("*").eq('status', 'pending').order('created_at').limit(1).execute()
            
            if not result.data:
                return None
            
            file_record = result.data[0]
            
            # Atomically claim the file
            update_result = self.supabase.table('processing_files').update({
                'status': 'processing',
                'worker_id': worker_id,
                'started_at': datetime.now().isoformat()
            }).eq('id', file_record['id']).eq('status', 'pending').execute()
            
            if update_result.data:
                # Update job counters
                self._update_job_counters(file_record['job_id'])
                print(f"🔄 Worker {worker_id} claimed file: {file_record['file_name']}")
                return update_result.data[0]
            
            return None
            
        except Exception as e:
            print(f"❌ Failed to claim file: {e}")
            return None
    
    def mark_file_completed(self, file_id: str, worker_id: str, results: Dict[str, Any]):
        """Mark file as completed with processing results"""
        
        try:
            # Update file record
            self.supabase.table('processing_files').update({
                'status': 'completed',
                'completed_at': datetime.now().isoformat(),
                'processing_duration_seconds': results.get('processing_duration_seconds'),
                'tiles_detected': results.get('tiles_detected', 0),
                'products_extracted': results.get('products_extracted', 0),
                'extracted_data': results.get('products_preview', [])
            }).eq('id', file_id).eq('worker_id', worker_id).execute()
            
            # Get job_id for counter update
            file_result = self.supabase.table('processing_files').select('job_id').eq('id', file_id).execute()
            if file_result.data:
                job_id = file_result.data[0]['job_id']
                self._update_job_counters(job_id)
            
            print(f"✅ File {file_id} marked completed")
            
        except Exception as e:
            print(f"❌ Failed to mark file completed: {e}")
            raise
    
    def mark_file_failed(self, file_id: str, worker_id: str, error_message: str):
        """Mark file as failed with error details"""
        
        try:
            self.supabase.table('processing_files').update({
                'status': 'failed',
                'completed_at': datetime.now().isoformat(),
                'error_message': error_message
            }).eq('id', file_id).eq('worker_id', worker_id).execute()
            
            # Get job_id for counter update
            file_result = self.supabase.table('processing_files').select('job_id').eq('id', file_id).execute()
            if file_result.data:
                job_id = file_result.data[0]['job_id']
                self._update_job_counters(job_id)
            
            print(f"❌ File {file_id} marked failed: {error_message}")
            
        except Exception as e:
            print(f"❌ Failed to mark file failed: {e}")
    
    def _update_job_counters(self, job_id: str):
        """Update job progress counters based on file statuses"""
        
        try:
            # Count files by status
            counts = {}
            for status in ['pending', 'processing', 'completed', 'failed']:
                result = self.supabase.table('processing_files').select('id', count='exact').eq('job_id', job_id).eq('status', status).execute()
                counts[f'files_{status}'] = result.count or 0
            
            # Check if job is complete
            if counts['files_pending'] == 0 and counts['files_processing'] == 0:
                if counts['files_completed'] > 0:
                    job_status = 'completed'
                    completed_at = datetime.now().isoformat()
                else:
                    job_status = 'failed' 
                    completed_at = datetime.now().isoformat()
                
                counts['status'] = job_status
                counts['completed_at'] = completed_at
            elif counts['files_processing'] > 0:
                counts['status'] = 'processing'
                if not self._job_has_started(job_id):
                    counts['started_at'] = datetime.now().isoformat()
            
            # Update job record
            self.supabase.table('processing_jobs').update(counts).eq('id', job_id).execute()
            
        except Exception as e:
            print(f"⚠️ Failed to update job counters: {e}")
    
    def _job_has_started(self, job_id: str) -> bool:
        """Check if job has a started_at timestamp"""
        result = self.supabase.table('processing_jobs').select('started_at').eq('id', job_id).execute()
        return result.data and result.data[0]['started_at'] is not None
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current job status and progress"""
        
        try:
            result = self.supabase.table('processing_jobs').select("*").eq('id', job_id).execute()
            
            if result.data:
                job = result.data[0]
                
                # Calculate completion percentage
                total = job['total_files']
                completed = job['files_completed'] + job['files_failed']
                job['completion_percentage'] = (completed / total * 100) if total > 0 else 0
                
                return job
            
            return None
            
        except Exception as e:
            print(f"❌ Failed to get job status: {e}")
            return None
    
    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get all active (pending/processing) jobs"""
        
        try:
            result = self.supabase.table('job_summary').select("*").in_('status', ['pending', 'processing']).order('created_at', desc=True).execute()
            return result.data
            
        except Exception as e:
            print(f"❌ Failed to get active jobs: {e}")
            return []
    
    def save_extracted_products(self, file_id: str, job_id: str, products: List[Dict[str, Any]]):
        """Save extracted product data to database"""
        
        if not products:
            return
        
        try:
            # Prepare product records
            product_records = []
            for product in products:
                product_record = {
                    'file_id': file_id,
                    'job_id': job_id,
                    'product_name': product.get('product_name'),
                    'brand': product.get('brand'),
                    'manufacturer': product.get('manufacturer'),
                    'category': product.get('category'),
                    'subcategory': product.get('subcategory'),
                    'price': product.get('price'),
                    'price_numeric': product.get('price_numeric'),
                    'currency': product.get('currency', 'EUR'),
                    'weight': product.get('weight'),
                    'weight_numeric': product.get('weight_numeric'),
                    'weight_unit': product.get('weight_unit'),
                    'quantity': product.get('quantity'),
                    'quantity_numeric': product.get('quantity_numeric'),
                    'organic_certified': product.get('organic_certified', False),
                    'bio_label': product.get('bio_label', False),
                    'extraction_method': product.get('extraction_method', 'AI_Consensus'),
                    'ai_confidence_score': product.get('ai_confidence_score'),
                    'data_quality_score': product.get('data_quality_score'),
                    'manual_review_required': product.get('manual_review_required', False),
                    'extracted_image_filename': product.get('extracted_image_filename'),
                    'tile_position': product.get('tile_position'),
                    'ocr_text_raw': product.get('ocr_text_raw'),
                    'processing_metadata': product.get('processing_metadata', {})
                }
                product_records.append(product_record)
            
            # Insert products
            result = self.supabase.table('extracted_products').insert(product_records).execute()
            print(f"💾 Saved {len(product_records)} products to database")
            
        except Exception as e:
            print(f"❌ Failed to save products: {e}")
    
    def register_worker(self, worker_id: str, worker_name: str = None) -> bool:
        """Register or update worker in database"""
        
        try:
            worker_data = {
                'worker_id': worker_id,
                'worker_name': worker_name or f"Worker-{worker_id[:8]}",
                'status': 'idle',
                'last_seen': datetime.now().isoformat(),
                'last_heartbeat': datetime.now().isoformat()
            }
            
            # Upsert worker record
            result = self.supabase.table('processing_workers').upsert(worker_data, on_conflict='worker_id').execute()
            print(f"👷 Registered worker: {worker_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to register worker: {e}")
            return False
    
    def update_worker_heartbeat(self, worker_id: str):
        """Update worker heartbeat timestamp"""
        
        try:
            self.supabase.table('processing_workers').update({
                'last_heartbeat': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat()
            }).eq('worker_id', worker_id).execute()
            
        except Exception as e:
            print(f"⚠️ Failed to update worker heartbeat: {e}")
    
    def export_job_results_to_csv(self, job_id: str, output_path: str) -> bool:
        """Export job results to CSV file"""
        
        try:
            # Get all products for the job
            result = self.supabase.table('extracted_products').select("*").eq('job_id', job_id).execute()
            
            if not result.data:
                print(f"⚠️ No products found for job {job_id}")
                return False
            
            import csv
            
            # Define CSV headers
            headers = [
                'product_name', 'brand', 'manufacturer', 'category', 'subcategory',
                'price', 'price_numeric', 'currency', 'weight', 'weight_numeric', 'weight_unit',
                'quantity', 'quantity_numeric', 'organic_certified', 'bio_label',
                'extraction_method', 'ai_confidence_score', 'data_quality_score',
                'extracted_image_filename', 'tile_position', 'created_at'
            ]
            
            # Write CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                
                for product in result.data:
                    # Filter to only include defined headers
                    row = {key: product.get(key, '') for key in headers}
                    writer.writerow(row)
            
            print(f"📊 Exported {len(result.data)} products to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export CSV: {e}")
            return False