"""
File Processing Registry System
Manages large-scale image processing with status tracking and conflict prevention
"""

import sqlite3
import json
import time
import os
import uuid
import threading
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    STALLED = "stalled"

class ProcessingRegistry:
    """Central registry for tracking file processing status with conflict prevention"""
    
    def __init__(self, registry_path: str, stall_timeout_minutes: int = 30):
        self.registry_path = registry_path
        self.stall_timeout = timedelta(minutes=stall_timeout_minutes)
        self.worker_id = str(uuid.uuid4())[:8]
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with processing registry table"""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS processing_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size INTEGER,
                    status TEXT NOT NULL DEFAULT 'pending',
                    worker_id TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    processing_duration_seconds REAL,
                    tiles_detected INTEGER,
                    products_extracted INTEGER,
                    error_message TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for efficient queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON processing_registry(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_worker ON processing_registry(worker_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON processing_registry(file_path)')
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = sqlite3.connect(self.registry_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def register_files(self, file_paths: List[str]) -> int:
        """Register multiple files for processing"""
        registered_count = 0
        
        with self._get_connection() as conn:
            for file_path in file_paths:
                try:
                    file_name = os.path.basename(file_path)
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    
                    cursor = conn.execute('''
                        INSERT OR IGNORE INTO processing_registry 
                        (file_path, file_name, file_size, status)
                        VALUES (?, ?, ?, ?)
                    ''', (file_path, file_name, file_size, ProcessingStatus.PENDING.value))
                    
                    if cursor.lastrowid:
                        registered_count += 1
                        
                except Exception as e:
                    print(f"Warning: Failed to register {file_path}: {e}")
        
        print(f"📝 Registered {registered_count} new files for processing")
        return registered_count
    
    def claim_next_file(self) -> Optional[Dict[str, Any]]:
        """Atomically claim the next pending file for processing"""
        with self._lock:
            # First, reset any stalled files
            self._reset_stalled_files()
            
            with self._get_connection() as conn:
                # Find next pending file
                cursor = conn.execute('''
                    SELECT id, file_path, file_name, file_size
                    FROM processing_registry 
                    WHERE status = ? 
                    ORDER BY created_at ASC 
                    LIMIT 1
                ''', (ProcessingStatus.PENDING.value,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Atomically claim the file
                now = datetime.now().isoformat()
                conn.execute('''
                    UPDATE processing_registry 
                    SET status = ?, worker_id = ?, started_at = ?, updated_at = ?
                    WHERE id = ? AND status = ?
                ''', (ProcessingStatus.PROCESSING.value, self.worker_id, now, now, row['id'], ProcessingStatus.PENDING.value))
                
                if conn.total_changes > 0:
                    return {
                        'id': row['id'],
                        'file_path': row['file_path'],
                        'file_name': row['file_name'], 
                        'file_size': row['file_size']
                    }
                
                return None
    
    def mark_completed(self, file_id: int, results: Dict[str, Any]):
        """Mark file processing as completed with results"""
        now = datetime.now().isoformat()
        
        # Calculate processing duration
        with self._get_connection() as conn:
            cursor = conn.execute('SELECT started_at FROM processing_registry WHERE id = ?', (file_id,))
            row = cursor.fetchone()
            
            duration = 0
            if row and row['started_at']:
                start_time = datetime.fromisoformat(row['started_at'])
                duration = (datetime.now() - start_time).total_seconds()
            
            conn.execute('''
                UPDATE processing_registry 
                SET status = ?, completed_at = ?, processing_duration_seconds = ?,
                    tiles_detected = ?, products_extracted = ?, metadata = ?, updated_at = ?
                WHERE id = ? AND worker_id = ?
            ''', (
                ProcessingStatus.COMPLETED.value, now, duration,
                results.get('tiles_detected', 0),
                results.get('products_extracted', 0), 
                json.dumps(results.get('metadata', {})),
                now, file_id, self.worker_id
            ))
    
    def mark_failed(self, file_id: int, error_message: str):
        """Mark file processing as failed with error details"""
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            conn.execute('''
                UPDATE processing_registry 
                SET status = ?, error_message = ?, completed_at = ?, updated_at = ?
                WHERE id = ? AND worker_id = ?
            ''', (ProcessingStatus.FAILED.value, error_message, now, now, file_id, self.worker_id))
    
    def _reset_stalled_files(self):
        """Reset files that have been processing too long (stalled)"""
        stall_time = (datetime.now() - self.stall_timeout).isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.execute('''
                UPDATE processing_registry 
                SET status = ?, worker_id = NULL, started_at = NULL, updated_at = ?
                WHERE status = ? AND started_at < ?
            ''', (ProcessingStatus.PENDING.value, datetime.now().isoformat(), 
                  ProcessingStatus.PROCESSING.value, stall_time))
            
            if cursor.rowcount > 0:
                print(f"⚠️  Reset {cursor.rowcount} stalled files to pending")
    
    def reset_failed_to_pending(self):
        """Reset all failed files back to pending status"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                UPDATE processing_registry 
                SET status = ?, worker_id = NULL, started_at = NULL, 
                    completed_at = NULL, error_message = NULL, updated_at = ?
                WHERE status = ?
            ''', (ProcessingStatus.PENDING.value, datetime.now().isoformat(), 
                  ProcessingStatus.FAILED.value))
            
            if cursor.rowcount > 0:
                print(f"🔄 Reset {cursor.rowcount} failed files to pending")
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Get current processing statistics"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT status, COUNT(*) as count 
                FROM processing_registry 
                GROUP BY status
            ''')
            
            stats = {status.value: 0 for status in ProcessingStatus}
            for row in cursor:
                stats[row['status']] = row['count']
                
            # Add totals
            stats['total'] = sum(stats.values())
            stats['remaining'] = stats['pending'] + stats['processing']
            
            return stats
    
    def get_processing_progress(self) -> Dict[str, Any]:
        """Get detailed processing progress information"""
        stats = self.get_processing_stats()
        
        with self._get_connection() as conn:
            # Get processing rates
            cursor = conn.execute('''
                SELECT 
                    AVG(processing_duration_seconds) as avg_duration,
                    SUM(tiles_detected) as total_tiles,
                    SUM(products_extracted) as total_products
                FROM processing_registry 
                WHERE status = ?
            ''', (ProcessingStatus.COMPLETED.value,))
            
            row = cursor.fetchone()
            
            return {
                'status_counts': stats,
                'progress_percent': (stats['completed'] / max(stats['total'], 1)) * 100,
                'avg_processing_time': row['avg_duration'] or 0,
                'total_tiles_detected': row['total_tiles'] or 0,
                'total_products_extracted': row['total_products'] or 0,
                'worker_id': self.worker_id
            }
    
    def export_results_to_csv(self, output_path: str):
        """Export all completed processing results to CSV"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT 
                    file_path, file_name, status, worker_id,
                    started_at, completed_at, processing_duration_seconds,
                    tiles_detected, products_extracted, error_message, metadata
                FROM processing_registry 
                ORDER BY created_at ASC
            ''')
            
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'file_path', 'file_name', 'status', 'worker_id',
                    'started_at', 'completed_at', 'processing_duration_seconds', 
                    'tiles_detected', 'products_extracted', 'error_message', 'metadata'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in cursor:
                    writer.writerow(dict(row))
        
        print(f"📊 Exported processing results to {output_path}")