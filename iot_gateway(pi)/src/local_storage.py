# src/local_storage.py
"""
Local storage module
Uses SQLite to store sensor data
"""

import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalStorage:
    """
    Local SQLite storage
    
    Features:
    1. Store all received sensor data
    2. Query by time and node
    3. Automatically clean up stale data
    4. Cache not-yet-uploaded data when offline
    """
    
    def __init__(
        self, 
        db_path: str = "data/sensor_data.db",
        max_records: int = 100000
    ):
        """
        Initialize storage
        
        Args:
            db_path: database file path
            max_records: maximum record count
        """
        self.db_path = db_path
        self.max_records = max_records
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sensor data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                value REAL,
                quality REAL,
                is_anomaly INTEGER,
                raw_data TEXT,
                uploaded INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_node_timestamp 
            ON sensor_data(node_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_uploaded 
            ON sensor_data(uploaded)
        """)
        
        # Create system log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT,
                message TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def save_sensor_data(self, data: Dict[str, Any]) -> bool:
        """
        Save sensor data
        
        Args:
            data: processed sensor data
            
        Returns:
            whether save succeeded
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            node_id = data.get("node_id", "unknown")
            timestamp = data.get("timestamp", datetime.now().isoformat())
            
            # Save each sensor reading
            sensors = data.get("sensors", {})
            for sensor_type, sensor_info in sensors.items():
                cursor.execute("""
                    INSERT INTO sensor_data 
                    (node_id, timestamp, sensor_type, value, quality, 
                     is_anomaly, raw_data, uploaded)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """, (
                    node_id,
                    timestamp,
                    sensor_type,
                    sensor_info.get("value"),
                    sensor_info.get("quality", 1.0),
                    1 if sensor_info.get("is_anomaly", False) else 0,
                    json.dumps(sensor_info)
                ))
            
            conn.commit()
            conn.close()
            
            # Check whether cleanup is needed
            self._cleanup_if_needed()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save sensor data: {e}")
            return False
    
    def get_unuploaded_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get data not uploaded yet
        
        Args:
            limit: maximum number of returned records
            
        Returns:
            list of unuploaded records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, node_id, timestamp, sensor_type, 
                       value, quality, is_anomaly
                FROM sensor_data
                WHERE uploaded = 0
                ORDER BY timestamp ASC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Group rows by timestamp
            data_dict = {}
            for row in rows:
                id_, node_id, timestamp, sensor_type, value, quality, is_anomaly = row
                
                key = f"{node_id}_{timestamp}"
                if key not in data_dict:
                    data_dict[key] = {
                        "ids": [],
                        "node_id": node_id,
                        "timestamp": timestamp,
                        "sensors": {}
                    }
                
                data_dict[key]["ids"].append(id_)
                data_dict[key]["sensors"][sensor_type] = {
                    "value": value,
                    "quality": quality,
                    "is_anomaly": bool(is_anomaly)
                }
            
            return list(data_dict.values())
            
        except Exception as e:
            logger.error(f"Failed to get unuploaded data: {e}")
            return []
    
    def mark_as_uploaded(self, record_ids: List[int]):
        """Mark records as uploaded"""
        if not record_ids:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            placeholders = ",".join(["?" for _ in record_ids])
            cursor.execute(f"""
                UPDATE sensor_data
                SET uploaded = 1
                WHERE id IN ({placeholders})
            """, record_ids)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Marked {len(record_ids)} records as uploaded")
            
        except Exception as e:
            logger.error(f"Failed to mark data as uploaded: {e}")
    
    def get_recent_data(
        self, 
        node_id: Optional[str] = None,
        sensor_type: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get recent data
        
        Args:
            node_id: node ID (optional)
            sensor_type: sensor type (optional)
            hours: time range in hours
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            since = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            query = """
                SELECT node_id, timestamp, sensor_type, value, quality
                FROM sensor_data
                WHERE timestamp > ?
            """
            params = [since]
            
            if node_id:
                query += " AND node_id = ?"
                params.append(node_id)
            
            if sensor_type:
                query += " AND sensor_type = ?"
                params.append(sensor_type)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "node_id": row[0],
                    "timestamp": row[1],
                    "sensor_type": row[2],
                    "value": row[3],
                    "quality": row[4]
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Failed to get recent data: {e}")
            return []
    
    def _cleanup_if_needed(self):
        """Clean stale data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current record count
            cursor.execute("SELECT COUNT(*) FROM sensor_data")
            count = cursor.fetchone()[0]
            
            if count > self.max_records:
                # Delete oldest uploaded rows
                delete_count = count - self.max_records + 1000
                cursor.execute("""
                    DELETE FROM sensor_data
                    WHERE id IN (
                        SELECT id FROM sensor_data
                        WHERE uploaded = 1
                        ORDER BY timestamp ASC
                        LIMIT ?
                    )
                """, (delete_count,))
                
                conn.commit()
                logger.info(f"Cleaned up {delete_count} old records")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to cleanup database: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total records
            cursor.execute("SELECT COUNT(*) FROM sensor_data")
            total = cursor.fetchone()[0]
            
            # Unuploaded records
            cursor.execute("SELECT COUNT(*) FROM sensor_data WHERE uploaded = 0")
            unuploaded = cursor.fetchone()[0]
            
            # Record counts by node
            cursor.execute("""
                SELECT node_id, COUNT(*) 
                FROM sensor_data 
                GROUP BY node_id
            """)
            by_node = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_records": total,
                "unuploaded": unuploaded,
                "by_node": by_node,
                "database_path": self.db_path
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}