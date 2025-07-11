"""
Authentication and Security Module
==================================

Provides user authentication, session management, and security features
for the forensic dashboard. Includes audit logging and access control.
"""

import hashlib
import hmac
import secrets
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import streamlit as st
from cryptography.fernet import Fernet
import jwt
import logging


@dataclass
class User:
    """User data model."""
    username: str
    email: str
    role: str
    permissions: List[str]
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    created_at: Optional[datetime] = None


@dataclass
class AuditLog:
    """Audit log entry model."""
    timestamp: datetime
    user: str
    action: str
    resource: str
    ip_address: str
    user_agent: str
    success: bool
    details: Optional[Dict] = None


class SecurityManager:
    """Handles security operations like encryption and hashing."""
    
    def __init__(self, secret_key: str):
        """Initialize with secret key for encryption."""
        self.secret_key = secret_key.encode()
        self.fernet = Fernet(Fernet.generate_key())
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 with SHA-256
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return key.hex(), salt
    
    def verify_password(self, password: str, hashed: str, salt: str) -> bool:
        """Verify password against hash."""
        key, _ = self.hash_password(password, salt)
        return hmac.compare_digest(key, hashed)
    
    def generate_token(self, user_data: Dict, expires_hours: int = 24) -> str:
        """Generate JWT token for user session."""
        payload = {
            'user_data': user_data,
            'exp': datetime.utcnow() + timedelta(hours=expires_hours),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload['user_data']
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()


class DatabaseManager:
    """Manages database operations for authentication."""
    
    def __init__(self, db_path: str):
        """Initialize database connection."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'investigator',
                    permissions TEXT NOT NULL DEFAULT '[]',
                    failed_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP NULL,
                    last_login TIMESTAMP NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Audit log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    username TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource TEXT NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    success BOOLEAN NOT NULL,
                    details TEXT
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (username) REFERENCES users (username)
                )
            """)
            
            conn.commit()
    
    def create_user(self, username: str, email: str, password_hash: str, 
                   password_salt: str, role: str = 'investigator', 
                   permissions: List[str] = None) -> bool:
        """Create a new user."""
        if permissions is None:
            permissions = ['read']
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (username, email, password_hash, password_salt, role, permissions)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (username, email, password_hash, password_salt, role, json.dumps(permissions)))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
    
    def get_user(self, username: str) -> Optional[User]:
        """Retrieve user by username."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT username, email, role, permissions, last_login, 
                       failed_attempts, locked_until, created_at
                FROM users WHERE username = ?
            """, (username,))
            
            row = cursor.fetchone()
            if row:
                return User(
                    username=row[0],
                    email=row[1],
                    role=row[2],
                    permissions=json.loads(row[3]),
                    last_login=datetime.fromisoformat(row[4]) if row[4] else None,
                    failed_attempts=row[5],
                    locked_until=datetime.fromisoformat(row[6]) if row[6] else None,
                    created_at=datetime.fromisoformat(row[7]) if row[7] else None
                )
        return None
    
    def get_user_credentials(self, username: str) -> Optional[Tuple[str, str]]:
        """Get user password hash and salt."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT password_hash, password_salt FROM users WHERE username = ?
            """, (username,))
            
            row = cursor.fetchone()
            if row:
                return row[0], row[1]
        return None
    
    def update_login_attempt(self, username: str, success: bool):
        """Update user login attempt information."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if success:
                cursor.execute("""
                    UPDATE users SET 
                        last_login = CURRENT_TIMESTAMP,
                        failed_attempts = 0,
                        locked_until = NULL
                    WHERE username = ?
                """, (username,))
            else:
                cursor.execute("""
                    UPDATE users SET 
                        failed_attempts = failed_attempts + 1,
                        locked_until = CASE 
                            WHEN failed_attempts >= 4 THEN datetime('now', '+1 hour')
                            ELSE locked_until
                        END
                    WHERE username = ?
                """, (username,))
            
            conn.commit()
    
    def log_audit_event(self, audit_log: AuditLog):
        """Log audit event to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_log (timestamp, username, action, resource, 
                                     ip_address, user_agent, success, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_log.timestamp.isoformat(),
                audit_log.user,
                audit_log.action,
                audit_log.resource,
                audit_log.ip_address,
                audit_log.user_agent,
                audit_log.success,
                json.dumps(audit_log.details) if audit_log.details else None
            ))
            conn.commit()


class AuthenticationManager:
    """Main authentication manager for the dashboard."""
    
    def __init__(self, secret_key: str, db_path: str):
        """Initialize authentication manager."""
        self.security = SecurityManager(secret_key)
        self.db = DatabaseManager(db_path)
        self.logger = logging.getLogger(__name__)
    
    def create_user(self, username: str, email: str, password: str, 
                   role: str = 'investigator', permissions: List[str] = None) -> bool:
        """Create a new user account."""
        password_hash, password_salt = self.security.hash_password(password)
        return self.db.create_user(username, email, password_hash, password_salt, role, permissions)
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user credentials."""
        # Get user
        user = self.db.get_user(username)
        if not user:
            self._log_audit("login", username, "authentication", False, {"reason": "user_not_found"})
            return None
        
        # Check if account is locked
        if user.locked_until and datetime.now() < user.locked_until:
            self._log_audit("login", username, "authentication", False, {"reason": "account_locked"})
            return None
        
        # Get credentials and verify password
        credentials = self.db.get_user_credentials(username)
        if not credentials:
            self._log_audit("login", username, "authentication", False, {"reason": "credentials_not_found"})
            return None
        
        password_hash, password_salt = credentials
        if self.security.verify_password(password, password_hash, password_salt):
            self.db.update_login_attempt(username, True)
            self._log_audit("login", username, "authentication", True)
            return user
        else:
            self.db.update_login_attempt(username, False)
            self._log_audit("login", username, "authentication", False, {"reason": "invalid_password"})
            return None
    
    def check_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in user.permissions or 'admin' in user.permissions
    
    def generate_session_token(self, user: User) -> str:
        """Generate session token for authenticated user."""
        user_data = {
            'username': user.username,
            'email': user.email,
            'role': user.role,
            'permissions': user.permissions
        }
        return self.security.generate_token(user_data)
    
    def verify_session_token(self, token: str) -> Optional[Dict]:
        """Verify session token."""
        return self.security.verify_token(token)
    
    def _log_audit(self, action: str, username: str, resource: str, 
                  success: bool, details: Optional[Dict] = None):
        """Log audit event."""
        # Get client info from Streamlit session if available
        ip_address = "unknown"
        user_agent = "unknown"
        
        if hasattr(st, 'session_state') and 'client_info' in st.session_state:
            client_info = st.session_state.client_info
            ip_address = client_info.get('ip', ip_address)
            user_agent = client_info.get('user_agent', user_agent)
        
        audit_log = AuditLog(
            timestamp=datetime.now(),
            user=username,
            action=action,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details
        )
        
        self.db.log_audit_event(audit_log)
    
    def get_audit_logs(self, limit: int = 100, username: Optional[str] = None,
                      action: Optional[str] = None, start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> List[Dict]:
        """Retrieve audit logs with optional filtering."""
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM audit_log WHERE 1=1"
            params = []
            
            if username:
                query += " AND username = ?"
                params.append(username)
            
            if action:
                query += " AND action = ?"
                params.append(action)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to dictionaries
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]


def init_default_users(auth_manager: AuthenticationManager):
    """Initialize default users for the system."""
    # Create admin user
    auth_manager.create_user(
        username="admin",
        email="admin@forensic-dashboard.local",
        password="admin123!",  # Should be changed in production
        role="admin",
        permissions=["admin", "read", "write", "export", "audit"]
    )
    
    # Create investigator user
    auth_manager.create_user(
        username="investigator",
        email="investigator@forensic-dashboard.local",
        password="invest123!",  # Should be changed in production
        role="investigator",
        permissions=["read", "export"]
    )
    
    # Create analyst user
    auth_manager.create_user(
        username="analyst",
        email="analyst@forensic-dashboard.local",
        password="analyst123!",  # Should be changed in production
        role="analyst",
        permissions=["read", "write"]
    )