"""
User management module with database integration
"""
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from passlib.context import CryptContext
from .db import get_conn
from .config import settings

logger = logging.getLogger("osint_api")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserManager:
    """Database-backed user management system"""
    
    def __init__(self):
        self.pwd_context = pwd_context
    
    async def create_user(self, username: str, email: str, password: str, 
                         is_superuser: bool = False) -> Dict[str, Any]:
        """Create a new user in the database"""
        try:
            hashed_password = self.pwd_context.hash(password)
            
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO users (username, email, hashed_password, is_superuser)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id, username, email, is_active, is_superuser, created_at
                    """, (username, email, hashed_password, is_superuser))
                    
                    result = cur.fetchone()
                    if result:
                        conn.commit()
                        logger.info(f"User {username} created successfully")
                        return {
                            "id": result[0],
                            "username": result[1],
                            "email": result[2],
                            "is_active": result[3],
                            "is_superuser": result[4],
                            "created_at": result[5]
                        }
        except Exception as e:
            logger.error(f"Error creating user {username}: {e}")
            raise
        return None
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, username, email, hashed_password, is_active, 
                               is_superuser, created_at, updated_at, last_login
                        FROM users WHERE username = %s
                    """, (username,))
                    
                    result = cur.fetchone()
                    if result:
                        return {
                            "id": result[0],
                            "username": result[1],
                            "email": result[2],
                            "hashed_password": result[3],
                            "is_active": result[4],
                            "is_superuser": result[5],
                            "created_at": result[6],
                            "updated_at": result[7],
                            "last_login": result[8]
                        }
        except Exception as e:
            logger.error(f"Error getting user {username}: {e}")
        return None
    
    async def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, username, email, hashed_password, is_active, 
                               is_superuser, created_at, updated_at, last_login
                        FROM users WHERE id = %s
                    """, (user_id,))
                    
                    result = cur.fetchone()
                    if result:
                        return {
                            "id": result[0],
                            "username": result[1],
                            "email": result[2],
                            "hashed_password": result[3],
                            "is_active": result[4],
                            "is_superuser": result[5],
                            "created_at": result[6],
                            "updated_at": result[7],
                            "last_login": result[8]
                        }
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
        return None
    
    async def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    async def update_last_login(self, user_id: int) -> None:
        """Update user's last login timestamp"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE users SET last_login = NOW(), updated_at = NOW()
                        WHERE id = %s
                    """, (user_id,))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error updating last login for user {user_id}: {e}")
    
    async def update_user(self, user_id: int, **kwargs) -> bool:
        """Update user information"""
        try:
            allowed_fields = ['email', 'is_active', 'is_superuser']
            updates = []
            values = []
            
            for field, value in kwargs.items():
                if field in allowed_fields:
                    updates.append(f"{field} = %s")
                    values.append(value)
            
            if not updates:
                return False
            
            values.append(user_id)
            query = f"UPDATE users SET {', '.join(updates)}, updated_at = NOW() WHERE id = %s"
            
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, values)
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}")
        return False
    
    async def change_password(self, user_id: int, new_password: str) -> bool:
        """Change user password"""
        try:
            hashed_password = self.pwd_context.hash(new_password)
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE users SET hashed_password = %s, updated_at = NOW()
                        WHERE id = %s
                    """, (hashed_password, user_id))
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Error changing password for user {user_id}: {e}")
        return False
    
    async def list_users(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List users with pagination"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, username, email, is_active, is_superuser, 
                               created_at, updated_at, last_login
                        FROM users
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """, (limit, offset))
                    
                    results = cur.fetchall()
                    return [
                        {
                            "id": row[0],
                            "username": row[1],
                            "email": row[2],
                            "is_active": row[3],
                            "is_superuser": row[4],
                            "created_at": row[5],
                            "updated_at": row[6],
                            "last_login": row[7]
                        }
                        for row in results
                    ]
        except Exception as e:
            logger.error(f"Error listing users: {e}")
        return []
    
    async def create_session(self, user_id: int, token_hash: str, expires_at: datetime) -> bool:
        """Create a user session"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO user_sessions (user_id, token_hash, expires_at)
                        VALUES (%s, %s, %s)
                    """, (user_id, token_hash, expires_at))
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error creating session for user {user_id}: {e}")
        return False
    
    async def get_session(self, token_hash: str) -> Optional[Dict[str, Any]]:
        """Get session by token hash"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT s.id, s.user_id, s.expires_at, s.is_active,
                               u.username, u.email, u.is_active as user_active
                        FROM user_sessions s
                        JOIN users u ON s.user_id = u.id
                        WHERE s.token_hash = %s AND s.is_active = TRUE
                    """, (token_hash,))
                    
                    result = cur.fetchone()
                    if result:
                        return {
                            "session_id": result[0],
                            "user_id": result[1],
                            "expires_at": result[2],
                            "is_active": result[3],
                            "username": result[4],
                            "email": result[5],
                            "user_active": result[6]
                        }
        except Exception as e:
            logger.error(f"Error getting session: {e}")
        return None
    
    async def invalidate_session(self, token_hash: str) -> bool:
        """Invalidate a session"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE user_sessions SET is_active = FALSE
                        WHERE token_hash = %s
                    """, (token_hash,))
                    conn.commit()
                    return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Error invalidating session: {e}")
        return False
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE user_sessions SET is_active = FALSE
                        WHERE expires_at < NOW() AND is_active = TRUE
                    """)
                    conn.commit()
                    return cur.rowcount
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
        return 0

# Global user manager instance
user_manager = UserManager()
