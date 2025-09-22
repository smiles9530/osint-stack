"""
Authentication module with database integration
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from .config import settings
from .user_management import user_manager

# JWT settings
SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes

# Security scheme
security = HTTPBearer()

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool

class UserInDB(User):
    hashed_password: str

def verify_password(plain_password, hashed_password):
    """Verify a password against its hash"""
    return user_manager.pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hash a password"""
    return user_manager.pwd_context.hash(password)

async def get_user(username: str):
    """Get user from database"""
    user_dict = await user_manager.get_user_by_username(username)
    if not user_dict:
        return None
    return UserInDB(**user_dict)

async def authenticate_user(username: str, password: str):
    """Authenticate user with database"""
    user = await get_user(username)
    if not user:
        return False
    if not await user_manager.verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = await get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def create_user(username: str, email: str, password: str, is_superuser: bool = False):
    """Create a new user"""
    return await user_manager.create_user(username, email, password, is_superuser)

async def update_user_last_login(user_id: int):
    """Update user's last login timestamp"""
    await user_manager.update_last_login(user_id)