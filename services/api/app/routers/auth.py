"""
Authentication router
Handles user authentication, registration, and management
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from ..auth import authenticate_user, create_access_token, get_current_active_user, create_user, update_user_last_login, Token, User
from ..schemas import LoginRequest, UserCreate, ErrorResponse, HTTPValidationError

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post(
    "/login", 
    response_model=Token,
    summary="User Login",
    description="Authenticate user credentials and return JWT access token for API access",
    responses={
        200: {
            "description": "Login successful", 
            "model": Token,
            "content": {
                "application/json": {
                    "example": {
                        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                        "token_type": "bearer"
                    }
                }
            }
        },
        401: {
            "description": "Invalid credentials", 
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "error": "Authentication failed",
                        "detail": "Incorrect username or password",
                        "request_id": "req_123456789",
                        "timestamp": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        422: {
            "description": "Validation error", 
            "model": HTTPValidationError,
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "username"],
                                "msg": "field required",
                                "type": "missing"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def login(login_data: LoginRequest):
    """Authenticate user and return JWT token"""
    user = await authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password"
        )
    
    # Update last login
    await update_user_last_login(user.id)
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post(
    "/users", 
    response_model=dict,
    summary="Create User",
    description="Create a new user account (admin only)",
    responses={
        200: {"description": "User created successfully"},
        403: {"description": "Insufficient permissions", "model": ErrorResponse},
        422: {"description": "Validation error", "model": HTTPValidationError}
    }
)
async def create_user_endpoint(
    user_data: UserCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new user (admin only)"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions"
        )
    
    try:
        user = await create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            is_superuser=user_data.is_superuser
        )
        return {
            "message": "User created successfully",
            "user_id": user.id,
            "username": user.username
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to create user: {str(e)}"
        )

@router.get("/users", response_model=dict)
async def list_users(
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_active_user)
):
    """List users (admin only)"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions"
        )
    
    # This would typically fetch from database
    return {
        "users": [],
        "total": 0,
        "limit": limit,
        "offset": offset
    }
