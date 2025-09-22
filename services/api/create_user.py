#!/usr/bin/env python3
"""
User creation script for OSINT Stack
Creates a new user in the database with proper password hashing
"""

import asyncio
import sys
import os
import getpass
from datetime import datetime

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.user_management import user_manager
from app.db import get_conn

async def create_user():
    """Interactive user creation script"""
    print("=== OSINT Stack User Creation ===")
    print()
    
    # Get user input
    username = input("Enter username: ").strip()
    if not username:
        print("Error: Username cannot be empty")
        return False
    
    email = input("Enter email: ").strip()
    if not email or '@' not in email:
        print("Error: Valid email is required")
        return False
    
    # Check if user already exists
    existing_user = await user_manager.get_user_by_username(username)
    if existing_user:
        print(f"Error: User '{username}' already exists")
        return False
    
    # Check if email already exists
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            if cur.fetchone():
                print(f"Error: Email '{email}' is already in use")
                return False
    
    password = getpass.getpass("Enter password: ")
    if len(password) < 6:
        print("Error: Password must be at least 6 characters long")
        return False
    
    confirm_password = getpass.getpass("Confirm password: ")
    if password != confirm_password:
        print("Error: Passwords do not match")
        return False
    
    # Ask for superuser privileges
    is_superuser_input = input("Make this user a superuser? (y/N): ").strip().lower()
    is_superuser = is_superuser_input in ['y', 'yes']
    
    print()
    print("Creating user...")
    
    try:
        # Create the user
        user_data = await user_manager.create_user(
            username=username,
            email=email,
            password=password,
            is_superuser=is_superuser
        )
        
        if user_data:
            print("✅ User created successfully!")
            print()
            print("User Details:")
            print(f"  ID: {user_data['id']}")
            print(f"  Username: {user_data['username']}")
            print(f"  Email: {user_data['email']}")
            print(f"  Active: {user_data['is_active']}")
            print(f"  Superuser: {user_data['is_superuser']}")
            print(f"  Created: {user_data['created_at']}")
            print()
            print("You can now use this user to log into the OSINT Stack API.")
            return True
        else:
            print("❌ Failed to create user")
            return False
            
    except Exception as e:
        print(f"❌ Error creating user: {e}")
        return False

async def list_users():
    """List existing users"""
    print("=== Existing Users ===")
    print()
    
    try:
        users = await user_manager.list_users(limit=50)
        if not users:
            print("No users found in the database.")
            return
        
        print(f"{'ID':<5} {'Username':<20} {'Email':<30} {'Active':<8} {'Superuser':<10} {'Created'}")
        print("-" * 90)
        
        for user in users:
            created_str = user['created_at'].strftime('%Y-%m-%d %H:%M') if user['created_at'] else 'N/A'
            print(f"{user['id']:<5} {user['username']:<20} {user['email']:<30} "
                  f"{'Yes' if user['is_active'] else 'No':<8} {'Yes' if user['is_superuser'] else 'No':<10} {created_str}")
        
        print()
        print(f"Total users: {len(users)}")
        
    except Exception as e:
        print(f"❌ Error listing users: {e}")

async def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        await list_users()
        return
    
    print("OSINT Stack User Management")
    print("=" * 30)
    print()
    print("1. Create new user")
    print("2. List existing users")
    print("3. Exit")
    print()
    
    choice = input("Select an option (1-3): ").strip()
    
    if choice == '1':
        success = await create_user()
        sys.exit(0 if success else 1)
    elif choice == '2':
        await list_users()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice. Please run the script again.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
