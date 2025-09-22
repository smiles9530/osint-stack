#!/usr/bin/env python3
"""
Simple user creation script using direct SQL
This script creates a user directly in the database without requiring all dependencies
"""

import psycopg2
import hashlib
import getpass
import sys
from datetime import datetime

def hash_password(password: str) -> str:
    """Simple password hashing using bcrypt-like approach"""
    import bcrypt
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_user_direct():
    """Create user directly using SQL"""
    print("=== OSINT Stack User Creation (Direct SQL) ===")
    print()
    
    # Database connection parameters
    # These should match your docker-compose.yml or environment settings
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'osint',
        'user': 'osint',
        'password': 'change_this_super_strong_password'
    }
    
    # Get user input
    username = input("Enter username: ").strip()
    if not username:
        print("Error: Username cannot be empty")
        return False
    
    email = input("Enter email: ").strip()
    if not email or '@' not in email:
        print("Error: Valid email is required")
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
    print("Connecting to database...")
    
    try:
        # Connect to database
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        # Check if user already exists
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            print(f"Error: User '{username}' already exists")
            return False
        
        # Check if email already exists
        cur.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            print(f"Error: Email '{email}' is already in use")
            return False
        
        print("Creating user...")
        
        # Hash password
        try:
            hashed_password = hash_password(password)
        except ImportError:
            print("Warning: bcrypt not available, using simple hash (not recommended for production)")
            # Fallback to simple hash (NOT SECURE for production)
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Insert user
        cur.execute("""
            INSERT INTO users (username, email, hashed_password, is_superuser, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, username, email, is_active, is_superuser, created_at
        """, (username, email, hashed_password, is_superuser, True, datetime.utcnow(), datetime.utcnow()))
        
        result = cur.fetchone()
        conn.commit()
        
        if result:
            print("✅ User created successfully!")
            print()
            print("User Details:")
            print(f"  ID: {result[0]}")
            print(f"  Username: {result[1]}")
            print(f"  Email: {result[2]}")
            print(f"  Active: {result[3]}")
            print(f"  Superuser: {result[4]}")
            print(f"  Created: {result[5]}")
            print()
            print("You can now use this user to log into the OSINT Stack API.")
            return True
        else:
            print("❌ Failed to create user")
            return False
            
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creating user: {e}")
        return False
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

def list_users_direct():
    """List users directly using SQL"""
    print("=== Existing Users ===")
    print()
    
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'osint',
        'user': 'osint',
        'password': 'change_this_super_strong_password'
    }
    
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT id, username, email, is_active, is_superuser, created_at
            FROM users
            ORDER BY created_at DESC
            LIMIT 50
        """)
        
        results = cur.fetchall()
        
        if not results:
            print("No users found in the database.")
            return
        
        print(f"{'ID':<5} {'Username':<20} {'Email':<30} {'Active':<8} {'Superuser':<10} {'Created'}")
        print("-" * 90)
        
        for row in results:
            created_str = row[5].strftime('%Y-%m-%d %H:%M') if row[5] else 'N/A'
            print(f"{row[0]:<5} {row[1]:<20} {row[2]:<30} "
                  f"{'Yes' if row[3] else 'No':<8} {'Yes' if row[4] else 'No':<10} {created_str}")
        
        print()
        print(f"Total users: {len(results)}")
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error listing users: {e}")
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        list_users_direct()
        return
    
    print("OSINT Stack User Management (Direct SQL)")
    print("=" * 40)
    print()
    print("1. Create new user")
    print("2. List existing users")
    print("3. Exit")
    print()
    
    choice = input("Select an option (1-3): ").strip()
    
    if choice == '1':
        success = create_user_direct()
        sys.exit(0 if success else 1)
    elif choice == '2':
        list_users_direct()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice. Please run the script again.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
