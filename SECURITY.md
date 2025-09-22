# Security Guide

## Security Improvements Made

This document outlines the security improvements implemented in the OSINT Stack to address vulnerabilities and follow security best practices.

### 🔒 Security Issues Fixed

#### 1. Hardcoded Credentials Removed
- **Issue**: Default passwords and API keys were hardcoded in configuration files
- **Fix**: Replaced with environment variable placeholders and secure random generation
- **Files affected**: `.env`, `runpod-deploy.sh`, `services/api/app/config.py`

#### 2. Auto Admin User Creation Disabled
- **Issue**: Admin users were automatically created with weak passwords
- **Fix**: Removed automatic user creation scripts
- **Files removed**: 
  - `create_admin_user.sql`
  - `services/api/create_user.sql`
- **Files modified**: `db/init/020-init-users.sql`

#### 3. CORS Policy Hardened
- **Issue**: CORS was configured to allow all origins (`*`)
- **Fix**: Restricted to specific allowed origins
- **Default allowed origins**: `http://localhost:3000`, `http://localhost:8080`
- **Files affected**: `services/api/app/config.py`, `services/forecast/main.py`

#### 4. Password Policy Strengthened
- **Previous**: Minimum 6 characters
- **New**: Minimum 12 characters with complexity requirements:
  - At least one uppercase letter
  - At least one lowercase letter
  - At least one digit
  - At least one special character
- **Files affected**: `services/api/app/validators.py`, user creation scripts

#### 5. Insecure Password Hashing Removed
- **Issue**: SHA256 fallback for password hashing
- **Fix**: Requires bcrypt for all password operations
- **Files affected**: `services/api/create_user_sql.py`

### 🛡️ Security Features

#### Strong Random Credential Generation
- All secrets now use cryptographically secure random generation
- Script provided: `scripts/init-secure-env.sh`
- Uses OpenSSL for entropy

#### Secure Configuration Requirements
- Mandatory environment variables for sensitive data
- No default passwords in production configuration
- Secure file permissions on environment files

#### Authentication & Authorization
- JWT-based authentication with configurable expiration
- Session management with database storage
- User role-based access control
- Secure password verification using bcrypt

### 🚀 Deployment Security

#### Secure Environment Setup
```bash
# Run the secure initialization script
./scripts/init-secure-env.sh
```

This script will:
- Generate strong random passwords for all services
- Create a secure `.env` file with proper permissions
- Display credentials for secure backup

#### Manual User Creation
Since auto user creation is disabled, create users manually:

```bash
# Using the secure user creation script
cd services/api
python create_user.py
```

Or via API endpoints (requires existing admin user).

### ⚠️ Security Recommendations

#### For Production Deployment:

1. **Environment Variables**
   - Use a secure secrets management system
   - Never commit `.env` files to version control
   - Rotate credentials regularly

2. **Network Security**
   - Use HTTPS in production (set `MINIO_SECURE=true`)
   - Configure proper firewall rules
   - Use VPN for administrative access

3. **Database Security**
   - Enable SSL/TLS for database connections
   - Use strong, unique passwords
   - Implement database user privilege restrictions

4. **Container Security**
   - Keep base images updated
   - Scan images for vulnerabilities
   - Use non-root users where possible

5. **Monitoring & Logging**
   - Enable security event logging
   - Monitor for failed authentication attempts
   - Set up alerts for suspicious activities

#### CORS Configuration
CORS has been properly configured for different environments:

**Development:**
```bash
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080
```

**Production:**
```bash
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com,https://www.yourdomain.com
```

See `docs/CORS_CONFIGURATION.md` for detailed configuration guide.

#### SSL/TLS Configuration
Enable HTTPS and secure connections:
```bash
MINIO_SECURE=true
N8N_PROTOCOL=https
# Add SSL certificates to nginx configuration
```

### 🔍 Security Checklist

Before deploying to production:

- [ ] Run `./scripts/init-secure-env.sh` to generate secure credentials
- [ ] Back up credentials in a secure password manager
- [ ] Create admin user manually with strong password
- [ ] Configure CORS origins for your domain
- [ ] Enable HTTPS/SSL for all services
- [ ] Set up monitoring and logging
- [ ] Review and harden container configurations
- [ ] Test authentication and authorization flows
- [ ] Conduct security review of custom code

### 📞 Security Contact

For security-related issues or questions:
- Review this documentation
- Check application logs for authentication errors
- Ensure all environment variables are properly set
- Verify password complexity requirements are met

### 🔄 Regular Security Maintenance

1. **Monthly**:
   - Review user accounts and permissions
   - Check for failed authentication attempts
   - Update container images

2. **Quarterly**:
   - Rotate API keys and secrets
   - Review CORS and security configurations
   - Audit user access patterns

3. **Annually**:
   - Full security audit
   - Penetration testing
   - Security training for administrators

---

## Emergency Procedures

### Password Reset
If admin access is lost:
1. Connect directly to the database
2. Use `services/api/create_user_sql.py` to create a new admin user
3. Disable old compromised accounts

### Credential Rotation
To rotate all credentials:
1. Generate new credentials using the secure script
2. Update `.env` file
3. Restart all services
4. Update any external integrations

### Security Incident Response
1. Isolate affected services
2. Review logs for unauthorized access
3. Change all potentially compromised credentials
4. Restore from clean backups if necessary
5. Document incident and apply lessons learned
