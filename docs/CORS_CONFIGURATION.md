# CORS Configuration Guide

This guide explains how to properly configure Cross-Origin Resource Sharing (CORS) for different deployment environments in the OSINT Stack.

## Overview

CORS is a security feature that controls which domains can access your API from web browsers. Proper CORS configuration is crucial for both security and functionality.

## Current Configuration

The OSINT Stack has been configured with secure CORS defaults:

### API Service (`services/api/app/main.py`)
- **Allowed Methods**: GET, POST, PUT, DELETE, OPTIONS, PATCH
- **Allowed Headers**: Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With
- **Exposed Headers**: X-Total-Count, X-Page-Count
- **Credentials**: Enabled (for authentication)

### Forecast Service (`services/forecast/main.py`)
- **Allowed Methods**: GET, POST, PUT, DELETE, OPTIONS
- **Allowed Headers**: All (*)
- **Credentials**: Enabled

## Environment-Specific Configuration

### Development Environment
```bash
CORS_ORIGINS=http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000,http://127.0.0.1:8080
```

**Use for:**
- Local development
- Testing with frontend running on localhost
- Development tools and debugging

### Staging Environment
```bash
CORS_ORIGINS=https://staging.yourdomain.com,https://staging-api.yourdomain.com,https://test.yourdomain.com
```

**Use for:**
- Pre-production testing
- QA environment
- Client demos

### Production Environment
```bash
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com,https://www.yourdomain.com,https://admin.yourdomain.com
```

**Use for:**
- Live production deployment
- Public-facing applications
- Customer environments

## Security Best Practices

### ✅ Do's

1. **Use Specific Origins**
   ```bash
   # Good - Specific domains
   CORS_ORIGINS=https://myapp.com,https://admin.myapp.com
   ```

2. **Use HTTPS in Production**
   ```bash
   # Good - Secure protocols
   CORS_ORIGINS=https://myapp.com
   ```

3. **Include All Necessary Subdomains**
   ```bash
   # Good - All required subdomains
   CORS_ORIGINS=https://app.mycompany.com,https://api.mycompany.com,https://admin.mycompany.com
   ```

4. **Separate Environments**
   ```bash
   # Development
   CORS_ORIGINS=http://localhost:3000,http://localhost:8080
   
   # Production
   CORS_ORIGINS=https://myapp.com,https://api.myapp.com
   ```

### ❌ Don'ts

1. **Never Use Wildcards in Production**
   ```bash
   # BAD - Allows any origin
   CORS_ORIGINS=*
   ```

2. **Don't Mix HTTP/HTTPS Carelessly**
   ```bash
   # BAD - Mixed protocols can cause issues
   CORS_ORIGINS=http://myapp.com,https://myapp.com
   ```

3. **Don't Include Unnecessary Origins**
   ```bash
   # BAD - Too permissive
   CORS_ORIGINS=https://myapp.com,https://anotherdomain.com,https://thirdparty.com
   ```

## Common Deployment Scenarios

### 1. Single Domain Deployment
If everything runs on one domain:
```bash
CORS_ORIGINS=https://myapp.com
```

### 2. API Subdomain
If API is on a subdomain:
```bash
CORS_ORIGINS=https://myapp.com,https://api.myapp.com
```

### 3. Multiple Frontend Apps
If you have multiple frontend applications:
```bash
CORS_ORIGINS=https://app.mycompany.com,https://admin.mycompany.com,https://dashboard.mycompany.com
```

### 4. CDN/Edge Deployment
If using CDN or edge deployment:
```bash
CORS_ORIGINS=https://myapp.com,https://cdn.myapp.com,https://edge.myapp.com
```

### 5. Load Balancer Setup
If using load balancers:
```bash
CORS_ORIGINS=https://myapp.com,https://lb.myapp.com
```

## RunPod/Cloud Deployment

For RunPod or other cloud deployments, you'll need to:

1. **Get Your Public IP/Domain**
   ```bash
   # For RunPod, this might be something like:
   CORS_ORIGINS=https://12345-8000.proxy.runpod.net,https://12345-3000.proxy.runpod.net
   ```

2. **Update After Deployment**
   Since cloud IPs may change, update CORS after deployment:
   ```bash
   # SSH into your instance and update
   echo "CORS_ORIGINS=https://$(curl -s ifconfig.me):8000,https://$(curl -s ifconfig.me):3000" >> .env
   ```

## Configuration Examples

### Example 1: Company with App and Admin
```bash
CORS_ORIGINS=https://app.acme.com,https://admin.acme.com,https://api.acme.com
```

### Example 2: SaaS with Multiple Tenants
```bash
CORS_ORIGINS=https://tenant1.myplatform.com,https://tenant2.myplatform.com,https://admin.myplatform.com
```

### Example 3: Development Team Setup
```bash
# Development
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:8080

# Staging
CORS_ORIGINS=https://dev.myapp.com,https://staging.myapp.com

# Production
CORS_ORIGINS=https://myapp.com,https://www.myapp.com
```

## Troubleshooting CORS Issues

### Common Error Messages

1. **"Access to fetch at ... has been blocked by CORS policy"**
   - **Solution**: Add the requesting origin to `CORS_ORIGINS`

2. **"CORS policy: No 'Access-Control-Allow-Origin' header"**
   - **Solution**: Ensure CORS middleware is properly configured

3. **"CORS policy: The request client is not a secure context"**
   - **Solution**: Use HTTPS in production environments

### Debugging Steps

1. **Check Browser Console**
   ```javascript
   // Check current origin
   console.log(window.location.origin);
   ```

2. **Verify Configuration**
   ```bash
   # Check current CORS setting
   echo $CORS_ORIGINS
   ```

3. **Test with curl**
   ```bash
   # Test CORS headers
   curl -H "Origin: https://yourdomain.com" \
        -H "Access-Control-Request-Method: GET" \
        -H "Access-Control-Request-Headers: Content-Type" \
        -X OPTIONS \
        http://your-api-url/api/health
   ```

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `CORS_ORIGINS` | Comma-separated list of allowed origins | `https://app.com,https://admin.app.com` |

## Quick Setup Commands

### For Development
```bash
echo "CORS_ORIGINS=http://localhost:3000,http://localhost:8080" >> .env
```

### For Production (Replace with your domain)
```bash
echo "CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com" >> .env
```

### For Dynamic IP (Cloud deployment)
```bash
PUBLIC_IP=$(curl -s ifconfig.me)
echo "CORS_ORIGINS=https://${PUBLIC_IP}:3000,https://${PUBLIC_IP}:8080" >> .env
```

## Testing Your CORS Configuration

### Browser Test
1. Open browser developer tools
2. Go to your frontend application
3. Check for CORS errors in console
4. Verify API calls work properly

### API Test
```bash
# Test preflight request
curl -X OPTIONS \
     -H "Origin: https://yourdomain.com" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type,Authorization" \
     http://your-api-url/api/health
```

## Security Considerations

1. **Principle of Least Privilege**: Only allow origins that actually need access
2. **Regular Audits**: Review and update CORS configuration regularly
3. **Environment Separation**: Use different CORS settings for different environments
4. **Monitoring**: Log and monitor CORS-related errors
5. **Documentation**: Keep CORS configuration documented and up-to-date

---

For more information about CORS, see:
- [MDN CORS Documentation](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)
- [FastAPI CORS Middleware](https://fastapi.tiangolo.com/tutorial/cors/)
