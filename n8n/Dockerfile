FROM n8nio/n8n:latest

# Switch to root to install packages
USER root

# Install Chrome dependencies and performance tools for Alpine
RUN apk add --no-cache \
    chromium \
    nss \
    freetype \
    freetype-dev \
    harfbuzz \
    ca-certificates \
    ttf-freefont \
    # Performance monitoring tools
    htop \
    iotop \
    # Additional Node.js performance packages
    && rm -rf /var/cache/apk/*

# Switch back to node user
USER node

# Install Puppeteer and performance packages
RUN npm install puppeteer@latest \
    && npm cache clean --force

# Ensure n8n is properly installed and accessible
RUN which n8n || echo "n8n not found in PATH"
RUN ls -la /usr/local/bin/ | grep n8n || echo "n8n not in /usr/local/bin/"
RUN find / -name "n8n" -type f 2>/dev/null | head -5 || echo "n8n not found anywhere"

# Set Puppeteer to use system Chrome
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser

# Set display for headless Chrome
ENV DISPLAY=:99

# Optimized Chrome arguments for Docker environment
ENV PUPPETEER_ARGS="--no-sandbox --disable-setuid-sandbox --disable-dev-shm-usage --disable-accelerated-2d-canvas --no-first-run --no-zygote --disable-gpu --disable-web-security --disable-features=VizDisplayCompositor --memory-pressure-off --max_old_space_size=4096 --disable-background-timer-throttling --disable-backgrounding-occluded-windows --disable-renderer-backgrounding"

# Node.js performance optimizations
ENV NODE_OPTIONS="--max-old-space-size=4096 --enable-source-maps"
ENV NODE_ENV=production

# N8N Performance optimizations
ENV N8N_METRICS=true
ENV N8N_DIAGNOSTICS_ENABLED=true

# PostgreSQL Database Configuration
ENV DB_TYPE=postgresdb
ENV DB_POSTGRESDB_POOL_SIZE=20
ENV DB_POSTGRESDB_CONNECTION_TIMEOUT=30000
ENV DB_POSTGRESDB_ACQUIRE_TIMEOUT=30000
ENV DB_POSTGRESDB_TIMEOUT=30000

# Queue Configuration
ENV QUEUE_BULL_REDIS_HOST=redis
ENV QUEUE_BULL_REDIS_PORT=6379
ENV QUEUE_BULL_REDIS_DB=1

# Execution settings
ENV EXECUTIONS_TIMEOUT=3600
ENV EXECUTIONS_TIMEOUT_MAX=7200
