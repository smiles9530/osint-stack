"""
Database connection pooling module
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
import asyncpg
from .config import settings

logger = logging.getLogger("osint_api")

class DatabasePool:
    """PostgreSQL connection pool manager"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def create_pool(self):
        """Create optimized connection pool for high load"""
        try:
            self.pool = await asyncpg.create_pool(
                settings.database_url,
                min_size=10,  # Increased minimum connections
                max_size=50,  # Increased maximum connections for high load
                max_queries=100000,  # Increased query limit
                max_inactive_connection_lifetime=600.0,  # 10 minutes
                command_timeout=30,  # Reduced timeout for faster failure detection
                server_settings={
                    'application_name': 'osint_stack_api',
                    'statement_timeout': '30000',  # 30 seconds
                    'idle_in_transaction_session_timeout': '300000',  # 5 minutes
                },
                # Connection pool optimizations
                setup=self._setup_connection,
                init=self._init_connection
            )
            logger.info("High-performance database connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    
    async def _setup_connection(self, conn):
        """Setup connection with optimizations"""
        # Enable query statistics
        await conn.execute("LOAD 'pg_stat_statements'")
        
        # Set connection-level optimizations for high load
        await conn.execute("SET work_mem = '256MB'")
        await conn.execute("SET maintenance_work_mem = '1GB'")
        await conn.execute("SET effective_cache_size = '4GB'")
        await conn.execute("SET random_page_cost = 1.1")
        await conn.execute("SET effective_io_concurrency = 200")
        await conn.execute("SET max_parallel_workers_per_gather = 4")
        await conn.execute("SET max_parallel_workers = 8")
        
        # Enable JIT compilation for complex queries
        await conn.execute("SET jit = on")
        await conn.execute("SET jit_above_cost = 100000")
        
        # Optimize for read-heavy workloads
        await conn.execute("SET default_statistics_target = 100")
        await conn.execute("SET log_statement = 'none'")  # Disable query logging for performance
        
        logger.debug("Connection setup completed with high-load optimizations")
    
    async def _init_connection(self, conn):
        """Initialize connection with additional optimizations"""
        # Set session-level optimizations
        await conn.execute("SET session_replication_role = 'replica'")  # Disable triggers for faster inserts
        await conn.execute("SET synchronous_commit = 'off'")  # Async commits for better performance
        
        logger.debug("Connection initialized with session optimizations")
    
    async def close_pool(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get connection from pool"""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")
        
        connection = None
        try:
            connection = await self.pool.acquire()
            yield connection
        finally:
            if connection:
                await self.pool.release(connection)
    
    async def execute(self, query: str, *args):
        """Execute query with connection from pool"""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)
    
    async def fetchrow(self, query: str, *args):
        """Fetch single row with connection from pool"""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetch(self, query: str, *args):
        """Fetch multiple rows with connection from pool"""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchval(self, query: str, *args):
        """Fetch single value with connection from pool"""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)

# Global database pool instance
db_pool = DatabasePool()
