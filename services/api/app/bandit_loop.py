"""
Bandit loop service for adaptive learning using Thompson sampling
"""

import random
import json
from typing import Dict, Any, List, Optional
from .db_pool import db_pool
from .enhanced_error_handling import APIError, ErrorHandler
import logging

logger = logging.getLogger(__name__)

class BanditLoop:
    """Thompson sampling bandit loop for adaptive learning"""
    
    async def select_option(self, keys: List[str]) -> Dict[str, Any]:
        """
        Select the best option using Thompson sampling
        
        Args:
            keys: List of bandit keys to choose from
            
        Returns:
            Dictionary with selected key and confidence
        """
        try:
            async with db_pool.get_connection() as conn:
                # Get bandit state for all keys
                rows = await conn.fetch("""
                    SELECT key, count, success, COALESCE(params, '{}') as params 
                    FROM bandit_state 
                    WHERE key = ANY($1)
                """, keys)
                
                if not rows:
                    return {"ok": False, "error": "No bandit keys found"}
                
                # Apply Thompson sampling
                best_key = None
                best_theta = -1
                
                for row in rows:
                    count = int(row['count'])
                    success = int(row['success'])
                    
                    # Beta distribution parameters
                    alpha = success + 1
                    beta = max(1, count - success + 1)
                    
                    # Sample from Beta distribution
                    theta = random.betavariate(alpha, beta)
                    
                    if theta > best_theta:
                        best_theta = theta
                        best_key = row['key']
                
                return {
                    "ok": True,
                    "selected": best_key,
                    "theta": best_theta,
                    "confidence": best_theta
                }
                
        except Exception as e:
            logger.error(f"Bandit selection failed: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    async def update_reward(self, key: str, reward: float) -> Dict[str, Any]:
        """
        Update bandit state with reward feedback
        
        Args:
            key: Bandit key to update
            reward: Reward value (0.0 to 1.0)
            
        Returns:
            Success status
        """
        try:
            # Clamp reward to [0, 1]
            reward = max(0.0, min(1.0, reward))
            
            async with db_pool.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO bandit_state(key, count, success, params)
                    VALUES ($1, 1, $2, '{}')
                    ON CONFLICT (key) DO UPDATE
                    SET count = bandit_state.count + 1,
                        success = bandit_state.success + $3,
                        updated_at = now()
                """, key, 1 if reward >= 0.5 else 0, 1 if reward >= 0.5 else 0)
            
            return {"ok": True, "message": "Bandit state updated"}
            
        except Exception as e:
            logger.error(f"Bandit update failed: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    async def get_bandit_stats(self) -> Dict[str, Any]:
        """Get current bandit statistics"""
        try:
            async with db_pool.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT key, count, success, params, updated_at
                    FROM bandit_state
                    ORDER BY updated_at DESC
                """)
                
                stats = []
                for row in rows:
                    count = int(row['count'])
                    success = int(row['success'])
                    failure = count - success
                    
                    stats.append({
                        "key": row['key'],
                        "count": count,
                        "success": success,
                        "failure": failure,
                        "success_rate": success / count if count > 0 else 0,
                        "params": json.loads(row['params']) if row['params'] else {},
                        "updated_at": row['updated_at'].isoformat()
                    })
                
                return {"ok": True, "stats": stats}
                
        except Exception as e:
            logger.error(f"Failed to get bandit stats: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    async def reset_bandit_state(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Reset bandit state for a specific key or all keys
        
        Args:
            key: Specific key to reset, or None for all keys
            
        Returns:
            Success status
        """
        try:
            async with db_pool.get_connection() as conn:
                if key:
                    await conn.execute("""
                        DELETE FROM bandit_state WHERE key = $1
                    """, key)
                    message = f"Reset bandit state for key: {key}"
                else:
                    await conn.execute("DELETE FROM bandit_state")
                    message = "Reset all bandit states"
                
                return {"ok": True, "message": message}
                
        except Exception as e:
            logger.error(f"Failed to reset bandit state: {str(e)}")
            return {"ok": False, "error": str(e)}
    
    async def get_source_weights(self) -> Dict[str, float]:
        """Get current source weights from bandit state"""
        try:
            async with db_pool.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT key, count, success, params
                    FROM bandit_state
                    WHERE key LIKE 'source:%'
                """)
                
                weights = {}
                for row in rows:
                    count = int(row['count'])
                    success = int(row['success'])
                    success_rate = success / count if count > 0 else 0.5
                    
                    # Extract source name from key
                    source_name = row['key'].replace('source:', '')
                    weights[source_name] = success_rate
                
                return weights
                
        except Exception as e:
            logger.error(f"Failed to get source weights: {str(e)}")
            return {}
    
    async def get_prompt_variant(self) -> Dict[str, Any]:
        """Get the best prompt variant for AI analysis"""
        try:
            prompt_keys = ["prompt:variantA", "prompt:variantB", "prompt:variantC"]
            result = await self.select_option(prompt_keys)
            
            if result["ok"]:
                # Get the parameters for the selected variant
                async with db_pool.get_connection() as conn:
                    row = await conn.fetchrow("""
                        SELECT params FROM bandit_state WHERE key = $1
                    """, result["selected"])
                    
                    if row and row['params']:
                        params = json.loads(row['params'])
                        result["params"] = params
                    else:
                        result["params"] = {"temperature": 0.1, "model": "llama3.1:8b"}
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get prompt variant: {str(e)}")
            return {"ok": False, "error": str(e)}

# Global instance
bandit_loop = BanditLoop()
