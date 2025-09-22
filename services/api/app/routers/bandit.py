"""
Enhanced Bandit router
Handles advanced contextual bandit operations with multiple algorithms and performance monitoring
"""

import asyncio
import json
import logging
import time
import math
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Union
from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ..auth import get_current_active_user, User
from ..schemas import BanditSelection, BanditUpdate, BanditStats, ErrorResponse
from ..enhanced_error_handling import ErrorHandler, APIError
from ..cache import cache
from ..monitoring import monitoring_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bandit", tags=["Bandit"])

# Advanced bandit algorithms
class AdvancedBanditAlgorithms:
    """Collection of advanced bandit algorithms"""
    
    @staticmethod
    def ucb(key_stats: Dict[str, Dict], total_selections: int, exploration_factor: float = 2.0) -> str:
        """Upper Confidence Bound algorithm"""
        if not key_stats:
            return None
        
        ucb_scores = {}
        for key, stats in key_stats.items():
            if stats["count"] == 0:
                ucb_scores[key] = float('inf')  # Prioritize unexplored arms
            else:
                confidence_radius = math.sqrt(exploration_factor * math.log(total_selections) / stats["count"])
                ucb_scores[key] = stats["mean_reward"] + confidence_radius
        
        return max(ucb_scores.keys(), key=lambda k: ucb_scores[k])
    
    @staticmethod
    def thompson_sampling(key_stats: Dict[str, Dict], alpha: float = 1.0, beta: float = 1.0) -> str:
        """Thompson Sampling algorithm"""
        if not key_stats:
            return None
        
        # Sample from Beta distribution for each key
        samples = {}
        for key, stats in key_stats.items():
            # Beta distribution parameters
            a = alpha + stats["count"] * stats["mean_reward"]
            b = beta + stats["count"] * (1 - stats["mean_reward"])
            samples[key] = np.random.beta(a, b)
        
        return max(samples.keys(), key=lambda k: samples[k])
    
    @staticmethod
    def linucb(key_stats: Dict[str, Dict], context: Dict[str, Any], 
               alpha: float = 1.0, lambda_reg: float = 1.0) -> str:
        """Linear Upper Confidence Bound algorithm"""
        if not key_stats or not context:
            return AdvancedBanditAlgorithms.ucb(key_stats, sum(s["count"] for s in key_stats.values()))
        
        # Convert context to feature vector
        context_vector = np.array(list(context.values())).reshape(1, -1)
        
        # For simplicity, use linear regression to estimate rewards
        # In practice, you'd maintain separate models for each arm
        linucb_scores = {}
        for key, stats in key_stats.items():
            if stats["count"] == 0:
                linucb_scores[key] = float('inf')
            else:
                # Simple linear model (in practice, use proper LinUCB implementation)
                estimated_reward = np.mean(list(context.values())) * stats["mean_reward"]
                confidence_radius = alpha * math.sqrt(stats["count"])
                linucb_scores[key] = estimated_reward + confidence_radius
        
        return max(linucb_scores.keys(), key=lambda k: linucb_scores[k])
    
    @staticmethod
    def epsilon_greedy(key_stats: Dict[str, Dict], epsilon: float = 0.1) -> str:
        """Epsilon-Greedy algorithm"""
        if not key_stats:
            return None
        
        if np.random.random() < epsilon:
            # Explore: choose randomly
            return np.random.choice(list(key_stats.keys()))
        else:
            # Exploit: choose best
            return max(key_stats.keys(), key=lambda k: key_stats[k]["mean_reward"])

# Performance monitoring for bandit operations
class BanditPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "total_selections": 0,
            "total_rewards": 0,
            "avg_reward": 0.0,
            "regret": 0.0,
            "exploration_rate": 0.0,
            "algorithm_performance": {}
        }
        self.selection_history = []
        self.reward_history = []
    
    def record_selection(self, key: str, algorithm: str, ucb_score: float, context: Dict[str, Any]):
        """Record a selection for performance tracking"""
        self.metrics["total_selections"] += 1
        self.selection_history.append({
            "key": key,
            "algorithm": algorithm,
            "ucb_score": ucb_score,
            "context": context,
            "timestamp": datetime.utcnow()
        })
        
        # Update algorithm performance
        if algorithm not in self.metrics["algorithm_performance"]:
            self.metrics["algorithm_performance"][algorithm] = {
                "selections": 0,
                "avg_reward": 0.0,
                "success_rate": 0.0
            }
        
        self.metrics["algorithm_performance"][algorithm]["selections"] += 1
    
    def record_reward(self, key: str, reward: float, context: Dict[str, Any]):
        """Record a reward for performance tracking"""
        self.metrics["total_rewards"] += 1
        self.reward_history.append({
            "key": key,
            "reward": reward,
            "context": context,
            "timestamp": datetime.utcnow()
        })
        
        # Update average reward
        total_rewards = len(self.reward_history)
        current_avg = self.metrics["avg_reward"]
        self.metrics["avg_reward"] = (current_avg * (total_rewards - 1) + reward) / total_rewards
        
        # Update regret (simplified calculation)
        if self.selection_history:
            recent_selection = self.selection_history[-1]
            if recent_selection["key"] == key:
                # Calculate regret as difference from optimal reward
                optimal_reward = 1.0  # Assume optimal reward is 1.0
                regret = optimal_reward - reward
                self.metrics["regret"] += regret
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        # Calculate exploration rate
        if self.selection_history:
            recent_selections = self.selection_history[-100:]  # Last 100 selections
            exploration_count = sum(1 for s in recent_selections if s["algorithm"] == "exploration")
            self.metrics["exploration_rate"] = exploration_count / len(recent_selections)
        
        # Calculate algorithm-specific metrics
        for algorithm, perf in self.metrics["algorithm_performance"].items():
            algorithm_rewards = [r["reward"] for r in self.reward_history 
                               if any(s["key"] == r["key"] and s["algorithm"] == algorithm 
                                     for s in self.selection_history)]
            
            if algorithm_rewards:
                perf["avg_reward"] = np.mean(algorithm_rewards)
                perf["success_rate"] = sum(1 for r in algorithm_rewards if r > 0.5) / len(algorithm_rewards)
        
        return self.metrics

performance_monitor = BanditPerformanceMonitor()

# Advanced bandit service
class AdvancedBanditService:
    def __init__(self):
        self.algorithms = AdvancedBanditAlgorithms()
        self.context_cache = {}
        self.performance_cache = {}
    
    async def select_arm_advanced(
        self, 
    keys: List[str],
        context: Optional[Dict[str, Any]] = None,
        algorithm: str = "ucb",
        user_id: int = None
    ) -> Dict[str, Any]:
        """Advanced arm selection with multiple algorithms and context awareness"""
        try:
            if not keys:
                ErrorHandler.raise_bad_request("No keys provided")
            
            # Get historical performance data
            key_stats = await self._get_key_statistics(keys, user_id)
            
            # Select algorithm
            selected_key = None
            confidence = 0.0
            algorithm_used = algorithm
            
            if algorithm == "ucb":
                selected_key = self.algorithms.ucb(key_stats, sum(s["count"] for s in key_stats.values()))
                confidence = key_stats[selected_key]["ucb_score"] if selected_key in key_stats else 0.0
            elif algorithm == "thompson":
                selected_key = self.algorithms.thompson_sampling(key_stats)
                confidence = key_stats[selected_key]["mean_reward"] if selected_key in key_stats else 0.0
            elif algorithm == "linucb":
                selected_key = self.algorithms.linucb(key_stats, context or {})
                confidence = key_stats[selected_key]["mean_reward"] if selected_key in key_stats else 0.0
            elif algorithm == "epsilon_greedy":
                selected_key = self.algorithms.epsilon_greedy(key_stats)
                confidence = key_stats[selected_key]["mean_reward"] if selected_key in key_stats else 0.0
            else:
                # Default to UCB
                selected_key = self.algorithms.ucb(key_stats, sum(s["count"] for s in key_stats.values()))
                confidence = key_stats[selected_key]["ucb_score"] if selected_key in key_stats else 0.0
                algorithm_used = "ucb"
            
            # Calculate UCB score for tracking
            ucb_score = 0.0
            if selected_key in key_stats:
                total_selections = sum(s["count"] for s in key_stats.values())
                if key_stats[selected_key]["count"] > 0:
                    ucb_score = (key_stats[selected_key]["mean_reward"] + 
                               math.sqrt(2 * math.log(total_selections) / key_stats[selected_key]["count"]))
                else:
                    ucb_score = float('inf')
            
            # Record selection for performance tracking
            performance_monitor.record_selection(selected_key, algorithm_used, ucb_score, context or {})
            
            # Store selection in database
            await self._store_selection(selected_key, context, ucb_score, user_id, algorithm_used)
            
            return {
                "selected_key": selected_key,
                "confidence": min(confidence, 1.0),
                "ucb_score": ucb_score,
                "algorithm_used": algorithm_used,
                "available_keys": keys,
                "key_stats": key_stats,
                "context_used": context,
                "selection_reason": f"{algorithm_used}_algorithm"
            }
        
        except Exception as e:
            logger.error(f"Advanced arm selection failed: {e}")
            raise APIError(
                status_code=500,
                error_code="BANDIT_SELECTION_FAILED",
                message="Advanced arm selection failed",
                detail=str(e)
            )
    
    async def _get_key_statistics(self, keys: List[str], user_id: Optional[int]) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive statistics for keys"""
        try:
            from .. import db
            
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    # Get reward statistics
                    query = """
                        SELECT key_name, reward, context_data, created_at
                        FROM bandit_rewards
                        WHERE key_name = ANY(%s)
                    """
                    params = [keys]
                    
                    if user_id:
                        query += " AND user_id = %s"
                        params.append(user_id)
                    
                    query += " ORDER BY created_at DESC LIMIT 1000"
                    
                    cur.execute(query, params)
                    rewards_data = cur.fetchall()
                    
                    # Get selection statistics
                    selection_query = """
                        SELECT key_name, ucb_score, context_data, created_at
                        FROM bandit_selections
                        WHERE key_name = ANY(%s)
                    """
                    selection_params = [keys]
                    
                    if user_id:
                        selection_query += " AND user_id = %s"
                        selection_params.append(user_id)
                    
                    selection_query += " ORDER BY created_at DESC LIMIT 1000"
                    
                    cur.execute(selection_query, selection_params)
                    selections_data = cur.fetchall()
            
            # Calculate statistics for each key
            key_stats = {}
            current_time = datetime.utcnow()
            
            for key in keys:
                key_rewards = [row[1] for row in rewards_data if row[0] == key]
                key_selections = [row[1] for row in selections_data if row[0] == key]
                
                if key_rewards:
                    mean_reward = np.mean(key_rewards)
                    count = len(key_rewards)
                    std_reward = np.std(key_rewards)
                    
                    # Calculate UCB score
                    total_selections = sum(len([r for r in rewards_data if r[0] == k]) for k in keys)
                    if count > 0:
                        ucb_score = mean_reward + math.sqrt(2 * math.log(total_selections) / count)
                    else:
                        ucb_score = float('inf')
                else:
                    mean_reward = 0.5  # Default optimistic value
                    count = 0
                    std_reward = 0.0
                    ucb_score = 1.0  # High exploration for new keys
                
                key_stats[key] = {
                    "mean_reward": mean_reward,
                    "count": count,
                    "std_reward": std_reward,
                    "ucb_score": ucb_score,
                    "selection_count": len(key_selections),
                    "last_selected": max([row[3] for row in selections_data if row[0] == key], default=None),
                    "last_rewarded": max([row[3] for row in rewards_data if row[0] == key], default=None)
                }
            
            return key_stats
            
        except Exception as e:
            logger.error(f"Failed to get key statistics: {e}")
            return {}
    
    async def _store_selection(
        self, 
        key: str, 
        context: Optional[Dict[str, Any]], 
        ucb_score: float, 
        user_id: Optional[int],
        algorithm: str
    ):
        """Store selection in database"""
        try:
            from .. import db
            
            with db.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO bandit_selections 
                        (key_name, context_data, ucb_score, user_id, algorithm_used, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        key,
                        json.dumps(context) if context else None,
                        ucb_score,
                        user_id,
                        algorithm,
                        datetime.utcnow()
                    ))
        except Exception as e:
            logger.error(f"Failed to store selection: {e}")
    
    async def _invalidate_key_cache(self, key: str, user_id: int):
        """Invalidate cache for a specific key"""
        try:
            cache_key = f"bandit_stats:{user_id}:{key}"
            await cache.delete(cache_key)
        except Exception as e:
            logger.warning(f"Failed to invalidate cache for key {key}: {e}")

bandit_service = AdvancedBanditService()

# Legacy endpoints for backward compatibility
@router.post(
    "/select/legacy",
    summary="Legacy Bandit Selection",
    description="Legacy endpoint for backward compatibility",
    deprecated=True
)
async def select_bandit_option_legacy(
    keys: List[str],
    context: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Legacy bandit selection endpoint"""
    return await select_bandit_option_advanced(keys, context, "ucb", current_user)

@router.post(
    "/update/legacy",
    summary="Legacy Reward Update",
    description="Legacy endpoint for backward compatibility",
    deprecated=True
)
async def update_bandit_reward_legacy(
    key: str,
    reward: float,
    context: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Legacy reward update endpoint"""
    return await update_bandit_reward_advanced(key, reward, context, "implicit", current_user)

@router.get(
    "/stats/legacy",
    summary="Legacy Bandit Statistics",
    description="Legacy endpoint for backward compatibility",
    deprecated=True
)
async def get_bandit_stats_legacy(
    current_user: User = Depends(get_current_active_user)
):
    """Legacy bandit statistics endpoint"""
    return await get_bandit_stats_advanced(30, True, True, current_user)

@router.post(
    "/select",
    summary="Advanced Bandit Selection",
    description="Select option using advanced contextual bandit algorithms with multiple strategies",
    responses={
        200: {"description": "Selection completed successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Selection failed", "model": ErrorResponse}
    }
)
async def select_bandit_option_advanced(
    keys: List[str],
    context: Optional[Dict[str, Any]] = None,
    algorithm: str = Query(default="ucb", description="Algorithm to use: ucb, thompson, linucb, epsilon_greedy"),
    current_user: User = Depends(get_current_active_user)
):
    """Advanced bandit selection with multiple algorithms"""
    try:
        result = await bandit_service.select_arm_advanced(
            keys=keys,
            context=context,
            algorithm=algorithm,
            user_id=current_user.id
        )
        
        return result
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Bandit selection failed: {e}")
        ErrorHandler.raise_internal_server_error("Bandit selection failed", str(e))

@router.post(
    "/select/batch",
    summary="Batch Bandit Selection",
    description="Select multiple options in parallel with different algorithms",
    responses={
        200: {"description": "Batch selection completed successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Batch selection failed", "model": ErrorResponse}
    }
)
async def select_bandit_batch(
    requests: List[Dict[str, Any]],
    current_user: User = Depends(get_current_active_user)
):
    """Batch bandit selection for multiple contexts"""
    try:
        if not requests or len(requests) == 0:
            ErrorHandler.raise_bad_request("No selection requests provided")
        
        if len(requests) > 50:
            ErrorHandler.raise_bad_request("Too many requests (max 50)")
        
        # Process requests in parallel
        tasks = []
        for req in requests:
            task = bandit_service.select_arm_advanced(
                keys=req.get("keys", []),
                context=req.get("context"),
                algorithm=req.get("algorithm", "ucb"),
                user_id=current_user.id
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append({
                    "index": i,
                    "error": str(result)
                })
            else:
                successful_results.append(result)
        
        return {
            "total_requests": len(requests),
            "successful_selections": len(successful_results),
            "failed_selections": len(failed_results),
            "results": successful_results,
            "failures": failed_results
        }
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Batch bandit selection failed: {e}")
        ErrorHandler.raise_internal_server_error("Batch bandit selection failed", str(e))

@router.post(
    "/update",
    summary="Update Bandit Reward",
    description="Update reward for a selected key with advanced tracking and analysis",
    responses={
        200: {"description": "Reward updated successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Reward update failed", "model": ErrorResponse}
    }
)
async def update_bandit_reward_advanced(
    key: str,
    reward: float,
    context: Optional[Dict[str, Any]] = None,
    feedback_type: str = Query(default="implicit", description="Type of feedback: implicit, explicit, binary"),
    current_user: User = Depends(get_current_active_user)
):
    """Advanced bandit reward update with performance tracking"""
    try:
        if not (0.0 <= reward <= 1.0):
            ErrorHandler.raise_bad_request("Reward must be between 0.0 and 1.0")
        
        if not key or len(key.strip()) == 0:
            ErrorHandler.raise_bad_request("Key must be provided")
        
        # Store the reward
        from .. import db
        
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO bandit_rewards 
                    (key_name, reward, context_data, user_id, feedback_type, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    key,
                    reward,
                    json.dumps(context) if context else None,
                    current_user.id,
                    feedback_type,
                    datetime.utcnow()
                ))
        
        # Record reward for performance tracking
        performance_monitor.record_reward(key, reward, context or {})
        
        # Update key statistics cache
        await bandit_service._invalidate_key_cache(key, current_user.id)
        
        return {
            "key": key,
            "reward": reward,
            "context": context,
            "feedback_type": feedback_type,
            "status": "updated",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Bandit reward update failed: {e}")
        ErrorHandler.raise_internal_server_error("Bandit reward update failed", str(e))

@router.post(
    "/update/batch",
    summary="Batch Reward Update",
    description="Update multiple rewards in parallel",
    responses={
        200: {"description": "Batch rewards updated successfully"},
        400: {"description": "Invalid input", "model": ErrorResponse},
        500: {"description": "Batch reward update failed", "model": ErrorResponse}
    }
)
async def update_bandit_rewards_batch(
    rewards: List[Dict[str, Any]],
    current_user: User = Depends(get_current_active_user)
):
    """Batch reward update for multiple keys"""
    try:
        if not rewards or len(rewards) == 0:
            ErrorHandler.raise_bad_request("No rewards provided")
        
        if len(rewards) > 100:
            ErrorHandler.raise_bad_request("Too many rewards (max 100)")
        
        # Validate all rewards
        for i, reward_data in enumerate(rewards):
            if "key" not in reward_data or "reward" not in reward_data:
                ErrorHandler.raise_bad_request(f"Reward {i} missing required fields: key, reward")
            
            if not (0.0 <= reward_data["reward"] <= 1.0):
                ErrorHandler.raise_bad_request(f"Reward {i} must be between 0.0 and 1.0")
        
        # Process rewards in parallel
        tasks = []
        for reward_data in rewards:
            task = update_bandit_reward_advanced(
                key=reward_data["key"],
                reward=reward_data["reward"],
                context=reward_data.get("context"),
                feedback_type=reward_data.get("feedback_type", "implicit"),
                current_user=current_user
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_updates = []
        failed_updates = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_updates.append({
                    "index": i,
                    "error": str(result)
                })
            else:
                successful_updates.append(result)
        
        return {
            "total_rewards": len(rewards),
            "successful_updates": len(successful_updates),
            "failed_updates": len(failed_updates),
            "results": successful_updates,
            "failures": failed_updates
        }
        
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Batch reward update failed: {e}")
        ErrorHandler.raise_internal_server_error("Batch reward update failed", str(e))

@router.get(
    "/stats",
    summary="Advanced Bandit Statistics",
    description="Get comprehensive bandit statistics with performance analysis and algorithm comparison",
    responses={
        200: {"description": "Statistics retrieved successfully"},
        500: {"description": "Failed to retrieve statistics", "model": ErrorResponse}
    }
)
async def get_bandit_stats_advanced(
    days: int = Query(default=30, description="Number of days to analyze", ge=1, le=365),
    include_performance: bool = Query(default=True, description="Include performance metrics"),
    include_algorithm_comparison: bool = Query(default=True, description="Include algorithm comparison"),
    current_user: User = Depends(get_current_active_user)
):
    """Advanced bandit statistics with comprehensive analysis"""
    try:
        from .. import db
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        with db.get_conn() as conn:
            with conn.cursor() as cur:
                # Get comprehensive statistics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_selections,
                        COUNT(DISTINCT key_name) as unique_keys,
                        COUNT(DISTINCT algorithm_used) as algorithms_used,
                        AVG(ucb_score) as avg_ucb_score,
                        MAX(ucb_score) as max_ucb_score
                    FROM bandit_selections
                    WHERE user_id = %s AND created_at >= %s
                """, (current_user.id, start_date))
                
                selection_stats = cur.fetchone()
                
                # Get reward statistics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_rewards,
                        AVG(reward) as average_reward,
                        MIN(reward) as min_reward,
                        MAX(reward) as max_reward,
                        STDDEV(reward) as reward_stddev,
                        COUNT(DISTINCT feedback_type) as feedback_types
                    FROM bandit_rewards
                    WHERE user_id = %s AND created_at >= %s
                """, (current_user.id, start_date))
                
                reward_stats = cur.fetchone()
                
                # Get algorithm performance
                algorithm_stats = {}
                if include_algorithm_comparison:
                    cur.execute("""
                        SELECT 
                            algorithm_used,
                            COUNT(*) as selections,
                            AVG(ucb_score) as avg_ucb_score,
                            COUNT(DISTINCT key_name) as unique_keys
                        FROM bandit_selections
                        WHERE user_id = %s AND created_at >= %s
                        GROUP BY algorithm_used
                        ORDER BY selections DESC
                    """, (current_user.id, start_date))
                    
                    for row in cur.fetchall():
                        algorithm_stats[row[0]] = {
                            "selections": row[1],
                            "avg_ucb_score": float(row[2]) if row[2] else 0.0,
                            "unique_keys": row[3]
                        }
                
                # Get per-key comprehensive statistics
                cur.execute("""
                    SELECT 
                        bs.key_name,
                        COUNT(bs.*) as selection_count,
                        AVG(bs.ucb_score) as avg_ucb_score,
                        MAX(bs.ucb_score) as max_ucb_score,
                        COUNT(br.*) as reward_count,
                        AVG(br.reward) as avg_reward,
                        MIN(br.reward) as min_reward,
                        MAX(br.reward) as max_reward,
                        STDDEV(br.reward) as reward_stddev,
                        MAX(bs.created_at) as last_selected,
                        MAX(br.created_at) as last_rewarded
                    FROM bandit_selections bs
                    LEFT JOIN bandit_rewards br ON bs.key_name = br.key_name 
                        AND br.user_id = bs.user_id 
                        AND br.created_at >= %s
                    WHERE bs.user_id = %s AND bs.created_at >= %s
                    GROUP BY bs.key_name
                    ORDER BY selection_count DESC
                """, (start_date, current_user.id, start_date))
                
                key_stats = []
                for row in cur.fetchall():
                    key_stats.append({
                        "key_name": row[0],
                        "selection_count": row[1],
                        "avg_ucb_score": float(row[2]) if row[2] else 0.0,
                        "max_ucb_score": float(row[3]) if row[3] else 0.0,
                        "reward_count": row[4] if row[4] else 0,
                        "avg_reward": float(row[5]) if row[5] else 0.0,
                        "min_reward": float(row[6]) if row[6] else 0.0,
                        "max_reward": float(row[7]) if row[7] else 0.0,
                        "reward_stddev": float(row[8]) if row[8] else 0.0,
                        "last_selected": row[9].isoformat() if row[9] else None,
                        "last_rewarded": row[10].isoformat() if row[10] else None
                    })
                
                # Get time series data
                cur.execute("""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as selections,
                        AVG(ucb_score) as avg_ucb_score
                    FROM bandit_selections
                    WHERE user_id = %s AND created_at >= %s
                    GROUP BY DATE(created_at)
                    ORDER BY date
                """, (current_user.id, start_date))
                
                time_series = []
                for row in cur.fetchall():
                    time_series.append({
                        "date": row[0].isoformat(),
                        "selections": row[1],
                        "avg_ucb_score": float(row[2]) if row[2] else 0.0
                    })
        
        # Get performance metrics if requested
        performance_metrics = {}
        if include_performance:
            performance_metrics = performance_monitor.get_performance_metrics()
        
        # Calculate additional metrics
        total_selections = selection_stats[0] if selection_stats[0] else 0
        total_rewards = reward_stats[0] if reward_stats[0] else 0
        
        # Calculate conversion rate
        conversion_rate = (total_rewards / total_selections) if total_selections > 0 else 0.0
        
        # Calculate exploration rate
        exploration_rate = 0.0
        if key_stats:
            total_explorations = sum(1 for key in key_stats if key["selection_count"] == 1)
            exploration_rate = total_explorations / len(key_stats)
        
        # Calculate regret (simplified)
        regret = 0.0
        if key_stats and total_rewards > 0:
            optimal_reward = max(key["avg_reward"] for key in key_stats if key["avg_reward"] > 0)
            actual_rewards = [key["avg_reward"] for key in key_stats if key["avg_reward"] > 0]
            regret = sum(optimal_reward - reward for reward in actual_rewards)
        
        stats = {
            "summary": {
                "total_selections": total_selections,
                "unique_keys": selection_stats[1] if selection_stats[1] else 0,
                "algorithms_used": selection_stats[2] if selection_stats[2] else 0,
                "total_rewards": total_rewards,
                "average_reward": float(reward_stats[1]) if reward_stats[1] else 0.0,
                "min_reward": float(reward_stats[2]) if reward_stats[2] else 0.0,
                "max_reward": float(reward_stats[3]) if reward_stats[3] else 0.0,
                "reward_stddev": float(reward_stats[4]) if reward_stats[4] else 0.0,
                "feedback_types": reward_stats[5] if reward_stats[5] else 0,
                "conversion_rate": conversion_rate,
                "exploration_rate": exploration_rate,
                "regret": regret
            },
            "algorithm_performance": algorithm_stats,
            "key_statistics": key_stats,
            "time_series": time_series,
            "performance_metrics": performance_metrics,
            "analysis_period": {
                "days": days,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get bandit stats: {e}")
        ErrorHandler.raise_internal_server_error("Failed to get bandit statistics", str(e))

@router.get(
    "/performance",
    summary="Bandit Performance Metrics",
    description="Get real-time performance metrics for bandit operations",
    responses={
        200: {"description": "Performance metrics retrieved successfully"},
        500: {"description": "Failed to retrieve metrics", "model": ErrorResponse}
    }
)
async def get_bandit_performance(
    current_user: User = Depends(get_current_active_user)
):
    """Get bandit performance metrics"""
    try:
        # Get performance metrics
        performance_metrics = performance_monitor.get_performance_metrics()
        
        # Get system health
        system_health = await monitoring_service.get_system_health()
        
        return {
            "bandit_metrics": performance_metrics,
            "system_health": system_health.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get bandit performance: {e}")
        ErrorHandler.raise_internal_server_error("Failed to retrieve performance metrics", str(e))

@router.get(
    "/algorithms",
    summary="Available Algorithms",
    description="Get information about available bandit algorithms",
    responses={
        200: {"description": "Algorithms retrieved successfully"}
    }
)
async def get_available_algorithms():
    """Get information about available bandit algorithms"""
    algorithms = {
        "ucb": {
            "name": "Upper Confidence Bound",
            "description": "Balances exploration and exploitation using confidence intervals",
            "best_for": "Stationary environments with unknown reward distributions",
            "parameters": ["exploration_factor"],
            "complexity": "O(n)"
        },
        "thompson": {
            "name": "Thompson Sampling",
            "description": "Bayesian approach that samples from posterior distributions",
            "best_for": "Non-stationary environments and contextual bandits",
            "parameters": ["alpha", "beta"],
            "complexity": "O(n)"
        },
        "linucb": {
            "name": "Linear Upper Confidence Bound",
            "description": "Contextual bandit algorithm using linear models",
            "best_for": "Contextual bandits with feature vectors",
            "parameters": ["alpha", "lambda_reg"],
            "complexity": "O(d²)"
        },
        "epsilon_greedy": {
            "name": "Epsilon-Greedy",
            "description": "Simple strategy that explores with probability epsilon",
            "best_for": "Simple environments and quick prototyping",
            "parameters": ["epsilon"],
            "complexity": "O(n)"
        }
    }
    
    return {
        "algorithms": algorithms,
        "default_algorithm": "ucb",
        "recommended_algorithm": "thompson"
    }
