import os, random, json, datetime as dt
import psycopg2
import pandas as pd
import numpy as np
import asyncio
import logging
import time
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import pickle
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")

# Redis connection for caching and model storage
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=2, decode_responses=True)

def pg():
    return psycopg2.connect(DATABASE_URL)

app = FastAPI(
    title="Advanced Forecasting & Bandit Service",
    description="High-performance forecasting with ensemble ML models and contextual bandits",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Advanced Forecasting Models ----------

class SeriesPoint(BaseModel):
    date: str
    value: float
    metadata: Optional[Dict[str, Any]] = None

class ForecastRequest(BaseModel):
    series: List[SeriesPoint]
    horizon_days: int = Field(default=7, ge=1, le=365)
    model_type: str = Field(default="ensemble", pattern="^(naive|linear|rf|gb|ensemble|auto)$")
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    include_uncertainty: bool = True
    seasonal_periods: Optional[int] = None
    external_features: Optional[Dict[str, List[float]]] = None

class ForecastResponse(BaseModel):
    success: bool
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    accuracy_metrics: Optional[Dict[str, float]] = None
    processing_time: float
    cached: bool = False
    error: Optional[str] = None

class ModelPerformance(BaseModel):
    model_name: str
    mae: float
    mse: float
    rmse: float
    r2: float
    training_time: float

# Advanced forecasting models
class EnsembleForecaster:
    def __init__(self):
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def create_features(self, df: pd.DataFrame, horizon: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """Create advanced features for time series forecasting"""
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Basic time features
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['day_of_month'] = df['ds'].dt.day
        df['month'] = df['ds'].dt.month
        df['year'] = df['ds'].dt.year
        
        # Lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            if len(df) > lag:
                df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14, 30]:
            if len(df) > window:
                df[f'rolling_mean_{window}'] = df['y'].rolling(window=window).mean()
                df[f'rolling_std_{window}'] = df['y'].rolling(window=window).std()
                df[f'rolling_max_{window}'] = df['y'].rolling(window=window).max()
                df[f'rolling_min_{window}'] = df['y'].rolling(window=window).min()
        
        # Trend features
        df['trend'] = np.arange(len(df))
        df['trend_squared'] = df['trend'] ** 2
        
        # Difference features
        df['diff_1'] = df['y'].diff()
        df['diff_7'] = df['y'].diff(7)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("Not enough data for feature creation")
        
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['ds', 'y']]
        X = df[feature_cols].values
        y = df['y'].values
        
        return X, y
    
    def fit(self, df: pd.DataFrame) -> Dict[str, ModelPerformance]:
        """Fit all models and return performance metrics"""
        try:
            X, y = self.create_features(df)
            
            if len(X) < 10:
                raise ValueError("Not enough data for training")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            performance = {}
            
            for name, model in self.models.items():
                start_time = time.time()
                
                try:
                    model.fit(X_scaled, y)
                    y_pred = model.predict(X_scaled)
                    
                    mae = mean_absolute_error(y, y_pred)
                    mse = mean_squared_error(y, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = model.score(X_scaled, y)
                    
                    performance[name] = ModelPerformance(
                        model_name=name,
                        mae=mae,
                        mse=mse,
                        rmse=rmse,
                        r2=r2,
                        training_time=time.time() - start_time
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to train {name}: {str(e)}")
                    continue
            
            self.is_fitted = True
            return performance
            
        except Exception as e:
            logger.error(f"Error in ensemble fitting: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame, horizon: int = 7) -> List[Dict[str, Any]]:
        """Generate ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        try:
            X, _ = self.create_features(df)
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models
            predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)
                    predictions[name] = pred
                except:
                    continue
            
            if not predictions:
                raise ValueError("No models available for prediction")
            
            # Ensemble prediction (weighted average)
            weights = {name: 1.0 for name in predictions.keys()}
            ensemble_pred = np.average(list(predictions.values()), axis=0, weights=list(weights.values()))
            
            # Calculate uncertainty (standard deviation across models)
            uncertainty = np.std(list(predictions.values()), axis=0)
            
            # Generate future predictions
            last_date = pd.to_datetime(df['ds']).max()
            future_predictions = []
            
            for i in range(1, horizon + 1):
                future_date = last_date + pd.Timedelta(days=i)
                
                # Simple extrapolation for future points
                if len(ensemble_pred) > 0:
                    point_forecast = float(ensemble_pred[-1])
                    uncertainty_val = float(uncertainty[-1]) if len(uncertainty) > 0 else 0.1
                else:
                    point_forecast = float(df['y'].mean())
                    uncertainty_val = float(df['y'].std())
                
                future_predictions.append({
                    "date": future_date.date().isoformat(),
                    "point": point_forecast,
                    "lower_bound": point_forecast - 1.96 * uncertainty_val,
                    "upper_bound": point_forecast + 1.96 * uncertainty_val,
                    "uncertainty": uncertainty_val,
                    "model_agreement": 1.0 - (uncertainty_val / max(abs(point_forecast), 1e-6))
                })
            
            return future_predictions
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            raise

# Global forecaster instance
forecaster = EnsembleForecaster()

class ParallelForecastProcessor:
    """High-performance parallel forecasting processor"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0.0
        }
        logger.info(f"Initialized ParallelForecastProcessor with {self.max_workers} workers")
    
    async def process_forecasts_batch(self, forecast_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple forecast requests in parallel
        
        Args:
            forecast_requests: List of forecast request dictionaries
        
        Returns:
            List of forecast results
        """
        start_time = time.time()
        logger.info(f"Starting parallel processing of {len(forecast_requests)} forecast requests")
        
        # Process forecasts in parallel
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all forecast tasks
            future_to_request = {
                executor.submit(self._process_single_forecast, request): request 
                for request in forecast_requests
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_request):
                request = future_to_request[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.stats["successful"] += 1
                except Exception as e:
                    logger.error(f"Error processing forecast request: {e}")
                    results.append({
                        "success": False,
                        "error": str(e),
                        "request_id": request.get("id", "unknown")
                    })
                    self.stats["failed"] += 1
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats["total_processed"] += len(forecast_requests)
        self.stats["total_time"] += processing_time
        
        logger.info(f"Parallel forecasting completed: {self.stats['successful']}/{len(forecast_requests)} successful in {processing_time:.2f}s")
        
        return results
    
    def _process_single_forecast(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single forecast request synchronously
        This runs in a separate process
        """
        try:
            # Create a new forecaster instance for this process
            local_forecaster = EnsembleForecaster()
            
            # Convert request to DataFrame
            series_data = request.get("series", [])
            df = pd.DataFrame([{"ds": p["date"], "y": p["value"]} for p in series_data])
            df["ds"] = pd.to_datetime(df["ds"])
            df = df.sort_values("ds").reset_index(drop=True)
            
            if len(df) < 3:
                raise ValueError("Need at least 3 data points for forecasting")
            
            # Train models and get performance
            performance = local_forecaster.fit(df)
            
            # Generate predictions
            horizon = request.get("horizon_days", 7)
            predictions = local_forecaster.predict(df, horizon)
            
            # Prepare result
            result = {
                "success": True,
                "request_id": request.get("id"),
                "predictions": predictions,
                "model_info": {
                    "ensemble_models": list(performance.keys()),
                    "best_model": max(performance.keys(), key=lambda k: performance[k].r2) if performance else None,
                    "total_models": len(performance),
                    "training_samples": len(df)
                },
                "accuracy_metrics": {
                    "avg_r2": np.mean([p.r2 for p in performance.values()]) if performance else 0,
                    "avg_mae": np.mean([p.mae for p in performance.values()]) if performance else 0,
                    "avg_rmse": np.mean([p.rmse for p in performance.values()]) if performance else 0
                } if performance else {}
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing single forecast: {e}")
            return {
                "success": False,
                "error": str(e),
                "request_id": request.get("id", "unknown")
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            "avg_processing_time": self.stats["total_time"] / max(1, self.stats["total_processed"]),
            "success_rate": self.stats["successful"] / max(1, self.stats["total_processed"]),
            "parallel_workers": self.max_workers
        }

# Global parallel processor
parallel_forecast_processor = ParallelForecastProcessor()

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_advanced(request: ForecastRequest) -> ForecastResponse:
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([{"ds": p.date, "y": p.value} for p in request.series])
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").reset_index(drop=True)
        
        if len(df) < 3:
            raise ValueError("Need at least 3 data points for forecasting")
        
        # Check cache
        cache_key = f"forecast:{hash(str(request.dict()))}"
        cached_result = redis_client.get(cache_key)
        if cached_result:
            cached_data = json.loads(cached_result)
            return ForecastResponse(
                success=True,
                predictions=cached_data["predictions"],
                model_info=cached_data["model_info"],
                accuracy_metrics=cached_data.get("accuracy_metrics"),
                processing_time=time.time() - start_time,
                cached=True
            )
        
        # Train models and get performance
        performance = forecaster.fit(df)
        
        # Generate predictions
        predictions = forecaster.predict(df, request.horizon_days)
        
        # Prepare model info
        model_info = {
            "ensemble_models": list(performance.keys()),
            "best_model": max(performance.keys(), key=lambda k: performance[k].r2) if performance else None,
            "total_models": len(performance),
            "training_samples": len(df)
        }
        
        # Calculate overall accuracy metrics
        accuracy_metrics = {}
        if performance:
            avg_r2 = np.mean([p.r2 for p in performance.values()])
            avg_mae = np.mean([p.mae for p in performance.values()])
            avg_rmse = np.mean([p.rmse for p in performance.values()])
            
            accuracy_metrics = {
                "avg_r2": float(avg_r2),
                "avg_mae": float(avg_mae),
                "avg_rmse": float(avg_rmse),
                "model_performance": {name: p.dict() for name, p in performance.items()}
            }
        
        # Cache result
        result_data = {
            "predictions": predictions,
            "model_info": model_info,
            "accuracy_metrics": accuracy_metrics
        }
        redis_client.setex(cache_key, 3600, json.dumps(result_data))  # Cache for 1 hour
        
        return ForecastResponse(
            success=True,
            predictions=predictions,
            model_info=model_info,
            accuracy_metrics=accuracy_metrics,
            processing_time=time.time() - start_time,
            cached=False
        )
        
    except Exception as e:
        logger.error(f"Forecasting error: {str(e)}")
        return ForecastResponse(
            success=False,
            predictions=[],
            model_info={},
            processing_time=time.time() - start_time,
            error=str(e)
        )

@app.get("/forecast/models")
async def get_available_models():
    """Get information about available forecasting models"""
    return {
        "models": {
            "linear": "Linear regression with trend",
            "ridge": "Ridge regression (L2 regularization)",
            "lasso": "Lasso regression (L1 regularization)",
            "rf": "Random Forest ensemble",
            "gb": "Gradient Boosting",
            "ensemble": "Weighted ensemble of all models"
        },
        "features": [
            "Time-based features (day, month, year)",
            "Lag features (1, 2, 3, 7, 14, 30 days)",
            "Rolling statistics (mean, std, max, min)",
            "Trend and difference features",
            "External features support"
        ]
    }

@app.post("/forecast/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Retrain models with latest data"""
    background_tasks.add_task(retrain_models_task)
    return {"message": "Model retraining started in background"}

async def retrain_models_task():
    """Background task to retrain models"""
    try:
        # This would typically fetch latest data and retrain
        logger.info("Retraining models with latest data...")
        # Implementation would go here
        logger.info("Model retraining completed")
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")

# ---------- Advanced Contextual Bandit System ----------

class ContextualBanditRequest(BaseModel):
    keys: List[str]
    context: Optional[Dict[str, Any]] = None
    exploration_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    algorithm: str = Field(default="thompson", pattern="^(thompson|ucb|epsilon_greedy|linucb)$")

class ContextualBanditResponse(BaseModel):
    success: bool
    selected_key: Optional[str] = None
    confidence: float = 0.0
    exploration_used: bool = False
    algorithm_used: str = ""
    context_features: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BanditUpdateRequest(BaseModel):
    key: str
    reward: float = Field(ge=0.0, le=1.0)
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class BanditStatsResponse(BaseModel):
    total_arms: int
    total_pulls: int
    best_arm: Optional[str] = None
    worst_arm: Optional[str] = None
    average_reward: float
    confidence_scores: Dict[str, float]
    exploration_rate: float

class LinUCB:
    """Linear Upper Confidence Bound algorithm for contextual bandits"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.arms = {}  # arm_id -> (A, b, features)
    
    def add_arm(self, arm_id: str, feature_dim: int):
        """Add a new arm with feature dimension"""
        self.arms[arm_id] = {
            'A': np.eye(feature_dim),
            'b': np.zeros(feature_dim),
            'features': feature_dim
        }
    
    def select_arm(self, context: np.ndarray, available_arms: List[str]) -> Tuple[str, float]:
        """Select arm using LinUCB algorithm"""
        if not available_arms:
            raise ValueError("No arms available")
        
        best_arm = None
        best_ucb = -np.inf
        
        for arm_id in available_arms:
            if arm_id not in self.arms:
                self.add_arm(arm_id, len(context))
            
            arm_data = self.arms[arm_id]
            A = arm_data['A']
            b = arm_data['b']
            
            # Calculate UCB
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
            ucb = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = arm_id
        
        return best_arm, best_ucb
    
    def update(self, arm_id: str, context: np.ndarray, reward: float):
        """Update arm parameters with new observation"""
        if arm_id not in self.arms:
            self.add_arm(arm_id, len(context))
        
        arm_data = self.arms[arm_id]
        arm_data['A'] += np.outer(context, context)
        arm_data['b'] += reward * context

class AdvancedBanditSystem:
    def __init__(self):
        self.linucb = LinUCB()
        self.arm_stats = {}  # arm_id -> (successes, failures, total_reward)
        self.context_history = []
        
    def extract_context_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from context"""
        features = []
        
        # Time-based features
        now = datetime.now()
        features.extend([
            now.hour / 24.0,  # Hour of day (0-1)
            now.weekday() / 7.0,  # Day of week (0-1)
            now.day / 31.0,  # Day of month (0-1)
        ])
        
        # Context features
        if context:
            # Numerical features
            for key, value in context.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, str):
                    # Hash string to numerical value
                    features.append(hash(value) % 1000 / 1000.0)
                elif isinstance(value, bool):
                    features.append(1.0 if value else 0.0)
        
        # Pad or truncate to fixed dimension
        target_dim = 10
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        else:
            features = features[:target_dim]
        
        return np.array(features, dtype=np.float32)
    
    def thompson_sampling(self, available_arms: List[str]) -> Tuple[str, float]:
        """Thompson sampling selection"""
        best_arm = None
        best_theta = -1
        
        for arm_id in available_arms:
            if arm_id not in self.arm_stats:
                self.arm_stats[arm_id] = {'successes': 0, 'failures': 0, 'total_reward': 0.0}
            
            stats = self.arm_stats[arm_id]
            a = stats['successes'] + 1
            b = stats['failures'] + 1
            theta = random.betavariate(a, b)
            
            if theta > best_theta:
                best_theta = theta
                best_arm = arm_id
        
        return best_arm, best_theta
    
    def epsilon_greedy(self, available_arms: List[str], epsilon: float) -> Tuple[str, float]:
        """Epsilon-greedy selection"""
        if random.random() < epsilon:
            # Explore
            selected = random.choice(available_arms)
            return selected, 0.5  # Random confidence
        else:
            # Exploit - choose best known arm
            best_arm = None
            best_avg_reward = -1
            
            for arm_id in available_arms:
                if arm_id not in self.arm_stats:
                    self.arm_stats[arm_id] = {'successes': 0, 'failures': 0, 'total_reward': 0.0}
                
                stats = self.arm_stats[arm_id]
                total_pulls = stats['successes'] + stats['failures']
                if total_pulls > 0:
                    avg_reward = stats['total_reward'] / total_pulls
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        best_arm = arm_id
            
            if best_arm is None:
                best_arm = random.choice(available_arms)
                best_avg_reward = 0.5
            
            return best_arm, best_avg_reward
    
    def select_arm(self, request: ContextualBanditRequest) -> ContextualBanditResponse:
        """Select arm using specified algorithm"""
        try:
            available_arms = request.keys
            if not available_arms:
                return ContextualBanditResponse(
                    success=False,
                    error="No arms provided"
                )
            
            context_features = None
            if request.context:
                context_features = self.extract_context_features(request.context)
                self.context_history.append(context_features)
            
            selected_arm = None
            confidence = 0.0
            exploration_used = False
            
            if request.algorithm == "thompson":
                selected_arm, confidence = self.thompson_sampling(available_arms)
                
            elif request.algorithm == "epsilon_greedy":
                selected_arm, confidence = self.epsilon_greedy(available_arms, request.exploration_rate)
                exploration_used = random.random() < request.exploration_rate
                
            elif request.algorithm == "linucb":
                if context_features is not None:
                    selected_arm, confidence = self.linucb.select_arm(context_features, available_arms)
                else:
                    # Fallback to Thompson sampling if no context
                    selected_arm, confidence = self.thompson_sampling(available_arms)
            
            else:
                return ContextualBanditResponse(
                    success=False,
                    error=f"Unknown algorithm: {request.algorithm}"
                )
            
            return ContextualBanditResponse(
                success=True,
                selected_key=selected_arm,
                confidence=float(confidence),
                exploration_used=exploration_used,
                algorithm_used=request.algorithm,
                context_features=request.context
            )
            
        except Exception as e:
            logger.error(f"Bandit selection error: {str(e)}")
            return ContextualBanditResponse(
                success=False,
                error=str(e)
            )
    
    def update_arm(self, request: BanditUpdateRequest):
        """Update arm with new reward"""
        try:
            arm_id = request.key
            reward = request.reward
            
            # Update basic stats
            if arm_id not in self.arm_stats:
                self.arm_stats[arm_id] = {'successes': 0, 'failures': 0, 'total_reward': 0.0}
            
            stats = self.arm_stats[arm_id]
            stats['total_reward'] += reward
            
            if reward >= 0.5:
                stats['successes'] += 1
            else:
                stats['failures'] += 1
            
            # Update LinUCB if context provided
            if request.context:
                context_features = self.extract_context_features(request.context)
                self.linucb.update(arm_id, context_features, reward)
            
            # Store in database
            with pg() as conn, conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO bandit_state(key, count, success, params, updated_at)
                    VALUES (%s, 1, %s, %s, now())
                    ON CONFLICT (key) DO UPDATE
                    SET count = bandit_state.count + 1,
                        success = bandit_state.success + %s,
                        params = %s,
                        updated_at = now()
                """, (
                    arm_id,
                    1 if reward >= 0.5 else 0,
                    json.dumps(request.context or {}),
                    1 if reward >= 0.5 else 0,
                    json.dumps(request.context or {})
                ))
                conn.commit()
            
            return {"success": True, "message": "Arm updated successfully"}
            
        except Exception as e:
            logger.error(f"Bandit update error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_stats(self) -> BanditStatsResponse:
        """Get bandit system statistics"""
        try:
            total_arms = len(self.arm_stats)
            total_pulls = sum(stats['successes'] + stats['failures'] for stats in self.arm_stats.values())
            
            if total_arms == 0:
                return BanditStatsResponse(
                    total_arms=0,
                    total_pulls=0,
                    average_reward=0.0,
                    confidence_scores={},
                    exploration_rate=0.0
                )
            
            # Calculate average reward and find best/worst arms
            arm_rewards = {}
            for arm_id, stats in self.arm_stats.items():
                total_pulls_arm = stats['successes'] + stats['failures']
                if total_pulls_arm > 0:
                    arm_rewards[arm_id] = stats['total_reward'] / total_pulls_arm
                else:
                    arm_rewards[arm_id] = 0.0
            
            average_reward = sum(arm_rewards.values()) / len(arm_rewards) if arm_rewards else 0.0
            best_arm = max(arm_rewards.keys(), key=lambda k: arm_rewards[k]) if arm_rewards else None
            worst_arm = min(arm_rewards.keys(), key=lambda k: arm_rewards[k]) if arm_rewards else None
            
            # Calculate confidence scores (Thompson sampling parameters)
            confidence_scores = {}
            for arm_id, stats in self.arm_stats.items():
                a = stats['successes'] + 1
                b = stats['failures'] + 1
                confidence_scores[arm_id] = a / (a + b)  # Expected value of Beta distribution
            
            return BanditStatsResponse(
                total_arms=total_arms,
                total_pulls=total_pulls,
                best_arm=best_arm,
                worst_arm=worst_arm,
                average_reward=average_reward,
                confidence_scores=confidence_scores,
                exploration_rate=0.1  # Default exploration rate
            )
            
        except Exception as e:
            logger.error(f"Error getting bandit stats: {str(e)}")
            return BanditStatsResponse(
                total_arms=0,
                total_pulls=0,
                average_reward=0.0,
                confidence_scores={},
                exploration_rate=0.0
            )

# Global bandit system
bandit_system = AdvancedBanditSystem()

@app.post("/bandit/select", response_model=ContextualBanditResponse)
async def bandit_select_advanced(request: ContextualBanditRequest) -> ContextualBanditResponse:
    """Advanced contextual bandit selection"""
    return bandit_system.select_arm(request)

@app.post("/bandit/update")
async def bandit_update_advanced(request: BanditUpdateRequest):
    """Update bandit arm with reward and context"""
    return bandit_system.update_arm(request)

@app.get("/bandit/stats", response_model=BanditStatsResponse)
async def bandit_stats():
    """Get comprehensive bandit statistics"""
    return bandit_system.get_stats()

@app.post("/forecast/batch")
async def forecast_batch_parallel(request: List[ForecastRequest]) -> List[Dict[str, Any]]:
    """
    Process multiple forecast requests in parallel for high performance
    
    - **requests**: List of forecast requests (max 50)
    - **parallel_workers**: Number of parallel workers (auto if not specified)
    
    Uses multiprocessing for optimal CPU utilization.
    """
    try:
        if not request:
            raise HTTPException(status_code=400, detail="No forecast requests provided")
        
        if len(request) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 forecast requests allowed per batch")
        
        # Convert Pydantic models to dictionaries
        forecast_requests = []
        for i, req in enumerate(request):
            forecast_requests.append({
                "id": f"forecast_{i}",
                "series": [{"date": p.date, "value": p.value} for p in req.series],
                "horizon_days": req.horizon_days,
                "model_type": req.model_type,
                "confidence_level": req.confidence_level,
                "include_uncertainty": req.include_uncertainty,
                "seasonal_periods": req.seasonal_periods,
                "external_features": req.external_features
            })
        
        # Process forecasts in parallel
        results = await parallel_forecast_processor.process_forecasts_batch(forecast_requests)
        
        return {
            "success": True,
            "total_requests": len(request),
            "successful": len([r for r in results if r.get("success", False)]),
            "failed": len([r for r in results if not r.get("success", False)]),
            "results": results,
            "performance_stats": parallel_forecast_processor.get_performance_stats()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch forecasting: {e}")
        raise HTTPException(status_code=500, detail="Failed to process forecast batch")


@app.get("/forecast/performance")
async def get_forecast_performance():
    """Get forecasting performance metrics"""
    try:
        stats = parallel_forecast_processor.get_performance_stats()
        
        # Add system metrics
        import psutil
        stats.update({
            "system_cpu_count": psutil.cpu_count(),
            "system_memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "system_cpu_percent": psutil.cpu_percent(interval=1),
            "system_memory_percent": psutil.virtual_memory().percent
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting forecast performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")


@app.get("/bandit/arms")
async def get_bandit_arms():
    """Get all available bandit arms with their statistics"""
    try:
        with pg() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT key, count, success, params, updated_at
                FROM bandit_state
                ORDER BY success::float / NULLIF(count, 0) DESC
            """)
            rows = cur.fetchall()
        
        arms = []
        for row in rows:
            key, count, success, params, updated_at = row
            success_rate = success / count if count > 0 else 0.0
            
            arms.append({
                "key": key,
                "total_pulls": count,
                "successes": success,
                "success_rate": success_rate,
                "params": json.loads(params) if params else {},
                "last_updated": updated_at.isoformat() if updated_at else None
            })
        
        return {"arms": arms, "total_arms": len(arms)}
        
    except Exception as e:
        logger.error(f"Error getting bandit arms: {str(e)}")
        return {"arms": [], "total_arms": 0, "error": str(e)}

@app.delete("/bandit/reset")
async def reset_bandit_system():
    """Reset bandit system (clear all data)"""
    try:
        bandit_system.arm_stats.clear()
        bandit_system.context_history.clear()
        bandit_system.linucb = LinUCB()
        
        with pg() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM bandit_state")
            conn.commit()
        
        return {"message": "Bandit system reset successfully"}
        
    except Exception as e:
        logger.error(f"Error resetting bandit system: {str(e)}")
        return {"error": str(e)}
