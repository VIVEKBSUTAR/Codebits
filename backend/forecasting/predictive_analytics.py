from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Optional import for prophet - graceful degradation if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet library not available. Forecasting will use fallback methods.")

from causal_engine.causal_graph import get_causal_graph
from models.prediction_model import PredictionItemModel
from utils.logger import SystemLogger

logger = SystemLogger(module_name="forecasting")

class AnomalyDetector:
    """Advanced anomaly detection using multiple statistical methods"""

    def __init__(self, window_size: int = 20, z_threshold: float = 2.5):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.scaler = StandardScaler()

    def detect_anomalies(self, values: List[float]) -> Dict:
        """
        Detect anomalies using Z-score and IQR methods
        Returns anomaly scores and detection flags
        """
        if len(values) < self.window_size:
            return {"anomalies": [], "scores": [], "method": "insufficient_data"}

        # Convert to numpy array
        data = np.array(values)

        # Method 1: Rolling Z-score
        z_anomalies = self._detect_z_score_anomalies(data)

        # Method 2: Interquartile Range (IQR)
        iqr_anomalies = self._detect_iqr_anomalies(data)

        # Method 3: Statistical Process Control
        spc_anomalies = self._detect_spc_anomalies(data)

        # Combine methods (ensemble approach)
        combined_anomalies = []
        anomaly_scores = []

        for i in range(len(data)):
            z_flag = i in z_anomalies
            iqr_flag = i in iqr_anomalies
            spc_flag = i in spc_anomalies

            # Voting system: anomaly if 2+ methods agree
            vote_count = sum([z_flag, iqr_flag, spc_flag])
            is_anomaly = vote_count >= 2

            combined_anomalies.append(is_anomaly)
            anomaly_scores.append(vote_count / 3.0)  # Normalized score

        return {
            "anomalies": combined_anomalies,
            "scores": anomaly_scores,
            "method": "ensemble",
            "details": {
                "z_score_anomalies": len(z_anomalies),
                "iqr_anomalies": len(iqr_anomalies),
                "spc_anomalies": len(spc_anomalies)
            }
        }

    def _detect_z_score_anomalies(self, data: np.ndarray) -> List[int]:
        """Rolling Z-score anomaly detection"""
        anomalies = []

        for i in range(self.window_size, len(data)):
            window = data[i-self.window_size:i]
            current_value = data[i]

            mean = np.mean(window)
            std = np.std(window)

            if std > 0:
                z_score = abs((current_value - mean) / std)
                if z_score > self.z_threshold:
                    anomalies.append(i)

        return anomalies

    def _detect_iqr_anomalies(self, data: np.ndarray) -> List[int]:
        """IQR-based anomaly detection"""
        anomalies = []

        for i in range(self.window_size, len(data)):
            window = data[i-self.window_size:i]
            current_value = data[i]

            Q1 = np.percentile(window, 25)
            Q3 = np.percentile(window, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if current_value < lower_bound or current_value > upper_bound:
                anomalies.append(i)

        return anomalies

    def _detect_spc_anomalies(self, data: np.ndarray) -> List[int]:
        """Statistical Process Control anomaly detection"""
        anomalies = []

        for i in range(self.window_size, len(data)):
            window = data[i-self.window_size:i]
            current_value = data[i]

            mean = np.mean(window)
            std = np.std(window)

            # Control limits (3-sigma rule)
            ucl = mean + 3 * std  # Upper Control Limit
            lcl = mean - 3 * std  # Lower Control Limit

            if current_value > ucl or current_value < lcl:
                anomalies.append(i)

        return anomalies

class TimeSeriesForecaster:
    """Advanced time series forecasting using Facebook Prophet"""

    def __init__(self):
        self.models = {}  # Store trained models per zone/metric
        self.anomaly_detector = AnomalyDetector()

    def generate_forecast(self, zone: str, metric: str, periods: int = 24) -> Dict:
        """
        Generate probabilistic forecast using Prophet with anomaly detection
        Falls back to statistical methods if Prophet is not available

        Args:
            zone: Target zone name
            metric: Risk metric (flood, traffic, emergency, overall)
            periods: Number of hours to forecast ahead
        """
        try:
            # Get historical risk data for this zone
            historical_data = self._get_historical_risk_data(zone, metric)

            if len(historical_data) < 10:
                return self._generate_fallback_forecast(zone, metric, periods)

            # Check if Prophet is available
            if not PROPHET_AVAILABLE:
                logger.log(f"Prophet not available, using statistical fallback for {zone}/{metric}")
                return self._generate_statistical_forecast(zone, metric, periods, historical_data)

            # Prepare data for Prophet
            df = self._prepare_prophet_data(historical_data)

            # Train Prophet model
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=0.1,
                holidays_prior_scale=0.1,
                daily_seasonality=True,
                weekly_seasonality=False,
                yearly_seasonality=False,
                interval_width=0.8
            )

            model.fit(df)

            # Generate future dataframe
            future = model.make_future_dataframe(periods=periods, freq='H')
            forecast = model.predict(future)

            # Extract forecasted values
            forecast_values = forecast['yhat'].tail(periods).tolist()
            upper_bounds = forecast['yhat_upper'].tail(periods).tolist()
            lower_bounds = forecast['yhat_lower'].tail(periods).tolist()

            # Detect anomalies in historical + forecasted data
            all_values = historical_data + forecast_values
            anomaly_results = self.anomaly_detector.detect_anomalies(all_values)

            # Generate timestamps for forecast
            start_time = datetime.now()
            forecast_timestamps = [
                (start_time + timedelta(hours=i)).strftime("%H:%M")
                for i in range(1, periods + 1)
            ]

            # Identify upcoming anomalies
            upcoming_anomalies = []
            historical_length = len(historical_data)

            for i, (timestamp, is_anomaly, score) in enumerate(zip(
                forecast_timestamps,
                anomaly_results["anomalies"][historical_length:],
                anomaly_results["scores"][historical_length:]
            )):
                if is_anomaly:
                    upcoming_anomalies.append({
                        "timestamp": timestamp,
                        "predicted_value": forecast_values[i],
                        "anomaly_score": score,
                        "severity": "high" if score > 0.8 else "medium"
                    })

            return {
                "zone": zone,
                "metric": metric,
                "forecast_horizon_hours": periods,
                "predictions": [
                    {
                        "timestamp": ts,
                        "predicted_value": round(val, 3),
                        "upper_bound": round(upper, 3),
                        "lower_bound": round(lower, 3),
                        "confidence_interval": round(upper - lower, 3)
                    }
                    for ts, val, upper, lower in zip(
                        forecast_timestamps, forecast_values, upper_bounds, lower_bounds
                    )
                ],
                "anomaly_detection": {
                    "upcoming_anomalies": upcoming_anomalies,
                    "anomaly_method": anomaly_results["method"],
                    "detection_details": anomaly_results["details"]
                },
                "model_performance": {
                    "training_samples": len(historical_data),
                    "forecast_uncertainty": round(np.mean([u - l for u, l in zip(upper_bounds, lower_bounds)]), 3),
                    "method": "prophet"
                }
            }

        except Exception as e:
            logger.log(f"Forecasting error for {zone}/{metric}: {str(e)}")
            return self._generate_fallback_forecast(zone, metric, periods)

    def _get_historical_risk_data(self, zone: str, metric: str) -> List[float]:
        """
        Generate historical risk data based on current Bayesian inference and real time.
        Produces time-varying patterns that change with each request based on current hour.
        In production, this would query actual historical database.
        """
        try:
            graph = get_causal_graph(zone)
            current_probs = graph.run_inference()

            # Map metric names
            prob_key = {
                'flood': 'Flooding',
                'traffic': 'TrafficCongestion',
                'emergency': 'EmergencyDelay',
                'overall': 'overall'
            }.get(metric, 'Flooding')

            if prob_key == 'overall':
                base_risk = (
                    current_probs.get('Flooding', 0.2) * 0.35 +
                    current_probs.get('TrafficCongestion', 0.3) * 0.40 +
                    current_probs.get('EmergencyDelay', 0.1) * 0.25
                )
            else:
                base_risk = current_probs.get(prob_key, 0.2)

            # Use current time to create a seed that changes every hour
            now = datetime.now()
            time_seed = int(now.timestamp() // 3600)  # Changes hourly
            zone_offset = hash(zone + metric) % 1000
            rng = np.random.RandomState(time_seed + zone_offset)

            # Zone-specific characteristics that affect patterns
            zone_profiles = {
                'Bibwewadi':  {'flood_bias': 0.05, 'traffic_peak': 0.15, 'volatility': 0.06},
                'Katraj':     {'flood_bias': 0.08, 'traffic_peak': 0.12, 'volatility': 0.07},
                'Hadapsar':   {'flood_bias': 0.03, 'traffic_peak': 0.20, 'volatility': 0.05},
                'Kothrud':    {'flood_bias': 0.04, 'traffic_peak': 0.18, 'volatility': 0.04},
                'Warje':      {'flood_bias': 0.06, 'traffic_peak': 0.10, 'volatility': 0.05},
                'Sinhagad':   {'flood_bias': 0.07, 'traffic_peak': 0.08, 'volatility': 0.06},
                'Pimpri':     {'flood_bias': 0.04, 'traffic_peak': 0.22, 'volatility': 0.07},
                'Chinchwad':  {'flood_bias': 0.05, 'traffic_peak': 0.19, 'volatility': 0.06},
                'Hinjewadi':  {'flood_bias': 0.02, 'traffic_peak': 0.25, 'volatility': 0.05},
                'Wakad':      {'flood_bias': 0.03, 'traffic_peak': 0.16, 'volatility': 0.05},
                'Baner':      {'flood_bias': 0.03, 'traffic_peak': 0.14, 'volatility': 0.04},
                'Shivajinagar': {'flood_bias': 0.06, 'traffic_peak': 0.20, 'volatility': 0.08},
            }
            profile = zone_profiles.get(zone, {'flood_bias': 0.04, 'traffic_peak': 0.15, 'volatility': 0.05})

            # Metric-specific daily patterns
            metric_patterns = {
                'flood': lambda h: profile['flood_bias'] * (1.5 if 14 <= h <= 18 else 0.5),  # Afternoon rain
                'traffic': lambda h: profile['traffic_peak'] * (1.8 if 8 <= h <= 10 or 17 <= h <= 20 else 0.4),  # Rush hours
                'emergency': lambda h: 0.05 * (1.3 if 22 <= h or h <= 4 else 0.7),  # Night incidents
                'overall': lambda h: 0.08 * (1.2 if 8 <= h <= 10 or 17 <= h <= 20 else 0.8),
            }
            pattern_fn = metric_patterns.get(metric, metric_patterns['overall'])

            # Generate 48 hours of data ending at current time
            historical_data = []
            current_hour = now.hour

            # Random walk component for realistic auto-correlation
            walk = 0.0
            for i in range(48):
                hour_of_day = (current_hour - 48 + i) % 24

                # Base risk from Bayesian model
                val = base_risk

                # Add time-of-day pattern specific to metric
                val += pattern_fn(hour_of_day)

                # Random walk for auto-correlated noise
                walk += rng.normal(0, profile['volatility'] * 0.3)
                walk *= 0.92  # Mean reversion
                val += walk

                # Small i.i.d. noise
                val += rng.normal(0, profile['volatility'] * 0.5)

                # Ensure bounds [0, 1]
                historical_data.append(float(max(0.01, min(0.95, val))))

            return historical_data

        except Exception:
            # Fallback with time-varying data
            now = datetime.now()
            rng = np.random.RandomState(int(now.timestamp() // 3600))
            return [float(max(0.01, min(0.95, 0.25 + 0.08 * np.sin(2 * np.pi * i / 24) + rng.normal(0, 0.04)))) for i in range(48)]

    def _prepare_prophet_data(self, historical_data: List[float]) -> pd.DataFrame:
        """Prepare data in Prophet's required format"""
        # Generate timestamps for historical data
        end_time = datetime.now()
        timestamps = [
            end_time - timedelta(hours=len(historical_data) - i - 1)
            for i in range(len(historical_data))
        ]

        return pd.DataFrame({
            'ds': timestamps,
            'y': historical_data
        })

    def _generate_statistical_forecast(self, zone: str, metric: str, periods: int, historical_data: List[float]) -> Dict:
        """Generate sophisticated statistical forecast when Prophet is not available"""

        # Convert to numpy array for analysis
        data = np.array(historical_data)

        # Calculate statistical properties
        mean_value = np.mean(data)
        std_value = np.std(data)
        trend = np.polyfit(range(len(data)), data, 1)[0]  # Linear trend

        # Detect seasonality (daily pattern)
        if len(data) >= 24:
            # Calculate hourly averages for daily seasonality
            daily_pattern = []
            for hour in range(24):
                hour_values = [data[i] for i in range(hour, len(data), 24) if i < len(data)]
                if hour_values:
                    daily_pattern.append(np.mean(hour_values))
                else:
                    daily_pattern.append(mean_value)

            # Normalize daily pattern
            pattern_mean = np.mean(daily_pattern)
            daily_pattern = [(x - pattern_mean) for x in daily_pattern]
        else:
            daily_pattern = [0] * 24

        # Generate forecast timestamps
        start_time = datetime.now()
        forecast_timestamps = [
            (start_time + timedelta(hours=i)).strftime("%H:%M")
            for i in range(1, periods + 1)
        ]

        # Generate predictions
        predictions = []
        forecast_values = []

        for i in range(periods):
            # Base forecast using trend
            base_forecast = mean_value + trend * (len(data) + i)

            # Add daily seasonality
            hour_of_day = (datetime.now().hour + i + 1) % 24
            seasonal_component = daily_pattern[hour_of_day] if hour_of_day < len(daily_pattern) else 0

            # Add some noise decay over time
            noise_factor = max(0.5, 1.0 - (i / periods) * 0.3)

            # Combine components
            predicted_value = base_forecast + seasonal_component

            # Calculate confidence bounds based on historical volatility
            confidence_width = std_value * 1.96 * noise_factor  # 95% confidence
            upper_bound = predicted_value + confidence_width
            lower_bound = predicted_value - confidence_width

            # Ensure bounds are within [0, 1] for probabilities
            predicted_value = max(0, min(1, predicted_value))
            upper_bound = max(0, min(1, upper_bound))
            lower_bound = max(0, min(1, lower_bound))

            forecast_values.append(predicted_value)

            predictions.append({
                "timestamp": forecast_timestamps[i],
                "predicted_value": round(predicted_value, 3),
                "upper_bound": round(upper_bound, 3),
                "lower_bound": round(lower_bound, 3),
                "confidence_interval": round(upper_bound - lower_bound, 3)
            })

        # Perform anomaly detection on combined historical + forecasted data
        all_values = list(historical_data) + forecast_values
        anomaly_results = self.anomaly_detector.detect_anomalies(all_values)

        # Identify upcoming anomalies in forecast period
        upcoming_anomalies = []
        historical_length = len(historical_data)

        for i, (timestamp, is_anomaly, score) in enumerate(zip(
            forecast_timestamps,
            anomaly_results["anomalies"][historical_length:],
            anomaly_results["scores"][historical_length:]
        )):
            if is_anomaly:
                upcoming_anomalies.append({
                    "timestamp": timestamp,
                    "predicted_value": forecast_values[i],
                    "anomaly_score": score,
                    "severity": "high" if score > 0.8 else "medium"
                })

        return {
            "zone": zone,
            "metric": metric,
            "forecast_horizon_hours": periods,
            "predictions": predictions,
            "anomaly_detection": {
                "upcoming_anomalies": upcoming_anomalies,
                "anomaly_method": anomaly_results["method"],
                "detection_details": anomaly_results["details"]
            },
            "model_performance": {
                "training_samples": len(historical_data),
                "forecast_uncertainty": round(std_value, 3),
                "method": "statistical_trend_seasonal",
                "trend_coefficient": round(trend, 4),
                "seasonality_detected": len(historical_data) >= 24
            }
        }

    def _generate_fallback_forecast(self, zone: str, metric: str, periods: int) -> Dict:
        """Generate simple fallback forecast when Prophet fails"""
        # Simple linear trend with seasonal component
        start_time = datetime.now()
        forecast_timestamps = [
            (start_time + timedelta(hours=i)).strftime("%H:%M")
            for i in range(1, periods + 1)
        ]

        base_risk = 0.3
        predictions = []

        for i, ts in enumerate(forecast_timestamps):
            # Simple pattern with noise
            seasonal = 0.1 * np.sin(2 * np.pi * i / 24)  # Daily pattern
            trend = base_risk + seasonal + np.random.normal(0, 0.02)

            predictions.append({
                "timestamp": ts,
                "predicted_value": round(max(0, min(1, trend)), 3),
                "upper_bound": round(max(0, min(1, trend + 0.1)), 3),
                "lower_bound": round(max(0, min(1, trend - 0.1)), 3),
                "confidence_interval": 0.2
            })

        return {
            "zone": zone,
            "metric": metric,
            "forecast_horizon_hours": periods,
            "predictions": predictions,
            "anomaly_detection": {
                "upcoming_anomalies": [],
                "anomaly_method": "fallback",
                "detection_details": {}
            },
            "model_performance": {
                "training_samples": 0,
                "forecast_uncertainty": 0.2,
                "method": "simple_fallback"
            }
        }

# Global forecaster instance
forecaster = TimeSeriesForecaster()

def get_predictive_forecast(zone: str, metric: str = "overall", hours: int = 24) -> Dict:
    """
    Main interface for predictive forecasting with anomaly detection

    Args:
        zone: Target zone name
        metric: Risk metric to forecast (flood, traffic, emergency, overall)
        hours: Forecast horizon in hours

    Returns:
        Comprehensive forecast with anomaly alerts
    """
    return forecaster.generate_forecast(zone, metric, hours)