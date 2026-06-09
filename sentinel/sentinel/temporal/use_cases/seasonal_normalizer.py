import math
from datetime import datetime

from sentinel.domain.models import SeasonalBucket, WindowSnapshot


def bucket_index_for(
    timestamp: datetime,
    bucket_count: int,
    duration_seconds: int,
) -> int:
    epoch_seconds = int(timestamp.timestamp())
    return (epoch_seconds // duration_seconds) % bucket_count


def hour_of_week(timestamp: datetime) -> int:
    """Return the hour-of-week (0–167) for a UTC timestamp."""
    return timestamp.weekday() * 24 + timestamp.hour


def compute_z_scores(
    snapshot: WindowSnapshot,
    bucket: SeasonalBucket,
) -> dict[str, float]:
    """Compute z-scores for each snapshot dimension vs the bucket baseline.

    Returns 0.0 for any dimension when n_observations < 2 (std undefined).
    """
    if bucket.n_observations < 2:
        return {
            "anomaly_rate": 0.0,
            "volume": 0.0,
            "avg_score": 0.0,
            "avg_latency": 0.0,
        }

    def _z(value: float, mean: float, var: float) -> float:
        std = math.sqrt(var) if var > 0 else 0.0
        if std == 0.0:
            return 0.0
        return (value - mean) / std

    return {
        "anomaly_rate": _z(
            snapshot.anomaly_rate,
            bucket.mean_anomaly_rate,
            bucket.var_anomaly_rate,
        ),
        "volume": _z(
            float(snapshot.total_events),
            bucket.mean_volume,
            bucket.var_volume,
        ),
        "avg_score": _z(
            snapshot.avg_score,
            bucket.mean_avg_score,
            bucket.var_avg_score,
        ),
        "avg_latency": _z(
            snapshot.avg_latency_ms,
            bucket.mean_avg_latency_ms,
            bucket.var_avg_latency_ms,
        ),
        "events_per_second": _z(
            snapshot.events_per_second,
            bucket.mean_events_per_second,
            bucket.var_events_per_second,
        ),
    }


def update_bucket(
    bucket: SeasonalBucket,
    snapshot: WindowSnapshot,
    ema_alpha: float,
) -> SeasonalBucket:
    """Update bucket statistics with EMA for mean and variance."""
    a = ema_alpha

    def _ema_update(
        old_mean: float,
        old_var: float,
        new_value: float,
    ) -> tuple[float, float]:
        new_mean = (1 - a) * old_mean + a * new_value
        new_var = (1 - a) * old_var + a * (new_value - new_mean) ** 2
        return new_mean, new_var

    new_rate_mean, new_rate_var = _ema_update(
        bucket.mean_anomaly_rate, bucket.var_anomaly_rate, snapshot.anomaly_rate
    )
    new_vol_mean, new_vol_var = _ema_update(
        bucket.mean_volume, bucket.var_volume, float(snapshot.total_events)
    )
    new_score_mean, new_score_var = _ema_update(
        bucket.mean_avg_score, bucket.var_avg_score, snapshot.avg_score
    )
    new_lat_mean, new_lat_var = _ema_update(
        bucket.mean_avg_latency_ms, bucket.var_avg_latency_ms, snapshot.avg_latency_ms
    )
    new_eps_mean, new_eps_var = _ema_update(
        bucket.mean_events_per_second, bucket.var_events_per_second, snapshot.events_per_second
    )

    return SeasonalBucket(
        n_observations=bucket.n_observations + 1,
        mean_anomaly_rate=new_rate_mean,
        var_anomaly_rate=new_rate_var,
        mean_volume=new_vol_mean,
        var_volume=new_vol_var,
        mean_avg_score=new_score_mean,
        var_avg_score=new_score_var,
        mean_avg_latency_ms=new_lat_mean,
        var_avg_latency_ms=new_lat_var,
        mean_events_per_second=new_eps_mean,
        var_events_per_second=new_eps_var,
    )
