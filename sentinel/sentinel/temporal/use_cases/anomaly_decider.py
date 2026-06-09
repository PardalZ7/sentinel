from dataclasses import dataclass

from sentinel.config.schema import TemporalLayerConfig


@dataclass
class AnomalyDecision:
    is_anomalous: bool
    is_rate_anomaly: bool
    is_volume_anomaly: bool
    is_score_anomaly: bool
    is_latency_anomaly: bool
    is_throughput_anomaly: bool
    z_scores: dict[str, float]


def decide(
    z_scores: dict[str, float],
    config: TemporalLayerConfig,
    bucket_confidence: int,
) -> AnomalyDecision:
    """Evaluate z-scores against thresholds.

    Returns is_anomalous=False during WARMUP (bucket_confidence below min_observations).
    """
    if bucket_confidence < config.bucket.min_observations:
        return AnomalyDecision(
            is_anomalous=False,
            is_rate_anomaly=False,
            is_volume_anomaly=False,
            is_score_anomaly=False,
            is_latency_anomaly=False,
            is_throughput_anomaly=False,
            z_scores=z_scores,
        )

    is_rate = abs(z_scores.get("anomaly_rate", 0.0)) >= config.anomaly_rate_z_threshold
    is_vol = abs(z_scores.get("volume", 0.0)) >= config.volume_z_threshold
    is_score = abs(z_scores.get("avg_score", 0.0)) >= config.avg_score_z_threshold
    is_lat = abs(z_scores.get("avg_latency", 0.0)) >= config.avg_latency_z_threshold
    is_throughput = abs(z_scores.get("events_per_second", 0.0)) >= config.throughput_z_threshold

    return AnomalyDecision(
        is_anomalous=is_rate or is_vol or is_score or is_lat or is_throughput,
        is_rate_anomaly=is_rate,
        is_volume_anomaly=is_vol,
        is_score_anomaly=is_score,
        is_latency_anomaly=is_lat,
        is_throughput_anomaly=is_throughput,
        z_scores=z_scores,
    )
