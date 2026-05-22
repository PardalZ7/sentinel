"""Concrete anomaly detector implementations.

Each module exposes a class that implements IDetector (the Strategy interface
defined in domain/ports/detector.py).  New algorithms are added here without
touching the core agent logic.

Available detectors:
  IsolationForestDetector  — operational anomaly detection
  ConditionalVAEDetector   — semantic detection via conditional VAE
  MAFDetector              — density estimation via MADE (Masked Autoencoder)
  NRIDetector              — graph-based (stub, stable interface)
"""
