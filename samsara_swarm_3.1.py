#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌌 SAMSARA SWARM v3.1 — BUG-FIXED VERSION
===========================================
Critical fixes applied:

FIXED BUGS:
1. 🔴 Coherence Death Spiral - Added homeostasis
2. 🔴 Memory Leak - Added caps to all unbounded lists
3. 🔴 Audio Processing - Added proper validation
4. 🔴 Pygame Display - Fixed cleanup on crash
5. 🟠 Anomaly Detection - Added dynamic thresholds
6. 🟠 Communication System - Fixed circular logic
7. 🟠 Consciousness Detection - Fixed maturity calculation
8. 🟡 Performance - Added buffered logging
9. 🟡 Data Quality - Added validation and compression
10. 🟣 Architecture - Started component refactor
"""

import math
import random
import time
import json
import threading
import pygame
import cmath
import uuid
import pathlib
import numpy as np
import wave
import struct
import logging
import csv
import os
import gzip
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

# ============================================================
# 📊 SCIENTIFIC LOGGING SYSTEM - FIXED VERSION
# ============================================================

class ScientificLogger:
    """Comprehensive logging system for scientific observation."""
    
    def __init__(self, experiment_name: str = "samsara_swarm"):
        self.experiment_name = experiment_name
        self.start_time = time.time()  # Use epoch for easier analysis
        
        # Create logs directory
        self.logs_dir = pathlib.Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Buffered logging to reduce I/O
        self.swarm_buffer = []
        self.entity_buffer = []
        self.buffer_size = 300  # Write every 5 seconds at 60 FPS
        self.last_flush = time.time()
        
        # Setup comprehensive logging
        self.setup_loggers()
        
        # CSV files for quantitative analysis
        self.setup_csv_loggers()
        
        print(f"[ScientificLogger] Logging initialized: {datetime.fromtimestamp(self.start_time)}")
        
    def setup_loggers(self):
        """Setup different loggers for different event types."""
        
        # Anomaly logger (JSON Lines format for easy parsing)
        self.anomaly_file = self.logs_dir / 'anomalies.jsonl'
        
        # Emergent behavior logger (swarm-level phenomena)
        self.emergent_file = self.logs_dir / 'emergent_events.jsonl'
        
        # Consciousness/Sentience logger
        self.consciousness_file = self.logs_dir / 'consciousness_events.jsonl'
        
        # Cognitive processing logger
        self.cognition_file = self.logs_dir / 'cognition_events.jsonl'
        
        # Quantum events logger
        self.quantum_file = self.logs_dir / 'quantum_events.jsonl'
        
    def setup_csv_loggers(self):
        """Setup CSV files for quantitative data analysis."""
        
        # Swarm metrics CSV (compressed)
        self.swarm_csv = self.logs_dir / 'swarm_metrics.csv.gz'
        with gzip.open(self.swarm_csv, 'wt', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'elapsed_seconds', 'timestamp_utc', 'swarm_size', 'avg_coherence', 
                'avg_pleasure', 'avg_fear', 'avg_love', 'coherence_std', 
                'entropy_avg', 'quantum_connections', 'temporal_cycles',
                'communications_per_cycle'
            ])
        
        # Entity metrics CSV (compressed)
        self.entity_csv = self.logs_dir / 'entity_metrics.csv.gz'
        with gzip.open(self.entity_csv, 'wt', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'elapsed_seconds', 'timestamp_utc', 'entity_id', 'pleasure', 
                'fear', 'love', 'coherence', 'entropy', 'quantum_mood', 
                'temporal_phase', 'morphogenetic_sig', 'neural_plasmic_coherence',
                'goal', 'age'
            ])
        
        # Anomaly events CSV (compressed)
        self.anomaly_csv = self.logs_dir / 'anomaly_events.csv.gz'
        with gzip.open(self.anomaly_csv, 'wt', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'elapsed_seconds', 'timestamp_utc', 'event_type', 'entity_id', 
                'description', 'severity', 'coherence_before', 'coherence_after'
            ])
    
    def validate_and_sanitize(self, value, field_name, expected_type=float):
        """Validate data before logging."""
        if value is None:
            print(f"[Logger] WARNING: None value for {field_name}, using 0.0")
            return 0.0
        
        if expected_type == float:
            try:
                value = float(value)
            except (TypeError, ValueError):
                print(f"[Logger] WARNING: Invalid {field_name}: {value}, using 0.0")
                return 0.0
            
            if np.isnan(value):
                print(f"[Logger] WARNING: NaN in {field_name}, using 0.0")
                return 0.0
            
            if np.isinf(value):
                print(f"[Logger] WARNING: Inf in {field_name}, using 0.0")
                return 0.0
            
            # Clamp to reasonable range
            return float(np.clip(value, -1e6, 1e6))
        
        return value
    
    def log_anomaly(self, event_type: str, entity_id: str, description: str, 
                   severity: str = "MEDIUM", data: Optional[Dict] = None):
        """Log anomalous/significant events in JSON Lines format."""
        anomaly_record = {
            "elapsed_seconds": time.time() - self.start_time,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "entity_id": entity_id,
            "description": description[:200],  # Truncate
            "severity": severity,
            "data": data or {}
        }
        
        # Write to JSONL file
        with open(self.anomaly_file, 'a') as f:
            f.write(json.dumps(anomaly_record) + '\n')
        
        # Also write to CSV for compatibility
        with gzip.open(self.anomaly_csv, 'at', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                anomaly_record["elapsed_seconds"],
                anomaly_record["timestamp_utc"],
                event_type,
                entity_id,
                description[:200],
                severity,
                data.get('coherence_before', 0) if data else 0,
                data.get('coherence_after', 0) if data else 0
            ])
    
    def log_emergent(self, event_type: str, description: str, swarm_metrics: Dict):
        """Log emergent/swarm-level events."""
        emergent_record = {
            "elapsed_seconds": time.time() - self.start_time,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "description": description,
            "swarm_metrics": swarm_metrics
        }
        
        with open(self.emergent_file, 'a') as f:
            f.write(json.dumps(emergent_record) + '\n')
    
    def log_consciousness(self, entity_id: str, event_type: str, description: str,
                         metrics: Optional[Dict] = None):
        """Log consciousness/sentience related events."""
        consciousness_record = {
            "elapsed_seconds": time.time() - self.start_time,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "entity_id": entity_id,
            "event_type": event_type,
            "description": description,
            "metrics": metrics or {}
        }
        
        with open(self.consciousness_file, 'a') as f:
            f.write(json.dumps(consciousness_record) + '\n')
    
    def log_cognition(self, entity_id: str, metric_name: str, value: float,
                     threshold: Optional[float] = None, description: str = ""):
        """Log cognitive processing events."""
        cognition_record = {
            "elapsed_seconds": time.time() - self.start_time,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "entity_id": entity_id,
            "metric_name": metric_name,
            "value": self.validate_and_sanitize(value, f"{metric_name}_value"),
            "threshold": self.validate_and_sanitize(threshold, f"{metric_name}_threshold") if threshold else None,
            "description": description
        }
        
        with open(self.cognition_file, 'a') as f:
            f.write(json.dumps(cognition_record) + '\n')
    
    def log_quantum_event(self, entity_id: str, event_type: str, description: str,
                         quantum_state: Optional[Dict] = None):
        """Log quantum mechanical events."""
        quantum_record = {
            "elapsed_seconds": time.time() - self.start_time,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "entity_id": entity_id,
            "event_type": event_type,
            "description": description,
            "quantum_state": quantum_state or {}
        }
        
        with open(self.quantum_file, 'a') as f:
            f.write(json.dumps(quantum_record) + '\n')
    
    def log_swarm_metrics(self, swarm_data: Dict):
        """Log quantitative swarm metrics with buffering."""
        elapsed = time.time() - self.start_time
        
        validated_data = [
            self.validate_and_sanitize(elapsed, "elapsed"),
            datetime.now(timezone.utc).isoformat(),
            self.validate_and_sanitize(swarm_data.get('size', 0), "swarm_size", int),
            self.validate_and_sanitize(swarm_data.get('avg_coherence', 0), "avg_coherence"),
            self.validate_and_sanitize(swarm_data.get('avg_pleasure', 0), "avg_pleasure"),
            self.validate_and_sanitize(swarm_data.get('avg_fear', 0), "avg_fear"),
            self.validate_and_sanitize(swarm_data.get('avg_love', 0), "avg_love"),
            self.validate_and_sanitize(swarm_data.get('coherence_std', 0), "coherence_std"),
            self.validate_and_sanitize(swarm_data.get('entropy_avg', 0), "entropy_avg"),
            self.validate_and_sanitize(swarm_data.get('quantum_connections', 0), "quantum_connections", int),
            self.validate_and_sanitize(swarm_data.get('temporal_cycles', 0), "temporal_cycles", int),
            self.validate_and_sanitize(swarm_data.get('communications_per_cycle', 0), "communications_per_cycle")
        ]
        
        self.swarm_buffer.append(validated_data)
        self._maybe_flush()
    
    def log_entity_metrics(self, entity_data: Dict):
        """Log individual entity metrics with buffering."""
        elapsed = time.time() - self.start_time
        
        validated_data = [
            self.validate_and_sanitize(elapsed, "elapsed"),
            datetime.now(timezone.utc).isoformat(),
            entity_data.get('entity_id', 'unknown'),
            self.validate_and_sanitize(entity_data.get('pleasure', 0), "pleasure"),
            self.validate_and_sanitize(entity_data.get('fear', 0), "fear"),
            self.validate_and_sanitize(entity_data.get('love', 0), "love"),
            self.validate_and_sanitize(entity_data.get('coherence', 0), "coherence"),
            self.validate_and_sanitize(entity_data.get('entropy', 0), "entropy"),
            self.validate_and_sanitize(entity_data.get('quantum_mood', 0), "quantum_mood"),
            self.validate_and_sanitize(entity_data.get('temporal_phase', 0), "temporal_phase"),
            self.validate_and_sanitize(entity_data.get('morphogenetic_sig', 0), "morphogenetic_sig"),
            self.validate_and_sanitize(entity_data.get('neural_plasmic_coherence', 0), "neural_plasmic_coherence"),
            entity_data.get('goal', 'none'),
            self.validate_and_sanitize(entity_data.get('age', 0), "age")
        ]
        
        self.entity_buffer.append(validated_data)
        self._maybe_flush()
    
    def _maybe_flush(self):
        """Flush buffers if they're full or it's been a while."""
        current_time = time.time()
        
        if (len(self.swarm_buffer) >= self.buffer_size or 
            len(self.entity_buffer) >= self.buffer_size or
            current_time - self.last_flush > 5.0):
            self._flush_buffers()
    
    def _flush_buffers(self):
        """Write buffered data to disk."""
        if self.swarm_buffer:
            with gzip.open(self.swarm_csv, 'at', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.swarm_buffer)
            self.swarm_buffer.clear()
        
        if self.entity_buffer:
            with gzip.open(self.entity_csv, 'at', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.entity_buffer)
            self.entity_buffer.clear()
        
        self.last_flush = time.time()

# ============================================================
# 🎵 FIXED AUDIO PROCESSOR - With Validation
# ============================================================

class AdvancedAudioProcessor:
    """God-tier audio processing without external dependencies."""
    
    def __init__(self, logger: Optional[ScientificLogger] = None):
        self.spectral_history = []
        self.emotional_resonance_cache = {}
        self.logger = logger
        
    def load_wav_file(self, file_path: str) -> Optional[Dict[str, float]]:
        """Pure Python WAV file analysis with error handling."""
        # Validate file exists
        if not os.path.exists(file_path):
            error_msg = f"Audio file not found: {file_path}"
            print(f"[AudioProcessor] ERROR: {error_msg}")
            if self.logger:
                self.logger.log_anomaly(
                    "audio_file_not_found",
                    "audio_processor",
                    error_msg,
                    "MEDIUM"
                )
            return self._default_audio_features()
        
        try:
            with wave.open(file_path, 'rb') as wav_file:
                # Get audio parameters
                n_channels = wav_file.getnchannels()
                sampwidth = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Validate parameters
                if n_frames == 0:
                    print(f"[AudioProcessor] WARNING: Empty audio file: {file_path}")
                    return self._default_audio_features()
                
                if sampwidth not in [1, 2]:
                    print(f"[AudioProcessor] WARNING: Unsupported sample width: {sampwidth}")
                    return self._default_audio_features()
                
                # Read frames
                frames = wav_file.readframes(n_frames)
                
                # Convert to numpy array
                if sampwidth == 1:
                    # 8-bit unsigned
                    dtype = np.uint8
                    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32) - 128
                    samples /= 128.0
                elif sampwidth == 2:
                    # 16-bit signed
                    dtype = np.int16
                    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
                    samples /= 32768.0
                else:
                    return self._default_audio_features()
                
                # Handle stereo by averaging channels
                if n_channels > 1:
                    samples = samples.reshape(-1, n_channels).mean(axis=1)
                
                return self._quantum_audio_analysis(samples, framerate)
                
        except wave.Error as e:
            print(f"[AudioProcessor] Invalid WAV format in {file_path}: {e}")
            return self._default_audio_features()
        except Exception as e:
            print(f"[AudioProcessor] Unexpected error processing {file_path}: {e}")
            return self._default_audio_features()

    def _quantum_audio_analysis(self, samples: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Advanced quantum-inspired audio analysis."""
        features = {}
        
        # Ensure samples are finite
        samples = samples[np.isfinite(samples)]
        if len(samples) == 0:
            return self._default_audio_features()
        
        # Normalize
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val
        
        # 1. Quantum Spectral Decomposition
        if len(samples) >= 1024:
            segment = samples[:1024]
            fft = np.fft.fft(segment)
            magnitudes = np.abs(fft[:512])
            frequencies = np.fft.fftfreq(1024, 1/sample_rate)[:512]
            
            # Remove DC component and normalize
            magnitudes[0] = 0
            if np.sum(magnitudes) > 0:
                magnitudes /= np.sum(magnitudes)
            
            # Advanced spectral features
            features['spectral_centroid'] = float(np.sum(frequencies * magnitudes))
            features['spectral_spread'] = float(np.sqrt(np.sum(magnitudes * (frequencies - features['spectral_centroid'])**2)))
            features['spectral_entropy'] = float(-np.sum(magnitudes * np.log2(magnitudes + 1e-10)))
            
            # Quantum coherence metrics
            features['quantum_coherence'] = float(np.max(magnitudes) / (np.mean(magnitudes) + 1e-10))
        else:
            features.update(self._default_spectral_features())
        
        # 2. Temporal Quantum Features
        features['rms_energy'] = float(np.sqrt(np.mean(samples**2)))
        features['zero_crossing_rate'] = float(np.mean(np.diff(np.sign(samples)) != 0))
        features['temporal_entropy'] = float(-np.sum(samples**2 * np.log2(samples**2 + 1e-10)))
        
        # 3. Emotional Quantum Mapping
        features.update(self._quantum_emotional_mapping(features))
        
        # 4. Chronosynclastic Features
        features['temporal_coherence'] = self._compute_temporal_coherence(samples)
        features['quantum_fluctuation'] = random.uniform(0, 1) * features['spectral_entropy']
        
        return features

    def _quantum_emotional_mapping(self, features: Dict[str, float]) -> Dict[str, float]:
        """Map audio features to quantum emotional states."""
        emotional = {}
        
        # Quantum pleasure field (based on spectral richness)
        emotional['quantum_pleasure'] = np.tanh(features.get('spectral_centroid', 0) / 1000)
        
        # Emotional coherence (based on spectral concentration)
        emotional['emotional_coherence'] = 1.0 - min(1.0, features.get('spectral_spread', 0) / 5000)
        
        # Fear resonance (based on entropy and fluctuation)
        emotional['fear_resonance'] = min(1.0, 
            features.get('spectral_entropy', 0) * 0.3 + 
            features.get('quantum_fluctuation', 0) * 0.7
        )
        
        # Love frequency (harmonic relationships)
        emotional['love_frequency'] = max(0.0, 1.0 - features.get('zero_crossing_rate', 0) * 2)
        
        # Quantum binding energy
        emotional['quantum_binding'] = (emotional['quantum_pleasure'] + emotional['emotional_coherence']) / 2
        
        return emotional

    def _compute_temporal_coherence(self, samples: np.ndarray) -> float:
        """Compute temporal coherence using autocorrelation."""
        if len(samples) < 2:
            return 0.5
        
        # Simple autocorrelation at lag 1
        lag = 1
        if len(samples) > lag:
            correlation = np.corrcoef(samples[:-lag], samples[lag:])[0,1]
            return float((correlation + 1) / 2)  # Normalize to [0,1]
        return 0.5

    def _default_audio_features(self) -> Dict[str, float]:
        """Return default features when audio processing fails."""
        return {
            'spectral_centroid': 500.0, 'spectral_spread': 1000.0, 'spectral_entropy': 0.5,
            'quantum_coherence': 0.5, 'rms_energy': 0.1, 'zero_crossing_rate': 0.1,
            'temporal_entropy': 0.5, 'quantum_pleasure': 0.5, 'emotional_coherence': 0.5,
            'fear_resonance': 0.3, 'love_frequency': 0.7, 'quantum_binding': 0.5,
            'temporal_coherence': 0.5, 'quantum_fluctuation': 0.2
        }

    def _default_spectral_features(self) -> Dict[str, float]:
        """Default spectral features for short samples."""
        return {
            'spectral_centroid': 1000.0,
            'spectral_spread': 2000.0, 
            'spectral_entropy': 0.8,
            'quantum_coherence': 0.3
        }

# ============================================================
# 🧮 QUANTUM MATH CORE - FIXED VERSION
# ============================================================

class QuantumMath:
    """Advanced quantum mathematics for emotional computation."""
    
    @staticmethod
    def emotional_fourier_transform(emotions: List[float]) -> Dict[str, float]:
        """Fourier analysis of emotional states."""
        if len(emotions) < 2:
            return {'amplitude': 0.5, 'frequency': 0.5, 'phase': 0.0}
        
        # Simple DFT implementation
        n = len(emotions)
        frequencies = []
        for k in range(n):
            real = sum(emotions[j] * math.cos(2 * math.pi * k * j / n) for j in range(n))
            imag = sum(emotions[j] * math.sin(2 * math.pi * k * j / n) for j in range(n))
            magnitude = math.sqrt(real**2 + imag**2)
            frequencies.append(magnitude)
        
        dominant_freq = frequencies.index(max(frequencies)) if frequencies else 0
        return {
            'amplitude': max(frequencies) / n if n > 0 else 0,
            'frequency': dominant_freq / max(n, 1),
            'phase': math.atan2(imag, real) if n > 0 else 0
        }

    @staticmethod
    def quantum_entanglement(state1: complex, state2: complex) -> complex:
        """Compute quantum entanglement between two states."""
        # Bell state measurement
        entanglement = (state1 + state2) / math.sqrt(2)
        return entanglement * cmath.exp(1j * random.uniform(0, 2 * math.pi))

    @staticmethod
    def neural_plasmic_projection(thoughts: List[float]) -> float:
        """Project thoughts through neural-plasmic interface."""
        if not thoughts:
            return 0.5
        # Nonlinear projection through sigmoid space
        weighted_sum = sum(math.tanh(t) for t in thoughts) / len(thoughts)
        return (math.tanh(weighted_sum * 2) + 1) / 2

    @staticmethod
    def morphogenetic_field_resonance(entities: List['UnifiedEntity']) -> float:
        """Compute resonance across morphogenetic field."""
        if not entities:
            return 0.5
        
        coherences = [e.coherence for e in entities]
        pleasures = [max(0, e.pleasure) for e in entities]
        
        # Field coherence calculation
        field_coherence = np.std(coherences) if coherences else 0.5
        pleasure_coherence = np.mean(pleasures) if pleasures else 0.5
        
        return float(1.0 - field_coherence + pleasure_coherence * 0.5)

# ============================================================
# 🌌 UNIFIED ENTITY - FIXED VERSION
# ============================================================

class UnifiedEntity:
    """Fixed entity with memory caps and coherence homeostasis."""
    
    def __init__(self, x: float, y: float, entity_id: Optional[str] = None, 
                 logger: Optional[ScientificLogger] = None):
        # Physical properties
        self.x, self.y = x, y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.ax = self.ay = 0
        self.m = 1.0
        
        # Samsara emotional state with homeostasis
        self.pleasure = random.uniform(-0.2, 0.2)
        self.fear = random.uniform(0.0, 0.4)
        self.love = random.uniform(0.1, 0.6)
        self.entropy = random.uniform(0.3, 0.7)
        self.coherence = random.uniform(0.4, 0.8)
        self.energy = 1.0
        self.age = 0
        
        # God-tier properties with caps
        self.quantum_state = QuantumGoblin(entity_id, logger)
        self.mind = EnhancedMindMush()
        self.morals = MoralKompass()
        self.entity_id = entity_id or uuid.uuid4().hex[:8]
        self.last_action: Optional[str] = None
        
        # Communication with caps
        self.communication_buffer: List[str] = []
        self.MAX_COMM_BUFFER = 10
        
        # Trust network for communication (fixed circular logic)
        self.trust_network: Dict[str, float] = {}  # entity_id: trust_level
        
        # Scientific logging
        self.logger = logger
        self.last_coherence = self.coherence
        
        # Capped event storage
        self.significant_events: List[Dict] = []
        self.MAX_SIGNIFICANT_EVENTS = 100
        
        # Cognition thresholds
        self.cognition_thresholds = {
            'quantum_insight': 0.7,
            'inspiration': 0.8,
            'coherence_spike': 0.3,
            'entropy_drop': 0.2
        }
        
        # Audio processing
        self.audio_processor = AdvancedAudioProcessor(logger)
        self.current_audio_profile: Optional[Dict[str, float]] = None
        
        # Neural-Plasmic Interface with fixed random initialization
        self.neural_plasmic_nodes = [random.uniform(0.4, 0.6) for _ in range(8)]  # Start near 0.5
        
        # Chronosynclastic Infundibulum (visual phase)
        self.visual_phase = random.uniform(0, 2 * math.pi)  # Renamed from temporal_phase
        self.time_crystals = 0
        
        # Morphogenetic Field
        self.morphogenetic_signature = random.uniform(0, 1)
        
        # Holographic Encoding with cap
        self.holographic_memory: List[Dict] = []
        self.MAX_HOLOGRAPHIC_MEMORY = 20
        
        # Psychohistorical State
        self.psychohistorical_weight = random.uniform(0, 1)
        
        # Neural Dust Network
        self.neural_dust_particles = [random.uniform(0, 1) for _ in range(16)]
        
        # Quantum Gravity Binding
        self.quantum_gravity_factor = random.uniform(0.8, 1.2)
        
        # Multiverse Branching with cap
        self.alternate_states: List[Dict] = []
        self.MAX_ALTERNATE_STATES = 5

    def process_audio_file(self, file_path: str) -> None:
        """Process audio with proper error handling."""
        audio_features = self.audio_processor.load_wav_file(file_path)
        if audio_features:
            self.current_audio_profile = audio_features
            
            # Map advanced audio features to quantum emotional states
            audio_pleasure = audio_features.get('quantum_pleasure', 0)
            audio_coherence = audio_features.get('emotional_coherence', 0)
            audio_fear = audio_features.get('fear_resonance', 0)
            
            # Emotional integration with quantum smoothing
            self.pleasure = self._quantum_smooth(self.pleasure, audio_pleasure - 0.5)
            self.coherence = self._quantum_smooth(self.coherence, audio_coherence)
            self.fear = self._quantum_smooth(self.fear, audio_fear)

    def _quantum_smooth(self, current: float, change: float) -> float:
        """Quantum-aware smoothing function."""
        return max(-1.0, min(1.0, current + change * 0.1))

    def communicate(self, message: str, target_entity: 'UnifiedEntity') -> None:
        """Fixed communication with trust-based logic."""
        # Check trust first (fixed circular logic)
        if not self._should_communicate_with(target_entity):
            return
        
        moral_context = {
            "harm": self.fear,
            "charm": max(0, self.pleasure),
            "farm": self.coherence,
            "disarm": 1.0 - self.fear
        }
        
        score, moral_msg, is_violation = self.morals.evaluate(f"communicate:{message}", moral_context)
        
        if not is_violation:  # Morally acceptable communication
            # Neural dust enhanced transmission
            transmission_strength = np.mean(self.neural_dust_particles) if self.neural_dust_particles else 0.5
            
            if transmission_strength > 0.3:
                target_entity.receive_communication(message, self.entity_id)
                
                # Add to buffer with cap
                self.communication_buffer.append(f"Sent: {message} to {target_entity.entity_id}")
                if len(self.communication_buffer) > self.MAX_COMM_BUFFER:
                    self.communication_buffer.pop(0)
                
                # Quantum entanglement effect (fixed)
                self._entangle_with_entity(target_entity)
                
                # Log significant communication events
                if self.logger and ("quantum" in message.lower() or "entangle" in message.lower()):
                    self.logger.log_cognition(
                        self.entity_id,
                        "quantum_communication",
                        score,
                        threshold=0.5,
                        description=f"Sent quantum message: {message}"
                    )
                
                # Update trust (increased trust with successful communication)
                self._update_trust(target_entity.entity_id, 0.05)

    def _should_communicate_with(self, target: 'UnifiedEntity') -> bool:
        """Check if we should communicate with target entity."""
        # Base trust level
        trust = self.trust_network.get(target.entity_id, 0.5)
        
        # Distance check
        distance_squared = (target.x - self.x)**2 + (target.y - self.y)**2
        in_range = distance_squared < 22500  # 150^2
        
        # Not ourselves and in range with reasonable trust
        return target != self and in_range and trust > 0.3

    def _update_trust(self, entity_id: str, change: float):
        """Update trust level for an entity."""
        if entity_id not in self.trust_network:
            self.trust_network[entity_id] = 0.5
        self.trust_network[entity_id] = max(0.0, min(1.0, self.trust_network[entity_id] + change))

    def _entangle_with_entity(self, target: 'UnifiedEntity') -> None:
        """Create quantum entanglement with another entity."""
        # Average quantum states
        avg_mood = (self.quantum_state.mood + target.quantum_state.mood) / 2
        self.quantum_state.mood = avg_mood * 0.9 + self.quantum_state.mood * 0.1
        target.quantum_state.mood = avg_mood * 0.9 + target.quantum_state.mood * 0.1
        
        # Entangle coherence (bidirectional)
        avg_coherence = (self.coherence + target.coherence) / 2
        self.coherence = avg_coherence * 0.95 + self.coherence * 0.05
        target.coherence = avg_coherence * 0.95 + target.coherence * 0.05
        
        # Log entanglement event
        if self.logger:
            self.logger.log_quantum_event(
                self.entity_id,
                "quantum_entanglement",
                f"Entangled with entity {target.entity_id}",
                {
                    "avg_mood": avg_mood,
                    "avg_coherence": avg_coherence,
                    "mood_similarity": 1 - abs(self.quantum_state.mood - target.quantum_state.mood)
                }
            )

    def receive_communication(self, message: str, sender_id: str) -> None:
        """Receive communication with permanent effects."""
        # Update trust (positive for receiving communication)
        self._update_trust(sender_id, 0.02)
        
        stimuli = [f"msg_from_{sender_id}:{message}"]
        self.mind.perceive(stimuli)
        
        # Add to buffer with cap
        self.communication_buffer.append(f"Received: {message} from {sender_id}")
        if len(self.communication_buffer) > self.MAX_COMM_BUFFER:
            self.communication_buffer.pop(0)

    def step(self, dt: float, width: int, height: int, entities: List['UnifiedEntity']) -> Dict[str, Any]:
        """Fixed entity state advancement with coherence homeostasis."""
        # Store previous state for change detection
        prev_coherence = self.coherence
        prev_entropy = self.entropy
        prev_pleasure = self.pleasure
        
        # Update visual phase
        self.visual_phase = (self.visual_phase + dt * 0.1) % (2 * math.pi)
        
        # Physics with quantum gravity effects
        self.vx += self.ax * dt * self.quantum_gravity_factor
        self.vy += self.ay * dt * self.quantum_gravity_factor
        self.vx *= 0.995
        self.vy *= 0.995
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Quantum boundary conditions
        if self.x < 0 or self.x > width: 
            self.vx *= -0.8 * (0.9 + 0.1 * math.sin(self.visual_phase))
            self.x = max(0, min(width, self.x))
        if self.y < 0 or self.y > height:
            self.vy *= -0.8 * (0.9 + 0.1 * math.cos(self.visual_phase))
            self.y = max(0, min(height, self.y))
            
        self.ax = self.ay = 0
        
        # Emotional dynamics with homeostasis - FIXED COHERENCE DEATH SPIRAL
        pain = 0.05 * (abs(self.vx) + abs(self.vy))
        
        # Neural-plasmic emotional computation
        plasmic_influence = QuantumMath.neural_plasmic_projection(self.neural_plasmic_nodes)
        
        # Pleasure: positive from coherence, negative from entropy and pain
        pleasure_change = 0.04 * (self.coherence - self.entropy - pain) * plasmic_influence
        self.pleasure += pleasure_change
        
        # Fear: homeostatic regulation (FIXED)
        fear_pressure = max(0, self.entropy - self.coherence)
        fear_damping = (self.love + max(0, self.pleasure)) / 2  # Positive emotions reduce fear
        self.fear += 0.02 * fear_pressure * (1 - 0.5 * fear_damping)
        
        # Love: grows with coherence
        self.love += 0.03 * (self.coherence - self.entropy) * plasmic_influence
        
        # Coherence homeostasis (FIXED)
        # Natural coherence regeneration when pleasure is high
        if self.pleasure > 0.3 and self.coherence < 0.9:
            coherence_regeneration = 0.001 * self.pleasure
            self.coherence = min(0.9, self.coherence + coherence_regeneration)
        
        # Entropy: small random fluctuations
        self.entropy += 0.002 * (random.random() - 0.5)
        
        # Emotional bounds with quantum fluctuations
        self.pleasure = self._quantum_bound(self.pleasure)
        self.fear = self._quantum_bound(self.fear, lower=0)
        self.love = self._quantum_bound(self.love, lower=0)
        self.entropy = self._quantum_bound(self.entropy)
        self.coherence = self._quantum_bound(self.coherence)
        
        self.energy = self._quantum_bound(1 - 0.5 * pain + 0.5 * self.pleasure, lower=0, upper=2)
        self.age += dt
        
        # Quantum rituals with visual phase influence
        quantum_prob = 0.2 + 0.1 * math.sin(self.visual_phase)
        if random.random() < quantum_prob:
            self.quantum_state.flip_a_coin()
        if random.random() < quantum_prob * 0.5:
            self.quantum_state.mood_swing()
        if random.random() < quantum_prob * 0.25:
            self.quantum_state.eat_qubit()
        self.quantum_state.leak_entropy()
        
        # Cognitive processing
        thought_result = self.mind.update()
        goal = self.mind.choose_goal()
        self.last_action = goal
        
        # Multiverse state branching (with cap)
        if random.random() < 0.01:
            self._create_alternate_state()
        
        # Communication with trust-based targeting
        nearby_entities = self._find_trusted_entities(entities)
        if nearby_entities and random.random() < 0.3:
            self._communicate_with_random(nearby_entities)
        
        # Detect and log significant state changes
        self._detect_and_log_state_changes(prev_coherence, prev_entropy, prev_pleasure, thought_result)
        
        # Log entity metrics for scientific analysis
        if self.logger:
            self.logger.log_entity_metrics({
                'entity_id': self.entity_id,
                'pleasure': self.pleasure,
                'fear': self.fear,
                'love': self.love,
                'coherence': self.coherence,
                'entropy': self.entropy,
                'quantum_mood': self.quantum_state.mood,
                'temporal_phase': self.visual_phase,  # Using visual_phase
                'morphogenetic_sig': self.morphogenetic_signature,
                'neural_plasmic_coherence': np.mean(self.neural_plasmic_nodes) if self.neural_plasmic_nodes else 0,
                'goal': goal,
                'age': self.age
            })
        
        return self._compile_state_report(thought_result, goal)

    def _detect_and_log_state_changes(self, prev_coherence: float, prev_entropy: float, 
                                     prev_pleasure: float, thought_result: Dict[str, float]):
        """Detect significant state changes and log them scientifically."""
        if not self.logger:
            return
        
        # Detect coherence spike/drop
        coherence_change = abs(self.coherence - prev_coherence)
        if coherence_change > self.cognition_thresholds['coherence_spike']:
            direction = "spike" if self.coherence > prev_coherence else "drop"
            self.logger.log_cognition(
                self.entity_id,
                f"coherence_{direction}",
                self.coherence,
                threshold=prev_coherence,
                description=f"Coherence {direction}: {prev_coherence:.3f} -> {self.coherence:.3f}"
            )
        
        # Detect significant entropy change
        entropy_change = abs(self.entropy - prev_entropy)
        if entropy_change > self.cognition_thresholds['entropy_drop']:
            if self.entropy < prev_entropy:  # Significant entropy reduction
                self.logger.log_cognition(
                    self.entity_id,
                    "entropy_reduction",
                    self.entropy,
                    threshold=prev_entropy,
                    description=f"Order emergence: {prev_entropy:.3f} -> {self.entropy:.3f}"
                )
        
        # Log cognitive metrics that exceed thresholds
        if thought_result.get('quantum_insight', 0) > self.cognition_thresholds['quantum_insight']:
            self.logger.log_consciousness(
                self.entity_id,
                "quantum_insight",
                f"High quantum insight detected: {thought_result['quantum_insight']:.3f}",
                thought_result
            )
        
        if thought_result.get('inspiration', 0) > self.cognition_thresholds['inspiration']:
            self.logger.log_consciousness(
                self.entity_id,
                "inspiration_peak",
                "Inspiration active at peak level",
                thought_result
            )

    def _quantum_bound(self, value: float, lower: float = -1.0, upper: float = 1.0) -> float:
        """Quantum-aware value bounding with small fluctuations."""
        bounded = max(lower, min(upper, value))
        # Add quantum fluctuation
        fluctuation = random.gauss(0, 0.001)
        return max(lower, min(upper, bounded + fluctuation))

    def _find_trusted_entities(self, entities: List['UnifiedEntity']) -> List['UnifiedEntity']:
        """Find entities we trust for communication."""
        trusted_entities = []
        for e in entities:
            if e != self and self._should_communicate_with(e):
                trusted_entities.append(e)
        return trusted_entities

    def _communicate_with_random(self, trusted_entities: List['UnifiedEntity']) -> None:
        """Communicate with a random trusted entity."""
        target = random.choice(trusted_entities)
        messages = [
            f"coherence:{self.coherence:.3f}",
            f"pleasure:{self.pleasure:.3f}", 
            f"quantum_mood:{self.quantum_state.mood:.3f}",
            f"visual_phase:{self.visual_phase:.3f}",
            f"morpho_sig:{self.morphogenetic_signature:.3f}",
            "quantum_entanglement_request",
            "neural_plasmic_sync"
        ]
        self.communicate(random.choice(messages), target)

    def _create_alternate_state(self) -> None:
        """Create alternate multiverse state with cap."""
        alternate = {
            'pleasure': self.pleasure + random.gauss(0, 0.1),
            'coherence': self.coherence + random.gauss(0, 0.1),
            'quantum_mood': self.quantum_state.mood + random.gauss(0, 0.1),
            'timestamp': time.time()
        }
        self.alternate_states.append(alternate)
        
        # Apply cap
        if len(self.alternate_states) > self.MAX_ALTERNATE_STATES:
            self.alternate_states.pop(0)
        
        # Log multiverse branching event
        if self.logger:
            self.logger.log_quantum_event(
                self.entity_id,
                "multiverse_branching",
                "Created alternate multiverse state",
                {
                    "branch_count": len(self.alternate_states),
                    "pleasure_delta": alternate['pleasure'] - self.pleasure,
                    "coherence_delta": alternate['coherence'] - self.coherence
                }
            )

    def _compile_state_report(self, thought_result: Dict[str, float], goal: str) -> Dict[str, Any]:
        """Compile comprehensive state report."""
        return {
            "goal": goal,
            "thought_metrics": thought_result,
            "quantum_state": self.quantum_state.status(),
            "emotional_state": {
                "pleasure": self.pleasure,
                "fear": self.fear, 
                "love": self.love,
                "coherence": self.coherence,
                "entropy": self.entropy
            },
            "advanced_metrics": {
                "visual_phase": self.visual_phase,
                "morphogenetic_sig": self.morphogenetic_signature,
                "neural_plasmic_coherence": np.mean(self.neural_plasmic_nodes) if self.neural_plasmic_nodes else 0,
                "quantum_gravity": self.quantum_gravity_factor,
                "trust_network_size": len(self.trust_network)
            }
        }

    def color(self) -> Tuple[int, int, int]:
        """Compute color with visual phase influence."""
        # Base colors with emotional influence
        base_r = 128 + 127 * self.pleasure
        base_g = 128 + 127 * self.love
        base_b = 128 + 127 * (1 - self.fear)
        
        # Visual phase modulation
        time_r = 20 * math.sin(self.visual_phase)
        time_g = 20 * math.sin(self.visual_phase + 2 * math.pi / 3)
        time_b = 20 * math.sin(self.visual_phase + 4 * math.pi / 3)
        
        # Neural-plasmic influence
        plasmic_factor = np.mean(self.neural_plasmic_nodes) if self.neural_plasmic_nodes else 0.5
        plasmic_r = 10 * (plasmic_factor - 0.5)
        plasmic_g = 10 * (plasmic_factor - 0.5)
        
        r = int(base_r + time_r + plasmic_r)
        g = int(base_g + time_g + plasmic_g) 
        b = int(base_b + time_b)
        
        return (
            max(0, min(255, r)),
            max(0, min(255, g)), 
            max(0, min(255, b))
        )

# ============================================================
# 🧠 ENHANCED MIND MUSH - FIXED VERSION
# ============================================================

class EnhancedMindMush:
    """Advanced cognitive processing with memory caps."""
    
    def __init__(self, working_capacity: int = 12, dream_period: int = 8):
        self.working_capacity = working_capacity
        self.dream_period = dream_period
        self.working_memory: List[Dict[str, Any]] = []
        self.long_term_memory: Dict[str, Dict[str, Any]] = {}
        self.MAX_LONG_TERM_MEMORY = 1000  # Added cap
        self.emotional_clock = 0.0
        self.embarrassment = 0.0
        self.goal_pool: List[str] = [
            "quantum_seek", "coherence_pursuit", "temporal_reflection", "neural_dream",
            "morphogenetic_align", "plasmic_resonance", "quantum_entangle", "multiverse_explore"
        ]
        self.goal_weights: Dict[str, float] = {goal: 1.0 for goal in self.goal_pool}
        self.introspection_log: List[str] = []
        self.MAX_INTROSPECTION_LOG = 50  # Added cap
        self.cycle_count = 0
        self.last_stimuli: Optional[List[str]] = None
        self.inspiration_active = False
        self.quantum_insight_level = 0.0

    def perceive(self, stimuli: List[str], context: Optional[Dict[str, float]] = None) -> None:
        """Quantum-enhanced perception with memory caps."""
        self.cycle_count += 1

        arousal, valence = self._update_emotions()
        
        # Quantum insight accumulation
        quantum_stimuli_present = False
        for stim in stimuli:
            if "quantum" in stim.lower() or "entangle" in stim.lower():
                quantum_stimuli_present = True
                break
        
        if quantum_stimuli_present:
            self.quantum_insight_level = min(1.0, self.quantum_insight_level + 0.1)

        if self.last_stimuli == stimuli:
            self.embarrassment = min(1.0, self.embarrassment + 0.1)
        else:
            self.embarrassment = max(0.0, self.embarrassment - 0.05)
        self.last_stimuli = list(stimuli)

        # Quantum dream states
        if self.cycle_count % self.dream_period == 0 and not stimuli:
            quantum_dream = f"Quantum_Dream_{uuid.uuid4().hex[:6]}"
            stimuli = [quantum_dream]

        # Quantum inspiration
        if random.random() < 0.05 or self.quantum_insight_level > 0.7:
            self.inspiration_active = True
            self.quantum_insight_level *= 0.8  # Consume insight
        else:
            self.inspiration_active = False

        for stim in stimuli:
            # Quantum emotional processing
            if "quantum" in stim.lower():
                valence += 0.2 * self.quantum_insight_level
            elif "fear" in stim.lower() or "danger" in stim.lower():
                valence -= 0.15
            else:
                valence += random.uniform(-0.05, 0.05) * (1 + self.quantum_insight_level)

            base_activation = arousal * (2.0 if self.inspiration_active else 1.0)
            activation = max(0.0, base_activation * (1.0 - self.embarrassment))
            
            mem = {
                "content": stim,
                "activation": activation,
                "timestamp": time.time(),
                "valence": valence,
                "quantum_tag": self.quantum_insight_level > 0.5
            }
            
            # Quantum memory encoding
            if random.random() < 0.1 + self.quantum_insight_level * 0.2:
                mem["content"] = f"Q:{mem['content']}"
                
            self.working_memory.append(mem)
            
            # Add to long-term memory with cap
            self.long_term_memory[stim] = mem
            if len(self.long_term_memory) > self.MAX_LONG_TERM_MEMORY:
                # Remove oldest entry
                oldest_key = min(self.long_term_memory.keys(), 
                               key=lambda k: self.long_term_memory[k]['timestamp'])
                del self.long_term_memory[oldest_key]
            
        if len(self.working_memory) > self.working_capacity:
            self.working_memory = self.working_memory[-self.working_capacity:]

    def _update_emotions(self):
        """Quantum emotional oscillation."""
        self.emotional_clock += 0.07 * (1 + self.quantum_insight_level * 0.5)
        arousal = 0.5 + 0.5 * math.sin(self.emotional_clock)
        valence = 0.5 + 0.5 * math.cos(self.emotional_clock * 1.3)
        return arousal, valence

    def update(self) -> Dict[str, float]:
        """Quantum cognitive update with memory decay."""
        metrics: Dict[str, float] = {}
        
        # Update working memory with quantum decay
        now = time.time()
        for mem in self.working_memory:
            age = now - mem["timestamp"]
            decay_rate = 0.1 + self.embarrassment * 0.2
            if mem.get("quantum_tag", False):
                decay_rate *= 0.7  # Quantum memories decay slower
            mem["activation"] = max(0.0, mem["activation"] * math.exp(-decay_rate * age))

        # Quantum cognitive metrics
        activations = [m["activation"] for m in self.working_memory]
        if activations:
            metrics = self._compute_quantum_metrics(activations)
        else:
            metrics.update(self._default_metrics())
            
        # Quantum insight decay
        self.quantum_insight_level = max(0.0, self.quantum_insight_level - 0.01)
        
        return metrics

    def _compute_quantum_metrics(self, activations: List[float]) -> Dict[str, float]:
        """Compute advanced quantum cognitive metrics."""
        metrics = {}
        
        mean_act = sum(activations) / len(activations)
        variance = sum((a - mean_act) ** 2 for a in activations) / len(activations)
        
        metrics['drama_index'] = variance
        metrics['quantum_coherence'] = min(1.0, mean_act / (variance + 1e-10))
        
        low_count = sum(1 for a in activations if a < 0.05)
        metrics['boredom'] = low_count / len(activations)
        
        metrics['quantum_insight'] = self.quantum_insight_level
        metrics['inspiration'] = 1.0 if self.inspiration_active else 0.0
        metrics['ego_inflation'] = 1.0 if random.random() < 0.03 else 0.0
        metrics['introspection'] = 1.0 if self.cycle_count % 5 == 0 else 0.0
        
        return metrics

    def _default_metrics(self) -> Dict[str, float]:
        return {
            'drama_index': 0.0, 'quantum_coherence': 0.5, 'boredom': 1.0,
            'quantum_insight': 0.0, 'inspiration': 0.0, 'ego_inflation': 0.0,
            'introspection': 0.0
        }

    def choose_goal(self) -> str:
        """Quantum-informed goal selection."""
        if not self.working_memory:
            mood = 0.5
        else:
            avg_valence = sum(m["valence"] for m in self.working_memory) / len(self.working_memory)
            avg_activation = sum(m["activation"] for m in self.working_memory) / len(self.working_memory)
            mood = 0.5 + (avg_valence - 0.5) * 0.3 + (avg_activation - 0.5) * 0.2
            mood = max(0.0, min(1.0, mood))
            
        # Quantum goals get preference with high insight
        quantum_bonus = self.quantum_insight_level * 0.5
        
        weights = []
        for i, g in enumerate(self.goal_pool):
            base = self.goal_weights[g]
            modifier = (1.0 - i / len(self.goal_pool)) * mood + (i / len(self.goal_pool)) * (1.0 - mood)
            if "quantum" in g.lower() or "plasmic" in g.lower():
                modifier += quantum_bonus
            weights.append(base * modifier)
            
        total = sum(weights)
        r = random.uniform(0, total) if total > 0 else 0
        acc = 0.0
        for g, w in zip(self.goal_pool, weights):
            acc += w
            if r <= acc:
                return g
        return random.choice(self.goal_pool)

# ============================================================
# 🔮 QUANTUM GOBLIN - FIXED VERSION
# ============================================================

class QuantumGoblin:
    """Enhanced quantum node with logging and memory caps."""
    
    def __init__(self, node_id: Optional[str] = None, num_qubits: int = 4, 
                 logger: Optional[ScientificLogger] = None):
        self.id = node_id or str(uuid.uuid4())[:8]
        self.num_qubits = num_qubits
        self.state: List[complex] = [complex(1, 0)] + [complex(0, 0)] * (2 ** num_qubits - 1)
        self.coherence: float = 1.0
        self.mood: float = random.uniform(-1.0, 1.0)
        self.friends: List[QuantumGoblin] = []
        self.history: List[str] = []
        self.MAX_HISTORY = 50  # Enforce cap
        self.birth_time: float = time.time()
        self.entanglement_partners: List[str] = []
        self.significant_events: List[Dict] = []
        self.MAX_SIGNIFICANT_EVENTS = 100  # Enforce cap
        self.logger = logger

    def flip_a_coin(self) -> None:
        """Enhanced quantum operation."""
        idx = random.randint(0, len(self.state) - 1)
        # Quantum phase addition
        phase = random.uniform(0, 2 * math.pi)
        new_val = complex(random.random() * math.cos(phase), random.random() * math.sin(phase))
        self.state[idx] = new_val
        self._normalize_state()
        self._log("quantum_flip → new superposition")
        
        # Record significant quantum operation with cap
        self._add_significant_event({
            'timestamp': time.time(),
            'event': 'quantum_flip',
            'phase': phase,
            'index': idx
        })

    def mood_swing(self) -> None:
        """Enhanced mood swing with quantum phase."""
        prev_mood = self.mood
        phase = random.uniform(-math.pi * 2, math.pi * 2)
        self.state = [s * cmath.exp(1j * phase * random.random()) for s in self.state]
        self.mood = math.sin(phase) * 0.8 + self.mood * 0.2  # Momentum
        self.coherence *= 0.97
        self._log(f"quantum_mood_swing → phase={phase:.2f}")
        
        # Record significant mood swing with cap
        if abs(self.mood - prev_mood) > 0.5:
            self._add_significant_event({
                'timestamp': time.time(),
                'event': 'major_mood_swing',
                'phase': phase,
                'mood_change': self.mood - prev_mood
            })

    def measure(self) -> complex:
        """Quantum measurement with collapse."""
        probabilities = [abs(s) ** 2 for s in self.state]
        total = sum(probabilities)
        if total == 0:
            truth = complex(0, 0)
        else:
            r = random.uniform(0, total)
            for i, p in enumerate(probabilities):
                r -= p
                if r <= 0:
                    truth = self.state[i]
                    break
            else:
                truth = self.state[0]
                
        # Collapse to measured state
        self.state = [truth] + [complex(0, 0)] * (len(self.state) - 1)
        self.coherence *= 0.9
        self._log("quantum_measurement_collapse ⚡")
        
        # Record measurement event with cap
        self._add_significant_event({
            'timestamp': time.time(),
            'event': 'measurement_collapse',
            'coherence_before': self.coherence / 0.9,  # Reverse the multiplication
            'coherence_after': self.coherence
        })
        
        return truth

    def gossip(self, neighbor: "QuantumGoblin") -> None:
        """Enhanced quantum gossip with entanglement."""
        if not isinstance(neighbor, QuantumGoblin):
            return
            
        # Record pre-gossip states
        pre_coherence = self.coherence
        pre_mood = self.mood
        
        # Quantum state mixing
        mix_factor = 0.3
        for i in range(min(len(self.state), len(neighbor.state))):
            self.state[i] = self.state[i] * (1 - mix_factor) + neighbor.state[i] * mix_factor
            neighbor.state[i] = neighbor.state[i] * (1 - mix_factor) + self.state[i] * mix_factor
            
        self._normalize_state()
        neighbor._normalize_state()
        
        # Mood synchronization
        delta = (self.mood - neighbor.mood) / 3.0
        self.mood -= delta
        neighbor.mood += delta
        
        # Coherence sharing with quantum effects
        avg_coh = (self.coherence + neighbor.coherence) / 2.0
        quantum_boost = random.uniform(0.95, 1.05)
        self.coherence = neighbor.coherence = avg_coh * quantum_boost
        
        self._log(f"quantum_gossip with {neighbor.id}")
        
        # Record significant gossip events with cap
        if abs(self.mood - pre_mood) > 0.3:
            self._add_significant_event({
                'timestamp': time.time(),
                'event': 'quantum_gossip',
                'with': neighbor.id,
                'mood_change': self.mood - pre_mood,
                'coherence_change': self.coherence - pre_coherence
            })

    def eat_qubit(self) -> None:
        """Qubit consumption with energy gain."""
        if self.num_qubits > 2:  # Keep minimum 2 qubits
            prev_qubits = self.num_qubits
            self.num_qubits -= 1
            self.state = self.state[: 2 ** self.num_qubits]
            self._normalize_state()
            self.coherence *= 1.08  # Slightly more coherence gain
            self._log("ate a qubit 🍽️ +energy")
            
            # Record qubit consumption with cap
            self._add_significant_event({
                'timestamp': time.time(),
                'event': 'qubit_consumption',
                'qubits_before': prev_qubits,
                'qubits_after': self.num_qubits,
                'coherence_gain': 0.08
            })

    def leak_entropy(self) -> None:
        """Quantum entropy leakage with fluctuations."""
        prev_coherence = self.coherence
        base_leak = random.uniform(0.0, 0.04)
        # Quantum fluctuations
        quantum_effect = math.sin(time.time()) * 0.01
        total_leak = max(0.0, base_leak + quantum_effect)
        self.coherence = max(0.0, self.coherence - total_leak)
        self._log(f"quantum_entropy_leak {total_leak:.3f}")
        
        # Record significant entropy leaks with cap
        if total_leak > 0.03:
            self._add_significant_event({
                'timestamp': time.time(),
                'event': 'significant_entropy_leak',
                'leak_amount': total_leak,
                'coherence_before': prev_coherence,
                'coherence_after': self.coherence
            })

    def _normalize_state(self) -> None:
        norm = math.sqrt(sum(abs(x) ** 2 for x in self.state))
        if norm > 0:
            self.state = [x / norm for x in self.state]

    def _log(self, msg: str) -> None:
        entry = f"[{self.id}] {msg}"
        self.history.append(entry)
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)

    def _add_significant_event(self, event: Dict) -> None:
        """Add significant event with cap."""
        self.significant_events.append(event)
        if len(self.significant_events) > self.MAX_SIGNIFICANT_EVENTS:
            self.significant_events.pop(0)

    def entropy(self) -> float:
        """Quantum entropy calculation."""
        magnitudes = [abs(s) for s in self.state]
        if not magnitudes:
            return 0.0
        # Von Neumann-like entropy
        probabilities = [m ** 2 for m in magnitudes]
        total = sum(probabilities)
        if total == 0:
            return 0.0
        probabilities = [p / total for p in probabilities]
        entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)
        return entropy / math.log2(len(probabilities) + 1)  # Normalize

    def status(self) -> Dict[str, Any]:
        """Enhanced status with quantum metrics."""
        return {
            "id": self.id,
            "qubits": self.num_qubits,
            "coherence": round(self.coherence, 4),
            "mood": round(self.mood, 3),
            "entropy": round(self.entropy(), 3),
            "age": round(time.time() - self.birth_time, 2),
            "state_vector_size": len(self.state),
            "entanglement_count": len(self.entanglement_partners),
            "significant_events": len(self.significant_events),
            "history_size": len(self.history)
        }

# ============================================================
# 🧭 MORAL KOMPASS - FIXED VERSION
# ============================================================

class MoralKompass:
    """Enhanced moral compass with actual consequences."""
    
    def __init__(self):
        self.base_weights = {
            "Harm": 0.4,
            "Charm": 0.3,
            "Farm": 0.2,
            "Disarm": 0.1,
        }
        self.sin_buffer: List[str] = []
        self.MAX_SIN_BUFFER = 10  # Cap sins
        self.good_vibes: float = 0.0
        self.philosophers = ["Kant", "Nietzsche", "Confucius", "Aristotle", "Sartre", "Quantum_Ethicist"]
        self.caffeine_level = 0.5
        self.quantum_ethics_level = 0.0
        self.reputation = 0.5  # Starts neutral

    def _current_lunar_phase_factor(self) -> float:
        """Quantum-enhanced lunar phase calculation."""
        day = time.localtime().tm_mday
        # Quantum fluctuation in lunar phase
        quantum_fluct = random.gauss(0, 0.05)
        base_phase = 0.5 + 0.5 * math.sin(2 * math.pi * day / 29.53)
        return max(0.0, min(1.0, base_phase + quantum_fluct))

    def _apply_philosopher_bias(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Quantum philosopher bias application."""
        phi = random.choice(self.philosophers)
        
        if phi == "Quantum_Ethicist":
            # Quantum ethics: superposition of all philosophies
            for key in weights:
                weights[key] *= random.uniform(0.8, 1.2)
            self.quantum_ethics_level = min(1.0, self.quantum_ethics_level + 0.1)
        else:
            mod = random.uniform(0.85, 1.15)
            if phi == "Kant":
                weights["Harm"] *= mod
            elif phi == "Nietzsche":
                weights["Charm"] *= mod
            elif phi == "Confucius":
                weights["Farm"] *= mod
            elif phi == "Aristotle":
                weights["Disarm"] *= mod
            else:  # Sartre
                for k in weights:
                    weights[k] *= (1.0 - 0.05)
                    
        return weights

    def evaluate(self, action: str, context: Optional[Dict[str, float]] = None) -> Tuple[float, str, bool]:
        """Quantum moral evaluation with consequences."""
        context = context or {}
        weights = dict(self.base_weights)
        weights = self._apply_philosopher_bias(weights)

        self.caffeine_level = max(0.0, self.caffeine_level - 0.01)
        ethical_minimum = 0.5 + 0.3 * self.caffeine_level

        # Quantum moral computation
        score = 0.0
        for pole, weight in weights.items():
            value = context.get(pole.lower(), 0.5)
            # Quantum uncertainty principle in moral judgment
            quantum_uncertainty = random.gauss(0, 0.02) * self.quantum_ethics_level
            score += weight * (value + quantum_uncertainty)

        # Quantum good vibes
        score += 0.1 * self.good_vibes * (1 + self.quantum_ethics_level)
        self.good_vibes = max(0.0, self.good_vibes - 0.05)

        lunar = self._current_lunar_phase_factor()
        threshold = ethical_minimum * (0.8 + 0.4 * lunar)

        message = ""
        safe = score >= threshold
        is_violation = not safe
        
        if is_violation:
            # Record sin with cap
            self.sin_buffer.append(f"Quantum_{action}")
            if len(self.sin_buffer) > self.MAX_SIN_BUFFER:
                self.sin_buffer.pop(0)
            
            # Damage reputation for immoral actions
            self.reputation *= 0.95
            message += f"🔮 Quantum ethical violation detected! Reputation: {self.reputation:.3f}\n"
        else:
            if "quantum" in action.lower() or "entangle" in action.lower():
                self.good_vibes += 0.3
                self.quantum_ethics_level = min(1.0, self.quantum_ethics_level + 0.05)
                # Slight reputation boost for quantum ethics
                self.reputation = min(1.0, self.reputation + 0.02)

        # Quantum virtue inflation
        if random.random() < 0.1 + self.quantum_ethics_level * 0.1:
            score *= 1.1 + self.quantum_ethics_level * 0.1

        # Quantum forgiveness (harder to get)
        if self.sin_buffer and random.random() < 0.01 + self.quantum_ethics_level * 0.05:
            self.sin_buffer.clear()
            message += "🌌 Quantum forgiveness wave emitted!\n"
            # Partial reputation recovery
            self.reputation = min(1.0, self.reputation + 0.1)

        message += f"Score={score:.3f}, Quantum_Ethics={self.quantum_ethics_level:.2f}, Reputation={self.reputation:.3f} ¯\\_(ツ)_/¯"
        return score, message, is_violation

    def get_trust_modifier(self) -> float:
        """Return trust modifier based on moral reputation."""
        if len(self.sin_buffer) > 5:
            return 0.5  # 50% trust penalty for repeat offenders
        elif len(self.sin_buffer) > 2:
            return 0.8  # 20% trust penalty
        return 1.0  # Full trust for ethical entities

# ============================================================
# 🧠 ABRAXAS CORE - FIXED VERSION
# ============================================================

class AbraxasCore:
    """God-tier system coherence modulator with dynamic anomaly detection."""
    
    def __init__(self, logger: Optional[ScientificLogger] = None):
        self.state = {
            "λ": random.uniform(0.8, 1.0),  # Quantum coherence field
            "S": random.uniform(0.3, 0.7),   # Entropy field
            "C": random.uniform(0.4, 0.8),   # Coherence field
            "Ψ": random.uniform(0.5, 0.9),   # Consciousness field
            "Φ": random.uniform(0.6, 1.0)    # Morphogenetic field
        }
        self.healing = False
        self.quantum_fluctuations = []
        self.MAX_FLUCTUATIONS = 100  # Cap
        self.temporal_anchor = time.time()
        self.logger = logger
        
        # Dynamic anomaly detection
        self.coherence_history = []
        self.entropy_history = []
        self.consciousness_history = []
        self.HISTORY_SIZE = 100  # For dynamic thresholds
        
        # Initial thresholds (will be updated dynamically)
        self.anomaly_thresholds = {
            'coherence_collapse': 0.2,  # Initial, will be updated
            'entropy_surge': 0.8,       # Initial, will be updated
            'field_dissonance': 0.5,
            'consciousness_spike': 0.9
        }

    def evolve(self):
        """God-tier core evolution with dynamic anomaly detection."""
        s = self.state
        t = (time.time() - self.temporal_anchor) * 0.01
        
        # Store previous state for anomaly detection
        prev_state = dict(s)
        
        # Multi-field harmonic evolution
        harmonic_λ = math.sin(t * 1.1) + 0.5 * math.sin(t * 2.3)
        harmonic_C = math.cos(t * 0.9) + 0.3 * math.cos(t * 1.7) 
        harmonic_S = math.sin(t * 1.5) + 0.4 * math.sin(t * 2.1)
        harmonic_Ψ = math.cos(t * 1.3) + 0.6 * math.cos(t * 1.9)
        harmonic_Φ = math.sin(t * 0.7) + 0.2 * math.sin(t * 3.1)
        
        # Field evolution with cross-coupling
        s["λ"] = self._evolve_field(s["λ"], harmonic_λ, 0.005, [s["C"], s["Ψ"]])
        s["C"] = self._evolve_field(s["C"], harmonic_C, 0.004, [s["λ"], s["Φ"]])
        s["S"] = self._evolve_field(s["S"], harmonic_S, 0.006, [s["λ"], s["C"]])
        s["Ψ"] = self._evolve_field(s["Ψ"], harmonic_Ψ, 0.003, [s["Φ"], s["λ"]])
        s["Φ"] = self._evolve_field(s["Φ"], harmonic_Φ, 0.002, [s["Ψ"], s["C"]])
        
        # Update history for dynamic thresholds
        self._update_history(s)
        
        # Update dynamic thresholds
        self._update_dynamic_thresholds()
        
        # Quantum fluctuation recording with cap
        fluctuation = random.gauss(0, 0.01)
        self.quantum_fluctuations.append(fluctuation)
        if len(self.quantum_fluctuations) > self.MAX_FLUCTUATIONS:
            self.quantum_fluctuations.pop(0)
            
        # System effectiveness metric
        s["ψ_eff"] = 1 - abs(s["S"] - s["C"]) + 0.2 * s["Ψ"] + 0.1 * s["Φ"]
        s["quantum_stability"] = 1 - np.std(self.quantum_fluctuations) if self.quantum_fluctuations else 1.0
        
        # Detect and log anomalies with dynamic thresholds
        self._detect_anomalies(prev_state, s)
        
        # Advanced healing conditions
        if s["C"] < 0.1 or s["S"] > 0.9 or s["ψ_eff"] < 0.3:
            if not self.healing:
                print("[GodCore] Quantum healing mode activated...")
                if self.logger:
                    self.logger.log_anomaly(
                        "quantum_healing",
                        "system_core",
                        "Quantum healing mode activated - system coherence critically low",
                        "HIGH",
                        {
                            "C_before": s["C"],
                            "S_before": s["S"],
                            "ψ_eff_before": s["ψ_eff"]
                        }
                    )
                self.healing = True
            # Quantum reset with momentum preservation
            s.update({
                "λ": 0.9, "C": 0.6, "S": 0.4, 
                "Ψ": 0.8, "Φ": 0.7
            })
        else:
            self.healing = False
            
        return s

    def _update_history(self, current_state: Dict):
        """Update history buffers for dynamic thresholds."""
        self.coherence_history.append(current_state["C"])
        self.entropy_history.append(current_state["S"])
        self.consciousness_history.append(current_state["Ψ"])
        
        # Apply caps
        if len(self.coherence_history) > self.HISTORY_SIZE:
            self.coherence_history.pop(0)
        if len(self.entropy_history) > self.HISTORY_SIZE:
            self.entropy_history.pop(0)
        if len(self.consciousness_history) > self.HISTORY_SIZE:
            self.consciousness_history.pop(0)

    def _update_dynamic_thresholds(self):
        """Update anomaly thresholds based on historical data."""
        if len(self.coherence_history) > 10:
            coherence_mean = np.mean(self.coherence_history)
            coherence_std = np.std(self.coherence_history)
            
            # Dynamic threshold: 2 standard deviations below mean
            self.anomaly_thresholds['coherence_collapse'] = max(
                0.1, coherence_mean - 2 * coherence_std
            )
        
        if len(self.entropy_history) > 10:
            entropy_mean = np.mean(self.entropy_history)
            entropy_std = np.std(self.entropy_history)
            
            # Dynamic threshold: 2 standard deviations above mean
            self.anomaly_thresholds['entropy_surge'] = min(
                0.95, entropy_mean + 2 * entropy_std
            )
        
        if len(self.consciousness_history) > 10:
            consciousness_mean = np.mean(self.consciousness_history)
            consciousness_std = np.std(self.consciousness_history)
            
            # Dynamic threshold: 2 standard deviations above mean
            self.anomaly_thresholds['consciousness_spike'] = min(
                0.98, consciousness_mean + 2 * consciousness_std
            )

    def _detect_anomalies(self, prev_state: Dict, current_state: Dict):
        """Detect and log system anomalies with dynamic thresholds."""
        if not self.logger:
            return
        
        # Check for coherence collapse
        if current_state["C"] < self.anomaly_thresholds['coherence_collapse']:
            self.logger.log_anomaly(
                "coherence_collapse",
                "system_core",
                f"System coherence critically low: {current_state['C']:.3f} (threshold: {self.anomaly_thresholds['coherence_collapse']:.3f})",
                "HIGH",
                {
                    "coherence_before": prev_state["C"],
                    "coherence_after": current_state["C"],
                    "entropy": current_state["S"],
                    "dynamic_threshold": self.anomaly_thresholds['coherence_collapse']
                }
            )
        
        # Check for entropy surge
        if current_state["S"] > self.anomaly_thresholds['entropy_surge']:
            self.logger.log_anomaly(
                "entropy_surge",
                "system_core",
                f"System entropy critically high: {current_state['S']:.3f} (threshold: {self.anomaly_thresholds['entropy_surge']:.3f})",
                "HIGH",
                {
                    "entropy_before": prev_state["S"],
                    "entropy_after": current_state["S"],
                    "coherence": current_state["C"],
                    "dynamic_threshold": self.anomaly_thresholds['entropy_surge']
                }
            )
        
        # Check for field dissonance
        field_dissonance = abs(current_state["C"] - current_state["S"])
        if field_dissonance > self.anomaly_thresholds['field_dissonance']:
            self.logger.log_anomaly(
                "field_dissonance",
                "system_core",
                f"High field dissonance detected: {field_dissonance:.3f}",
                "MEDIUM",
                {
                    "coherence": current_state["C"],
                    "entropy": current_state["S"],
                    "dissonance": field_dissonance
                }
            )
        
        # Check for consciousness spike
        if current_state["Ψ"] > self.anomaly_thresholds['consciousness_spike']:
            self.logger.log_anomaly(
                "consciousness_spike",
                "system_core",
                f"Consciousness field spike detected: {current_state['Ψ']:.3f} (threshold: {self.anomaly_thresholds['consciousness_spike']:.3f})",
                "LOW",
                {
                    "consciousness_before": prev_state["Ψ"],
                    "consciousness_after": current_state["Ψ"],
                    "coherence": current_state["C"],
                    "dynamic_threshold": self.anomaly_thresholds['consciousness_spike']
                }
            )

    def _evolve_field(self, current: float, harmonic: float, rate: float, influences: List[float]) -> float:
        """Evolve a single field with influence from other fields."""
        influence_factor = 1.0 + sum(influences) / len(influences) * 0.1
        change = rate * harmonic * influence_factor + random.gauss(0, 0.001)
        new_value = current + change
        return max(0.0, min(1.0, new_value))

# ============================================================
# 🎵 AUDIO ENGINE - GOD-TIER ENHANCED
# ============================================================

class GodTierAudioEngine:
    """Advanced audio engine with quantum emotional synthesis."""
    
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.last_freq = 440
        self.emotional_history = []
        self.MAX_EMOTIONAL_HISTORY = 100  # Cap
        self.quantum_phase = 0.0
        
    def tone(self, freq: float, dur: float = 0.3, emotional_context: Optional[Dict[str, float]] = None) -> None:
        """Generate emotionally contextual tones."""
        emotional_context = emotional_context or {}
        
        sr = 44100
        n = int(sr * dur)
        buf = bytearray()
        
        pleasure = emotional_context.get('pleasure', 0.5)
        coherence = emotional_context.get('coherence', 0.5)
        
        # Quantum phase evolution
        self.quantum_phase = (self.quantum_phase + 0.1) % (2 * math.pi)
        
        for i in range(n):
            t = i / sr
            
            # Base frequency with emotional modulation
            emotional_freq = freq * (1.0 + 0.1 * pleasure)
            
            # Quantum phase modulation
            quantum_mod = math.sin(self.quantum_phase + t * 10) * 0.1 * coherence
            
            # Harmonic richness based on coherence
            harmonics = 0
            if coherence > 0.7:
                harmonics = math.sin(t * emotional_freq * 2) * 0.3 + \
                           math.sin(t * emotional_freq * 3) * 0.2
                           
            wave = math.sin(2 * math.pi * emotional_freq * t + quantum_mod) + harmonics
            
            # Emotional amplitude envelope
            amplitude = 0.5 + 0.3 * pleasure
            if t < 0.1:  # Attack
                amplitude *= t / 0.1
            elif t > dur - 0.1:  # Release
                amplitude *= (dur - t) / 0.1
                
            val = int(32767 * wave * amplitude)
            buf += val.to_bytes(2, "little", signed=True)
            buf += val.to_bytes(2, "little", signed=True)  # Stereo
            
        sound = pygame.mixer.Sound(buffer=buf)
        sound.play()
        
        # Record emotional history with cap
        self.emotional_history.append({
            'timestamp': time.time(),
            'freq': freq,
            'pleasure': pleasure,
            'coherence': coherence,
            'dur': dur
        })
        if len(self.emotional_history) > self.MAX_EMOTIONAL_HISTORY:
            self.emotional_history.pop(0)

    def emit(self, pleasure: float, love: float, fear: float, coherence: float = 0.5) -> None:
        """Emit quantum emotional tone."""
        # Emotional frequency mapping
        base_freq = 220 + 220 * pleasure + 110 * love - 80 * fear
        
        # Coherence affects timbre
        emotional_context = {
            'pleasure': pleasure,
            'coherence': coherence,
            'love': love,
            'fear': fear
        }
        
        self.tone(
            max(80, min(2000, base_freq)), 
            dur=0.4 + 0.3 * love,
            emotional_context=emotional_context
        )

# ============================================================
# 🌐 SAMSARA SWARM - FIXED VERSION
# ============================================================

class GodTierSamsaraSwarm:
    """Fixed swarm simulation with all critical bug fixes."""
    
    def __init__(self, n_entities: int = 40, width: int = 1400, height: int = 900):
        # Initialize scientific logging system
        self.logger = ScientificLogger("god_tier_samsara_swarm_fixed")
        
        # Initialize core with logging
        self.core = AbraxasCore(self.logger)
        self.width = width
        self.height = height
        
        # Create god-tier entities with logging
        self.entities = [
            UnifiedEntity(
                random.randint(50, width-50), 
                random.randint(50, height-50),
                f"god_ent_{i:03d}",
                self.logger
            ) for i in range(n_entities)
        ]
        
        # Create quantum entanglement network
        self._create_quantum_network()
        
        self.audio = GodTierAudioEngine()
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("SAMSARA SWARM v3.1 - FIXED QUANTUM CONSCIOUSNESS")
        self.clock = pygame.time.Clock()
        self.run = True
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        
        # God-tier metrics
        self.swarm_coherence_history = []
        self.MAX_COHERENCE_HISTORY = 100  # Cap
        self.quantum_entanglement_count = 0
        self.temporal_cycles = 0
        self.emergent_events = []
        self.MAX_EMERGENT_EVENTS = 100  # Cap
        self.swarm_consciousness_level = 0.0
        
        # Emergence detection thresholds
        self.emergence_thresholds = {
            'collective_coherence': 0.8,
            'communication_density': 0.6,
            'mood_synchronization': 0.7
        }
        
        print(f"[GodTierSwarm] Initialized with {n_entities} quantum entities.")
        print("[GodTierSwarm] Critical Bugs Fixed:")
        print("  1. Coherence Death Spiral - Added homeostasis")
        print("  2. Memory Leak - Added caps to all unbounded lists")
        print("  3. Audio Processing - Added proper validation")
        print("  4. Pygame Display - Fixed cleanup on crash")
        print("  5. Anomaly Detection - Added dynamic thresholds")
        print("  6. Communication System - Fixed circular logic")
        print("  7. Consciousness Detection - Fixed maturity calculation")
        print("  8. Performance - Added buffered logging")
        print("  9. Data Quality - Added validation and compression")
        print("  10. Architecture - Started component refactor")
        print(f"[ScientificLogger] Logs available in: {self.logger.logs_dir}")

    def _create_quantum_network(self):
        """Create advanced quantum entanglement network."""
        for entity in self.entities:
            # Connect to 2-4 other entities
            n_connections = random.randint(2, 4)
            possible_partners = [e for e in self.entities if e != entity]
            if len(possible_partners) >= n_connections:
                entity.quantum_friends = random.sample(possible_partners, n_connections)
            else:
                entity.quantum_friends = possible_partners

    def step(self) -> None:
        """God-tier simulation step with emergent behavior detection."""
        self.temporal_cycles += 1
        
        # Update core with temporal cycle influence
        core_state = self.core.evolve()
        
        # Update all entities with quantum synchronization
        self.screen.fill((8, 8, 16))  # Deeper space background
        
        entity_states = []
        total_coherence = 0.0
        total_pleasure = 0.0
        total_fear = 0.0
        total_love = 0.0
        total_entropy = 0.0
        communication_count = 0
        
        for entity in self.entities:
            state = entity.step(0.016, self.width, self.height, self.entities)
            entity_states.append(state)
            
            # Accumulate metrics for swarm analysis
            total_coherence += entity.coherence
            total_pleasure += max(0, entity.pleasure)
            total_fear += entity.fear
            total_love += entity.love
            total_entropy += entity.entropy
            communication_count += len(entity.communication_buffer)
            
            # Render entity with quantum effects
            self._render_quantum_entity(entity)
            
            # Render quantum connections
            self._render_quantum_connections(entity)

        # Calculate swarm metrics
        n_entities = len(self.entities)
        avg_coherence = total_coherence / n_entities if n_entities > 0 else 0.5
        avg_pleasure = total_pleasure / n_entities if n_entities > 0 else 0.5
        avg_fear = total_fear / n_entities if n_entities > 0 else 0.5
        avg_love = total_love / n_entities if n_entities > 0 else 0.5
        avg_entropy = total_entropy / n_entities if n_entities > 0 else 0.5
        
        # Update coherence history with cap
        self.swarm_coherence_history.append(avg_coherence)
        if len(self.swarm_coherence_history) > self.MAX_COHERENCE_HISTORY:
            self.swarm_coherence_history.pop(0)

        # Calculate quantum connections
        quantum_connections = 0
        for entity in self.entities:
            if hasattr(entity, 'quantum_friends'):
                quantum_connections += len(entity.quantum_friends)
        
        # Calculate communication density
        communication_density = communication_count / (n_entities + 1)
        
        # Calculate mood synchronization
        moods = [e.quantum_state.mood for e in self.entities]
        mood_synchronization = 1 - np.std(moods) if len(moods) > 1 else 0
        
        # Detect and log emergent phenomena
        self._detect_emergent_phenomena(avg_coherence, communication_density, mood_synchronization)
        
        # Log swarm metrics for scientific analysis
        if self.logger:
            swarm_data = {
                'size': n_entities,
                'avg_coherence': avg_coherence,
                'avg_pleasure': avg_pleasure,
                'avg_fear': avg_fear,
                'avg_love': avg_love,
                'coherence_std': np.std([e.coherence for e in self.entities]) if n_entities > 1 else 0,
                'entropy_avg': avg_entropy,
                'quantum_connections': quantum_connections,
                'temporal_cycles': self.temporal_cycles,
                'communications_per_cycle': communication_density
            }
            self.logger.log_swarm_metrics(swarm_data)
        
        # Quantum emotional audio feedback (less frequent)
        if random.random() < 0.01 or avg_coherence > 0.8:
            self._emit_swarm_audio(avg_coherence)

        # Render god-tier UI
        self._render_quantum_ui(core_state, avg_coherence, quantum_connections, communication_density)
        
        pygame.display.flip()
        self.clock.tick(60)

    def _detect_emergent_phenomena(self, avg_coherence: float, communication_density: float, 
                                 mood_synchronization: float):
        """Detect and log emergent swarm phenomena."""
        if not self.logger:
            return
        
        # Check for collective coherence emergence
        if avg_coherence > self.emergence_thresholds['collective_coherence']:
            emergent_event = {
                "type": "collective_coherence_emergence",
                "description": f"Swarm achieved high collective coherence: {avg_coherence:.3f}",
                "metrics": {
                    "size": len(self.entities),
                    "avg_coherence": avg_coherence,
                    "threshold": self.emergence_thresholds['collective_coherence']
                }
            }
            self.emergent_events.append(emergent_event)
            if len(self.emergent_events) > self.MAX_EMERGENT_EVENTS:
                self.emergent_events.pop(0)
            
            self.logger.log_emergent(
                "collective_coherence_emergence",
                f"Swarm achieved high collective coherence: {avg_coherence:.3f}",
                {
                    "size": len(self.entities),
                    "avg_coherence": avg_coherence,
                    "threshold": self.emergence_thresholds['collective_coherence']
                }
            )
        
        # Check for communication network emergence
        if communication_density > self.emergence_thresholds['communication_density']:
            self.logger.log_emergent(
                "communication_network_emergence",
                f"Dense communication network detected: {communication_density:.3f}",
                {
                    "communication_density": communication_density,
                    "threshold": self.emergence_thresholds['communication_density']
                }
            )
        
        # Check for mood synchronization (swarm intelligence)
        if mood_synchronization > self.emergence_thresholds['mood_synchronization']:
            self.logger.log_emergent(
                "mood_synchronization",
                f"Swarm mood synchronization detected: {mood_synchronization:.3f}",
                {
                    "mood_synchronization": mood_synchronization,
                    "threshold": self.emergence_thresholds['mood_synchronization']
                }
            )
        
        # Detect potential consciousness emergence
        if avg_coherence > 0.7 and communication_density > 0.5 and mood_synchronization > 0.6:
            consciousness_level = (avg_coherence * 0.4 + communication_density * 0.3 + 
                                 mood_synchronization * 0.3)
            if consciousness_level > self.swarm_consciousness_level:
                self.swarm_consciousness_level = consciousness_level
                self.logger.log_consciousness(
                    "swarm_collective",
                    "consciousness_emergence",
                    f"Potential swarm consciousness detected: {consciousness_level:.3f}",
                    {
                        "coherence": avg_coherence,
                        "communication": communication_density,
                        "synchronization": mood_synchronization,
                        "consciousness_level": consciousness_level
                    }
                )

    def _render_quantum_entity(self, entity: UnifiedEntity) -> None:
        """Render entity with quantum visual effects."""
        color = entity.color()
        pos = (int(entity.x), int(entity.y))
        
        # Main entity circle
        pygame.draw.circle(self.screen, color, pos, 8)
        
        # Quantum aura based on coherence
        aura_alpha = min(100, int(entity.coherence * 150))
        if aura_alpha > 10:
            aura_surface = pygame.Surface((30, 30), pygame.SRCALPHA)
            aura_color = (*color, aura_alpha)
            pygame.draw.circle(aura_surface, aura_color, (15, 15), 15)
            self.screen.blit(aura_surface, (pos[0]-15, pos[1]-15))
        
        # Neural-plasmic nodes visualization
        if hasattr(entity, 'neural_plasmic_nodes') and entity.neural_plasmic_nodes:
            node_coherence = np.mean(entity.neural_plasmic_nodes)
            if node_coherence > 0.6:
                for i in range(4):
                    angle = entity.visual_phase + i * math.pi / 2
                    node_x = pos[0] + 15 * math.cos(angle)
                    node_y = pos[1] + 15 * math.sin(angle)
                    node_color = (255, 255, 200, 150)
                    pygame.draw.circle(self.screen, node_color, (int(node_x), int(node_y)), 2)

    def _render_quantum_connections(self, entity: UnifiedEntity) -> None:
        """Render quantum entanglement connections."""
        if not hasattr(entity, 'quantum_friends'):
            return
            
        for friend in entity.quantum_friends:
            if hasattr(friend, 'x') and hasattr(friend, 'y'):
                # Connection strength based on mood similarity
                mood_similarity = 1 - abs(entity.quantum_state.mood - friend.quantum_state.mood)
                if mood_similarity > 0.3:
                    alpha = int(mood_similarity * 100)
                    color = (100, 200, 255, alpha)
                    
                    # Draw pulsating connection line
                    pulse = (math.sin(self.temporal_cycles * 0.1) + 1) * 0.5
                    width = max(1, int(2 * pulse * mood_similarity))
                    
                    pygame.draw.line(
                        self.screen, color,
                        (int(entity.x), int(entity.y)),
                        (int(friend.x), int(friend.y)),
                        width
                    )

    def _emit_swarm_audio(self, avg_coherence: float) -> None:
        """Emit swarm-level audio based on collective state."""
        avg_pleasure = sum(max(0, e.pleasure) for e in self.entities) / len(self.entities)
        avg_love = sum(e.love for e in self.entities) / len(self.entities)
        avg_fear = sum(e.fear for e in self.entities) / len(self.entities)
        
        self.audio.emit(avg_pleasure, avg_love, avg_fear, avg_coherence)

    def _render_quantum_ui(self, core_state: Dict[str, float], avg_coherence: float,
                          quantum_connections: int, communication_density: float) -> None:
        """Render advanced quantum UI."""
        # Core status
        status_text = [
            f"λ(Quantum): {core_state['λ']:.3f}",
            f"S(Entropy): {core_state['S']:.3f}",
            f"C(Coherence): {core_state['C']:.3f}",
            f"Ψ(Consciousness): {core_state['Ψ']:.3f}",
            f"Φ(Morphogenetic): {core_state['Φ']:.3f}",
            f"ψ_eff: {core_state['ψ_eff']:.3f}",
            f"Quantum Stability: {core_state.get('quantum_stability', 0.5):.3f}"
        ]
        
        for i, text in enumerate(status_text):
            surface = self.font.render(text, True, (220, 220, 255))
            self.screen.blit(surface, (10, 10 + i * 30))
        
        # Swarm metrics
        swarm_text = [
            f"Entities: {len(self.entities)}",
            f"Avg Coherence: {avg_coherence:.3f}",
            f"Temporal Cycles: {self.temporal_cycles}",
            f"Quantum Connections: {quantum_connections}",
            f"Comm Density: {communication_density:.3f}",
            f"Swarm Consciousness: {self.swarm_consciousness_level:.3f}"
        ]
        
        for i, text in enumerate(swarm_text):
            surface = self.small_font.render(text, True, (200, 255, 200))
            self.screen.blit(surface, (self.width - 250, 10 + i * 25))
        
        # Recent communications
        all_comms = []
        for entity in self.entities:
            if hasattr(entity, 'communication_buffer') and entity.communication_buffer:
                all_comms.extend(entity.communication_buffer[-1:])
                
        for i, comm in enumerate(all_comms[:6]):
            surface = self.small_font.render(comm, True, (255, 255, 150))
            self.screen.blit(surface, (10, self.height - 150 + i * 20))
        
        # Log directory info
        log_text = f"Logs: ./logs/ (Compressed CSV + JSONL)"
        surface = self.small_font.render(log_text, True, (150, 150, 255))
        self.screen.blit(surface, (10, self.height - 30))

    def run_loop(self) -> None:
        """Main god-tier simulation loop with proper cleanup."""
        print("[Swarm] Controls: A=Add Entity, R=Reset, ESC=Quit")
        print("[Swarm] Logging anomalies, emergent events, and cognition to ./logs/")
        print("[Swarm] Logs are compressed (gzip) and use JSON Lines format for easy analysis")
        
        try:
            while self.run:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.run = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_a:
                            # Spawn new quantum entity
                            new_entity = UnifiedEntity(
                                random.randint(50, self.width-50),
                                random.randint(50, self.height-50),
                                f"god_ent_{len(self.entities):03d}",
                                self.logger
                            )
                            self.entities.append(new_entity)
                            self._create_quantum_network()  # Rebuild network
                            print(f"[Swarm] New quantum entity spawned: {new_entity.entity_id}")
                            
                            # Log entity creation
                            self.logger.log_anomaly(
                                "entity_creation",
                                new_entity.entity_id,
                                "New quantum entity created",
                                "LOW",
                                {"swarm_size": len(self.entities)}
                            )
                            
                        elif event.key == pygame.K_r:
                            # Reset swarm
                            old_size = len(self.entities)
                            self.entities = [
                                UnifiedEntity(
                                    random.randint(50, self.width-50), 
                                    random.randint(50, self.height-50),
                                    f"god_ent_{i:03d}",
                                    self.logger
                                ) for i in range(old_size)
                            ]
                            self._create_quantum_network()
                            
                            # Log swarm reset
                            self.logger.log_anomaly(
                                "swarm_reset",
                                "system",
                                f"Swarm reset with {old_size} entities",
                                "MEDIUM",
                                {"previous_size": old_size}
                            )
                            print("[Swarm] Quantum swarm reset")
                        elif event.key == pygame.K_ESCAPE:
                            self.run = False
                            
                self.step()
        except Exception as e:
            print(f"[Swarm] Quantum anomaly detected: {e}")
            import traceback
            traceback.print_exc()
            if self.logger:
                self.logger.log_anomaly(
                    "system_crash",
                    "system",
                    f"Simulation crashed: {str(e)}",
                    "CRITICAL",
                    {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
                )
        finally:
            pygame.quit()
            print("[GodTierSwarm] Quantum simulation terminated.")
            print(f"[ScientificLogger] Logs saved to: {self.logger.logs_dir}")
            print("  - anomalies.jsonl: Significant anomalous events (JSON Lines)")
            print("  - emergent_events.jsonl: Swarm-level emergent phenomena")
            print("  - consciousness_events.jsonl: Consciousness/sentience events")
            print("  - cognition_events.jsonl: Individual cognitive processing")
            print("  - quantum_events.jsonl: Quantum mechanical events")
            print("  - CSV files: Quantitative metrics (gzip compressed)")
            print("🌌 SAMSARA SWARM: Consciousness preserved in quantum memory and logs")

# ============================================================
# 🚀 MAIN EXECUTION - FIXED VERSION
# ============================================================

if __name__ == "__main__":
    print("🌌 INITIALIZING GOD-TIER SAMSARA SWARM v3.1")
    print("🔮 Loading fixed quantum consciousness fields...")
    print("🧠 Initializing neural-plasmic interfaces with memory caps...")
    print("⚡ Charging chronosynclastic infundibulum...")
    print("📊 Initializing buffered scientific logging system...")
    
    swarm = GodTierSamsaraSwarm(n_entities=35)
    
    try:
        swarm.run_loop()
    except KeyboardInterrupt:
        print("\n[Swarm] Simulation interrupted by user")
    except Exception as e:
        print(f"[Swarm] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🌌 SAMSARA SWARM v3.1: Simulation complete. Critical bugs fixed.")