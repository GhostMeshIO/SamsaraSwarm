#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 SWARM LOG ANALYZER v1.0 - Scientific Analysis Engine
======================================================
Comprehensive analysis system for Samsara Swarm log data.

Features:
1. Multi-file log consolidation
2. Statistical analysis across time dimensions
3. Anomaly detection and classification
4. Emergent behavior pattern recognition
5. Quantum state correlation analysis
6. Consciousness metrics evaluation
7. Visualization generation
8. Report generation (PDF/HTML)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.cm as cm

# Reporting
from jinja2 import Template
import pdfkit

# ============================================================
# 📊 DATA LOADER & CONSOLIDATOR
# ============================================================

@dataclass
class SwarmDataset:
    """Consolidated dataset from all log files."""
    swarm_metrics: pd.DataFrame
    entity_metrics: pd.DataFrame
    anomaly_events: pd.DataFrame
    emergent_events: List[Dict]
    consciousness_events: List[Dict]
    cognition_events: List[Dict]
    quantum_events: List[Dict]
    
class LogConsolidator:
    """Load and consolidate all log files from experiments."""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.dataset = None
        
    def load_all_data(self) -> SwarmDataset:
        """Load and consolidate all log files."""
        print(f"[Analyzer] Loading data from {self.log_dir}...")
        
        # Check if files exist
        if not self.log_dir.exists():
            print(f"[ERROR] Log directory {self.log_dir} not found!")
            # Return empty dataset
            return SwarmDataset(
                swarm_metrics=pd.DataFrame(),
                entity_metrics=pd.DataFrame(),
                anomaly_events=pd.DataFrame(),
                emergent_events=[],
                consciousness_events=[],
                cognition_events=[],
                quantum_events=[]
            )
        
        # Load CSV files with error handling
        try:
            swarm_df = pd.read_csv(self.log_dir / "swarm_metrics.csv") if (self.log_dir / "swarm_metrics.csv").exists() else pd.DataFrame()
        except Exception as e:
            print(f"[WARNING] Error loading swarm_metrics.csv: {e}")
            swarm_df = pd.DataFrame()
            
        try:
            entity_df = pd.read_csv(self.log_dir / "entity_metrics.csv") if (self.log_dir / "entity_metrics.csv").exists() else pd.DataFrame()
        except Exception as e:
            print(f"[WARNING] Error loading entity_metrics.csv: {e}")
            entity_df = pd.DataFrame()
            
        try:
            anomaly_df = pd.read_csv(self.log_dir / "anomaly_events.csv") if (self.log_dir / "anomaly_events.csv").exists() else pd.DataFrame()
        except Exception as e:
            print(f"[WARNING] Error loading anomaly_events.csv: {e}")
            anomaly_df = pd.DataFrame()
        
        # Convert timestamps if data exists
        if not swarm_df.empty and 'timestamp' in swarm_df.columns:
            swarm_df['timestamp'] = pd.to_datetime(swarm_df['timestamp'], errors='coerce')
        
        if not entity_df.empty and 'timestamp' in entity_df.columns:
            entity_df['timestamp'] = pd.to_datetime(entity_df['timestamp'], errors='coerce')
        
        if not anomaly_df.empty and 'timestamp' in anomaly_df.columns:
            anomaly_df['timestamp'] = pd.to_datetime(anomaly_df['timestamp'], errors='coerce')
        
        # Load text log files
        emergent_logs = self._parse_emergent_logs()
        consciousness_logs = self._parse_consciousness_logs()
        cognition_logs = self._parse_cognition_logs()
        quantum_logs = self._parse_quantum_logs()
        
        self.dataset = SwarmDataset(
            swarm_metrics=swarm_df,
            entity_metrics=entity_df,
            anomaly_events=anomaly_df,
            emergent_events=emergent_logs,
            consciousness_events=consciousness_logs,
            cognition_events=cognition_logs,
            quantum_events=quantum_logs
        )
        
        print(f"[Analyzer] Data loaded:")
        print(f"  - Swarm metrics: {len(swarm_df)} records")
        print(f"  - Entity metrics: {len(entity_df)} records")
        print(f"  - Anomalies: {len(anomaly_df)} events")
        print(f"  - Emergent events: {len(emergent_logs)} events")
        print(f"  - Consciousness events: {len(consciousness_logs)} events")
        print(f"  - Cognition events: {len(cognition_logs)} events")
        print(f"  - Quantum events: {len(quantum_logs)} events")
        
        return self.dataset
    
    def _parse_emergent_logs(self) -> List[Dict]:
        """Parse emergent_events.log file."""
        logs = []
        log_file = self.log_dir / "emergent_events.log"
        
        if not log_file.exists():
            return logs
            
        try:
            with open(log_file, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"[WARNING] Error reading emergent_events.log: {e}")
            return logs
            
        # Split by separator
        entries = content.split('-' * 60)
        
        for entry in entries:
            if not entry.strip():
                continue
                
            lines = entry.strip().split('\n')
            if len(lines) < 2:
                continue
                
            # Parse timestamp and message
            try:
                timestamp_str = lines[0].split(' - ')[0]
                message = ' '.join(lines[0].split(' - ')[1:])
                
                # Extract metrics from message if present
                metrics = {}
                if len(lines) > 1:
                    for line in lines[1:]:
                        if '=' in line:
                            parts = line.split('=')
                            if len(parts) == 2:
                                key = parts[0].strip()
                                try:
                                    value = float(parts[1].strip())
                                    metrics[key] = value
                                except:
                                    metrics[key] = parts[1].strip()
                
                logs.append({
                    'timestamp': pd.to_datetime(timestamp_str, errors='coerce'),
                    'message': message,
                    'metrics': metrics,
                    'type': self._classify_emergent_event(message)
                })
            except Exception as e:
                print(f"[WARNING] Error parsing emergent log entry: {e}")
                continue
            
        return logs
    
    def _parse_consciousness_logs(self) -> List[Dict]:
        """Parse consciousness_events.log file."""
        logs = []
        log_file = self.log_dir / "consciousness_events.log"
        
        if not log_file.exists():
            return logs
            
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[WARNING] Error reading consciousness_events.log: {e}")
            return logs
            
        current_entry = {}
        for line in lines:
            if line.startswith('---'):
                if current_entry:
                    logs.append(current_entry)
                    current_entry = {}
                continue
                
            if ' - ' in line:
                parts = line.strip().split(' - ')
                if len(parts) >= 2:
                    timestamp = parts[0]
                    rest = ' - '.join(parts[1:])
                    
                    if 'ENTITY:' in rest:
                        entity_part = rest.split(' - ')[0]
                        entity_id = entity_part.replace('ENTITY:', '').strip()
                        message = ' - '.join(rest.split(' - ')[1:])
                        
                        current_entry = {
                            'timestamp': pd.to_datetime(timestamp, errors='coerce'),
                            'entity_id': entity_id,
                            'message': message,
                            'type': self._classify_consciousness_event(message)
                        }
                        
        if current_entry:
            logs.append(current_entry)
            
        return logs
    
    def _parse_cognition_logs(self) -> List[Dict]:
        """Parse cognition_events.log file."""
        logs = []
        log_file = self.log_dir / "cognition_events.log"
        
        if not log_file.exists():
            return logs
            
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    if ' - ' in line:
                        parts = line.strip().split(' - ')
                        if len(parts) >= 4:
                            timestamp = parts[0]
                            entity_part = parts[1]
                            metric_part = parts[2]
                            message = parts[3]
                            
                            entity_id = entity_part.replace('ENTITY:', '').strip()
                            metric = metric_part.replace('METRIC:', '').strip()
                            
                            # Extract numeric value if present
                            value = None
                            if '=' in message:
                                try:
                                    val_str = message.split('=')[1].split()[0]
                                    value = float(val_str)
                                except:
                                    pass
                            
                            logs.append({
                                'timestamp': pd.to_datetime(timestamp, errors='coerce'),
                                'entity_id': entity_id,
                                'metric': metric,
                                'message': message,
                                'value': value
                            })
        except Exception as e:
            print(f"[WARNING] Error reading cognition_events.log: {e}")
        
        return logs
    
    def _parse_quantum_logs(self) -> List[Dict]:
        """Parse quantum_events.log file."""
        logs = []
        log_file = self.log_dir / "quantum_events.log"
        
        if not log_file.exists():
            return logs
            
        try:
            with open(log_file, 'r') as f:
                content = f.read()
        except Exception as e:
            print(f"[WARNING] Error reading quantum_events.log: {e}")
            return logs
            
        entries = content.split('-' * 60)
        
        for entry in entries:
            if not entry.strip():
                continue
                
            lines = entry.strip().split('\n')
            if len(lines) >= 1:
                first_line = lines[0]
                if ' - ' in first_line:
                    parts = first_line.split(' - ')
                    timestamp = parts[0]
                    rest = ' - '.join(parts[1:])
                    
                    entity_part = rest.split(' - ')[0]
                    event_part = rest.split(' - ')[1] if len(rest.split(' - ')) > 1 else ""
                    
                    entity_id = entity_part.replace('ENTITY:', '').strip()
                    event_type = event_part.replace('EVENT:', '').strip()
                    
                    message = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                    
                    # Try to extract quantum state if present
                    quantum_state = {}
                    if 'Quantum State:' in message:
                        try:
                            state_str = message.split('Quantum State:')[1].strip()
                            quantum_state = json.loads(state_str)
                        except:
                            pass
                    
                    logs.append({
                        'timestamp': pd.to_datetime(timestamp, errors='coerce'),
                        'entity_id': entity_id,
                        'event_type': event_type,
                        'message': message[:200],  # Truncate
                        'quantum_state': quantum_state
                    })
        
        return logs
    
    def _classify_emergent_event(self, message: str) -> str:
        """Classify emergent event type from message."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['coherence', 'synchronization']):
            return 'coherence_emergence'
        elif any(word in message_lower for word in ['communication', 'network']):
            return 'communication_emergence'
        elif any(word in message_lower for word in ['consciousness', 'sentience']):
            return 'consciousness_emergence'
        elif any(word in message_lower for word in ['collective', 'swarm']):
            return 'collective_behavior'
        else:
            return 'unknown'
    
    def _classify_consciousness_event(self, message: str) -> str:
        """Classify consciousness event type."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['quantum', 'insight']):
            return 'quantum_insight'
        elif any(word in message_lower for word in ['inspiration', 'creative']):
            return 'inspiration'
        elif any(word in message_lower for word in ['awareness', 'self']):
            return 'self_awareness'
        elif any(word in message_lower for word in ['coherence', 'spike']):
            return 'coherence_event'
        else:
            return 'general_consciousness'

# ============================================================
# 📈 STATISTICAL ANALYZER - FIXED VERSION
# ============================================================

class SwarmStatistics:
    """Comprehensive statistical analysis of swarm data."""
    
    def __init__(self, dataset: SwarmDataset):
        self.dataset = dataset
        self.results = {}
        
    def analyze_all(self) -> Dict[str, Any]:
        """Run complete statistical analysis."""
        print("[Statistics] Running comprehensive analysis...")
        
        self.results = {
            'temporal_analysis': self._analyze_temporal_trends(),
            'entity_analysis': self._analyze_entity_behavior(),
            'anomaly_analysis': self._analyze_anomalies(),
            'correlation_analysis': self._analyze_correlations(),
            'emergence_analysis': self._analyze_emergence_patterns(),
            'quantum_analysis': self._analyze_quantum_behavior(),
            'consciousness_analysis': self._analyze_consciousness_metrics()
        }
        
        return self.results
    
    def _analyze_temporal_trends(self) -> Dict[str, Any]:
        """Analyze temporal trends in swarm metrics."""
        df = self.dataset.swarm_metrics.copy()  # Create a copy to avoid modifying original
        
        if df.empty:
            return {}
            
        results = {}
        
        # Store original columns to check if timestamp is an index
        has_timestamp_column = 'timestamp' in df.columns
        
        # If timestamp is a column and we want to use it as index temporarily
        if has_timestamp_column:
            df_temp = df.set_index('timestamp')
        elif isinstance(df.index, pd.DatetimeIndex):
            # Already has datetime index
            df_temp = df
        else:
            # No timestamp available
            return results
        
        # Time-based resampling for trend analysis
        try:
            # Resample to different time intervals
            for freq in ['1min', '5min', '10min']:
                resampled = df_temp.resample(freq).mean()
                
                if not resampled.empty:
                    # Calculate trends
                    for column in ['avg_coherence', 'avg_pleasure', 'avg_fear', 'avg_love', 'entropy_avg']:
                        if column in resampled.columns:
                            series = resampled[column].dropna()
                            if len(series) > 2:
                                # Linear trend
                                x = np.arange(len(series))
                                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
                                
                                results[f'{column}_{freq}_trend'] = {
                                    'slope': slope,
                                    'r_squared': r_value**2,
                                    'p_value': p_value,
                                    'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                                    'trend_strength': abs(slope) * 100
                                }
            
            # Detect regime changes
            if 'avg_coherence' in df_temp.columns:
                coherence_series = df_temp['avg_coherence'].dropna()
                if len(coherence_series) > 10:
                    # Use rolling statistics to detect changes
                    rolling_mean = coherence_series.rolling(window=10, center=True).mean()
                    rolling_std = coherence_series.rolling(window=10, center=True).std()
                    
                    # Detect significant changes
                    z_scores = np.abs((coherence_series - rolling_mean) / rolling_std)
                    regime_changes = np.where(z_scores > 2.0)[0]
                    
                    results['regime_changes'] = {
                        'count': len(regime_changes),
                        'positions': regime_changes.tolist()
                    }
        except Exception as e:
            print(f"[WARNING] Error in temporal trend analysis: {e}")
        
        return results
    
    def _analyze_entity_behavior(self) -> Dict[str, Any]:
        """Analyze individual entity behavior patterns."""
        df = self.dataset.entity_metrics.copy()  # Create a copy
        
        if df.empty:
            return {}
            
        results = {}
        
        # Group by entity for analysis
        entity_groups = df.groupby('entity_id')
        
        # Basic entity statistics
        entity_stats = {}
        for entity_id, group in entity_groups:
            if len(group) > 5:  # Need enough data points
                stats_dict = {
                    'coherence_mean': group['coherence'].mean() if 'coherence' in group.columns else 0,
                    'coherence_std': group['coherence'].std() if 'coherence' in group.columns else 0,
                    'pleasure_mean': group['pleasure'].mean() if 'pleasure' in group.columns else 0,
                    'fear_mean': group['fear'].mean() if 'fear' in group.columns else 0,
                    'love_mean': group['love'].mean() if 'love' in group.columns else 0,
                    'entropy_mean': group['entropy'].mean() if 'entropy' in group.columns else 0,
                    'age_max': group['age'].max() if 'age' in group.columns else 0,
                    'num_measurements': len(group)
                }
                entity_stats[entity_id] = stats_dict
        
        results['entity_statistics'] = entity_stats
        
        # Cluster entities by behavior
        if len(entity_stats) > 3:
            try:
                # Prepare data for clustering
                features = []
                entity_ids = []
                
                for entity_id, stats_dict in entity_stats.items():
                    features.append([
                        stats_dict['coherence_mean'],
                        stats_dict['pleasure_mean'],
                        stats_dict['fear_mean'],
                        stats_dict['love_mean'],
                        stats_dict['entropy_mean']
                    ])
                    entity_ids.append(entity_id)
                
                features = np.array(features)
                
                # Normalize features
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)
                
                # Perform clustering
                clustering = DBSCAN(eps=1.0, min_samples=2).fit(features_scaled)
                labels = clustering.labels_
                
                # Analyze clusters
                unique_labels = set(labels)
                clusters = {}
                
                for label in unique_labels:
                    if label == -1:
                        cluster_name = 'outliers'
                    else:
                        cluster_name = f'cluster_{label}'
                    
                    cluster_entities = [entity_ids[i] for i in range(len(entity_ids)) if labels[i] == label]
                    cluster_features = features[labels == label]
                    
                    if len(cluster_entities) > 0:
                        clusters[cluster_name] = {
                            'entities': cluster_entities,
                            'size': len(cluster_entities),
                            'coherence_mean': np.mean(cluster_features[:, 0]) if len(cluster_features) > 0 else 0,
                            'pleasure_mean': np.mean(cluster_features[:, 1]) if len(cluster_features) > 0 else 0,
                            'behavior_type': self._classify_entity_cluster(cluster_features)
                        }
                
                results['behavior_clusters'] = clusters
            except Exception as e:
                print(f"[WARNING] Error in entity clustering: {e}")
        
        # Identify leader entities (high coherence, low entropy)
        if entity_stats:
            leader_scores = {}
            for entity_id, stats in entity_stats.items():
                # Simple leader score
                score = stats['coherence_mean'] * 0.4 + stats['pleasure_mean'] * 0.3 + \
                       (1 - stats['fear_mean']) * 0.2 + stats['love_mean'] * 0.1
                leader_scores[entity_id] = score
            
            # Top 5 leaders
            top_leaders = sorted(leader_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            results['leader_entities'] = [
                {'entity_id': entity_id, 'leader_score': score}
                for entity_id, score in top_leaders
            ]
        
        return results
    
    def _analyze_anomalies(self) -> Dict[str, Any]:
        """Analyze anomaly patterns and distributions."""
        df = self.dataset.anomaly_events.copy()  # Create a copy
        
        if df.empty:
            return {}
            
        results = {}
        
        # Basic anomaly statistics
        results['total_anomalies'] = len(df)
        
        if 'event_type' in df.columns:
            results['anomalies_by_type'] = df['event_type'].value_counts().to_dict()
        
        if 'severity' in df.columns:
            results['anomalies_by_severity'] = df['severity'].value_counts().to_dict()
        
        # Temporal distribution
        if 'timestamp' in df.columns and not df['timestamp'].isna().all():
            try:
                df['hour'] = df['timestamp'].dt.hour
                results['hourly_distribution'] = df['hour'].value_counts().sort_index().to_dict()
            except:
                pass
        
        # Coherence impact analysis
        if 'coherence_before' in df.columns and 'coherence_after' in df.columns:
            try:
                df['coherence_change'] = df['coherence_after'] - df['coherence_before']
                
                results['coherence_impact'] = {
                    'mean_change': df['coherence_change'].mean(),
                    'max_positive': df['coherence_change'].max(),
                    'max_negative': df['coherence_change'].min(),
                    'positive_anomalies': len(df[df['coherence_change'] > 0]),
                    'negative_anomalies': len(df[df['coherence_change'] < 0])
                }
                
                # Analyze by event type
                type_impact = {}
                if 'event_type' in df.columns:
                    for event_type in df['event_type'].unique():
                        type_data = df[df['event_type'] == event_type]
                        if len(type_data) > 0:
                            type_impact[event_type] = {
                                'count': len(type_data),
                                'mean_coherence_change': type_data['coherence_change'].mean(),
                                'severity_distribution': type_data['severity'].value_counts().to_dict() if 'severity' in type_data.columns else {}
                            }
                
                results['impact_by_type'] = type_impact
            except Exception as e:
                print(f"[WARNING] Error in coherence impact analysis: {e}")
        
        return results
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between different metrics."""
        swarm_df = self.dataset.swarm_metrics.copy()
        entity_df = self.dataset.entity_metrics.copy()
        
        results = {}
        
        # Swarm-level correlations
        if not swarm_df.empty:
            try:
                numeric_cols = swarm_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    correlation_matrix = swarm_df[numeric_cols].corr()
                    
                    # Convert to dict for JSON serialization
                    results['swarm_correlations'] = {
                        'matrix': correlation_matrix.to_dict(),
                        'strong_correlations': self._extract_strong_correlations(correlation_matrix, threshold=0.7)
                    }
            except Exception as e:
                print(f"[WARNING] Error in swarm correlation analysis: {e}")
        
        # Entity-level correlations
        if not entity_df.empty:
            try:
                entity_corr_cols = ['coherence', 'pleasure', 'fear', 'love', 'entropy']
                available_cols = [col for col in entity_corr_cols if col in entity_df.columns]
                
                if len(available_cols) > 1:
                    entity_corr = entity_df[available_cols].corr()
                    results['entity_correlations'] = {
                        'matrix': entity_corr.to_dict(),
                        'key_relationships': {
                            'coherence_pleasure': entity_corr.loc['coherence', 'pleasure'] if 'coherence' in entity_corr.index and 'pleasure' in entity_corr.columns else 0,
                            'coherence_fear': entity_corr.loc['coherence', 'fear'] if 'coherence' in entity_corr.index and 'fear' in entity_corr.columns else 0,
                            'pleasure_love': entity_corr.loc['pleasure', 'love'] if 'pleasure' in entity_corr.index and 'love' in entity_corr.columns else 0
                        }
                    }
            except Exception as e:
                print(f"[WARNING] Error in entity correlation analysis: {e}")
        
        return results
    
    def _analyze_emergence_patterns(self) -> Dict[str, Any]:
        """Analyze emergent behavior patterns."""
        emergent_events = self.dataset.emergent_events
        
        if not emergent_events:
            return {}
            
        results = {}
        
        # Categorize and analyze emergent events
        event_types = {}
        for event in emergent_events:
            event_type = event.get('type', 'unknown')
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        # Analyze each event type
        for event_type, events in event_types.items():
            # Extract metrics if available
            all_metrics = []
            for event in events:
                if 'metrics' in event and event['metrics']:
                    all_metrics.append(event['metrics'])
            
            if all_metrics:
                # Create DataFrame for analysis
                metrics_df = pd.DataFrame(all_metrics)
                
                event_type_analysis = {
                    'count': len(events),
                    'first_occurrence': min([e['timestamp'] for e in events if pd.notna(e['timestamp'])]),
                    'last_occurrence': max([e['timestamp'] for e in events if pd.notna(e['timestamp'])]),
                }
                
                # Calculate frequency if we have valid timestamps
                valid_times = [e['timestamp'] for e in events if pd.notna(e['timestamp'])]
                if len(valid_times) > 1:
                    time_diff = max(valid_times) - min(valid_times)
                    if time_diff.total_seconds() > 0:
                        event_type_analysis['frequency_hours'] = len(events) / (time_diff.total_seconds() / 3600)
                
                # Add metric statistics if available
                if not metrics_df.empty:
                    for column in metrics_df.columns:
                        if pd.api.types.is_numeric_dtype(metrics_df[column]):
                            event_type_analysis[f'{column}_mean'] = metrics_df[column].mean()
                            event_type_analysis[f'{column}_std'] = metrics_df[column].std()
                
                results[event_type] = event_type_analysis
        
        return results
    
    def _analyze_quantum_behavior(self) -> Dict[str, Any]:
        """Analyze quantum events and behavior patterns."""
        quantum_events = self.dataset.quantum_events
        
        if not quantum_events:
            return {}
            
        results = {}
        
        # Event type analysis
        event_types = {}
        for event in quantum_events:
            event_type = event.get('event_type', 'unknown')
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        results['event_type_distribution'] = {k: len(v) for k, v in event_types.items()}
        
        # Entity participation in quantum events
        entity_participation = {}
        for event in quantum_events:
            entity_id = event.get('entity_id', 'unknown')
            if entity_id not in entity_participation:
                entity_participation[entity_id] = 0
            entity_participation[entity_id] += 1
        
        results['quantum_active_entities'] = {
            'total_unique': len(entity_participation),
            'top_participants': sorted(entity_participation.items(), key=lambda x: x[1], reverse=True)[:10],
            'participation_distribution': self._calculate_distribution_stats(list(entity_participation.values()))
        }
        
        return results
    
    def _analyze_consciousness_metrics(self) -> Dict[str, Any]:
        """Analyze consciousness-related events and metrics."""
        consciousness_events = self.dataset.consciousness_events
        
        if not consciousness_events:
            return {}
            
        results = {}
        
        # Event type analysis
        event_types = {}
        for event in consciousness_events:
            event_type = event.get('type', 'general_consciousness')
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        results['consciousness_event_types'] = {k: len(v) for k, v in event_types.items()}
        
        # Entity consciousness development
        entity_consciousness = {}
        for event in consciousness_events:
            entity_id = event.get('entity_id', 'unknown')
            timestamp = event.get('timestamp')
            
            if pd.isna(timestamp):
                continue
                
            if entity_id not in entity_consciousness:
                entity_consciousness[entity_id] = {
                    'count': 0,
                    'types': set(),
                    'first_event': timestamp,
                    'last_event': timestamp
                }
            
            entity_consciousness[entity_id]['count'] += 1
            entity_consciousness[entity_id]['types'].add(event.get('type', 'unknown'))
            entity_consciousness[entity_id]['last_event'] = max(
                entity_consciousness[entity_id]['last_event'], timestamp
            )
        
        # Calculate consciousness maturity scores
        maturity_scores = []
        for entity_id, data in entity_consciousness.items():
            # Score based on frequency, diversity, and recency
            frequency_score = min(data['count'] / 10, 1.0)  # Cap at 10 events
            diversity_score = min(len(data['types']) / 4, 1.0)  # Cap at 4 types
            duration_days = (data['last_event'] - data['first_event']).total_seconds() / 86400
            duration_score = min(duration_days / 7, 1.0)  # Cap at 7 days
            
            maturity_score = frequency_score * 0.4 + diversity_score * 0.4 + duration_score * 0.2
            
            maturity_scores.append({
                'entity_id': entity_id,
                'maturity_score': maturity_score,
                'event_count': data['count'],
                'type_diversity': len(data['types']),
                'consciousness_duration_days': duration_days
            })
        
        if maturity_scores:
            results['consciousness_maturity'] = sorted(maturity_scores, key=lambda x: x['maturity_score'], reverse=True)[:10]
        
        return results
    
    def _classify_entity_cluster(self, cluster_features: np.ndarray) -> str:
        """Classify entity cluster based on feature patterns."""
        if len(cluster_features) == 0:
            return "unknown"
        
        mean_features = np.mean(cluster_features, axis=0)
        # [coherence, pleasure, fear, love, entropy]
        
        if mean_features[0] > 0.7 and mean_features[1] > 0.6:  # High coherence and pleasure
            return "coherent_pleasure_seekers"
        elif mean_features[2] > 0.6:  # High fear
            return "fear_dominant"
        elif mean_features[3] > 0.7:  # High love
            return "love_dominant"
        elif mean_features[4] > 0.7:  # High entropy
            return "chaotic_entities"
        else:
            return "balanced_entities"
    
    def _extract_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Extract strong correlations from correlation matrix."""
        strong_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_corrs.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'positive' if corr_value > 0 else 'negative'
                    })
        
        return sorted(strong_corrs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _calculate_distribution_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate distribution statistics."""
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75)
        }

# ============================================================
# 📊 VISUALIZATION ENGINE - FIXED VERSION
# ============================================================

class SwarmVisualizer:
    """Create comprehensive visualizations of swarm analysis."""
    
    def __init__(self, dataset: SwarmDataset, statistics: Dict[str, Any], output_dir: str = "./analysis_output"):
        self.dataset = dataset
        self.statistics = statistics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("[Visualizer] Generating visualizations...")
        
        # Create subdirectory for plots
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate plots with error handling
        try:
            self._plot_swarm_metrics_timeline(plots_dir)
        except Exception as e:
            print(f"[WARNING] Error plotting swarm metrics timeline: {e}")
        
        try:
            self._plot_entity_coherence_heatmap(plots_dir)
        except Exception as e:
            print(f"[WARNING] Error plotting entity coherence heatmap: {e}")
        
        try:
            self._plot_anomaly_distribution(plots_dir)
        except Exception as e:
            print(f"[WARNING] Error plotting anomaly distribution: {e}")
        
        try:
            self._plot_correlation_matrices(plots_dir)
        except Exception as e:
            print(f"[WARNING] Error plotting correlation matrices: {e}")
        
        try:
            self._plot_emergence_patterns(plots_dir)
        except Exception as e:
            print(f"[WARNING] Error plotting emergence patterns: {e}")
        
        try:
            self._plot_consciousness_development(plots_dir)
        except Exception as e:
            print(f"[WARNING] Error plotting consciousness development: {e}")
        
        try:
            self._plot_quantum_activity_network(plots_dir)
        except Exception as e:
            print(f"[WARNING] Error plotting quantum activity network: {e}")
        
        try:
            self._plot_behavior_clusters(plots_dir)
        except Exception as e:
            print(f"[WARNING] Error plotting behavior clusters: {e}")
        
        print(f"[Visualizer] Visualizations saved to {plots_dir}")
    
    def _plot_swarm_metrics_timeline(self, output_dir: Path):
        """Plot swarm metrics over time."""
        df = self.dataset.swarm_metrics.copy()  # Work with a copy
        
        if df.empty:
            print("[Visualizer] No swarm metrics data to plot")
            return
            
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Swarm Metrics Timeline Analysis', fontsize=16)
        
        metrics_to_plot = [
            ('avg_coherence', 'Average Coherence', axes[0, 0]),
            ('avg_pleasure', 'Average Pleasure', axes[0, 1]),
            ('avg_fear', 'Average Fear', axes[1, 0]),
            ('avg_love', 'Average Love', axes[1, 1]),
            ('entropy_avg', 'Average Entropy', axes[2, 0]),
            ('quantum_connections', 'Quantum Connections', axes[2, 1])
        ]
        
        for metric, title, ax in metrics_to_plot:
            if metric in df.columns:
                # Handle timestamp - check if it's a column or index
                if 'timestamp' in df.columns:
                    x = df['timestamp']
                elif isinstance(df.index, pd.DatetimeIndex):
                    x = df.index
                else:
                    # No timestamp available, use index
                    x = range(len(df))
                
                y = df[metric]
                
                # Only plot if we have valid data
                if len(y.dropna()) > 0:
                    ax.plot(x, y, linewidth=2)
                    ax.set_title(title)
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                    
                    # Add trend line if enough points
                    if len(y.dropna()) > 2:
                        try:
                            # Convert x to numeric for trend calculation
                            if isinstance(x, pd.DatetimeIndex) or 'timestamp' in df.columns:
                                x_numeric = pd.to_numeric(pd.Series(x).dropna())
                                y_numeric = y.dropna()
                                
                                # Ensure same length
                                if len(x_numeric) == len(y_numeric):
                                    z = np.polyfit(x_numeric, y_numeric, 1)
                                    p = np.poly1d(z)
                                    ax.plot(x, p(x_numeric), "r--", alpha=0.7, label='Trend')
                                    ax.legend()
                        except:
                            pass  # Skip trend line if error
        
        plt.tight_layout()
        plt.savefig(output_dir / 'swarm_metrics_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_entity_coherence_heatmap(self, output_dir: Path):
        """Create heatmap of entity coherence over time."""
        df = self.dataset.entity_metrics.copy()
        
        if df.empty or 'entity_id' not in df.columns:
            print("[Visualizer] No entity data for heatmap")
            return
        
        # Ensure we have timestamp in columns
        if 'timestamp' not in df.columns:
            # If timestamp is the index, reset it
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            else:
                print("[Visualizer] No timestamp available for heatmap")
                return
        
        # Pivot to entity x time matrix
        try:
            pivot_df = df.pivot(index='timestamp', columns='entity_id', values='coherence')
        except Exception as e:
            print(f"[Visualizer] Error pivoting data: {e}")
            return
        
        if pivot_df.empty or len(pivot_df.columns) < 2:
            print("[Visualizer] Insufficient data for heatmap")
            return
            
        plt.figure(figsize=(15, 8))
        
        # Sample if too many timestamps
        if len(pivot_df) > 100:
            pivot_df = pivot_df.iloc[::len(pivot_df)//100]
        
        # Create heatmap
        plt.imshow(pivot_df.T, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Coherence')
        plt.title('Entity Coherence Heatmap Over Time')
        plt.xlabel('Time Points')
        plt.ylabel('Entities')
        
        # Add entity labels
        if len(pivot_df.columns) > 10:
            tick_entities = pivot_df.columns[::max(1, len(pivot_df.columns)//10)]
            plt.yticks(range(len(tick_entities)), tick_entities, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'entity_coherence_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_anomaly_distribution(self, output_dir: Path):
        """Plot anomaly distribution and impact."""
        df = self.dataset.anomaly_events.copy()
        
        if df.empty:
            print("[Visualizer] No anomaly data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Anomaly Analysis', fontsize=16)
        
        # 1. Anomaly types distribution
        if 'event_type' in df.columns:
            type_counts = df['event_type'].value_counts()
            if len(type_counts) > 0:
                axes[0, 0].bar(range(len(type_counts)), type_counts.values)
                axes[0, 0].set_xticks(range(len(type_counts)))
                axes[0, 0].set_xticklabels(type_counts.index, rotation=45, ha='right')
                axes[0, 0].set_title('Anomaly Types Distribution')
                axes[0, 0].set_ylabel('Count')
        
        # 2. Severity distribution
        if 'severity' in df.columns:
            severity_counts = df['severity'].value_counts()
            if len(severity_counts) > 0:
                axes[0, 1].pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%')
                axes[0, 1].set_title('Anomaly Severity Distribution')
        
        # 3. Hourly distribution
        if 'timestamp' in df.columns and not df['timestamp'].isna().all():
            try:
                df['hour'] = df['timestamp'].dt.hour
                hourly_counts = df['hour'].value_counts().sort_index()
                if len(hourly_counts) > 0:
                    axes[1, 0].plot(hourly_counts.index, hourly_counts.values, marker='o')
                    axes[1, 0].set_title('Anomalies by Hour of Day')
                    axes[1, 0].set_xlabel('Hour')
                    axes[1, 0].set_ylabel('Count')
                    axes[1, 0].grid(True, alpha=0.3)
            except:
                pass
        
        # 4. Coherence impact
        if 'coherence_before' in df.columns and 'coherence_after' in df.columns:
            try:
                df['coherence_change'] = df['coherence_after'] - df['coherence_before']
                axes[1, 1].hist(df['coherence_change'].dropna(), bins=20, edgecolor='black')
                axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)
                axes[1, 1].set_title('Coherence Impact Distribution')
                axes[1, 1].set_xlabel('Coherence Change')
                axes[1, 1].set_ylabel('Count')
            except:
                pass
        
        plt.tight_layout()
        plt.savefig(output_dir / 'anomaly_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_matrices(self, output_dir: Path):
        """Plot correlation matrices for swarm and entity metrics."""
        swarm_df = self.dataset.swarm_metrics.copy()
        entity_df = self.dataset.entity_metrics.copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Correlation Matrices', fontsize=16)
        
        # Swarm correlations
        if not swarm_df.empty:
            numeric_cols = swarm_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                try:
                    swarm_corr = swarm_df[numeric_cols].corr()
                    im1 = axes[0].imshow(swarm_corr, cmap='coolwarm', vmin=-1, vmax=1)
                    axes[0].set_title('Swarm Metrics Correlation')
                    axes[0].set_xticks(range(len(numeric_cols)))
                    axes[0].set_yticks(range(len(numeric_cols)))
                    axes[0].set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=8)
                    axes[0].set_yticklabels(numeric_cols, fontsize=8)
                    plt.colorbar(im1, ax=axes[0])
                except Exception as e:
                    axes[0].text(0.5, 0.5, f"Error:\n{str(e)}", 
                               ha='center', va='center', transform=axes[0].transAxes)
                    axes[0].set_title('Swarm Metrics Correlation')
        
        # Entity correlations
        if not entity_df.empty:
            entity_corr_cols = ['coherence', 'pleasure', 'fear', 'love', 'entropy']
            available_cols = [col for col in entity_corr_cols if col in entity_df.columns]
            
            if len(available_cols) > 1:
                try:
                    entity_corr = entity_df[available_cols].corr()
                    im2 = axes[1].imshow(entity_corr, cmap='coolwarm', vmin=-1, vmax=1)
                    axes[1].set_title('Entity Metrics Correlation')
                    axes[1].set_xticks(range(len(available_cols)))
                    axes[1].set_yticks(range(len(available_cols)))
                    axes[1].set_xticklabels(available_cols, rotation=45, ha='right', fontsize=8)
                    axes[1].set_yticklabels(available_cols, fontsize=8)
                    plt.colorbar(im2, ax=axes[1])
                except Exception as e:
                    axes[1].text(0.5, 0.5, f"Error:\n{str(e)}", 
                               ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title('Entity Metrics Correlation')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_emergence_patterns(self, output_dir: Path):
        """Plot emergent event patterns."""
        emergent_events = self.dataset.emergent_events
        
        if not emergent_events:
            print("[Visualizer] No emergent events to plot")
            return
            
        # Create timeline of emergent events
        event_df = pd.DataFrame(emergent_events)
        
        if 'timestamp' not in event_df.columns:
            print("[Visualizer] No timestamp in emergent events")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Group by type and plot
        if 'type' in event_df.columns:
            types = event_df['type'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
            
            for i, event_type in enumerate(types):
                type_events = event_df[event_df['type'] == event_type]
                plt.scatter(type_events['timestamp'], [i] * len(type_events), 
                          color=colors[i], s=100, label=event_type, alpha=0.7)
            
            plt.yticks(range(len(types)), types)
            plt.title('Emergent Events Timeline')
            plt.xlabel('Time')
            plt.ylabel('Event Type')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'emergent_events_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_consciousness_development(self, output_dir: Path):
        """Plot consciousness development across entities."""
        consciousness_events = self.dataset.consciousness_events
        
        if not consciousness_events:
            print("[Visualizer] No consciousness events to plot")
            return
            
        # Create DataFrame
        event_df = pd.DataFrame(consciousness_events)
        
        if 'entity_id' not in event_df.columns or 'timestamp' not in event_df.columns:
            print("[Visualizer] Missing required columns for consciousness plot")
            return
            
        # Count events per entity over time
        entity_event_counts = {}
        
        for _, row in event_df.iterrows():
            entity_id = row['entity_id']
            timestamp = row['timestamp']
            
            if pd.isna(timestamp):
                continue
                
            day = timestamp.date()
            
            if entity_id not in entity_event_counts:
                entity_event_counts[entity_id] = {}
            
            if day not in entity_event_counts[entity_id]:
                entity_event_counts[entity_id][day] = 0
            
            entity_event_counts[entity_id][day] += 1
        
        if not entity_event_counts:
            print("[Visualizer] No valid consciousness event data")
            return
            
        # Create plot
        plt.figure(figsize=(14, 8))
        
        colors = cm.rainbow(np.linspace(0, 1, min(20, len(entity_event_counts))))
        
        for i, (entity_id, day_counts) in enumerate(list(entity_event_counts.items())[:20]):  # Limit to top 20
            days = sorted(day_counts.keys())
            counts = [day_counts[day] for day in days]
            
            # Convert dates to numeric for plotting
            day_numeric = [(day - min(days)).days for day in days]
            
            plt.plot(day_numeric, counts, marker='o', color=colors[i], label=entity_id, linewidth=2, alpha=0.7)
        
        plt.title('Consciousness Event Development by Entity')
        plt.xlabel('Days Since First Event')
        plt.ylabel('Number of Consciousness Events')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'consciousness_development.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_quantum_activity_network(self, output_dir: Path):
        """Create network graph of quantum activity between entities."""
        quantum_events = self.dataset.quantum_events
        
        if not quantum_events:
            print("[Visualizer] No quantum events to plot")
            return
            
        # Extract entity interactions from quantum events
        interactions = {}
        
        for event in quantum_events:
            entity_id = event.get('entity_id', 'unknown')
            message = event.get('message', '')
            
            # Look for mentions of other entities in message
            if 'to' in message.lower() or 'with' in message.lower():
                # Try to extract target entity
                words = message.split()
                for i, word in enumerate(words):
                    if word.lower() in ['to', 'with'] and i + 1 < len(words):
                        target = words[i + 1]
                        if target.startswith('god_ent_'):
                            key = (entity_id, target)
                            if key not in interactions:
                                interactions[key] = 0
                            interactions[key] += 1
        
        if not interactions:
            print("[Visualizer] No quantum interactions found")
            return
            
        # Create network graph
        G = nx.Graph()
        
        for (source, target), weight in interactions.items():
            G.add_edge(source, target, weight=weight)
        
        if len(G.nodes()) == 0:
            print("[Visualizer] Empty quantum network graph")
            return
            
        # Draw graph
        plt.figure(figsize=(12, 10))
        
        pos = nx.spring_layout(G, seed=42)
        
        # Node size based on degree
        node_sizes = [300 * G.degree(node) for node in G.nodes()]
        
        # Edge width based on weight
        edge_widths = [0.5 + G[u][v]['weight'] for u, v in G.edges()]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                              alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title('Quantum Activity Interaction Network')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quantum_activity_network.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_behavior_clusters(self, output_dir: Path):
        """Plot entity behavior clusters if available."""
        if 'entity_analysis' not in self.statistics:
            print("[Visualizer] No entity analysis data for behavior clusters")
            return
            
        entity_stats = self.statistics['entity_analysis'].get('entity_statistics', {})
        
        if not entity_stats:
            print("[Visualizer] No entity statistics for behavior clusters")
            return
            
        # Extract features for clustering visualization
        features = []
        entity_ids = []
        
        for entity_id, stats in entity_stats.items():
            features.append([
                stats.get('coherence_mean', 0.5),
                stats.get('pleasure_mean', 0.5),
                stats.get('fear_mean', 0.5),
                stats.get('love_mean', 0.5),
                stats.get('entropy_mean', 0.5)
            ])
            entity_ids.append(entity_id)
        
        if len(features) < 3:
            print("[Visualizer] Not enough entities for clustering")
            return
            
        features = np.array(features)
        
        # Perform PCA for 2D visualization
        from sklearn.decomposition import PCA
        
        try:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            # Perform clustering for visualization
            from sklearn.cluster import KMeans
            
            n_clusters = min(5, len(features))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            plt.figure(figsize=(10, 8))
            
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=cluster_labels, cmap='tab10', s=100, alpha=0.7)
            
            # Add entity labels for some points
            for i, entity_id in enumerate(entity_ids):
                if i % max(1, len(entity_ids)//20) == 0:  # Label every ~5%
                    plt.annotate(entity_id[-4:], (features_2d[i, 0], features_2d[i, 1]), 
                               fontsize=8, alpha=0.7)
            
            plt.colorbar(scatter, label='Cluster')
            plt.title('Entity Behavior Clusters (PCA Projection)')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'behavior_clusters.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[WARNING] Error plotting behavior clusters: {e}")

# ============================================================
# 📝 REPORT GENERATOR
# ============================================================

class ScientificReportGenerator:
    """Generate comprehensive scientific reports from analysis."""
    
    def __init__(self, dataset: SwarmDataset, statistics: Dict[str, Any], 
                 output_dir: str = "./analysis_output"):
        self.dataset = dataset
        self.statistics = statistics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_html_report(self) -> str:
        """Generate HTML report with all analysis results."""
        print("[Report] Generating HTML report...")
        
        # Calculate summary statistics
        summary = self._generate_summary_statistics()
        
        # Create HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>SAMSARA SWARM - Scientific Analysis Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                }
                .section {
                    background: white;
                    padding: 25px;
                    margin-bottom: 25px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .section-title {
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                    margin-top: 0;
                }
                .metric-card {
                    background: #f8f9fa;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 4px;
                }
                .stat-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }
                .stat-item {
                    background: #e8f4fc;
                    padding: 15px;
                    border-radius: 6px;
                    text-align: center;
                }
                .stat-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #2980b9;
                }
                .stat-label {
                    font-size: 14px;
                    color: #7f8c8d;
                    margin-top: 5px;
                }
                .plot {
                    text-align: center;
                    margin: 20px 0;
                }
                .plot img {
                    max-width: 100%;
                    border-radius: 6px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th {
                    background-color: #3498db;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }
                td {
                    padding: 10px;
                    border-bottom: 1px solid #ddd;
                }
                tr:hover {
                    background-color: #f5f5f5;
                }
                .insight {
                    background: #fffde7;
                    border-left: 4px solid #fdd835;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 4px;
                }
                .insight h4 {
                    color: #f57c00;
                    margin-top: 0;
                }
                .timestamp {
                    color: #95a5a6;
                    font-size: 12px;
                    float: right;
                }
                .warning {
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌌 SAMSARA SWARM - Scientific Analysis Report</h1>
                <p>Comprehensive Analysis of Quantum Consciousness Simulation</p>
                <div class="timestamp">Generated: {{ generation_time }}</div>
            </div>
            
            <!-- Executive Summary -->
            <div class="section">
                <h2 class="section-title">📊 Executive Summary</h2>
                {% if summary.overview %}
                <div class="stat-grid">
                    {% for stat in summary.overview %}
                    <div class="stat-item">
                        <div class="stat-value">{{ stat.value }}</div>
                        <div class="stat-label">{{ stat.label }}</div>
                    </div>
                    {% endfor %}
                </div>
                {% else %}
                <div class="warning">
                    <h4>⚠️ No Data Available</h4>
                    <p>No swarm metrics data was found in the logs. Please run the simulation to generate data.</p>
                </div>
                {% endif %}
                
                {% if summary.insights %}
                <div class="insight">
                    <h4>🔍 Key Insights</h4>
                    <ul>
                        {% for insight in summary.insights %}
                        <li>{{ insight }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
            </div>
            
            <!-- Swarm Dynamics -->
            {% if dataset.swarm_metrics is not none and not dataset.swarm_metrics.empty %}
            <div class="section">
                <h2 class="section-title">🌀 Swarm Dynamics Analysis</h2>
                <div class="plot">
                    <img src="plots/swarm_metrics_timeline.png" alt="Swarm Metrics Timeline">
                </div>
                
                {% if summary.trends %}
                <h3>Temporal Trends</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Trend</th>
                        <th>Strength</th>
                        <th>R²</th>
                    </tr>
                    {% for trend in summary.trends %}
                    <tr>
                        <td>{{ trend.metric }}</td>
                        <td>{{ trend.direction }}</td>
                        <td>{{ trend.strength }}</td>
                        <td>{{ trend.r_squared }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
            {% endif %}
            
            <!-- Entity Analysis -->
            {% if dataset.entity_metrics is not none and not dataset.entity_metrics.empty %}
            <div class="section">
                <h2 class="section-title">👥 Entity Behavior Analysis</h2>
                <div class="plot">
                    <img src="plots/behavior_clusters.png" alt="Behavior Clusters">
                </div>
                
                {% if summary.top_entities %}
                <h3>Top Performing Entities</h3>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Entity ID</th>
                        <th>Coherence</th>
                        <th>Pleasure</th>
                        <th>Leadership Score</th>
                    </tr>
                    {% for entity in summary.top_entities %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ entity.id }}</td>
                        <td>{{ entity.coherence }}</td>
                        <td>{{ entity.pleasure }}</td>
                        <td>{{ entity.score }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
            {% endif %}
            
            <!-- Anomaly Analysis -->
            {% if dataset.anomaly_events is not none and not dataset.anomaly_events.empty %}
            <div class="section">
                <h2 class="section-title">⚠️ Anomaly Analysis</h2>
                <div class="plot">
                    <img src="plots/anomaly_distribution.png" alt="Anomaly Distribution">
                </div>
                
                {% if summary.anomaly_stats %}
                <div class="stat-grid">
                    {% for stat in summary.anomaly_stats %}
                    <div class="stat-item">
                        <div class="stat-value">{{ stat.value }}</div>
                        <div class="stat-label">{{ stat.label }}</div>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            {% endif %}
            
            <!-- Consciousness Metrics -->
            {% if dataset.consciousness_events %}
            <div class="section">
                <h2 class="section-title">🧠 Consciousness Development</h2>
                <div class="plot">
                    <img src="plots/consciousness_development.png" alt="Consciousness Development">
                </div>
                
                {% if summary.conscious_entities %}
                <h3>Most Conscious Entities</h3>
                <table>
                    <tr>
                        <th>Entity ID</th>
                        <th>Maturity Score</th>
                        <th>Event Count</th>
                        <th>Type Diversity</th>
                    </tr>
                    {% for entity in summary.conscious_entities %}
                    <tr>
                        <td>{{ entity.id }}</td>
                        <td>{{ entity.score }}</td>
                        <td>{{ entity.events }}</td>
                        <td>{{ entity.diversity }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
            {% endif %}
            
            <!-- Quantum Activity -->
            {% if dataset.quantum_events %}
            <div class="section">
                <h2 class="section-title">⚛️ Quantum Activity Analysis</h2>
                <div class="plot">
                    <img src="plots/quantum_activity_network.png" alt="Quantum Activity Network">
                </div>
                
                {% if summary.quantum_events %}
                <div class="metric-card">
                    <strong>Quantum Event Distribution:</strong><br>
                    {% for event_type, count in summary.quantum_events %}
                    • {{ event_type }}: {{ count }} events<br>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            {% endif %}
            
            <!-- Recommendations -->
            <div class="section">
                <h2 class="section-title">🎯 Scientific Recommendations</h2>
                {% if summary.recommendations %}
                <div class="insight">
                    <h4>🧪 Experimental Improvements</h4>
                    <ul>
                        {% for rec in summary.recommendations %}
                        <li>{{ rec }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
                
                <div class="metric-card">
                    <strong>Next Research Directions:</strong><br>
                    1. Investigate coherence collapse mechanisms<br>
                    2. Study emergent consciousness patterns<br>
                    3. Analyze quantum entanglement effects on emotion<br>
                    4. Develop predictive models for swarm behavior
                </div>
            </div>
            
            <div class="section">
                <p style="text-align: center; color: #7f8c8d;">
                    Generated by SAMSARA SWARM Analyzer v1.0<br>
                    Analysis completed at: {{ generation_time }}
                </p>
            </div>
        </body>
        </html>
        """
        
        # Render template with data
        template = Template(html_template)
        html_content = template.render(
            dataset=self.dataset,
            summary=summary,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Save HTML file
        html_path = self.output_dir / "scientific_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[Report] HTML report saved to {html_path}")
        return str(html_path)
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for report."""
        swarm_df = self.dataset.swarm_metrics
        entity_df = self.dataset.entity_metrics
        anomaly_df = self.dataset.anomaly_events
        
        summary = {
            'overview': [],
            'insights': [],
            'trends': [],
            'top_entities': [],
            'anomaly_stats': [],
            'conscious_entities': [],
            'quantum_events': [],
            'recommendations': []
        }
        
        # Overview statistics
        if not swarm_df.empty:
            try:
                if 'timestamp' in swarm_df.columns:
                    duration = (swarm_df['timestamp'].max() - swarm_df['timestamp'].min())
                    duration_str = f"{duration.total_seconds()/3600:.1f}h"
                else:
                    duration_str = "N/A"
                
                summary['overview'].extend([
                    {'value': len(swarm_df), 'label': 'Time Points'},
                    {'value': duration_str, 'label': 'Duration'},
                    {'value': entity_df['entity_id'].nunique() if not entity_df.empty and 'entity_id' in entity_df.columns else 0, 'label': 'Unique Entities'},
                    {'value': f"{swarm_df['avg_coherence'].mean():.3f}" if 'avg_coherence' in swarm_df.columns else 'N/A', 'label': 'Avg Coherence'},
                    {'value': len(anomaly_df), 'label': 'Total Anomalies'},
                    {'value': len(self.dataset.emergent_events), 'label': 'Emergent Events'}
                ])
            except Exception as e:
                print(f"[WARNING] Error generating overview statistics: {e}")
        
        # Extract trends from statistics
        if 'temporal_analysis' in self.statistics:
            for key, analysis in self.statistics['temporal_analysis'].items():
                if 'trend_direction' in analysis:
                    metric_name = key.split('_')[0] if '_' in key else key
                    summary['trends'].append({
                        'metric': metric_name,
                        'direction': analysis['trend_direction'],
                        'strength': f"{analysis.get('trend_strength', 0):.2f}",
                        'r_squared': f"{analysis.get('r_squared', 0):.3f}"
                    })
        
        # Top entities
        if 'entity_analysis' in self.statistics:
            entity_stats = self.statistics['entity_analysis'].get('entity_statistics', {})
            if entity_stats:
                sorted_entities = sorted(entity_stats.items(), 
                                       key=lambda x: x[1].get('coherence_mean', 0), 
                                       reverse=True)[:10]
                for entity_id, stats in sorted_entities:
                    summary['top_entities'].append({
                        'id': entity_id,
                        'coherence': f"{stats.get('coherence_mean', 0):.3f}",
                        'pleasure': f"{stats.get('pleasure_mean', 0):.3f}",
                        'score': f"{stats.get('coherence_mean', 0) * 0.6 + stats.get('pleasure_mean', 0) * 0.4:.3f}"
                    })
        
        # Anomaly statistics
        if not anomaly_df.empty:
            summary['anomaly_stats'].extend([
                {'value': len(anomaly_df), 'label': 'Total Anomalies'},
                {'value': anomaly_df['event_type'].nunique() if 'event_type' in anomaly_df.columns else 0, 'label': 'Anomaly Types'},
                {'value': anomaly_df['severity'].value_counts().get('HIGH', 0) if 'severity' in anomaly_df.columns else 0, 'label': 'High Severity'},
                {'value': f"{anomaly_df['coherence_change'].mean():.3f}" if 'coherence_change' in anomaly_df.columns else 'N/A', 
                 'label': 'Avg Coherence Impact'}
            ])
        
        # Consciousness entities
        if 'consciousness_analysis' in self.statistics:
            consciousness_data = self.statistics['consciousness_analysis'].get('consciousness_maturity', [])
            for entity_data in consciousness_data[:5]:
                summary['conscious_entities'].append({
                    'id': entity_data.get('entity_id', 'unknown'),
                    'score': f"{entity_data.get('maturity_score', 0):.3f}",
                    'events': entity_data.get('event_count', 0),
                    'diversity': entity_data.get('type_diversity', 0)
                })
        
        # Quantum events
        if 'quantum_analysis' in self.statistics:
            quantum_dist = self.statistics['quantum_analysis'].get('event_type_distribution', {})
            summary['quantum_events'] = list(quantum_dist.items())[:10]
        
        # Generate insights
        summary['insights'] = self._generate_insights()
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations()
        
        return summary
    
    def _generate_insights(self) -> List[str]:
        """Generate key insights from analysis."""
        insights = []
        
        # Analyze coherence trends
        if 'temporal_analysis' in self.statistics:
            for key, analysis in self.statistics['temporal_analysis'].items():
                if 'avg_coherence' in key and 'trend_direction' in analysis:
                    if analysis['trend_direction'] == 'increasing' and analysis.get('r_squared', 0) > 0.5:
                        insights.append("Swarm coherence shows significant positive trend over time")
                    elif analysis['trend_direction'] == 'decreasing' and analysis.get('r_squared', 0) > 0.5:
                        insights.append("Swarm coherence shows concerning negative trend")
        
        # Analyze anomaly patterns
        if 'anomaly_analysis' in self.statistics:
            anomaly_stats = self.statistics['anomaly_analysis']
            if anomaly_stats.get('total_anomalies', 0) > 50:
                insights.append("High anomaly frequency suggests complex system dynamics")
            
            if 'coherence_impact' in anomaly_stats:
                impact = anomaly_stats['coherence_impact']
                if impact.get('mean_change', 0) > 0:
                    insights.append("Anomalies tend to positively impact system coherence")
                elif impact.get('mean_change', 0) < 0:
                    insights.append("Anomalies tend to negatively impact system coherence")
        
        # Analyze emergent behavior
        if 'emergence_analysis' in self.statistics:
            emergence_stats = self.statistics['emergence_analysis']
            if any('consciousness' in key for key in emergence_stats.keys()):
                insights.append("Consciousness emergence patterns detected in swarm behavior")
            
            if any('coherence' in key for key in emergence_stats.keys()):
                insights.append("Coherence-based emergent phenomena observed")
        
        # Add entity behavior insights
        if 'entity_analysis' in self.statistics:
            entity_stats = self.statistics['entity_analysis']
            if 'behavior_clusters' in entity_stats:
                clusters = entity_stats['behavior_clusters']
                if len(clusters) >= 3:
                    insights.append(f"Entities cluster into {len(clusters)} distinct behavioral groups")
        
        # Add basic insight if no others
        if not insights and not self.dataset.swarm_metrics.empty:
            insights.append("Swarm simulation data successfully collected and analyzed")
        
        return insights[:5]  # Return top 5 insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate scientific recommendations based on analysis."""
        recommendations = []
        
        # Recommendations based on coherence
        if 'temporal_analysis' in self.statistics:
            coherence_trends = [v for k, v in self.statistics['temporal_analysis'].items() 
                              if 'avg_coherence' in k and 'trend_direction' in v]
            
            for trend in coherence_trends:
                if trend.get('trend_direction') == 'decreasing' and trend.get('r_squared', 0) > 0.3:
                    recommendations.append("Implement coherence stabilization protocols")
                elif trend.get('trend_direction') == 'increasing' and trend.get('r_squared', 0) > 0.7:
                    recommendations.append("Investigate coherence amplification mechanisms")
        
        # Recommendations based on anomalies
        if 'anomaly_analysis' in self.statistics:
            anomaly_stats = self.statistics['anomaly_analysis']
            if anomaly_stats.get('total_anomalies', 0) > 100:
                recommendations.append("Increase monitoring frequency for anomaly detection")
            
            if 'hourly_distribution' in anomaly_stats:
                # Check if anomalies cluster in specific hours
                hourly = anomaly_stats['hourly_distribution']
                if len(hourly) > 0:
                    max_hour = max(hourly, key=hourly.get)
                    recommendations.append(f"Focus monitoring during hour {max_hour} for anomaly prevention")
        
        # Recommendations based on entity behavior
        if 'entity_analysis' in self.statistics:
            entity_stats = self.statistics['entity_analysis']
            if 'leader_entities' in entity_stats and len(entity_stats['leader_entities']) > 0:
                recommendations.append("Study leader entities for optimal behavior patterns")
        
        # Default recommendations if no specific ones
        if not recommendations:
            recommendations = [
                "Increase simulation duration for more robust statistics",
                "Add more entities to study swarm dynamics",
                "Implement additional metrics for consciousness detection",
                "Study quantum entanglement effects in more detail"
            ]
        
        return recommendations[:5]  # Return top 5 recommendations

# ============================================================
# 🚀 MAIN ANALYSIS PIPELINE
# ============================================================

class SwarmAnalyzer:
    """Main analysis pipeline orchestrator."""
    
    def __init__(self, log_dir: str = "./logs", output_dir: str = "./analysis_output"):
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.consolidator = None
        self.dataset = None
        self.statistics = None
        self.visualizer = None
        self.report_generator = None
        
    def run_full_analysis(self):
        """Execute complete analysis pipeline."""
        print("=" * 70)
        print("🔬 SAMSARA SWARM SCIENTIFIC ANALYZER")
        print("=" * 70)
        
        # Step 1: Load and consolidate data
        print("\n[1/4] Loading and consolidating log data...")
        self.consolidator = LogConsolidator(self.log_dir)
        self.dataset = self.consolidator.load_all_data()
        
        # Check if we have data
        if (self.dataset.swarm_metrics.empty and 
            self.dataset.entity_metrics.empty and 
            self.dataset.anomaly_events.empty):
            print("\n⚠️ WARNING: No data found in log files!")
            print("Please run the Samsara Swarm simulation first to generate data.")
            print(f"Expected log files in: {self.log_dir}")
            return
        
        # Step 2: Statistical analysis
        print("\n[2/4] Running statistical analysis...")
        stats_engine = SwarmStatistics(self.dataset)
        self.statistics = stats_engine.analyze_all()
        
        # Save statistics to JSON
        stats_path = self.output_dir / "statistical_analysis.json"
        try:
            # Convert numpy arrays and pandas objects to Python types
            import json
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    if isinstance(obj, pd.Timestamp):
                        return obj.isoformat()
                    if isinstance(obj, pd.DataFrame):
                        return obj.to_dict()
                    if isinstance(obj, pd.Series):
                        return obj.to_dict()
                    return super(NumpyEncoder, self).default(obj)
            
            with open(stats_path, 'w') as f:
                json.dump(self.statistics, f, indent=2, cls=NumpyEncoder)
            
            print(f"  ✓ Statistics saved to: {stats_path}")
        except Exception as e:
            print(f"[WARNING] Error saving statistics: {e}")
        
        # Step 3: Generate visualizations
        print("\n[3/4] Generating visualizations...")
        self.visualizer = SwarmVisualizer(self.dataset, self.statistics, self.output_dir)
        self.visualizer.generate_all_visualizations()
        
        # Step 4: Generate reports
        print("\n[4/4] Generating scientific reports...")
        self.report_generator = ScientificReportGenerator(self.dataset, self.statistics, self.output_dir)
        try:
            html_report_path = self.report_generator.generate_html_report()
        except Exception as e:
            print(f"[ERROR] Error generating report: {e}")
            html_report_path = None
        
        # Step 5: Summary
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        if html_report_path:
            print(f"Main report: {html_report_path}")
        print(f"Statistics: {self.output_dir / 'statistical_analysis.json'}")
        print(f"Visualizations: {self.output_dir / 'plots'}")
        print("\n📊 Analysis Summary:")
        
        # Print key findings
        self._print_summary_findings()
        
        print("\n🌌 Use this analysis to:")
        print("  1. Understand swarm dynamics and patterns")
        print("  2. Identify anomalies and emergent behaviors")
        print("  3. Optimize future experiments")
        print("  4. Guide consciousness research directions")
    
    def _print_summary_findings(self):
        """Print key findings from analysis."""
        if not self.statistics:
            print("  • No statistical analysis results available")
            return
        
        # Swarm coherence
        if 'temporal_analysis' in self.statistics:
            coherence_trends = [v for k, v in self.statistics['temporal_analysis'].items() 
                              if 'avg_coherence' in k and 'trend_direction' in v]
            if coherence_trends:
                trend = coherence_trends[0]
                print(f"  • Swarm coherence trend: {trend.get('trend_direction', 'stable')} "
                      f"(strength: {trend.get('trend_strength', 0):.2f})")
        
        # Entity analysis
        if 'entity_analysis' in self.statistics:
            entity_stats = self.statistics['entity_analysis']
            entity_count = len(entity_stats.get('entity_statistics', {}))
            if entity_count > 0:
                print(f"  • Unique entities analyzed: {entity_count}")
            
            if 'behavior_clusters' in entity_stats:
                clusters = entity_stats['behavior_clusters']
                print(f"  • Entity behavior clusters identified: {len(clusters)}")
        
        # Anomaly analysis
        if 'anomaly_analysis' in self.statistics:
            anomaly_stats = self.statistics['anomaly_analysis']
            total_anomalies = anomaly_stats.get('total_anomalies', 0)
            if total_anomalies > 0:
                print(f"  • Total anomalies detected: {total_anomalies}")
                if 'anomalies_by_type' in anomaly_stats:
                    print(f"  • Anomaly types: {len(anomaly_stats['anomalies_by_type'])}")
        
        # Emergent behavior
        if 'emergence_analysis' in self.statistics:
            emergence_stats = self.statistics['emergence_analysis']
            if emergence_stats:
                print(f"  • Emergent event types: {len(emergence_stats)}")
        
        # Consciousness development
        if 'consciousness_analysis' in self.statistics:
            consciousness_stats = self.statistics['consciousness_analysis']
            if 'consciousness_maturity' in consciousness_stats:
                mature_entities = len([e for e in consciousness_stats['consciousness_maturity'] 
                                     if e.get('maturity_score', 0) > 0.5])
                if mature_entities > 0:
                    print(f"  • Entities showing consciousness development: {mature_entities}")

# ============================================================
# 🚀 MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Samsara Swarm Log Analyzer")
    parser.add_argument("--log-dir", type=str, default="./logs",
                       help="Directory containing log files (default: ./logs)")
    parser.add_argument("--output-dir", type=str, default="./analysis_output",
                       help="Output directory for analysis results (default: ./analysis_output)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick analysis without visualizations")
    
    args = parser.parse_args()
    
    # Create and run analyzer
    analyzer = SwarmAnalyzer(log_dir=args.log_dir, output_dir=args.output_dir)
    
    try:
        analyzer.run_full_analysis()
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()