#!/usr/bin/env python3
"""
Advanced Analytics Engine
Comprehensive behavior tracking, impact measurement, and predictive analytics
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import statistics
from enum import Enum
import hashlib
import math


class MetricType(Enum):
    ENGAGEMENT = "engagement"
    DECISION_QUALITY = "decision_quality"
    USER_SATISFACTION = "user_satisfaction"
    COMPLETION_RATE = "completion_rate"
    SUCCESS_RATE = "success_rate"
    TIME_TO_DECISION = "time_to_decision"
    CONFIDENCE_LEVEL = "confidence_level"
    PATTERN_CONSISTENCY = "pattern_consistency"


class EventType(Enum):
    DECISION_CREATED = "decision_created"
    DECISION_COMPLETED = "decision_completed"
    COMPLIMENT_REQUESTED = "compliment_requested"
    COMPLIMENT_RATED = "compliment_rated"
    PAGE_VIEW = "page_view"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    FEATURE_USED = "feature_used"
    OUTCOME_REPORTED = "outcome_reported"
    FEEDBACK_GIVEN = "feedback_given"


class SegmentType(Enum):
    NEW_USER = "new_user"
    ACTIVE_USER = "active_user"
    POWER_USER = "power_user"
    CHURNED_USER = "churned_user"
    HIGH_VALUE = "high_value"
    LOW_ENGAGEMENT = "low_engagement"


@dataclass
class AnalyticsEvent:
    id: str
    user_id: str
    session_id: str
    event_type: EventType
    timestamp: datetime
    properties: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False


@dataclass
class UserSegment:
    user_id: str
    segment_type: SegmentType
    confidence: float
    characteristics: Dict[str, Any]
    assigned_at: datetime
    expires_at: Optional[datetime] = None


@dataclass
class ConversionFunnel:
    name: str
    steps: List[str]
    conversion_rates: List[float]
    drop_off_points: List[str]
    total_users: int
    completed_users: int
    avg_time_to_complete: float


@dataclass
class ABTestVariant:
    variant_id: str
    name: str
    description: str
    traffic_allocation: float
    conversion_rate: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    statistical_significance: float


class AdvancedAnalyticsEngine:
    """Comprehensive analytics engine with ML-powered insights"""

    def __init__(self):
        self.events: List[AnalyticsEvent] = []
        self.user_segments: Dict[str, UserSegment] = {}
        self.metrics_cache: Dict[str, Any] = {}
        self.conversion_funnels: Dict[str, ConversionFunnel] = {}
        self.ab_tests: Dict[str, List[ABTestVariant]] = {}
        self.predictive_models: Dict[str, Any] = {}
        self.behavioral_patterns: Dict[str, Any] = {}
        self._initialize_analytics_framework()

    def _initialize_analytics_framework(self):
        """Initialize analytics framework and models"""
        self.predictive_models = {
            "churn_predictor": self._create_churn_model(),
            "engagement_scorer": self._create_engagement_model(),
            "value_predictor": self._create_value_model(),
            "satisfaction_model": self._create_satisfaction_model(),
            "conversion_optimizer": self._create_conversion_model()
        }

        self._setup_conversion_funnels()
        self._initialize_ab_tests()

    def _create_churn_model(self):
        """Create churn prediction model"""
        return {
            "features": [
                "days_since_last_visit",
                "total_decisions",
                "completion_rate",
                "avg_session_duration",
                "feature_adoption_rate",
                "satisfaction_score",
                "support_interactions"
            ],
            "thresholds": {
                "high_risk": 0.7,
                "medium_risk": 0.4,
                "low_risk": 0.2
            },
            "weights": {
                "days_since_last_visit": -0.3,
                "total_decisions": 0.2,
                "completion_rate": 0.25,
                "avg_session_duration": 0.15,
                "feature_adoption_rate": 0.1,
                "satisfaction_score": 0.2,
                "support_interactions": -0.1
            }
        }

    def _create_engagement_model(self):
        """Create user engagement scoring model"""
        return {
            "dimensions": {
                "frequency": 0.3,    # How often they use the app
                "depth": 0.25,       # How many features they use
                "duration": 0.2,     # How long they spend
                "recency": 0.15,     # How recently they were active
                "quality": 0.1       # Quality of interactions
            },
            "scoring_functions": {
                "frequency": lambda x: min(1.0, x / 10),  # Daily usage = 1.0
                "depth": lambda x: min(1.0, x / 8),       # All features = 1.0
                "duration": lambda x: min(1.0, x / 3600),  # 1 hour = 1.0
                "recency": lambda x: max(0, 1 - x / 30),  # 30 days ago = 0
                "quality": lambda x: x                     # Direct score 0-1
            }
        }

    def _create_value_model(self):
        """Create user lifetime value prediction model"""
        return {
            "value_indicators": [
                "premium_features_used",
                "decisions_completed",
                "positive_outcomes",
                "referrals_made",
                "feedback_provided",
                "session_consistency",
                "feature_exploration"
            ],
            "value_weights": {
                "premium_features_used": 0.25,
                "decisions_completed": 0.2,
                "positive_outcomes": 0.15,
                "referrals_made": 0.15,
                "feedback_provided": 0.1,
                "session_consistency": 0.1,
                "feature_exploration": 0.05
            }
        }

    def _create_satisfaction_model(self):
        """Create user satisfaction prediction model"""
        return {
            "satisfaction_factors": {
                "decision_success_rate": 0.3,
                "response_time": 0.2,
                "feature_availability": 0.15,
                "ease_of_use": 0.15,
                "personalization": 0.1,
                "support_quality": 0.1
            },
            "satisfaction_thresholds": {
                "highly_satisfied": 0.8,
                "satisfied": 0.6,
                "neutral": 0.4,
                "dissatisfied": 0.2
            }
        }

    def _create_conversion_model(self):
        """Create conversion optimization model"""
        return {
            "conversion_factors": [
                "time_on_landing_page",
                "features_explored",
                "decisions_started",
                "compliments_received",
                "social_proof_seen",
                "onboarding_completed"
            ],
            "optimization_rules": {
                "low_time_on_page": "show_value_proposition",
                "high_bounce_rate": "improve_first_impression",
                "low_feature_adoption": "guided_onboarding",
                "abandoned_decisions": "decision_assistance"
            }
        }

    def _setup_conversion_funnels(self):
        """Setup key conversion funnels for tracking"""
        self.conversion_funnels = {
            "user_onboarding": ConversionFunnel(
                name="User Onboarding",
                steps=[
                    "landing_page",
                    "signup",
                    "first_decision",
                    "decision_completion",
                    "return_visit"],
                conversion_rates=[],
                drop_off_points=[],
                total_users=0,
                completed_users=0,
                avg_time_to_complete=0),
            "decision_completion": ConversionFunnel(
                name="Decision Completion",
                steps=[
                    "decision_start",
                    "options_added",
                    "analysis_viewed",
                    "decision_made",
                    "outcome_reported"],
                conversion_rates=[],
                drop_off_points=[],
                total_users=0,
                completed_users=0,
                avg_time_to_complete=0),
            "feature_adoption": ConversionFunnel(
                name="Feature Adoption",
                steps=[
                    "feature_discovery",
                    "feature_trial",
                    "feature_use",
                    "feature_mastery",
                    "feature_advocacy"],
                conversion_rates=[],
                drop_off_points=[],
                total_users=0,
                completed_users=0,
                avg_time_to_complete=0)}

    def _initialize_ab_tests(self):
        """Initialize A/B test framework"""
        self.ab_tests = {
            "compliment_timing": [
                ABTestVariant("control", "Standard Timing", "Compliments shown after decisions", 0.5, 0.0, (0, 0), 0, 0),
                ABTestVariant("immediate", "Immediate Compliments", "Compliments shown immediately", 0.5, 0.0, (0, 0), 0, 0)
            ],
            "decision_ui": [
                ABTestVariant("classic", "Classic Interface", "Standard decision interface", 0.33, 0.0, (0, 0), 0, 0),
                ABTestVariant("guided", "Guided Interface", "Step-by-step guided interface", 0.33, 0.0, (0, 0), 0, 0),
                ABTestVariant("minimal", "Minimal Interface", "Simplified minimal interface", 0.34, 0.0, (0, 0), 0, 0)
            ]
        }

    def track_event(self, user_id: str, session_id: str, event_type: EventType,
                    properties: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Track analytics event with enrichment"""
        event_id = self._generate_event_id(user_id, event_type, datetime.now())

        # Enrich event with additional context
        enriched_properties = self._enrich_event_properties(user_id, event_type, properties)
        enriched_metadata = self._enrich_event_metadata(user_id, session_id, metadata or {})

        event = AnalyticsEvent(
            id=event_id,
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            timestamp=datetime.now(),
            properties=enriched_properties,
            metadata=enriched_metadata
        )

        self.events.append(event)
        self._process_event_realtime(event)
        self._update_user_segment(user_id)

        return event_id

    def _enrich_event_properties(
            self,
            user_id: str,
            event_type: EventType,
            properties: Dict) -> Dict:
        """Enrich event properties with context"""
        enriched = properties.copy()

        # Add user context
        user_segment = self.user_segments.get(user_id)
        if user_segment:
            enriched["user_segment"] = user_segment.segment_type.value
            enriched["segment_confidence"] = user_segment.confidence

        # Add temporal context
        now = datetime.now()
        enriched["hour_of_day"] = now.hour
        enriched["day_of_week"] = now.strftime("%A")
        enriched["is_weekend"] = now.weekday() >= 5
        enriched["month"] = now.month
        enriched["quarter"] = (now.month - 1) // 3 + 1

        # Add behavioral context
        recent_events = self._get_recent_events(user_id, hours=24)
        enriched["events_last_24h"] = len(recent_events)
        enriched["unique_sessions_last_24h"] = len(set(e.session_id for e in recent_events))

        return enriched

    def _enrich_event_metadata(self, user_id: str, session_id: str, metadata: Dict) -> Dict:
        """Enrich event metadata"""
        enriched = metadata.copy()

        # Add session context
        session_events = [e for e in self.events if e.session_id == session_id]
        enriched["session_event_count"] = len(session_events)

        if session_events:
            session_start = min(e.timestamp for e in session_events)
            enriched["session_duration"] = (datetime.now() - session_start).total_seconds()

        # Add user lifecycle stage
        user_events = [e for e in self.events if e.user_id == user_id]
        enriched["total_user_events"] = len(user_events)

        if user_events:
            first_event = min(e.timestamp for e in user_events)
            enriched["days_since_first_event"] = (datetime.now() - first_event).days

        return enriched

    def _process_event_realtime(self, event: AnalyticsEvent):
        """Process event for real-time insights"""
        # Update conversion funnels
        self._update_conversion_funnels(event)

        # Update A/B test metrics
        self._update_ab_test_metrics(event)

        # Trigger real-time alerts if needed
        self._check_realtime_alerts(event)

        # Update behavioral patterns
        self._update_behavioral_patterns(event)

    def _update_conversion_funnels(self, event: AnalyticsEvent):
        """Update conversion funnel metrics"""
        for funnel_name, funnel in self.conversion_funnels.items():
            if self._event_matches_funnel_step(event, funnel):
                # Logic to update funnel metrics would go here
                pass

    def _event_matches_funnel_step(self, event: AnalyticsEvent, funnel: ConversionFunnel) -> bool:
        """Check if event matches a funnel step"""
        # Implement funnel step matching logic
        event_mapping = {
            "landing_page": [EventType.PAGE_VIEW],
            "signup": [EventType.SESSION_START],
            "first_decision": [EventType.DECISION_CREATED],
            "decision_completion": [EventType.DECISION_COMPLETED],
            "return_visit": [EventType.SESSION_START]
        }

        for step in funnel.steps:
            if event.event_type in event_mapping.get(step, []):
                return True
        return False

    def _update_ab_test_metrics(self, event: AnalyticsEvent):
        """Update A/B test conversion metrics"""
        # Get user's test variants
        user_variants = event.metadata.get("ab_test_variants", {})

        for test_name, variant_id in user_variants.items():
            if test_name in self.ab_tests:
                variant = next(
                    (v for v in self.ab_tests[test_name] if v.variant_id == variant_id), None)
                if variant and self._is_conversion_event(event, test_name):
                    # Update variant metrics
                    variant.sample_size += 1
                    # Conversion logic would be implemented here

    def _is_conversion_event(self, event: AnalyticsEvent, test_name: str) -> bool:
        """Check if event represents a conversion for the test"""
        conversion_events = {
            "compliment_timing": [EventType.COMPLIMENT_RATED],
            "decision_ui": [EventType.DECISION_COMPLETED]
        }

        return event.event_type in conversion_events.get(test_name, [])

    def _check_realtime_alerts(self, event: AnalyticsEvent):
        """Check for real-time alert conditions"""
        # Implement real-time alerting logic
        pass

    def _update_behavioral_patterns(self, event: AnalyticsEvent):
        """Update user behavioral patterns"""
        user_id = event.user_id

        if user_id not in self.behavioral_patterns:
            self.behavioral_patterns[user_id] = {
                "session_patterns": defaultdict(list),
                "feature_usage": defaultdict(int),
                "temporal_patterns": defaultdict(int),
                "engagement_trends": []
            }

        patterns = self.behavioral_patterns[user_id]

        # Update session patterns
        patterns["session_patterns"][event.session_id].append({
            "event_type": event.event_type.value,
            "timestamp": event.timestamp,
            "properties": event.properties
        })

        # Update feature usage
        if "feature" in event.properties:
            patterns["feature_usage"][event.properties["feature"]] += 1

        # Update temporal patterns
        hour_key = f"hour_{event.timestamp.hour}"
        patterns["temporal_patterns"][hour_key] += 1

        day_key = f"day_{event.timestamp.strftime('%A')}"
        patterns["temporal_patterns"][day_key] += 1

    def _update_user_segment(self, user_id: str):
        """Update user segmentation based on recent activity"""
        user_events = [e for e in self.events if e.user_id == user_id]

        if not user_events:
            return

        # Calculate segmentation features
        features = self._calculate_segmentation_features(user_id, user_events)

        # Determine segment
        segment_type = self._classify_user_segment(features)
        confidence = self._calculate_segment_confidence(features, segment_type)

        # Update or create segment
        self.user_segments[user_id] = UserSegment(
            user_id=user_id,
            segment_type=segment_type,
            confidence=confidence,
            characteristics=features,
            assigned_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=30)
        )

    def _calculate_segmentation_features(self, user_id: str, events: List[AnalyticsEvent]) -> Dict:
        """Calculate features for user segmentation"""
        now = datetime.now()

        # Recency features
        last_event = max(events, key=lambda e: e.timestamp)
        days_since_last_activity = (now - last_event.timestamp).days

        # Frequency features
        events_last_30_days = [e for e in events if (now - e.timestamp).days <= 30]
        events_per_week = len(events_last_30_days) / 4.3  # Average weeks in month

        # Engagement features
        unique_sessions = len(set(e.session_id for e in events))
        avg_session_length = self._calculate_avg_session_length(events)

        # Feature adoption
        unique_features = len(set(e.properties.get("feature", "")
                              for e in events if e.properties.get("feature")))

        # Value indicators
        decisions_created = len([e for e in events if e.event_type == EventType.DECISION_CREATED])
        decisions_completed = len(
            [e for e in events if e.event_type == EventType.DECISION_COMPLETED])
        completion_rate = decisions_completed / max(1, decisions_created)

        return {
            "days_since_last_activity": days_since_last_activity,
            "events_per_week": events_per_week,
            "unique_sessions": unique_sessions,
            "avg_session_length": avg_session_length,
            "unique_features": unique_features,
            "decisions_created": decisions_created,
            "decisions_completed": decisions_completed,
            "completion_rate": completion_rate,
            "total_events": len(events),
            "account_age_days": (now - min(e.timestamp for e in events)).days
        }

    def _calculate_avg_session_length(self, events: List[AnalyticsEvent]) -> float:
        """Calculate average session length"""
        session_lengths = []

        for session_id in set(e.session_id for e in events):
            session_events = [e for e in events if e.session_id == session_id]
            if len(session_events) > 1:
                start_time = min(e.timestamp for e in session_events)
                end_time = max(e.timestamp for e in session_events)
                length = (end_time - start_time).total_seconds()
                session_lengths.append(length)

        return statistics.mean(session_lengths) if session_lengths else 0

    def _classify_user_segment(self, features: Dict) -> SegmentType:
        """Classify user into segment based on features"""
        # New user detection
        if features["account_age_days"] <= 7:
            return SegmentType.NEW_USER

        # Churned user detection
        if features["days_since_last_activity"] > 30:
            return SegmentType.CHURNED_USER

        # Power user detection
        if (features["events_per_week"] > 10 and
            features["unique_features"] > 5 and
                features["completion_rate"] > 0.7):
            return SegmentType.POWER_USER

        # High value user detection
        if (features["decisions_completed"] > 20 and
                features["completion_rate"] > 0.8):
            return SegmentType.HIGH_VALUE

        # Low engagement detection
        if (features["events_per_week"] < 1 or
                features["completion_rate"] < 0.3):
            return SegmentType.LOW_ENGAGEMENT

        # Default to active user
        return SegmentType.ACTIVE_USER

    def _calculate_segment_confidence(self, features: Dict, segment: SegmentType) -> float:
        """Calculate confidence score for segment assignment"""
        # Implement confidence calculation based on feature strength
        confidence_factors = {
            SegmentType.NEW_USER: features["account_age_days"] <= 7,
            SegmentType.CHURNED_USER: features["days_since_last_activity"] > 30,
            SegmentType.POWER_USER: features["events_per_week"] > 10 and features["unique_features"] > 5,
            SegmentType.HIGH_VALUE: features["decisions_completed"] > 20,
            SegmentType.LOW_ENGAGEMENT: features["events_per_week"] < 1}

        base_confidence = 0.8 if confidence_factors.get(segment, False) else 0.6

        # Adjust based on data quality
        data_quality = min(1.0, features["total_events"] / 50)  # More events = higher confidence

        return min(1.0, base_confidence * (0.5 + 0.5 * data_quality))

    def generate_analytics_dashboard(self, time_range: str = "30d") -> Dict:
        """Generate comprehensive analytics dashboard"""
        end_date = datetime.now()

        if time_range == "24h":
            start_date = end_date - timedelta(hours=24)
        elif time_range == "7d":
            start_date = end_date - timedelta(days=7)
        elif time_range == "30d":
            start_date = end_date - timedelta(days=30)
        elif time_range == "90d":
            start_date = end_date - timedelta(days=90)
        else:
            start_date = end_date - timedelta(days=30)

        filtered_events = [e for e in self.events if start_date <= e.timestamp <= end_date]

        dashboard = {
            "overview": self._generate_overview_metrics(filtered_events),
            "user_engagement": self._generate_engagement_metrics(filtered_events),
            "conversion_analytics": self._generate_conversion_metrics(filtered_events),
            "behavioral_insights": self._generate_behavioral_insights(filtered_events),
            "ab_test_results": self._generate_ab_test_results(),
            "predictive_insights": self._generate_predictive_insights(filtered_events),
            "segmentation_analysis": self._generate_segmentation_analysis(),
            "feature_adoption": self._generate_feature_adoption_metrics(filtered_events),
            "performance_metrics": self._generate_performance_metrics(filtered_events),
            "recommendations": self._generate_recommendations(filtered_events)
        }

        return dashboard

    def _generate_overview_metrics(self, events: List[AnalyticsEvent]) -> Dict:
        """Generate overview metrics"""
        unique_users = len(set(e.user_id for e in events))
        unique_sessions = len(set(e.session_id for e in events))
        total_events = len(events)

        decision_events = [e for e in events if e.event_type == EventType.DECISION_CREATED]
        completion_events = [e for e in events if e.event_type == EventType.DECISION_COMPLETED]

        return {
            "total_users": unique_users,
            "total_sessions": unique_sessions,
            "total_events": total_events,
            "decisions_created": len(decision_events),
            "decisions_completed": len(completion_events),
            "completion_rate": len(completion_events) / max(1, len(decision_events)),
            "events_per_user": total_events / max(1, unique_users),
            "events_per_session": total_events / max(1, unique_sessions)
        }

    def _generate_engagement_metrics(self, events: List[AnalyticsEvent]) -> Dict:
        """Generate user engagement metrics"""
        model = self.predictive_models["engagement_scorer"]

        user_engagement_scores = {}
        for user_id in set(e.user_id for e in events):
            user_events = [e for e in events if e.user_id == user_id]
            score = self._calculate_engagement_score(user_id, user_events, model)
            user_engagement_scores[user_id] = score

        avg_engagement = statistics.mean(
            user_engagement_scores.values()) if user_engagement_scores else 0

        return {
            "average_engagement_score": avg_engagement,
            "high_engagement_users": len([s for s in user_engagement_scores.values() if s > 0.7]),
            "low_engagement_users": len([s for s in user_engagement_scores.values() if s < 0.3]),
            "engagement_distribution": self._create_distribution(list(user_engagement_scores.values()))
        }

    def _calculate_engagement_score(
            self,
            user_id: str,
            events: List[AnalyticsEvent],
            model: Dict) -> float:
        """Calculate user engagement score"""
        if not events:
            return 0.0

        now = datetime.now()

        # Calculate dimension scores
        frequency = len(events) / max(1, (now - min(e.timestamp for e in events)).days)
        depth = len(set(e.properties.get("feature", "")
                    for e in events if e.properties.get("feature")))
        duration = self._calculate_avg_session_length(events)
        recency = (now - max(e.timestamp for e in events)).days

        # Quality based on completion rates
        decision_events = [e for e in events if e.event_type == EventType.DECISION_CREATED]
        completion_events = [e for e in events if e.event_type == EventType.DECISION_COMPLETED]
        quality = len(completion_events) / max(1, len(decision_events))

        # Apply scoring functions
        scores = {
            "frequency": model["scoring_functions"]["frequency"](frequency),
            "depth": model["scoring_functions"]["depth"](depth),
            "duration": model["scoring_functions"]["duration"](duration),
            "recency": model["scoring_functions"]["recency"](recency),
            "quality": model["scoring_functions"]["quality"](quality)
        }

        # Weighted average
        engagement_score = sum(scores[dim] * model["dimensions"][dim] for dim in scores)

        return min(1.0, max(0.0, engagement_score))

    def _create_distribution(self, values: List[float]) -> Dict:
        """Create distribution analysis of values"""
        if not values:
            return {}

        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "percentiles": {
                "25th": np.percentile(values, 25),
                "75th": np.percentile(values, 75),
                "90th": np.percentile(values, 90)
            }
        }

    def _generate_conversion_metrics(self, events: List[AnalyticsEvent]) -> Dict:
        """Generate conversion analytics"""
        conversion_metrics = {}

        for funnel_name, funnel in self.conversion_funnels.items():
            funnel_events = [e for e in events if self._event_matches_funnel_step(e, funnel)]
            # Calculate funnel metrics
            conversion_metrics[funnel_name] = {
                "total_entries": len(funnel_events),
                "conversion_rate": 0.0,  # Would be calculated based on funnel logic
                "drop_off_rate": 0.0,
                "avg_time_to_convert": 0.0
            }

        return conversion_metrics

    def _generate_behavioral_insights(self, events: List[AnalyticsEvent]) -> Dict:
        """Generate behavioral insights"""
        # Time-based patterns
        hourly_distribution = Counter(e.timestamp.hour for e in events)
        daily_distribution = Counter(e.timestamp.strftime("%A") for e in events)

        # Feature usage patterns
        feature_usage = Counter(e.properties.get("feature", "unknown")
                                for e in events if e.properties.get("feature"))

        # Session patterns
        session_lengths = []
        for session_id in set(e.session_id for e in events):
            session_events = [e for e in events if e.session_id == session_id]
            if len(session_events) > 1:
                start = min(e.timestamp for e in session_events)
                end = max(e.timestamp for e in session_events)
                session_lengths.append((end - start).total_seconds())

        return {
            "peak_hours": [hour for hour, count in hourly_distribution.most_common(3)],
            "peak_days": [day for day, count in daily_distribution.most_common(3)],
            "most_used_features": [feature for feature, count in feature_usage.most_common(5)],
            "average_session_length": statistics.mean(session_lengths) if session_lengths else 0,
            "session_length_distribution": self._create_distribution(session_lengths)
        }

    def _generate_ab_test_results(self) -> Dict:
        """Generate A/B test results"""
        results = {}

        for test_name, variants in self.ab_tests.items():
            results[test_name] = {
                "variants": [
                    {
                        "variant_id": v.variant_id,
                        "name": v.name,
                        "conversion_rate": v.conversion_rate,
                        "sample_size": v.sample_size,
                        "statistical_significance": v.statistical_significance
                    }
                    for v in variants
                ],
                "winner": max(variants, key=lambda v: v.conversion_rate).variant_id if variants else None,
                "confidence_level": max(v.statistical_significance for v in variants) if variants else 0
            }

        return results

    def _generate_predictive_insights(self, events: List[AnalyticsEvent]) -> Dict:
        """Generate predictive insights"""
        churn_model = self.predictive_models["churn_predictor"]

        # Calculate churn risk for active users
        churn_predictions = {}
        active_users = set(e.user_id for e in events if (datetime.now() - e.timestamp).days <= 7)

        for user_id in active_users:
            user_events = [e for e in self.events if e.user_id == user_id]
            churn_risk = self._calculate_churn_risk(user_id, user_events, churn_model)
            churn_predictions[user_id] = churn_risk

        high_churn_risk = [uid for uid, risk in churn_predictions.items() if risk > 0.7]

        return {
            "churn_predictions": {
                "high_risk_users": len(high_churn_risk),
                "average_churn_risk": statistics.mean(
                    churn_predictions.values()) if churn_predictions else 0,
                "risk_distribution": self._create_distribution(
                    list(
                        churn_predictions.values()))},
            "growth_predictions": {
                "projected_new_users_next_month": self._predict_new_users(),
                "projected_revenue_impact": self._predict_revenue_impact()}}

    def _calculate_churn_risk(
            self,
            user_id: str,
            events: List[AnalyticsEvent],
            model: Dict) -> float:
        """Calculate churn risk for user"""
        if not events:
            return 1.0

        now = datetime.now()

        # Calculate features
        features = {
            "days_since_last_visit": (now - max(e.timestamp for e in events)).days,
            "total_decisions": len([e for e in events if e.event_type == EventType.DECISION_CREATED]),
            "completion_rate": len([e for e in events if e.event_type == EventType.DECISION_COMPLETED]) /
            max(1, len([e for e in events if e.event_type == EventType.DECISION_CREATED])),
            "avg_session_duration": self._calculate_avg_session_length(events),
            "feature_adoption_rate": len(set(e.properties.get("feature", "") for e in events if e.properties.get("feature"))) / 8,
            "satisfaction_score": 0.7,  # Would be calculated from feedback
            "support_interactions": 0   # Would be calculated from support events
        }

        # Calculate weighted score
        risk_score = 0
        for feature_name, value in features.items():
            weight = model["weights"].get(feature_name, 0)
            normalized_value = min(1.0, max(0.0, value /
                                            100 if feature_name == "days_since_last_visit" else value))
            risk_score += weight * normalized_value

        return max(0.0, min(1.0, 0.5 + risk_score))

    def _predict_new_users(self) -> int:
        """Predict new users for next month"""
        # Simple growth prediction based on recent trends
        recent_events = [e for e in self.events if (datetime.now() - e.timestamp).days <= 30]
        new_user_events = [e for e in recent_events if e.event_type == EventType.SESSION_START]

        # Count unique new users (simplified)
        current_growth = len(set(e.user_id for e in new_user_events))

        # Simple projection with growth factor
        growth_factor = 1.1  # 10% growth assumption
        return int(current_growth * growth_factor)

    def _predict_revenue_impact(self) -> float:
        """Predict revenue impact"""
        # Simplified revenue prediction
        high_value_users = len([s for s in self.user_segments.values()
                               if s.segment_type == SegmentType.HIGH_VALUE])
        return high_value_users * 50  # Assume $50 value per high-value user

    def _generate_segmentation_analysis(self) -> Dict:
        """Generate user segmentation analysis"""
        segment_distribution = Counter(s.segment_type.value for s in self.user_segments.values())

        return {
            "segment_distribution": dict(segment_distribution),
            "total_segments": len(self.user_segments),
            "segment_confidence": {
                segment: statistics.mean([s.confidence for s in self.user_segments.values() if s.segment_type.value == segment])
                for segment in segment_distribution.keys()
            }
        }

    def _generate_feature_adoption_metrics(self, events: List[AnalyticsEvent]) -> Dict:
        """Generate feature adoption metrics"""
        feature_events = [e for e in events if e.event_type == EventType.FEATURE_USED]
        feature_usage = Counter(e.properties.get("feature", "unknown") for e in feature_events)

        total_users = len(set(e.user_id for e in events))

        adoption_rates = {}
        for feature, usage_count in feature_usage.items():
            unique_users = len(
                set(e.user_id for e in feature_events if e.properties.get("feature") == feature))
            adoption_rates[feature] = unique_users / max(1, total_users)

        return {
            "feature_adoption_rates": adoption_rates, "most_adopted_features": [
                f for f, rate in sorted(
                    adoption_rates.items(), key=lambda x: x[1], reverse=True)[
                    :5]], "least_adopted_features": [
                f for f, rate in sorted(
                    adoption_rates.items(), key=lambda x: x[1])[
                    :3]]}

    def _generate_performance_metrics(self, events: List[AnalyticsEvent]) -> Dict:
        """Generate performance metrics"""
        # Response time analysis (would be from actual performance data)
        response_times = [e.metadata.get("response_time", 200)
                          for e in events if "response_time" in e.metadata]

        return {
            "average_response_time": statistics.mean(response_times) if response_times else 200,
            "response_time_p95": np.percentile(response_times, 95) if response_times else 500,
            "error_rate": 0.01,  # Would be calculated from error events
            "uptime_percentage": 99.9  # Would be from monitoring
        }

    def _generate_recommendations(self, events: List[AnalyticsEvent]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Engagement recommendations
        avg_engagement = statistics.mean([
            self._calculate_engagement_score(uid, [e for e in events if e.user_id == uid], self.predictive_models["engagement_scorer"])
            for uid in set(e.user_id for e in events)
        ]) if events else 0

        if avg_engagement < 0.5:
            recommendations.append("Improve user onboarding to increase engagement")

        # Conversion recommendations
        decision_events = [e for e in events if e.event_type == EventType.DECISION_CREATED]
        completion_events = [e for e in events if e.event_type == EventType.DECISION_COMPLETED]
        completion_rate = len(completion_events) / max(1, len(decision_events))

        if completion_rate < 0.6:
            recommendations.append("Optimize decision completion flow to reduce abandonment")

        # Feature adoption recommendations
        feature_usage = Counter(e.properties.get("feature", "")
                                for e in events if e.properties.get("feature"))
        if len(feature_usage) < 5:
            recommendations.append("Promote feature discovery to increase adoption")

        return recommendations

    def _get_recent_events(self, user_id: str, hours: int = 24) -> List[AnalyticsEvent]:
        """Get recent events for user"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [e for e in self.events if e.user_id == user_id and e.timestamp > cutoff]

    def _generate_event_id(self, user_id: str, event_type: EventType, timestamp: datetime) -> str:
        """Generate unique event ID"""
        text = f"{user_id}_{event_type.value}_{timestamp.isoformat()}"
        return hashlib.md5(text.encode()).hexdigest()[:16]


# Initialize global analytics engine
analytics_engine = AdvancedAnalyticsEngine()

if __name__ == "__main__":
    # Test the analytics engine
    engine = AdvancedAnalyticsEngine()

    # Simulate some events
    test_events = [{"user_id": "user1",
                    "session_id": "sess1",
                    "event_type": EventType.SESSION_START,
                    "properties": {}},
                   {"user_id": "user1",
                    "session_id": "sess1",
                    "event_type": EventType.DECISION_CREATED,
                    "properties": {"category": "career"}},
                   {"user_id": "user1",
                    "session_id": "sess1",
                    "event_type": EventType.DECISION_COMPLETED,
                    "properties": {"outcome": "positive"}},
                   {"user_id": "user2",
                    "session_id": "sess2",
                    "event_type": EventType.COMPLIMENT_REQUESTED,
                    "properties": {"mood": "confident"}}]

    for event_data in test_events:
        engine.track_event(
            event_data["user_id"],
            event_data["session_id"],
            event_data["event_type"],
            event_data["properties"]
        )

    # Generate dashboard
    dashboard = engine.generate_analytics_dashboard("30d")
    print("Analytics Dashboard Generated:")
    print(f"Total Users: {dashboard['overview']['total_users']}")
    print(f"Total Events: {dashboard['overview']['total_events']}")
    print(f"Completion Rate: {dashboard['overview']['completion_rate']:.2%}")
    print(f"Recommendations: {dashboard['recommendations']}")
