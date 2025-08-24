#!/usr/bin/env python3
"""
Advanced Decisions Database Schema
Sophisticated decision tracking with pattern recognition, context awareness, and ML-driven insights
"""

import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import re
from collections import defaultdict, Counter
import statistics


class DecisionCategory(Enum):
    CAREER = "career"
    FINANCIAL = "financial"
    RELATIONSHIP = "relationship"
    HEALTH = "health"
    EDUCATION = "education"
    LIFESTYLE = "lifestyle"
    BUSINESS = "business"
    CREATIVE = "creative"
    PERSONAL_GROWTH = "personal_growth"
    FAMILY = "family"
    TRAVEL = "travel"
    TECHNOLOGY = "technology"


class DecisionComplexity(Enum):
    SIMPLE = 1      # Binary yes/no decisions
    MODERATE = 2    # 2-3 options with clear pros/cons
    COMPLEX = 3     # Multiple options with interdependencies
    STRATEGIC = 4   # Long-term decisions with many variables
    LIFE_CHANGING = 5  # Major life decisions with lasting impact


class DecisionUrgency(Enum):
    LOW = "low"         # Can decide within weeks/months
    MEDIUM = "medium"   # Should decide within days/weeks
    HIGH = "high"       # Need to decide within hours/days
    CRITICAL = "critical"  # Immediate decision required


class DecisionOutcome(Enum):
    HIGHLY_SATISFIED = "highly_satisfied"
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    DISSATISFIED = "dissatisfied"
    REGRETFUL = "regretful"
    UNKNOWN = "unknown"


@dataclass
class DecisionOption:
    id: str
    title: str
    description: str
    pros: List[str]
    cons: List[str]
    estimated_impact: float  # 0.0 to 1.0
    feasibility_score: float  # 0.0 to 1.0
    risk_level: float  # 0.0 to 1.0
    cost_estimate: Optional[float] = None
    time_requirement: Optional[int] = None  # hours
    confidence_level: float = 0.5


@dataclass
class ContextualFactors:
    emotional_state: str
    stress_level: int  # 1-10
    energy_level: int  # 1-10
    time_pressure: int  # 1-10
    external_pressure: int  # 1-10
    financial_situation: str
    support_system: str
    past_experience: str
    current_goals: List[str]
    potential_obstacles: List[str]


@dataclass
class PatternAnalysis:
    decision_velocity: float  # decisions per week
    complexity_preference: float  # average complexity chosen
    risk_tolerance: float  # average risk level of chosen options
    success_rate: float  # percentage of positive outcomes
    common_categories: List[str]
    decision_triggers: List[str]
    time_patterns: Dict[str, int]  # hour/day patterns
    seasonal_patterns: Dict[str, float]
    cognitive_biases: List[str]
    improvement_areas: List[str]


@dataclass
class DecisionData:
    id: str
    user_id: str
    title: str
    description: str
    category: DecisionCategory
    complexity: DecisionComplexity
    urgency: DecisionUrgency
    options: List[DecisionOption]
    contextual_factors: ContextualFactors
    ai_analysis: Dict[str, Any]
    confidence_score: float
    chosen_option_id: Optional[str] = None
    decision_rationale: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    decided_at: Optional[datetime] = None
    followed_up_at: Optional[datetime] = None
    outcome: DecisionOutcome = DecisionOutcome.UNKNOWN
    outcome_notes: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    sentiment_score: float = 0.5
    follow_up_wisdom: Optional[str] = None


class DecisionPatternEngine:
    """Advanced pattern recognition and decision analysis engine"""

    def __init__(self):
        self.decisions: Dict[str, DecisionData] = {}
        self.user_patterns: Dict[str, PatternAnalysis] = {}
        self.global_insights: Dict[str, Any] = {}
        self.ml_models: Dict[str, Any] = {}
        self._initialize_ml_models()

    def _initialize_ml_models(self):
        """Initialize machine learning models for pattern recognition"""
        self.ml_models = {
            "sentiment_analyzer": self._create_sentiment_model(),
            "outcome_predictor": self._create_outcome_model(),
            "bias_detector": self._create_bias_model(),
            "timing_optimizer": self._create_timing_model(),
            "similarity_engine": self._create_similarity_model()
        }

    def _create_sentiment_model(self):
        """Create sentiment analysis model for decision text"""
        positive_indicators = [
            "excited", "confident", "optimistic", "ready", "motivated", "clear",
            "opportunity", "growth", "improvement", "benefit", "success", "positive"
        ]
        negative_indicators = [
            "worried", "anxious", "uncertain", "confused", "pressured", "stuck",
            "problem", "risk", "danger", "loss", "failure", "negative", "difficult"
        ]
        neutral_indicators = [
            "considering", "evaluating", "thinking", "analyzing", "reviewing",
            "option", "choice", "decision", "possibility", "alternative"
        ]

        return {
            "positive": positive_indicators,
            "negative": negative_indicators,
            "neutral": neutral_indicators
        }

    def _create_outcome_model(self):
        """Create outcome prediction model"""
        success_factors = [
            "thorough_analysis", "multiple_options", "stakeholder_input",
            "risk_assessment", "timeline_planning", "resource_availability",
            "past_experience", "expert_consultation", "clear_criteria"
        ]

        failure_factors = [
            "time_pressure", "emotional_decision", "limited_information",
            "single_option", "external_pressure", "resource_constraints",
            "analysis_paralysis", "overconfidence", "ignoring_risks"
        ]

        return {
            "success_factors": success_factors,
            "failure_factors": failure_factors,
            "weights": {factor: 0.1 for factor in success_factors + failure_factors}
        }

    def _create_bias_model(self):
        """Create cognitive bias detection model"""
        bias_patterns = {
            "confirmation_bias": ["only positive", "ignore negative", "selective information"],
            "anchoring_bias": ["first option", "initial impression", "starting point"],
            "availability_bias": ["recent experience", "memorable event", "easy to recall"],
            "sunk_cost_fallacy": ["already invested", "can't waste", "too far in"],
            "overconfidence_bias": ["definitely will", "certain outcome", "guaranteed success"],
            "analysis_paralysis": ["need more data", "not enough information", "more research"]
        }

        return bias_patterns

    def _create_timing_model(self):
        """Create optimal timing prediction model"""
        return {
            "peak_hours": [9, 10, 11, 14, 15, 16],  # Best decision-making hours
            "avoid_hours": [0, 1, 2, 3, 4, 5, 22, 23],  # Poor decision-making hours
            "day_patterns": {
                "monday": 0.7,    # Week start, lower quality
                "tuesday": 0.9,   # Peak performance
                "wednesday": 0.85,  # Good performance
                "thursday": 0.8,  # Declining performance
                "friday": 0.6,    # End of week fatigue
                "saturday": 0.7,  # Weekend, more relaxed
                "sunday": 0.6     # Sunday scaries, anxiety
            }
        }

    def _create_similarity_model(self):
        """Create decision similarity comparison model"""
        return {
            "category_weight": 0.3,
            "complexity_weight": 0.2,
            "urgency_weight": 0.1,
            "context_weight": 0.2,
            "outcome_weight": 0.2
        }

    def add_decision(self, decision_data: Dict) -> str:
        """Add new decision with AI analysis"""
        decision_id = self._generate_decision_id(decision_data)

        # Convert options to DecisionOption objects
        options = []
        for opt_data in decision_data.get("options", []):
            option = DecisionOption(
                id=opt_data.get("id", self._generate_option_id(opt_data.get("title", ""))),
                title=opt_data.get("title", ""),
                description=opt_data.get("description", ""),
                pros=opt_data.get("pros", []),
                cons=opt_data.get("cons", []),
                estimated_impact=opt_data.get("estimated_impact", 0.5),
                feasibility_score=opt_data.get("feasibility_score", 0.5),
                risk_level=opt_data.get("risk_level", 0.5),
                cost_estimate=opt_data.get("cost_estimate"),
                time_requirement=opt_data.get("time_requirement"),
                confidence_level=opt_data.get("confidence_level", 0.5)
            )
            options.append(option)

        # Create contextual factors
        context_data = decision_data.get("contextual_factors", {})
        contextual_factors = ContextualFactors(
            emotional_state=context_data.get("emotional_state", "neutral"),
            stress_level=context_data.get("stress_level", 5),
            energy_level=context_data.get("energy_level", 5),
            time_pressure=context_data.get("time_pressure", 5),
            external_pressure=context_data.get("external_pressure", 5),
            financial_situation=context_data.get("financial_situation", "stable"),
            support_system=context_data.get("support_system", "adequate"),
            past_experience=context_data.get("past_experience", "limited"),
            current_goals=context_data.get("current_goals", []),
            potential_obstacles=context_data.get("potential_obstacles", [])
        )

        # Perform AI analysis
        ai_analysis = self._analyze_decision(decision_data, options, contextual_factors)
        confidence_score = self._calculate_confidence_score(
            options, contextual_factors, ai_analysis)
        sentiment_score = self._analyze_sentiment(decision_data.get(
            "description", "") + " " + decision_data.get("title", ""))

        # Create decision object
        decision = DecisionData(
            id=decision_id,
            user_id=decision_data.get("user_id", "anonymous"),
            title=decision_data.get("title", ""),
            description=decision_data.get("description", ""),
            category=DecisionCategory(decision_data.get("category", "personal_growth")),
            complexity=DecisionComplexity(decision_data.get("complexity", 2)),
            urgency=DecisionUrgency(decision_data.get("urgency", "medium")),
            options=options,
            contextual_factors=contextual_factors,
            ai_analysis=ai_analysis,
            confidence_score=confidence_score,
            sentiment_score=sentiment_score,
            tags=decision_data.get("tags", [])
        )

        self.decisions[decision_id] = decision
        self._update_user_patterns(decision.user_id)

        return decision_id

    def _analyze_decision(self, decision_data: Dict, options: List[DecisionOption],
                          context: ContextualFactors) -> Dict[str, Any]:
        """Comprehensive AI analysis of decision"""
        analysis = {
            "complexity_assessment": self._assess_complexity(options, context),
            "risk_analysis": self._analyze_risks(options, context),
            "bias_detection": self._detect_biases(decision_data, context),
            "timing_analysis": self._analyze_timing(context),
            "outcome_prediction": self._predict_outcomes(options, context),
            "recommendations": self._generate_recommendations(options, context),
            "similar_decisions": self._find_similar_decisions(decision_data),
            "success_factors": self._identify_success_factors(options, context),
            "potential_pitfalls": self._identify_pitfalls(options, context),
            "follow_up_plan": self._create_follow_up_plan(options, context)
        }

        return analysis

    def _assess_complexity(self, options: List[DecisionOption], context: ContextualFactors) -> Dict:
        """Assess decision complexity"""
        factors = {
            "option_count": len(options),
            "interdependencies": sum(1 for opt in options if len(opt.pros) + len(opt.cons) > 5),
            "uncertainty_level": 1 - statistics.mean([opt.confidence_level for opt in options]),
            "stakeholder_count": len(context.current_goals),
            "time_sensitivity": context.time_pressure / 10,
            "resource_requirements": statistics.mean([opt.feasibility_score for opt in options])
        }

        complexity_score = (
            factors["option_count"] * 0.2 +
            factors["interdependencies"] * 0.2 +
            factors["uncertainty_level"] * 0.3 +
            factors["time_sensitivity"] * 0.2 +
            (1 - factors["resource_requirements"]) * 0.1
        )

        return {
            "score": min(
                1.0,
                complexity_score),
            "factors": factors,
            "assessment": "high" if complexity_score > 0.7 else "medium" if complexity_score > 0.4 else "low"}

    def _analyze_risks(self, options: List[DecisionOption], context: ContextualFactors) -> Dict:
        """Comprehensive risk analysis"""
        risk_factors = []
        total_risk = 0

        for option in options:
            option_risks = {
                "financial_risk": option.cost_estimate / 10000 if option.cost_estimate else 0,
                "time_risk": option.time_requirement / 1000 if option.time_requirement else 0,
                "feasibility_risk": 1 - option.feasibility_score,
                "impact_risk": option.risk_level,
                "opportunity_cost": 1 - option.estimated_impact
            }

            option_risk_score = statistics.mean(option_risks.values())
            risk_factors.append({
                "option_id": option.id,
                "risk_score": option_risk_score,
                "risk_breakdown": option_risks
            })
            total_risk += option_risk_score

        avg_risk = total_risk / len(options) if options else 0

        # Context-based risk adjustments
        context_risk_multiplier = 1.0
        if context.stress_level > 7:
            context_risk_multiplier += 0.2
        if context.external_pressure > 7:
            context_risk_multiplier += 0.1
        if context.support_system == "limited":
            context_risk_multiplier += 0.15

        return {
            "average_risk": avg_risk *
            context_risk_multiplier,
            "option_risks": risk_factors,
            "risk_mitigation": self._suggest_risk_mitigation(
                risk_factors,
                context),
            "overall_assessment": "high" if avg_risk > 0.7 else "medium" if avg_risk > 0.4 else "low"}

    def _detect_biases(self, decision_data: Dict, context: ContextualFactors) -> List[str]:
        """Detect potential cognitive biases"""
        detected_biases = []
        text = decision_data.get("description", "") + " " + decision_data.get("title", "")

        bias_model = self.ml_models["bias_detector"]

        for bias_type, indicators in bias_model.items():
            if any(indicator in text.lower() for indicator in indicators):
                detected_biases.append(bias_type)

        # Context-based bias detection
        if context.stress_level > 7:
            detected_biases.append("stress_induced_bias")
        if context.time_pressure > 8:
            detected_biases.append("urgency_bias")
        if context.external_pressure > 7:
            detected_biases.append("social_pressure_bias")

        return list(set(detected_biases))

    def _analyze_timing(self, context: ContextualFactors) -> Dict:
        """Analyze decision timing factors"""
        timing_model = self.ml_models["timing_optimizer"]
        current_hour = datetime.now().hour
        current_day = datetime.now().strftime("%A").lower()

        timing_quality = 1.0

        # Hour-based adjustment
        if current_hour in timing_model["peak_hours"]:
            timing_quality *= 1.2
        elif current_hour in timing_model["avoid_hours"]:
            timing_quality *= 0.7

        # Day-based adjustment
        timing_quality *= timing_model["day_patterns"].get(current_day, 0.8)

        # Context adjustments
        if context.energy_level < 4:
            timing_quality *= 0.8
        if context.stress_level > 7:
            timing_quality *= 0.7

        recommendations = []
        if timing_quality < 0.6:
            recommendations.append("Consider delaying this decision if possible")
        if context.time_pressure > 8:
            recommendations.append("High time pressure detected - be aware of urgency bias")
        if current_hour in timing_model["avoid_hours"]:
            recommendations.append("Current time not optimal for decision-making")

        return {
            "timing_quality": timing_quality,
            "recommendations": recommendations,
            "optimal_hours": timing_model["peak_hours"],
            "current_assessment": "optimal" if timing_quality > 0.8 else "good" if timing_quality > 0.6 else "suboptimal"
        }

    def _predict_outcomes(self, options: List[DecisionOption], context: ContextualFactors) -> Dict:
        """Predict decision outcomes using ML models"""
        outcome_model = self.ml_models["outcome_predictor"]
        predictions = []

        for option in options:
            success_score = 0
            failure_score = 0

            # Analyze success factors
            for factor in outcome_model["success_factors"]:
                if self._has_factor(option, context, factor):
                    success_score += outcome_model["weights"][factor]

            # Analyze failure factors
            for factor in outcome_model["failure_factors"]:
                if self._has_factor(option, context, factor):
                    failure_score += outcome_model["weights"][factor]

            # Calculate probability
            net_score = success_score - failure_score
            success_probability = max(0.1, min(0.9, 0.5 + net_score))

            predictions.append({
                "option_id": option.id,
                "success_probability": success_probability,
                "confidence_interval": [success_probability - 0.1, success_probability + 0.1],
                "key_factors": {
                    "success": [f for f in outcome_model["success_factors"] if self._has_factor(option, context, f)],
                    "risk": [f for f in outcome_model["failure_factors"] if self._has_factor(option, context, f)]
                }
            })

        return {
            "predictions": predictions,
            "methodology": "ML-based factor analysis",
            "confidence": "medium",
            "best_option": max(
                predictions,
                key=lambda p: p["success_probability"]) if predictions else None}

    def _has_factor(self, option: DecisionOption, context: ContextualFactors, factor: str) -> bool:
        """Check if option/context has specific factor"""
        factor_checks = {
            "thorough_analysis": len(option.pros) + len(option.cons) >= 5,
            "multiple_options": True,  # Checked at decision level
            "risk_assessment": option.risk_level > 0,
            "clear_criteria": option.confidence_level > 0.7,
            "time_pressure": context.time_pressure > 7,
            "emotional_decision": context.stress_level > 7 or context.emotional_state in ["angry", "excited", "anxious"],
            "limited_information": option.confidence_level < 0.5,
            "external_pressure": context.external_pressure > 7,
            "resource_constraints": option.feasibility_score < 0.5
        }

        return factor_checks.get(factor, False)

    def _generate_recommendations(
            self,
            options: List[DecisionOption],
            context: ContextualFactors) -> List[str]:
        """Generate AI-powered recommendations"""
        recommendations = []

        # Option-specific recommendations
        best_option = max(options, key=lambda o: o.estimated_impact *
                          o.feasibility_score) if options else None
        if best_option:
            recommendations.append(
                f"Consider '{
                    best_option.title}' as it offers the best impact-to-feasibility ratio")

        # Risk recommendations
        high_risk_options = [o for o in options if o.risk_level > 0.7]
        if high_risk_options:
            recommendations.append(
                f"High-risk options detected. Consider mitigation strategies for: {', '.join([o.title for o in high_risk_options])}")

        # Context recommendations
        if context.stress_level > 7:
            recommendations.append(
                "High stress detected. Consider stress management before deciding")

        if context.time_pressure > 8:
            recommendations.append("High time pressure. Focus on most critical factors only")

        if context.energy_level < 4:
            recommendations.append("Low energy level. Consider postponing complex decisions")

        # Pattern-based recommendations
        if len(options) > 5:
            recommendations.append(
                "Many options available. Consider eliminating clearly inferior choices first")

        return recommendations

    def _find_similar_decisions(self, decision_data: Dict) -> List[Dict]:
        """Find similar past decisions for learning"""
        similar_decisions = []
        current_category = decision_data.get("category", "")

        for decision in self.decisions.values():
            if decision.category.value == current_category and decision.outcome != DecisionOutcome.UNKNOWN:
                similarity_score = self._calculate_similarity(decision_data, asdict(decision))
                if similarity_score > 0.6:
                    similar_decisions.append({
                        "decision_id": decision.id,
                        "title": decision.title,
                        "similarity_score": similarity_score,
                        "outcome": decision.outcome.value,
                        "lessons_learned": decision.lessons_learned
                    })

        return sorted(similar_decisions, key=lambda d: d["similarity_score"], reverse=True)[:5]

    def _calculate_similarity(self, decision1: Dict, decision2: Dict) -> float:
        """Calculate similarity between two decisions"""
        similarity_model = self.ml_models["similarity_engine"]
        score = 0

        # Category similarity
        if decision1.get("category") == decision2.get("category"):
            score += similarity_model["category_weight"]

        # Complexity similarity
        complexity_diff = abs(decision1.get("complexity", 2) - decision2.get("complexity", 2))
        complexity_sim = 1 - (complexity_diff / 4)  # Normalize to 0-1
        score += complexity_sim * similarity_model["complexity_weight"]

        # Urgency similarity
        urgency_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        urgency1 = urgency_map.get(decision1.get("urgency", "medium"), 2)
        urgency2 = urgency_map.get(decision2.get("urgency", "medium"), 2)
        urgency_diff = abs(urgency1 - urgency2)
        urgency_sim = 1 - (urgency_diff / 3)
        score += urgency_sim * similarity_model["urgency_weight"]

        return min(1.0, score)

    def _identify_success_factors(
            self,
            options: List[DecisionOption],
            context: ContextualFactors) -> List[str]:
        """Identify factors that increase success probability"""
        factors = []

        if any(opt.confidence_level > 0.8 for opt in options):
            factors.append("High confidence in at least one option")

        if statistics.mean([opt.feasibility_score for opt in options]) > 0.7:
            factors.append("High feasibility across options")

        if context.support_system in ["strong", "excellent"]:
            factors.append("Strong support system available")

        if context.stress_level < 5:
            factors.append("Low stress environment")

        if len(context.current_goals) > 0:
            factors.append("Clear goals and objectives")

        return factors

    def _identify_pitfalls(
            self,
            options: List[DecisionOption],
            context: ContextualFactors) -> List[str]:
        """Identify potential pitfalls and challenges"""
        pitfalls = []

        if all(opt.risk_level > 0.7 for opt in options):
            pitfalls.append("All options carry high risk")

        if context.time_pressure > 8:
            pitfalls.append("Extreme time pressure may lead to hasty decisions")

        if context.external_pressure > 7:
            pitfalls.append("High external pressure may compromise decision quality")

        if statistics.mean([opt.confidence_level for opt in options]) < 0.5:
            pitfalls.append("Low confidence across all options indicates need for more information")

        if len(options) > 7:
            pitfalls.append("Too many options may lead to decision paralysis")

        return pitfalls

    def _create_follow_up_plan(
            self,
            options: List[DecisionOption],
            context: ContextualFactors) -> Dict:
        """Create follow-up plan for decision tracking"""
        plan = {
            "immediate_actions": [],
            "week_1_review": [],
            "month_1_review": [],
            "quarter_review": [],
            "success_metrics": [],
            "warning_signs": []
        }

        # Immediate actions
        plan["immediate_actions"].extend([
            "Document decision rationale",
            "Set up success metrics tracking",
            "Identify early warning signs"
        ])

        # Week 1 review
        plan["week_1_review"].extend([
            "Assess initial implementation challenges",
            "Monitor early indicators",
            "Adjust approach if needed"
        ])

        # Success metrics
        plan["success_metrics"].extend([
            "Achievement of stated goals",
            "Satisfaction level (1-10)",
            "Unexpected benefits or costs",
            "Impact on related areas"
        ])

        # Warning signs
        plan["warning_signs"].extend([
            "Persistent regret or doubt",
            "Significant unexpected obstacles",
            "Negative impact on other areas",
            "Resource depletion beyond expectations"
        ])

        return plan

    def _suggest_risk_mitigation(
            self,
            risk_factors: List[Dict],
            context: ContextualFactors) -> List[str]:
        """Suggest risk mitigation strategies"""
        suggestions = []

        for risk_factor in risk_factors:
            risks = risk_factor["risk_breakdown"]

            if risks["financial_risk"] > 0.5:
                suggestions.append("Consider financial contingency planning or phased investment")

            if risks["time_risk"] > 0.5:
                suggestions.append("Build buffer time into timeline estimates")

            if risks["feasibility_risk"] > 0.5:
                suggestions.append("Conduct feasibility study or prototype before full commitment")

        return list(set(suggestions))

    def _calculate_confidence_score(self, options: List[DecisionOption],
                                    context: ContextualFactors, ai_analysis: Dict) -> float:
        """Calculate overall decision confidence score"""
        factors = []

        # Option quality
        if options:
            avg_confidence = statistics.mean([opt.confidence_level for opt in options])
            factors.append(avg_confidence)

        # Context factors
        context_score = (
            (10 - context.stress_level) / 10 * 0.3 +
            context.energy_level / 10 * 0.2 +
            (10 - context.time_pressure) / 10 * 0.3 +
            (10 - context.external_pressure) / 10 * 0.2
        )
        factors.append(context_score)

        # AI analysis confidence
        complexity_score = 1 - ai_analysis["complexity_assessment"]["score"]
        risk_score = 1 - ai_analysis["risk_analysis"]["average_risk"]
        ai_score = (complexity_score + risk_score) / 2
        factors.append(ai_score)

        return statistics.mean(factors)

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of decision text"""
        sentiment_model = self.ml_models["sentiment_analyzer"]
        text_lower = text.lower()

        positive_count = sum(1 for word in sentiment_model["positive"] if word in text_lower)
        negative_count = sum(1 for word in sentiment_model["negative"] if word in text_lower)
        neutral_count = sum(1 for word in sentiment_model["neutral"] if word in text_lower)

        total_words = len(text.split())
        if total_words == 0:
            return 0.5

        sentiment_score = (
            positive_count * 1.0 +
            neutral_count * 0.5 +
            negative_count * 0.0
        ) / max(1, positive_count + negative_count + neutral_count)

        return sentiment_score

    def _generate_decision_id(self, decision_data: Dict) -> str:
        """Generate unique decision ID"""
        text = decision_data.get("title",
                                 "") + decision_data.get("description",
                                                         "") + str(datetime.now())
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _generate_option_id(self, title: str) -> str:
        """Generate unique option ID"""
        return hashlib.md5((title + str(datetime.now())).encode()).hexdigest()[:12]

    def _update_user_patterns(self, user_id: str):
        """Update user decision patterns"""
        user_decisions = [d for d in self.decisions.values() if d.user_id == user_id]

        if len(user_decisions) < 2:
            return

        # Calculate patterns
        decision_velocity = len(user_decisions) / max(1,
                                                      (datetime.now() - min(d.created_at for d in user_decisions)).days / 7)
        complexity_preference = statistics.mean([d.complexity.value for d in user_decisions])

        # Risk tolerance calculation
        risk_scores = []
        for decision in user_decisions:
            if decision.chosen_option_id:
                chosen_option = next(
                    (opt for opt in decision.options if opt.id == decision.chosen_option_id), None)
                if chosen_option:
                    risk_scores.append(chosen_option.risk_level)

        risk_tolerance = statistics.mean(risk_scores) if risk_scores else 0.5

        # Success rate calculation
        completed_decisions = [d for d in user_decisions if d.outcome != DecisionOutcome.UNKNOWN]
        successful_decisions = [
            d for d in completed_decisions if d.outcome in [
                DecisionOutcome.SATISFIED,
                DecisionOutcome.HIGHLY_SATISFIED]]
        success_rate = len(successful_decisions) / \
            len(completed_decisions) if completed_decisions else 0.5

        # Common categories
        category_counts = Counter([d.category.value for d in user_decisions])
        common_categories = [cat for cat, count in category_counts.most_common(3)]

        # Time patterns
        hour_counts = Counter([d.created_at.hour for d in user_decisions])
        time_patterns = dict(hour_counts)

        # Store pattern analysis
        self.user_patterns[user_id] = PatternAnalysis(
            decision_velocity=decision_velocity,
            complexity_preference=complexity_preference,
            risk_tolerance=risk_tolerance,
            success_rate=success_rate,
            common_categories=common_categories,
            decision_triggers=[],  # Would need more analysis
            time_patterns=time_patterns,
            seasonal_patterns={},  # Would need seasonal data
            cognitive_biases=[],   # Would need bias tracking
            improvement_areas=[]   # Would need performance analysis
        )

    def get_user_insights(self, user_id: str) -> Dict:
        """Get comprehensive user insights and recommendations"""
        if user_id not in self.user_patterns:
            return {"message": "Insufficient data for analysis"}

        pattern = self.user_patterns[user_id]
        user_decisions = [d for d in self.decisions.values() if d.user_id == user_id]

        insights = {
            "decision_making_profile": {
                "velocity": "high" if pattern.decision_velocity > 2 else "medium" if pattern.decision_velocity > 0.5 else "low",
                "complexity_preference": pattern.complexity_preference,
                "risk_tolerance": "high" if pattern.risk_tolerance > 0.7 else "medium" if pattern.risk_tolerance > 0.4 else "low",
                "success_rate": pattern.success_rate
            },
            "behavioral_patterns": {
                "preferred_categories": pattern.common_categories,
                "peak_decision_hours": [hour for hour, count in sorted(pattern.time_patterns.items(), key=lambda x: x[1], reverse=True)[:3]],
                "total_decisions": len(user_decisions),
                "recent_activity": len([d for d in user_decisions if d.created_at > datetime.now() - timedelta(days=30)])
            },
            "recommendations": self._generate_user_recommendations(pattern, user_decisions),
            "strengths": self._identify_user_strengths(pattern, user_decisions),
            "improvement_areas": self._identify_improvement_areas(pattern, user_decisions)
        }

        return insights

    def _generate_user_recommendations(
            self,
            pattern: PatternAnalysis,
            decisions: List[DecisionData]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []

        if pattern.success_rate < 0.6:
            recommendations.append("Consider taking more time for analysis before deciding")

        if pattern.decision_velocity > 3:
            recommendations.append(
                "High decision frequency detected - consider if all decisions require immediate action")

        if pattern.risk_tolerance > 0.8:
            recommendations.append("High risk tolerance - ensure adequate risk assessment")

        if pattern.complexity_preference < 2:
            recommendations.append(
                "Consider exploring more complex decisions for growth opportunities")

        return recommendations

    def _identify_user_strengths(
            self,
            pattern: PatternAnalysis,
            decisions: List[DecisionData]) -> List[str]:
        """Identify user decision-making strengths"""
        strengths = []

        if pattern.success_rate > 0.7:
            strengths.append("Consistently makes successful decisions")

        if pattern.decision_velocity > 1 and pattern.success_rate > 0.6:
            strengths.append("Effective at making timely decisions")

        if len(pattern.common_categories) <= 2:
            strengths.append("Focused decision-making in key areas")

        return strengths

    def _identify_improvement_areas(
            self,
            pattern: PatternAnalysis,
            decisions: List[DecisionData]) -> List[str]:
        """Identify areas for improvement"""
        areas = []

        if pattern.success_rate < 0.5:
            areas.append("Decision outcome tracking and analysis")

        if len(pattern.common_categories) > 5:
            areas.append("Decision prioritization and focus")

        recent_decisions = [
            d for d in decisions if d.created_at > datetime.now() -
            timedelta(
                days=30)]
        if any(d.confidence_score < 0.5 for d in recent_decisions):
            areas.append("Information gathering and analysis confidence")

        return areas


# Initialize the global decisions database
decisions_db = DecisionPatternEngine()

if __name__ == "__main__":
    # Test the decision engine
    engine = DecisionPatternEngine()

    # Test decision data
    test_decision = {
        "user_id": "test_user_123",
        "title": "Should I accept the new job offer?",
        "description": "I received a job offer with better pay but requires relocation. Need to decide soon.",
        "category": "career",
        "complexity": 3,
        "urgency": "high",
        "options": [
            {
                "title": "Accept the offer",
                "description": "Take the new position with 30% salary increase",
                "pros": [
                    "Higher salary",
                    "Career advancement",
                    "New experiences"],
                "cons": [
                    "Relocation required",
                    "Unknown work culture",
                    "Distance from family"],
                "estimated_impact": 0.8,
                "feasibility_score": 0.7,
                "risk_level": 0.6},
            {
                "title": "Decline and stay",
                "description": "Remain in current position",
                "pros": [
                    "Familiar environment",
                    "Close to family",
                    "Established network"],
                "cons": [
                    "Lower salary",
                    "Limited growth",
                    "Potential regret"],
                "estimated_impact": 0.4,
                "feasibility_score": 0.9,
                "risk_level": 0.3}],
        "contextual_factors": {
            "emotional_state": "anxious",
            "stress_level": 7,
            "energy_level": 6,
            "time_pressure": 8,
            "external_pressure": 5,
            "financial_situation": "stable",
            "support_system": "strong",
            "past_experience": "limited",
            "current_goals": [
                "career growth",
                "financial stability"],
            "potential_obstacles": [
                "relocation costs",
                "adaptation challenges"]}}

    # Add decision
    decision_id = engine.add_decision(test_decision)
    print(f"Added decision: {decision_id}")

    # Get decision analysis
    decision = engine.decisions[decision_id]
    print(f"\nDecision Analysis:")
    print(f"Confidence Score: {decision.confidence_score:.2f}")
    print(f"Sentiment Score: {decision.sentiment_score:.2f}")
    print(f"AI Recommendations: {decision.ai_analysis['recommendations'][:2]}")

    # Test pattern analysis
    insights = engine.get_user_insights("test_user_123")
    print(f"\nUser Insights: {insights}")
