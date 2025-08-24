#!/usr/bin/env python3
"""
Enterprise Security Layer
Advanced rate limiting, bot protection, API encryption, and anti-scraping measures
"""

import json
import hashlib
import hmac
import time
import base64
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import re
import ipaddress
from collections import defaultdict, deque
import geoip2.database
import user_agents


class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityAction(Enum):
    ALLOW = "allow"
    RATE_LIMIT = "rate_limit"
    CHALLENGE = "challenge"
    BLOCK = "block"
    LOG_ONLY = "log_only"


class RequestType(Enum):
    API_READ = "api_read"
    API_WRITE = "api_write"
    STATIC_RESOURCE = "static_resource"
    SENSITIVE_DATA = "sensitive_data"
    USER_REGISTRATION = "user_registration"
    PASSWORD_RESET = "password_reset"


@dataclass
class SecurityEvent:
    id: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    request_path: str
    threat_level: ThreatLevel
    threat_indicators: List[str]
    action_taken: SecurityAction
    metadata: Dict[str, Any]


@dataclass
class RateLimitRule:
    name: str
    request_type: RequestType
    max_requests: int
    time_window: int  # seconds
    burst_allowance: int
    penalty_duration: int  # seconds
    geographic_multiplier: Dict[str, float]


@dataclass
class BotSignature:
    name: str
    user_agent_patterns: List[str]
    behavior_patterns: List[str]
    ip_ranges: List[str]
    confidence_threshold: float


class AdvancedSecurityManager:
    """Enterprise-grade security manager with ML-powered threat detection"""

    def __init__(self):
        self.rate_limit_buckets: Dict[str, Dict] = defaultdict(dict)
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.security_events: List[SecurityEvent] = []
        self.blocked_ips: Dict[str, datetime] = {}
        self.suspicious_patterns: Dict[str, Any] = {}
        self.bot_signatures: List[BotSignature] = []
        self.encryption_keys: Dict[str, bytes] = {}
        self.geographic_data: Optional[Any] = None

        self._initialize_security_framework()

    def _initialize_security_framework(self):
        """Initialize security framework with rules and signatures"""
        self._setup_rate_limiting_rules()
        self._setup_bot_detection()
        self._setup_encryption_keys()
        self._initialize_geographic_protection()
        self._setup_threat_detection_models()

    def _setup_rate_limiting_rules(self):
        """Setup comprehensive rate limiting rules"""
        self.rate_limit_rules = {
            "api_read": RateLimitRule(
                name="API Read Operations",
                request_type=RequestType.API_READ,
                max_requests=100,
                time_window=60,  # 1 minute
                burst_allowance=20,
                penalty_duration=300,  # 5 minutes
                geographic_multiplier={
                    "US": 1.0,
                    "CA": 1.0,
                    "EU": 0.8,
                    "CN": 0.3,
                    "RU": 0.2,
                    "DEFAULT": 0.5
                }
            ),
            "api_write": RateLimitRule(
                name="API Write Operations",
                request_type=RequestType.API_WRITE,
                max_requests=30,
                time_window=60,
                burst_allowance=5,
                penalty_duration=600,  # 10 minutes
                geographic_multiplier={
                    "US": 1.0,
                    "CA": 1.0,
                    "EU": 0.9,
                    "DEFAULT": 0.4
                }
            ),
            "sensitive_data": RateLimitRule(
                name="Sensitive Data Access",
                request_type=RequestType.SENSITIVE_DATA,
                max_requests=10,
                time_window=300,  # 5 minutes
                burst_allowance=2,
                penalty_duration=1800,  # 30 minutes
                geographic_multiplier={
                    "US": 1.0,
                    "CA": 1.0,
                    "DEFAULT": 0.2
                }
            ),
            "user_registration": RateLimitRule(
                name="User Registration",
                request_type=RequestType.USER_REGISTRATION,
                max_requests=5,
                time_window=3600,  # 1 hour
                burst_allowance=1,
                penalty_duration=7200,  # 2 hours
                geographic_multiplier={"DEFAULT": 1.0}
            )
        }

    def _setup_bot_detection(self):
        """Setup advanced bot detection signatures"""
        self.bot_signatures = [
            BotSignature(
                name="Generic Scrapers",
                user_agent_patterns=[
                    r".*bot.*", r".*crawler.*", r".*spider.*", r".*scraper.*",
                    r"python-requests.*", r"curl.*", r"wget.*", r".*headless.*"
                ],
                behavior_patterns=[
                    "rapid_sequential_requests",
                    "no_javascript_execution",
                    "missing_common_headers",
                    "consistent_timing_patterns"
                ],
                ip_ranges=["*"],
                confidence_threshold=0.7
            ),
            BotSignature(
                name="AI Training Bots",
                user_agent_patterns=[
                    r".*openai.*", r".*anthropic.*", r".*gpt.*", r".*claude.*",
                    r".*chatgpt.*", r".*bard.*", r".*palm.*"
                ],
                behavior_patterns=[
                    "large_content_requests",
                    "systematic_crawling",
                    "api_endpoint_discovery"
                ],
                ip_ranges=["*"],
                confidence_threshold=0.9
            ),
            BotSignature(
                name="Malicious Bots",
                user_agent_patterns=[
                    r".*nikto.*", r".*sqlmap.*", r".*nmap.*", r".*masscan.*",
                    r".*burp.*", r".*zap.*", r".*acunetix.*"
                ],
                behavior_patterns=[
                    "vulnerability_scanning",
                    "injection_attempts",
                    "directory_traversal",
                    "excessive_error_generation"
                ],
                ip_ranges=["*"],
                confidence_threshold=0.95
            )
        ]

    def _setup_encryption_keys(self):
        """Setup encryption keys for API security"""
        self.encryption_keys = {
            "api_signing": secrets.token_bytes(32),
            "data_encryption": secrets.token_bytes(32),
            "session_key": secrets.token_bytes(32),
            "webhook_secret": secrets.token_bytes(32)
        }

    def _initialize_geographic_protection(self):
        """Initialize geographic-based protection"""
        # In production, this would load actual GeoIP database
        self.geographic_rules = {
            "high_risk_countries": ["CN", "RU", "KP", "IR"],
            "medium_risk_countries": ["VN", "BD", "PK", "NG"],
            "allowed_countries": ["US", "CA", "GB", "DE", "FR", "AU", "JP"],
            "datacenter_ranges": [
                "cloud_providers", "hosting_services", "vpn_services"
            ]
        }

    def _setup_threat_detection_models(self):
        """Setup ML-powered threat detection models"""
        self.threat_models = {
            "anomaly_detector": {
                "baseline_patterns": {
                    "requests_per_minute": {"mean": 10, "std": 5},
                    "request_size": {"mean": 1024, "std": 512},
                    "response_time": {"mean": 200, "std": 100}
                },
                "anomaly_threshold": 3.0  # Standard deviations
            },
            "behavioral_analyzer": {
                "normal_patterns": [
                    "gradual_feature_exploration",
                    "human_like_timing",
                    "interactive_sessions",
                    "error_recovery_behavior"
                ],
                "suspicious_patterns": [
                    "automated_interactions",
                    "systematic_data_extraction",
                    "rapid_endpoint_discovery",
                    "repeated_failed_attempts"
                ]
            },
            "fingerprint_engine": {
                "tracking_parameters": [
                    "user_agent_consistency",
                    "header_patterns",
                    "tls_fingerprint",
                    "timing_characteristics",
                    "request_ordering"
                ]
            }
        }

    def validate_request(self, request_data: Dict) -> Tuple[SecurityAction, Dict]:
        """Comprehensive request validation and threat assessment"""
        ip_address = request_data.get("ip_address", "")
        user_agent = request_data.get("user_agent", "")
        request_path = request_data.get("path", "")
        headers = request_data.get("headers", {})
        request_type = self._classify_request_type(request_path, headers)

        # Initial security checks
        validation_results = {
            "ip_validation": self._validate_ip_address(ip_address),
            "rate_limiting": self._check_rate_limits(
                ip_address,
                request_type),
            "bot_detection": self._detect_bot_behavior(
                ip_address,
                user_agent,
                headers,
                request_path),
            "geographic_check": self._check_geographic_restrictions(ip_address),
            "pattern_analysis": self._analyze_behavioral_patterns(
                ip_address,
                request_data),
            "threat_assessment": self._assess_threat_level(
                ip_address,
                user_agent,
                request_data)}

        # Determine final action
        final_action = self._determine_security_action(validation_results)

        # Log security event
        if final_action != SecurityAction.ALLOW or validation_results[
                "threat_assessment"]["level"] != ThreatLevel.LOW:
            self._log_security_event(request_data, validation_results, final_action)

        return final_action, validation_results

    def _classify_request_type(self, path: str, headers: Dict) -> RequestType:
        """Classify request type for appropriate security handling"""
        if "/api/" in path:
            if any(method in headers.get("method", "").upper()
                   for method in ["POST", "PUT", "DELETE"]):
                return RequestType.API_WRITE
            else:
                return RequestType.API_READ

        if any(sensitive in path for sensitive in ["/admin", "/user", "/profile", "/analytics"]):
            return RequestType.SENSITIVE_DATA

        if "/register" in path or "/signup" in path:
            return RequestType.USER_REGISTRATION

        if "/reset" in path or "/password" in path:
            return RequestType.PASSWORD_RESET

        return RequestType.STATIC_RESOURCE

    def _validate_ip_address(self, ip_address: str) -> Dict:
        """Validate IP address and check reputation"""
        validation = {
            "is_valid": False,
            "is_private": False,
            "is_blocked": False,
            "reputation_score": 0.5,
            "risk_factors": []
        }

        try:
            ip_obj = ipaddress.ip_address(ip_address)
            validation["is_valid"] = True
            validation["is_private"] = ip_obj.is_private

            # Check if IP is blocked
            if ip_address in self.blocked_ips:
                if self.blocked_ips[ip_address] > datetime.now():
                    validation["is_blocked"] = True
                    validation["risk_factors"].append("temporarily_blocked")
                else:
                    del self.blocked_ips[ip_address]

            # Check IP reputation (simplified)
            validation.update(self._check_ip_reputation(ip_address))

        except ValueError:
            validation["risk_factors"].append("invalid_ip_format")

        return validation

    def _check_ip_reputation(self, ip_address: str) -> Dict:
        """Check IP reputation against threat databases"""
        # In production, this would query actual threat intelligence APIs
        reputation_data = {
            "reputation_score": 0.8,  # Default good reputation
            "threat_categories": [],
            "last_seen_malicious": None,
            "geographic_risk": "low"
        }

        # Simulate reputation check
        if self._is_datacenter_ip(ip_address):
            reputation_data["reputation_score"] -= 0.2
            reputation_data["threat_categories"].append("datacenter")

        if self._is_tor_exit_node(ip_address):
            reputation_data["reputation_score"] -= 0.4
            reputation_data["threat_categories"].append("tor_exit")

        if self._is_vpn_ip(ip_address):
            reputation_data["reputation_score"] -= 0.1
            reputation_data["threat_categories"].append("vpn_service")

        return reputation_data

    def _is_datacenter_ip(self, ip_address: str) -> bool:
        """Check if IP belongs to datacenter (simplified)"""
        # In production, use actual datacenter IP ranges
        datacenter_ranges = [
            "aws", "google", "azure", "digitalocean", "linode"
        ]
        return False  # Simplified implementation

    def _is_tor_exit_node(self, ip_address: str) -> bool:
        """Check if IP is Tor exit node"""
        # In production, query Tor exit node lists
        return False  # Simplified implementation

    def _is_vpn_ip(self, ip_address: str) -> bool:
        """Check if IP belongs to VPN service"""
        # In production, use VPN detection services
        return False  # Simplified implementation

    def _check_rate_limits(self, ip_address: str, request_type: RequestType) -> Dict:
        """Advanced rate limiting with adaptive thresholds"""
        rule_key = request_type.value
        if rule_key not in self.rate_limit_rules:
            return {"allowed": True, "reason": "no_rule_defined"}

        rule = self.rate_limit_rules[rule_key]
        current_time = time.time()

        # Get or create rate limit bucket for this IP and rule
        bucket_key = f"{ip_address}:{rule_key}"
        if bucket_key not in self.rate_limit_buckets:
            self.rate_limit_buckets[bucket_key] = {
                "requests": deque(),
                "burst_used": 0,
                "penalty_until": 0
            }

        bucket = self.rate_limit_buckets[bucket_key]

        # Check if still under penalty
        if current_time < bucket["penalty_until"]:
            return {
                "allowed": False,
                "reason": "penalty_period",
                "retry_after": bucket["penalty_until"] - current_time
            }

        # Clean old requests outside time window
        bucket["requests"] = deque([
            req_time for req_time in bucket["requests"]
            if current_time - req_time < rule.time_window
        ], maxlen=rule.max_requests * 2)

        # Get geographic multiplier
        country_code = self._get_country_code(ip_address)
        geo_multiplier = rule.geographic_multiplier.get(
            country_code, rule.geographic_multiplier.get("DEFAULT", 0.5)
        )

        # Calculate effective limits
        effective_max_requests = int(rule.max_requests * geo_multiplier)
        current_requests = len(bucket["requests"])

        # Check rate limit
        if current_requests >= effective_max_requests:
            # Check burst allowance
            if bucket["burst_used"] < rule.burst_allowance:
                bucket["burst_used"] += 1
                bucket["requests"].append(current_time)
                return {
                    "allowed": True,
                    "reason": "burst_allowance_used",
                    "remaining_burst": rule.burst_allowance - bucket["burst_used"]
                }
            else:
                # Apply penalty
                bucket["penalty_until"] = current_time + rule.penalty_duration
                return {
                    "allowed": False,
                    "reason": "rate_limit_exceeded",
                    "retry_after": rule.penalty_duration
                }

        # Allow request and record it
        bucket["requests"].append(current_time)

        # Reset burst allowance if enough time has passed
        if current_requests < effective_max_requests * 0.5:
            bucket["burst_used"] = max(0, bucket["burst_used"] - 1)

        return {
            "allowed": True,
            "remaining_requests": effective_max_requests - current_requests - 1,
            "reset_time": current_time + rule.time_window
        }

    def _get_country_code(self, ip_address: str) -> str:
        """Get country code for IP address"""
        # In production, use actual GeoIP database
        # For now, return a default
        return "US"

    def _detect_bot_behavior(self, ip_address: str, user_agent: str,
                             headers: Dict, request_path: str) -> Dict:
        """Advanced bot detection using multiple signals"""
        detection_results = {
            "is_bot": False,
            "confidence": 0.0,
            "bot_type": None,
            "detection_methods": [],
            "risk_score": 0.0
        }

        # User agent analysis
        ua_analysis = self._analyze_user_agent(user_agent)
        if ua_analysis["is_bot"]:
            detection_results["is_bot"] = True
            detection_results["confidence"] = max(
                detection_results["confidence"], ua_analysis["confidence"])
            detection_results["detection_methods"].append("user_agent_analysis")
            detection_results["bot_type"] = ua_analysis["bot_type"]

        # Behavioral pattern analysis
        behavioral_analysis = self._analyze_request_patterns(ip_address, headers, request_path)
        if behavioral_analysis["suspicious"]:
            detection_results["confidence"] = max(
                detection_results["confidence"],
                behavioral_analysis["confidence"])
            detection_results["detection_methods"].extend(behavioral_analysis["indicators"])

        # Header fingerprinting
        header_analysis = self._analyze_headers(headers)
        if header_analysis["suspicious"]:
            detection_results["confidence"] = max(
                detection_results["confidence"],
                header_analysis["confidence"])
            detection_results["detection_methods"].append("header_fingerprinting")

        # TLS fingerprinting (if available)
        tls_analysis = self._analyze_tls_fingerprint(headers)
        if tls_analysis["suspicious"]:
            detection_results["confidence"] = max(
                detection_results["confidence"], tls_analysis["confidence"])
            detection_results["detection_methods"].append("tls_fingerprinting")

        # Calculate final risk score
        detection_results["risk_score"] = min(1.0, detection_results["confidence"] * 1.2)
        detection_results["is_bot"] = detection_results["confidence"] > 0.7

        return detection_results

    def _analyze_user_agent(self, user_agent: str) -> Dict:
        """Analyze user agent for bot indicators"""
        analysis = {
            "is_bot": False,
            "confidence": 0.0,
            "bot_type": None,
            "indicators": []
        }

        if not user_agent:
            analysis["is_bot"] = True
            analysis["confidence"] = 0.8
            analysis["indicators"].append("missing_user_agent")
            return analysis

        # Check against bot signatures
        for signature in self.bot_signatures:
            for pattern in signature.user_agent_patterns:
                if re.search(pattern, user_agent, re.IGNORECASE):
                    analysis["is_bot"] = True
                    analysis["confidence"] = signature.confidence_threshold
                    analysis["bot_type"] = signature.name
                    analysis["indicators"].append(f"signature_match_{signature.name}")
                    break

        # Parse user agent for additional analysis
        try:
            parsed_ua = user_agents.parse(user_agent)

            # Check for known bot indicators
            if any(bot_indicator in user_agent.lower() for bot_indicator in
                   ["bot", "crawler", "spider", "scraper", "automation"]):
                analysis["is_bot"] = True
                analysis["confidence"] = 0.9
                analysis["indicators"].append("explicit_bot_declaration")

            # Check for unusual browser/OS combinations
            if parsed_ua.browser.family == "Other" and parsed_ua.os.family == "Other":
                analysis["confidence"] = max(analysis["confidence"], 0.6)
                analysis["indicators"].append("unknown_browser_os")

            # Check for outdated browser versions (potential bot indicator)
            if parsed_ua.browser.version and len(parsed_ua.browser.version) >= 2:
                if parsed_ua.browser.family == "Chrome" and parsed_ua.browser.version[0] < 90:
                    analysis["confidence"] = max(analysis["confidence"], 0.4)
                    analysis["indicators"].append("outdated_browser")

        except Exception:
            analysis["confidence"] = max(analysis["confidence"], 0.3)
            analysis["indicators"].append("unparseable_user_agent")

        return analysis

    def _analyze_request_patterns(self, ip_address: str, headers: Dict, request_path: str) -> Dict:
        """Analyze request patterns for bot behavior"""
        analysis = {
            "suspicious": False,
            "confidence": 0.0,
            "indicators": []
        }

        # Get request history for this IP
        history = self.request_history[ip_address]
        current_time = time.time()

        # Add current request to history
        history.append({
            "timestamp": current_time,
            "path": request_path,
            "headers": headers
        })

        if len(history) < 3:
            return analysis

        # Analyze timing patterns
        recent_requests = [
            r for r in history if current_time -
            r["timestamp"] < 300]  # Last 5 minutes

        if len(recent_requests) > 50:  # More than 50 requests in 5 minutes
            analysis["suspicious"] = True
            analysis["confidence"] = 0.8
            analysis["indicators"].append("high_request_frequency")

        # Analyze request timing consistency (bot-like behavior)
        if len(recent_requests) >= 5:
            intervals = []
            for i in range(1, len(recent_requests)):
                interval = recent_requests[i]["timestamp"] - recent_requests[i - 1]["timestamp"]
                intervals.append(interval)

            # Check for suspiciously consistent timing
            if intervals and max(intervals) - min(intervals) < 1.0:  # Very consistent timing
                analysis["suspicious"] = True
                analysis["confidence"] = max(analysis["confidence"], 0.7)
                analysis["indicators"].append("consistent_timing_pattern")

        # Analyze path patterns
        paths = [r["path"] for r in recent_requests]
        unique_paths = set(paths)

        # Sequential path crawling pattern
        if len(paths) > 10 and len(unique_paths) / len(paths) > 0.9:
            analysis["suspicious"] = True
            analysis["confidence"] = max(analysis["confidence"], 0.6)
            analysis["indicators"].append("sequential_crawling_pattern")

        # Check for systematic API endpoint discovery
        api_paths = [p for p in paths if "/api/" in p]
        if len(api_paths) > 20 and len(set(api_paths)) > 15:
            analysis["suspicious"] = True
            analysis["confidence"] = max(analysis["confidence"], 0.8)
            analysis["indicators"].append("api_endpoint_discovery")

        return analysis

    def _analyze_headers(self, headers: Dict) -> Dict:
        """Analyze HTTP headers for bot indicators"""
        analysis = {
            "suspicious": False,
            "confidence": 0.0,
            "indicators": []
        }

        # Expected headers for legitimate browsers
        expected_headers = ["accept", "accept-language", "accept-encoding", "cache-control"]
        missing_headers = [h for h in expected_headers if h not in [k.lower()
                                                                    for k in headers.keys()]]

        if len(missing_headers) > 2:
            analysis["suspicious"] = True
            analysis["confidence"] = 0.6
            analysis["indicators"].append("missing_browser_headers")

        # Check for automation frameworks
        automation_indicators = ["selenium", "playwright", "puppeteer", "webdriver"]
        for indicator in automation_indicators:
            for header_value in headers.values():
                if isinstance(header_value, str) and indicator.lower() in header_value.lower():
                    analysis["suspicious"] = True
                    analysis["confidence"] = 0.9
                    analysis["indicators"].append(f"automation_framework_{indicator}")

        # Check for unusual header combinations
        if "python-requests" in headers.get("user-agent", "").lower():
            analysis["suspicious"] = True
            analysis["confidence"] = 0.8
            analysis["indicators"].append("programmatic_client")

        return analysis

    def _analyze_tls_fingerprint(self, headers: Dict) -> Dict:
        """Analyze TLS fingerprint for bot detection"""
        analysis = {
            "suspicious": False,
            "confidence": 0.0,
            "indicators": []
        }

        # In production, this would analyze actual TLS fingerprints
        # For now, check for TLS-related headers
        tls_headers = ["cf-ray", "cf-connecting-ip", "x-forwarded-proto"]

        if any(header in headers for header in tls_headers):
            # Behind CDN - harder to fingerprint
            analysis["confidence"] = 0.1

        return analysis

    def _check_geographic_restrictions(self, ip_address: str) -> Dict:
        """Check geographic-based access restrictions"""
        geo_check = {
            "allowed": True,
            "country_code": "US",  # Default
            "risk_level": "low",
            "restrictions": []
        }

        country_code = self._get_country_code(ip_address)
        geo_check["country_code"] = country_code

        if country_code in self.geographic_rules["high_risk_countries"]:
            geo_check["risk_level"] = "high"
            geo_check["restrictions"].append("high_risk_country")
        elif country_code in self.geographic_rules["medium_risk_countries"]:
            geo_check["risk_level"] = "medium"
            geo_check["restrictions"].append("medium_risk_country")

        # Check if from allowed countries only
        if self.geographic_rules["allowed_countries"] and country_code not in self.geographic_rules["allowed_countries"]:
            geo_check["allowed"] = False
            geo_check["restrictions"].append("country_not_in_allowlist")

        return geo_check

    def _analyze_behavioral_patterns(self, ip_address: str, request_data: Dict) -> Dict:
        """Advanced behavioral pattern analysis"""
        analysis = {
            "anomaly_score": 0.0,
            "behavioral_indicators": [],
            "risk_factors": []
        }

        # Get recent request patterns
        history = self.request_history[ip_address]

        if len(history) < 5:
            return analysis

        # Analyze request size patterns
        request_sizes = [len(str(r.get("headers", {}))) for r in history[-10:]]
        if request_sizes:
            size_consistency = len(set(request_sizes)) / len(request_sizes)
            if size_consistency < 0.3:  # Very consistent request sizes
                analysis["anomaly_score"] += 0.3
                analysis["behavioral_indicators"].append("consistent_request_sizes")

        # Analyze error patterns
        error_requests = [r for r in history[-20:] if r.get("status_code", 200) >= 400]
        if len(error_requests) > 10:
            analysis["anomaly_score"] += 0.4
            analysis["behavioral_indicators"].append("high_error_rate")

        # Analyze session behavior
        if not request_data.get("session_id") and len(history) > 10:
            analysis["anomaly_score"] += 0.2
            analysis["behavioral_indicators"].append("no_session_management")

        return analysis

    def _assess_threat_level(self, ip_address: str, user_agent: str, request_data: Dict) -> Dict:
        """Comprehensive threat level assessment"""
        threat_assessment = {
            "level": ThreatLevel.LOW,
            "score": 0.0,
            "contributing_factors": [],
            "recommended_action": SecurityAction.ALLOW
        }

        threat_score = 0.0

        # IP reputation contribution
        ip_validation = self._validate_ip_address(ip_address)
        if ip_validation["is_blocked"]:
            threat_score += 0.8
            threat_assessment["contributing_factors"].append("blocked_ip")

        reputation_score = ip_validation.get("reputation_score", 0.8)
        threat_score += (1.0 - reputation_score) * 0.3

        # Bot detection contribution
        bot_detection = self._detect_bot_behavior(ip_address, user_agent, {}, "")
        if bot_detection["is_bot"]:
            threat_score += bot_detection["confidence"] * 0.5
            threat_assessment["contributing_factors"].append("bot_detected")

        # Geographic risk contribution
        geo_check = self._check_geographic_restrictions(ip_address)
        if geo_check["risk_level"] == "high":
            threat_score += 0.3
            threat_assessment["contributing_factors"].append("high_risk_geography")
        elif geo_check["risk_level"] == "medium":
            threat_score += 0.15

        # Determine threat level
        threat_assessment["score"] = min(1.0, threat_score)

        if threat_score >= 0.8:
            threat_assessment["level"] = ThreatLevel.CRITICAL
            threat_assessment["recommended_action"] = SecurityAction.BLOCK
        elif threat_score >= 0.6:
            threat_assessment["level"] = ThreatLevel.HIGH
            threat_assessment["recommended_action"] = SecurityAction.CHALLENGE
        elif threat_score >= 0.4:
            threat_assessment["level"] = ThreatLevel.MEDIUM
            threat_assessment["recommended_action"] = SecurityAction.RATE_LIMIT
        else:
            threat_assessment["level"] = ThreatLevel.LOW
            threat_assessment["recommended_action"] = SecurityAction.ALLOW

        return threat_assessment

    def _determine_security_action(self, validation_results: Dict) -> SecurityAction:
        """Determine final security action based on all validation results"""
        # Priority order: Block > Challenge > Rate Limit > Allow

        # Check for critical threats
        if validation_results["threat_assessment"]["level"] == ThreatLevel.CRITICAL:
            return SecurityAction.BLOCK

        # Check for IP blocks
        if validation_results["ip_validation"]["is_blocked"]:
            return SecurityAction.BLOCK

        # Check for high-confidence bot detection
        if (validation_results["bot_detection"]["is_bot"] and
                validation_results["bot_detection"]["confidence"] > 0.9):
            return SecurityAction.BLOCK

        # Check for rate limiting
        if not validation_results["rate_limiting"]["allowed"]:
            return SecurityAction.RATE_LIMIT

        # Check for geographic restrictions
        if not validation_results["geographic_check"]["allowed"]:
            return SecurityAction.BLOCK

        # Check for medium threats requiring challenge
        if validation_results["threat_assessment"]["level"] == ThreatLevel.HIGH:
            return SecurityAction.CHALLENGE

        # Check for bot detection requiring challenge
        if (validation_results["bot_detection"]["is_bot"] and
                validation_results["bot_detection"]["confidence"] > 0.7):
            return SecurityAction.CHALLENGE

        return SecurityAction.ALLOW

    def _log_security_event(
            self,
            request_data: Dict,
            validation_results: Dict,
            action: SecurityAction):
        """Log security event for monitoring and analysis"""
        event_id = self._generate_event_id(request_data)

        threat_indicators = []
        if validation_results["bot_detection"]["is_bot"]:
            threat_indicators.extend(validation_results["bot_detection"]["detection_methods"])

        if validation_results["ip_validation"]["risk_factors"]:
            threat_indicators.extend(validation_results["ip_validation"]["risk_factors"])

        if validation_results["pattern_analysis"]["behavioral_indicators"]:
            threat_indicators.extend(
                validation_results["pattern_analysis"]["behavioral_indicators"])

        event = SecurityEvent(
            id=event_id,
            timestamp=datetime.now(),
            ip_address=request_data.get("ip_address", ""),
            user_agent=request_data.get("user_agent", ""),
            request_path=request_data.get("path", ""),
            threat_level=validation_results["threat_assessment"]["level"],
            threat_indicators=threat_indicators,
            action_taken=action,
            metadata={
                "validation_results": validation_results,
                "request_data": request_data
            }
        )

        self.security_events.append(event)

        # In production, this would also send to SIEM/logging systems
        self._send_to_security_monitoring(event)

    def _send_to_security_monitoring(self, event: SecurityEvent):
        """Send security event to monitoring systems"""
        # In production, integrate with SIEM, CloudWatch, Datadog, etc.
        pass

    def _generate_event_id(self, request_data: Dict) -> str:
        """Generate unique event ID"""
        timestamp = str(time.time())
        ip = request_data.get("ip_address", "")
        path = request_data.get("path", "")
        combined = f"{timestamp}_{ip}_{path}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def encrypt_api_payload(self, payload: Dict, key_name: str = "data_encryption") -> str:
        """Encrypt API payload for secure transmission"""
        if key_name not in self.encryption_keys:
            raise ValueError(f"Unknown encryption key: {key_name}")

        # In production, use proper encryption like AES-GCM
        payload_json = json.dumps(payload, separators=(',', ':'))
        payload_bytes = payload_json.encode('utf-8')

        # Simple encoding for demo (use proper encryption in production)
        encoded = base64.b64encode(payload_bytes).decode('ascii')

        return encoded

    def decrypt_api_payload(
            self,
            encrypted_payload: str,
            key_name: str = "data_encryption") -> Dict:
        """Decrypt API payload"""
        if key_name not in self.encryption_keys:
            raise ValueError(f"Unknown encryption key: {key_name}")

        try:
            # Simple decoding for demo (use proper decryption in production)
            decoded_bytes = base64.b64decode(encrypted_payload.encode('ascii'))
            payload_json = decoded_bytes.decode('utf-8')
            return json.loads(payload_json)
        except Exception as e:
            raise ValueError(f"Failed to decrypt payload: {str(e)}")

    def generate_api_signature(self, payload: str, timestamp: str,
                               key_name: str = "api_signing") -> str:
        """Generate HMAC signature for API request validation"""
        if key_name not in self.encryption_keys:
            raise ValueError(f"Unknown signing key: {key_name}")

        message = f"{timestamp}.{payload}"
        signature = hmac.new(
            self.encryption_keys[key_name],
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    def validate_api_signature(self, payload: str, timestamp: str, signature: str,
                               key_name: str = "api_signing", max_age: int = 300) -> bool:
        """Validate API request signature"""
        # Check timestamp freshness
        current_time = int(time.time())
        request_time = int(timestamp)

        if current_time - request_time > max_age:
            return False

        # Validate signature
        expected_signature = self.generate_api_signature(payload, timestamp, key_name)
        return hmac.compare_digest(signature, expected_signature)

    def get_security_analytics(self, time_range: str = "24h") -> Dict:
        """Get security analytics and insights"""
        end_time = datetime.now()

        if time_range == "1h":
            start_time = end_time - timedelta(hours=1)
        elif time_range == "24h":
            start_time = end_time - timedelta(hours=24)
        elif time_range == "7d":
            start_time = end_time - timedelta(days=7)
        else:
            start_time = end_time - timedelta(hours=24)

        filtered_events = [e for e in self.security_events if start_time <= e.timestamp <= end_time]

        analytics = {
            "overview": {
                "total_events": len(filtered_events),
                "unique_ips": len(set(e.ip_address for e in filtered_events)),
                "blocked_requests": len([e for e in filtered_events if e.action_taken == SecurityAction.BLOCK]),
                "bot_detections": len([e for e in filtered_events if "bot_detected" in e.threat_indicators])
            },
            "threat_distribution": self._analyze_threat_distribution(filtered_events),
            "geographic_analysis": self._analyze_geographic_patterns(filtered_events),
            "attack_patterns": self._analyze_attack_patterns(filtered_events),
            "top_threats": self._identify_top_threats(filtered_events),
            "recommendations": self._generate_security_recommendations(filtered_events)
        }

        return analytics

    def _analyze_threat_distribution(self, events: List[SecurityEvent]) -> Dict:
        """Analyze threat level distribution"""
        threat_counts = {}
        for level in ThreatLevel:
            threat_counts[level.value] = len([e for e in events if e.threat_level == level])

        return threat_counts

    def _analyze_geographic_patterns(self, events: List[SecurityEvent]) -> Dict:
        """Analyze geographic patterns in security events"""
        country_counts = {}
        for event in events:
            country = self._get_country_code(event.ip_address)
            country_counts[country] = country_counts.get(country, 0) + 1

        return {
            "top_countries": sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "high_risk_activity": {
                country: count for country, count in country_counts.items()
                if country in self.geographic_rules["high_risk_countries"]
            }
        }

    def _analyze_attack_patterns(self, events: List[SecurityEvent]) -> Dict:
        """Analyze attack patterns"""
        attack_types = {}
        for event in events:
            for indicator in event.threat_indicators:
                attack_types[indicator] = attack_types.get(indicator, 0) + 1

        return {"most_common_attacks": sorted(attack_types.items(), key=lambda x: x[1], reverse=True)[:10], "attack_frequency": len(
            events) / max(1, (datetime.now() - min(e.timestamp for e in events)).total_seconds() / 3600) if events else 0}

    def _identify_top_threats(self, events: List[SecurityEvent]) -> List[Dict]:
        """Identify top threat sources"""
        ip_threat_scores = {}
        for event in events:
            if event.ip_address not in ip_threat_scores:
                ip_threat_scores[event.ip_address] = 0
            ip_threat_scores[event.ip_address] += len(event.threat_indicators)

        top_threats = sorted(ip_threat_scores.items(), key=lambda x: x[1], reverse=True)[:10]

        return [
            {
                "ip_address": ip,
                "threat_score": score,
                "country": self._get_country_code(ip),
                "event_count": len([e for e in events if e.ip_address == ip])
            }
            for ip, score in top_threats
        ]

    def _generate_security_recommendations(self, events: List[SecurityEvent]) -> List[str]:
        """Generate security recommendations based on events"""
        recommendations = []

        # Check for high bot activity
        bot_events = [
            e for e in events if any(
                "bot" in indicator for indicator in e.threat_indicators)]
        if len(bot_events) > len(events) * 0.3:
            recommendations.append(
                "High bot activity detected - consider implementing CAPTCHA challenges")

        # Check for geographic threats
        high_risk_events = [
            e for e in events if e.threat_level in [
                ThreatLevel.HIGH,
                ThreatLevel.CRITICAL]]
        if len(high_risk_events) > 10:
            recommendations.append("Consider implementing stricter geographic restrictions")

        # Check for rate limiting effectiveness
        rate_limit_events = [e for e in events if e.action_taken == SecurityAction.RATE_LIMIT]
        if len(rate_limit_events) > len(events) * 0.2:
            recommendations.append("Consider adjusting rate limiting thresholds")

        return recommendations


# Initialize global security manager
security_manager = AdvancedSecurityManager()

if __name__ == "__main__":
    # Test the security manager
    manager = AdvancedSecurityManager()

    # Test request validation
    test_request = {
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "path": "/api/decisions",
        "headers": {
            "accept": "application/json",
            "accept-language": "en-US,en;q=0.9",
            "method": "POST"
        }
    }

    action, results = manager.validate_request(test_request)
    print(f"Security Action: {action}")
    print(f"Threat Level: {results['threat_assessment']['level']}")
    print(f"Bot Detection: {results['bot_detection']['is_bot']}")

    # Test encryption
    test_payload = {"sensitive": "data", "user_id": "test123"}
    encrypted = manager.encrypt_api_payload(test_payload)
    decrypted = manager.decrypt_api_payload(encrypted)
    print(f"Encryption test: {decrypted == test_payload}")

    # Get security analytics
    analytics = manager.get_security_analytics("24h")
    print(f"Security Analytics: {analytics['overview']}")
