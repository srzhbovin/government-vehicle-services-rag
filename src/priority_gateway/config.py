"""Configuration loading and request-to-group classification."""

from __future__ import annotations

import hmac
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?}")


class ConfigurationError(ValueError):
    """Raised when the gateway policy is invalid."""


class AuthenticationError(PermissionError):
    """Raised when no configured client identity matches the request."""


@dataclass(frozen=True)
class GroupPolicy:
    name: str
    priority: int


@dataclass(frozen=True)
class MatchRule:
    group: str
    bearer_tokens: tuple[str, ...] = field(default=(), repr=False)
    api_key_prefixes: tuple[str, ...] = ()
    headers: tuple[tuple[str, str], ...] = ()

    def matches(self, headers: Mapping[str, str]) -> bool:
        normalized = {key.lower(): value for key, value in headers.items()}
        authorization = normalized.get("authorization", "")
        token = (
            authorization[7:].strip()
            if authorization.lower().startswith("bearer ")
            else ""
        )

        token_match = any(
            token and hmac.compare_digest(token, candidate)
            for candidate in self.bearer_tokens
            if candidate
        )
        prefix_match = any(
            token.startswith(prefix) for prefix in self.api_key_prefixes if prefix
        )
        header_match = bool(self.headers) and all(
            hmac.compare_digest(normalized.get(name.lower(), ""), expected)
            for name, expected in self.headers
        )
        return token_match or prefix_match or header_match


@dataclass(frozen=True)
class SchedulerConfig:
    strategy: str
    max_concurrency: int
    max_queue_size: int
    queue_timeout_seconds: float
    aging_interval_seconds: float
    starvation_timeout_seconds: float


@dataclass(frozen=True)
class GatewayConfig:
    upstream_url: str
    upstream_api_key: str | None = field(repr=False)
    upstream_timeout_seconds: float
    require_matching_rule: bool
    default_group: str
    groups: dict[str, GroupPolicy]
    rules: tuple[MatchRule, ...]
    scheduler: SchedulerConfig

    def classify(self, headers: Mapping[str, str]) -> GroupPolicy:
        for rule in self.rules:
            if rule.matches(headers):
                return self.groups[rule.group]
        if self.require_matching_rule:
            raise AuthenticationError("Unknown gateway API key")
        return self.groups[self.default_group]


class ConfigManager:
    """Loads policies from YAML and reloads them when the file changes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mtime_ns: int | None = None
        self._config: GatewayConfig | None = None
        self.last_reload_error: str | None = None

    def get(self) -> GatewayConfig:
        try:
            stat = self.path.stat()
        except OSError as error:
            self.last_reload_error = str(error)
            if self._config is not None:
                return self._config
            raise ConfigurationError(
                f"Cannot stat gateway policy {self.path}: {error}"
            ) from error
        if self._config is not None and stat.st_mtime_ns == self._mtime_ns:
            return self._config
        with self._lock:
            try:
                stat = self.path.stat()
            except OSError as error:
                self.last_reload_error = str(error)
                if self._config is not None:
                    return self._config
                raise ConfigurationError(
                    f"Cannot stat gateway policy {self.path}: {error}"
                ) from error
            if self._config is None or stat.st_mtime_ns != self._mtime_ns:
                try:
                    self._config = load_config(self.path)
                    self._mtime_ns = stat.st_mtime_ns
                    self.last_reload_error = None
                except ConfigurationError as error:
                    self.last_reload_error = str(error)
                    if self._config is None:
                        raise
        return self._config


def load_config(path: str | Path) -> GatewayConfig:
    source = Path(path)
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(
            f"Cannot load gateway policy {source}: {error}"
        ) from error
    document = _expand_environment(document)
    if not isinstance(document, dict):
        raise ConfigurationError("Gateway policy root must be a mapping")

    upstream = _mapping(document, "upstream")
    authentication = _mapping(document, "authentication")
    scheduler_data = _mapping(document, "scheduler")
    raw_groups = _mapping(document, "groups")
    if not raw_groups:
        raise ConfigurationError("At least one user group must be configured")

    groups: dict[str, GroupPolicy] = {}
    for name, raw_policy in raw_groups.items():
        if not isinstance(raw_policy, dict):
            raise ConfigurationError(f"Group {name!r} must be a mapping")
        try:
            priority = int(raw_policy.get("priority", 100))
        except (TypeError, ValueError) as error:
            raise ConfigurationError(
                f"Group {name!r} priority must be an integer"
            ) from error
        if priority < 0:
            raise ConfigurationError(f"Group {name!r} priority cannot be negative")
        groups[str(name)] = GroupPolicy(name=str(name), priority=priority)

    default_group = str(document.get("default_group", "standard"))
    if default_group not in groups:
        raise ConfigurationError(f"Unknown default_group: {default_group}")

    rules = []
    for raw_rule in document.get("rules") or []:
        if not isinstance(raw_rule, dict):
            raise ConfigurationError("Every rule must be a mapping")
        group = str(raw_rule.get("group", ""))
        if group not in groups:
            raise ConfigurationError(f"Rule refers to unknown group: {group}")
        match = raw_rule.get("match") or {}
        if not isinstance(match, dict):
            raise ConfigurationError("Rule match must be a mapping")
        raw_headers = match.get("headers") or {}
        if not isinstance(raw_headers, dict):
            raise ConfigurationError("Rule headers must be a mapping")
        rules.append(
            MatchRule(
                group=group,
                bearer_tokens=_strings(match.get("bearer_tokens")),
                api_key_prefixes=_strings(match.get("api_key_prefixes")),
                headers=tuple(
                    (str(key).lower(), str(value)) for key, value in raw_headers.items()
                ),
            )
        )

    strategy = str(scheduler_data.get("strategy", "priority_aging"))
    if strategy not in {"fifo", "priority_aging"}:
        raise ConfigurationError("scheduler.strategy must be fifo or priority_aging")
    scheduler = SchedulerConfig(
        strategy=strategy,
        max_concurrency=_positive_int(scheduler_data, "max_concurrency", 4),
        max_queue_size=_positive_int(scheduler_data, "max_queue_size", 200),
        queue_timeout_seconds=_positive_float(
            scheduler_data, "queue_timeout_seconds", 120.0
        ),
        aging_interval_seconds=_positive_float(
            scheduler_data, "aging_interval_seconds", 1.0
        ),
        starvation_timeout_seconds=_positive_float(
            scheduler_data, "starvation_timeout_seconds", 15.0
        ),
    )
    upstream_url = str(upstream.get("url", "http://litellm:4000")).rstrip("/")
    if not upstream_url:
        raise ConfigurationError("upstream.url cannot be empty")
    api_key = str(upstream.get("api_key") or "").strip() or None
    return GatewayConfig(
        upstream_url=upstream_url,
        upstream_api_key=api_key,
        upstream_timeout_seconds=_positive_float(upstream, "timeout_seconds", 180.0),
        require_matching_rule=_as_bool(authentication.get("required"), True),
        default_group=default_group,
        groups=groups,
        rules=tuple(rules),
        scheduler=scheduler,
    )


def _mapping(document: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = document.get(key) or {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, list):
        raise ConfigurationError("Rule token lists must be strings or lists")
    return tuple(str(item) for item in value if str(item))


def _positive_int(mapping: Mapping[str, Any], key: str, default: int) -> int:
    try:
        value = int(mapping.get(key, default))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{key} must be an integer") from error
    if value < 1:
        raise ConfigurationError(f"{key} must be positive")
    return value


def _positive_float(mapping: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = float(mapping.get(key, default))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{key} must be a number") from error
    if value <= 0:
        raise ConfigurationError(f"{key} must be positive")
    return value


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError("authentication.required must be a boolean")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        return os.environ.get(name, default or "")

    return _ENV_PATTERN.sub(replace, value)
