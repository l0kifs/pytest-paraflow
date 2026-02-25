"""Domain-specific exceptions for test distribution."""


class InvalidShardingConfigurationError(ValueError):
    """Raised when sharding options are invalid."""


class MissingShardingConfigurationError(ValueError):
    """Raised when sharding options are incomplete."""
