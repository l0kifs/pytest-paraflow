from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration settings.
    """
    model_config = SettingsConfigDict(
        env_prefix="PYTEST_PARAFLOW__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application settings
    app_name: str = Field(default="pytest-paraflow", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")

    # Paraflow option defaults
    shard_id: int | None = Field(
        default=None,
        description="Default value for --paraflow-shard-id",
    )
    num_shards: int | None = Field(
        default=None,
        description="Default value for --paraflow-num-shards",
    )
    target_shard_size: int | None = Field(
        default=None,
        description="Default value for --paraflow-target-shard-size",
    )
    group_marker: list[str] = Field(
        default_factory=list,
        description="Default values for --paraflow-group-marker",
    )


def get_settings() -> Settings:
    """
    Get configuration settings.
    """
    return Settings()
