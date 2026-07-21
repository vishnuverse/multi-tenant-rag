"""Typed application configuration loaded explicitly from the environment."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_USERS: dict[str, tuple[str, ...]] = {
    "alice@example.com": ("apple",),
    "bob@example.com": ("microsoft", "google"),
    "charlie@example.com": ("meta", "amazon"),
}
USERS: Mapping[str, tuple[str, ...]] = MappingProxyType(_DEFAULT_USERS.copy())

EmbeddingCost = tuple[float, float]
_DEFAULT_EMBEDDING_PRICING: dict[str, EmbeddingCost] = {
    "openai/text-embedding-3-small": (0.02, 0.0),
}

_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})
_FALSE_VALUES = frozenset({"false", "0", "no", "off"})


class ConfigurationError(RuntimeError):
    """Raised when configuration required for an operation is unavailable."""


def _normalize_users(
    users: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    normalized_users: dict[str, tuple[str, ...]] = {}
    for email, scopes in users.items():
        normalized_email = email.strip().lower()
        if not normalized_email:
            raise ConfigurationError("USERS contains a blank normalized email")
        if normalized_email in normalized_users:
            raise ConfigurationError("USERS contains a duplicate normalized email")

        normalized_scopes: list[str] = []
        seen_scopes: set[str] = set()
        for scope in scopes:
            normalized_scope = scope.strip().lower()
            if not normalized_scope:
                raise ConfigurationError("USERS contains a blank normalized scope")
            if normalized_scope not in seen_scopes:
                seen_scopes.add(normalized_scope)
                normalized_scopes.append(normalized_scope)
        normalized_users[normalized_email] = tuple(normalized_scopes)
    return normalized_users


def allowed_companies(email: str) -> tuple[str, ...]:
    """Return the default company scopes assigned to an email address."""

    return USERS.get(email.strip().lower(), ())


def _parse_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of true/1/yes/on or false/0/no/off, got {raw!r}"
    )


def _parse_positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_threshold(environ: Mapping[str, str]) -> float:
    name = "SEMANTIC_CACHE_THRESHOLD"
    raw = environ.get(name)
    try:
        value = 0.96 if raw is None else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number from 0 to 1") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number from 0 to 1")
    return value


def _parse_users(raw: str | None) -> dict[str, tuple[str, ...]]:
    if raw is None or not raw.strip():
        return _normalize_users(_DEFAULT_USERS)
    try:
        decoded = json.loads(raw, object_pairs_hook=_parse_user_object)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            "USERS must be a JSON object of email-to-scope lists"
        ) from exc
    if not isinstance(decoded, dict):
        raise ConfigurationError("USERS must be a JSON object of email-to-scope lists")
    return decoded


def _parse_user_object(
    pairs: list[tuple[str, object]],
) -> dict[str, tuple[str, ...]]:
    users: dict[str, tuple[str, ...]] = {}
    for email, scopes in pairs:
        if (
            not isinstance(email, str)
            or not isinstance(scopes, list)
            or not all(isinstance(scope, str) for scope in scopes)
        ):
            raise ConfigurationError(
                "USERS must be a JSON object of email-to-scope lists"
            )
        normalized_email = email.strip().lower()
        if normalized_email in users:
            raise ConfigurationError("USERS contains a duplicate normalized email")
        users[email] = tuple(scopes)
    return _normalize_users(users)


def _optional(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name, "").strip()
    return value or None


def _default_chat_model(environ: Mapping[str, str]) -> str:
    raw = environ.get("DEFAULT_CHAT_MODEL")
    if raw is None:
        legacy_names = [
            name for name in ("MODEL_FAST", "MODEL_ANSWER") if name in environ
        ]
        if legacy_names:
            legacy = " and ".join(legacy_names)
            raise ConfigurationError(
                f"{legacy} has been replaced by DEFAULT_CHAT_MODEL; "
                "set DEFAULT_CHAT_MODEL explicitly"
            )
        # Fast + grounded: ~3s turns on Apple Q&A; better prose than Flash Lite.
        return "openai/gpt-4.1-mini"

    model = raw.strip()
    if not model:
        raise ConfigurationError("DEFAULT_CHAT_MODEL cannot be blank")
    return model


def _path(environ: Mapping[str, str], name: str, default: Path) -> Path:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    return Path(raw).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings."""

    users: Mapping[str, tuple[str, ...]]
    model_embed: str
    default_chat_model: str
    local_embed_model: str
    reranker_model: str
    use_local_embedding: bool
    multi_query: bool
    hyde: bool
    contextual_chunking: bool
    langfuse_enabled: bool
    project_root: Path
    data_dir: Path
    pdf_dir: Path
    chroma_dir: Path
    bm25_index_path: Path
    logs_dir: Path
    chunk_max_chars: int
    chunk_min_chars: int
    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    final_top_k: int
    rrf_k: int
    embed_batch_size: int
    semantic_cache_threshold: float
    semantic_cache_max_entries: int
    semantic_cache_ttl_seconds: int
    model_catalog_ttl_seconds: int
    openrouter_base_url: str
    openrouter_app_url: str | None
    openrouter_app_title: str | None
    openrouter_api_key: str = field(repr=False)

    def __post_init__(self) -> None:
        """Defensively freeze nested configuration mappings."""

        users = MappingProxyType(_normalize_users(self.users))
        object.__setattr__(self, "users", users)

    def allowed_companies(self, email: str) -> tuple[str, ...]:
        """Return company scopes assigned to an email address."""

        return self.users.get(email.strip().lower(), ())

    def require_openrouter_api_key(self) -> str:
        """Return the key or fail when a hosted model call requires it."""

        if not self.openrouter_api_key:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required for hosted model calls"
            )
        return self.openrouter_api_key

    def embedding_cost(self, model: str) -> EmbeddingCost:
        """Return built-in embedding costs, defaulting unknown models to zero."""

        return _DEFAULT_EMBEDDING_PRICING.get(model, (0.0, 0.0))

    @property
    def checkpoint_db_path(self) -> Path:
        """SQLite file for LangGraph conversation checkpoints."""

        return self.data_dir / "checkpoints.sqlite"

    @property
    def chainlit_db_path(self) -> Path:
        """SQLite file for Chainlit thread/message persistence."""

        return self.data_dir / "chainlit.sqlite"


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Build settings from a supplied mapping or the process environment."""

    env = os.environ if environ is None else environ
    data_dir = _path(env, "DATA_DIR", PROJECT_ROOT / "data")
    fusion_top_k = _parse_positive_int(env, "FUSION_TOP_K", 20)
    final_top_k = _parse_positive_int(env, "FINAL_TOP_K", 5)
    if final_top_k > fusion_top_k:
        raise ValueError("FINAL_TOP_K must be less than or equal to FUSION_TOP_K")

    chunk_max_chars = _parse_positive_int(env, "CHUNK_MAX_CHARS", 1200)
    chunk_min_chars = _parse_positive_int(env, "CHUNK_MIN_CHARS", 200)
    if chunk_min_chars > chunk_max_chars:
        raise ValueError(
            "CHUNK_MIN_CHARS must be less than or equal to CHUNK_MAX_CHARS"
        )

    return Settings(
        users=MappingProxyType(_parse_users(env.get("USERS"))),
        model_embed=env.get("MODEL_EMBED", "openai/text-embedding-3-small"),
        default_chat_model=_default_chat_model(env),
        local_embed_model=env.get("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        reranker_model=env.get("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        use_local_embedding=_parse_bool(env, "USE_LOCAL_EMBEDDING", False),
        multi_query=_parse_bool(env, "MULTI_QUERY", True),
        hyde=_parse_bool(env, "HYDE", False),
        contextual_chunking=_parse_bool(env, "CONTEXTUAL_CHUNKING", False),
        langfuse_enabled=_parse_bool(env, "LANGFUSE_ENABLED", False),
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        pdf_dir=data_dir / "pdfs",
        chroma_dir=data_dir / "chroma",
        bm25_index_path=data_dir / "bm25_index.pkl",
        logs_dir=data_dir / "logs",
        chunk_max_chars=chunk_max_chars,
        chunk_min_chars=chunk_min_chars,
        dense_top_k=_parse_positive_int(env, "DENSE_TOP_K", 20),
        sparse_top_k=_parse_positive_int(env, "SPARSE_TOP_K", 20),
        fusion_top_k=fusion_top_k,
        final_top_k=final_top_k,
        rrf_k=_parse_positive_int(env, "RRF_K", 60),
        embed_batch_size=_parse_positive_int(env, "EMBED_BATCH_SIZE", 100),
        semantic_cache_threshold=_parse_threshold(env),
        semantic_cache_max_entries=_parse_positive_int(
            env, "SEMANTIC_CACHE_MAX_ENTRIES", 256
        ),
        semantic_cache_ttl_seconds=_parse_positive_int(
            env, "SEMANTIC_CACHE_TTL_SECONDS", 3600
        ),
        model_catalog_ttl_seconds=_parse_positive_int(
            env, "MODEL_CATALOG_TTL_SECONDS", 900
        ),
        openrouter_base_url=env.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ),
        openrouter_app_url=_optional(env, "OPENROUTER_APP_URL"),
        openrouter_app_title=_optional(env, "OPENROUTER_APP_TITLE"),
        openrouter_api_key=env.get("OPENROUTER_API_KEY", "").strip(),
    )
