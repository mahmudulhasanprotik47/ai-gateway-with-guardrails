"""API key authentication for client requests (Authorization: Bearer <key>)."""

import hashlib
import hmac
import logging
import re
from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# RFC 6750: "Bearer" 1*SP b64token. The scheme is compared case-insensitively
# in parse_bearer. No nested quantifiers, so matching stays linear in length.
_BEARER_RE = re.compile(r"([A-Za-z]+) +([A-Za-z0-9\-._~+/]+=*)")
_MAX_TOKEN_LENGTH = 512

_MALFORMED_DETAIL = "Missing or malformed Authorization header"
_INVALID_DETAIL = "Invalid API key"


@dataclass(frozen=True)
class AuthenticatedClient:
    """Identity of an authenticated caller.

    ``key_id`` is a non-secret label (a prefix of the key's SHA-256 digest),
    safe to use in logs and rate-limit buckets. The raw key never leaves this
    module.
    """

    key_id: str


def hash_api_key(key: str) -> bytes:
    """Return the SHA-256 digest used to store and compare API keys."""
    return hashlib.sha256(key.encode("utf-8")).digest()


def parse_bearer(header: str | None) -> str | None:
    """Extract the token from an Authorization header value.

    Returns None if the header is missing or empty, uses a scheme other than
    Bearer, or carries a token that is too long or contains characters outside
    RFC 6750's token set (which includes any inner whitespace or tabs).
    Surrounding whitespace is ignored.
    """
    if header is None:
        return None
    match = _BEARER_RE.fullmatch(header.strip())
    if match is None:
        return None
    scheme, token = match.groups()
    if scheme.lower() != "bearer" or len(token) > _MAX_TOKEN_LENGTH:
        return None
    return token


@lru_cache
def _known_key_digests(settings: Settings) -> tuple[bytes, ...]:
    """Digests of the configured keys, computed once per Settings instance.

    This is the one place to change when keys move to persistent storage:
    return the stored digests instead of hashing values from the environment.
    """
    return tuple(hash_api_key(key) for key in settings.api_keys)


def _reject(request: Request, reason: str, detail: str) -> HTTPException:
    """Log a failed attempt (never the header or key) and build the 401."""
    client_host = request.client.host if request.client else "unknown"
    # Path only: a query string could carry a key sent by a misbehaving client.
    logger.warning(
        "Rejected request to %s from %s: %s", request.url.path, client_host, reason
    )
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_api_key(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> AuthenticatedClient:
    """FastAPI dependency that admits only requests bearing a configured key.

    Every failure is a 401 with ``WWW-Authenticate: Bearer``. With no keys
    configured, every request is rejected (fail closed).
    """
    headers = request.headers.getlist("authorization")
    if not headers:
        raise _reject(request, "missing Authorization header", _MALFORMED_DETAIL)
    if len(headers) > 1:
        # Ambiguous: servers and proxies disagree on which header wins.
        raise _reject(request, "multiple Authorization headers", _MALFORMED_DETAIL)

    token = parse_bearer(headers[0])
    if token is None:
        raise _reject(request, "malformed Authorization header", _MALFORMED_DETAIL)

    presented = hash_api_key(token)
    found = False
    for known in _known_key_digests(settings):
        # Compare against every key without stopping early, so timing reveals
        # neither how many keys exist nor which one matched.
        found |= hmac.compare_digest(presented, known)
    if not found:
        raise _reject(request, "unknown API key", _INVALID_DETAIL)

    return AuthenticatedClient(key_id=presented.hex()[:16])
