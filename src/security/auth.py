from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jwt
from fastapi import HTTPException, Request, status

from config.settings import get_settings


def get_bearer_token(request: Request) -> str | None:
    """
    Extract Bearer token from Authorization header.

    Returns the token string or None if missing/invalid.
    """
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def verify_jwt(token: str, secret: str, algorithms: Iterable[str]) -> dict[str, Any]:
    """
    Verify and decode a JWT using HMAC algorithms.

    Raises HTTPException(401) for invalid tokens.
    """
    try:
        claims = jwt.decode(token, secret, algorithms=list(algorithms))  # type: ignore[arg-type]
        if not isinstance(claims, dict):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid JWT claims"
            )
        return claims
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT expired") from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid JWT") from e


def _extract_roles(claims: dict[str, Any]) -> set[str]:
    """
    Extract a set of roles from claims. Supports:
      - roles: List[str]
      - role: str (single role)
      - roles: comma-separated string
    """
    roles: set[str] = set()
    if "roles" in claims:
        raw = claims["roles"]
        if isinstance(raw, list):
            for r in raw:
                if isinstance(r, str) and r.strip():
                    roles.add(r.strip().lower())
        elif isinstance(raw, str):
            for r in raw.split(","):
                rs = r.strip()
                if rs:
                    roles.add(rs.lower())
    if "role" in claims and isinstance(claims["role"], str) and claims["role"].strip():
        roles.add(claims["role"].strip().lower())
    return roles


def require_roles(required: set[str]):
    """
    FastAPI dependency that enforces presence of at least one required role when RBAC is enabled.

    Behavior:
      - If settings.enable_rbac is False, this is a no-op (allows request).
      - If enabled, verifies JWT using APP_JWT_SECRET and allowed algorithms.
      - Denies with 403 if user roles do not include any of required.
      - Denies with 401 if Authorization is missing or invalid.

    Returns decoded claims dict for downstream handlers if verification succeeds.
    """

    required_norm = {r.strip().lower() for r in required if r and r.strip()}

    async def _dependency(request: Request) -> dict[str, Any]:
        settings = get_settings()
        if not getattr(settings, "enable_rbac", False):
            # RBAC disabled; allow through without token
            return {}

        token = get_bearer_token(request)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Missing bearer token"
            )

        secret = getattr(settings, "jwt_secret", None)
        algos: list[str] = getattr(settings, "jwt_algorithms", []) or ["HS256"]
        if not secret:
            # Security hardening: do not allow RBAC enabled without secret
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RBAC misconfigured"
            )

        claims = verify_jwt(token, secret, algos)
        roles = _extract_roles(claims)
        if not roles.intersection(required_norm):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return claims

    return _dependency
