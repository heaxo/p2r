from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import get_settings


async def require_token(
    x_auth_token: str | None = Header(default=None, alias="X-Auth-Token"),
    authorization: str | None = Header(default=None),
) -> None:
    """Check the request token with a simple string comparison.

    Supported forms:
    - X-Auth-Token: your-token
    - Authorization: Bearer your-token
    """

    expected = get_settings().auth_token
    provided = x_auth_token

    if not provided and authorization:
        prefix = "Bearer "
        provided = authorization[len(prefix):].strip() if authorization.startswith(prefix) else authorization.strip()

    if not expected or provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid auth token",
        )
