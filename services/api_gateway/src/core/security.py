"""JWT verification for the API gateway (middleware layer).

Provides ``JWTBearer``, a FastAPI-style callable that reads the ``Authorization:
Bearer`` header and verifies the token with the shared secret. This is the gateway's
authentication boundary: downstream services trust the ``x-user-id`` header that
``api/proxy.py`` derives from the claims returned here.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

from core.config import JWT_SECRET_KEY, JWT_ALGORITHM
from shared.logger import get_logger

logger = get_logger("api-gateway")

# auto_error=False: missing Authorization yields None instead of raising; we handle that below.
security = HTTPBearer(auto_error=False)

# Generic client-facing detail for every bad-token case, so responses do not reveal why.
_INVALID_TOKEN_DETAIL = "Invalid or expired token"


class JWTBearer:
    """Callable dependency that authenticates a request via a Bearer JWT.

    Attributes:
        auto_error: Stored for API compatibility; not used by ``__call__``.
        require_auth: If True, a missing token raises 401; if False, it returns None.
    """

    def __init__(
        self,
        auto_error: bool = True,
        require_auth: bool = True,
    ):
        self.auto_error = auto_error
        self.require_auth = require_auth

    async def __call__(self, request: Request) -> Optional[Dict[str, Any]]:
        """Verify the request's Bearer token.

        SECURITY: the signature and ``exp`` are checked against ``JWT_SECRET_KEY`` using
        only the configured algorithm. The ``type`` claim must be ``"access"``, so
        refresh tokens (also signed with the shared secret) are rejected.

        Args:
            request: Incoming request.

        Returns:
            The decoded JWT claims, or None when no token is sent and
            ``require_auth`` is False.

        Raises:
            HTTPException: 401 if the token is missing (when required), expired,
                invalid, or not an access token. The detail for bad tokens is always the
                generic "Invalid or expired token"; the real reason is only logged.
        """
        auth: Optional[HTTPAuthorizationCredentials] = await security(request)
        if auth is None:
            if self.require_auth:
                raise HTTPException(
                    status_code=401,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return None

        token = auth.credentials
        try:
            # SECURITY: algorithms is pinned to the configured one (blocks alg=none / confusion).
            # NOTE: core/config.py refuses to start without JWT_SECRET_KEY.
            payload = jwt.decode(jwt=token, key=JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        except ExpiredSignatureError:
            logger.warning("Rejected token: expired")
            raise HTTPException(status_code=401, detail=_INVALID_TOKEN_DETAIL)
        except InvalidTokenError as e:
            logger.warning("Rejected token: invalid (%s)", e)
            raise HTTPException(status_code=401, detail=_INVALID_TOKEN_DETAIL)
        except Exception as e:
            logger.warning("Rejected token: unexpected decode error (%s)", e)
            raise HTTPException(status_code=401, detail=_INVALID_TOKEN_DETAIL)

        # The auth service issues type=access|refresh; only access tokens may call APIs.
        if payload.get("type") != "access":
            logger.warning(
                "Rejected token: type claim is %r, expected 'access'", payload.get("type")
            )
            raise HTTPException(status_code=401, detail=_INVALID_TOKEN_DETAIL)
        return payload

# Convenience instances: require_auth rejects missing tokens, optional_auth lets them through.
require_auth = JWTBearer(require_auth=True)
optional_auth = JWTBearer(require_auth=False)