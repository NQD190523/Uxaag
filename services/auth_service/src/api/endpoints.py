"""
Auth Service API Endpoints
Port: 8006

Layer: router (thin HTTP handlers). Mounted at ``/api/v1/auth``; the gateway exposes
it as ``/auth/*`` with ``protected: false``, so these endpoints do their own
authentication where needed (``/me`` verifies the Bearer JWT, ``/refresh`` uses the
HttpOnly refresh-token cookie). Business logic lives in ``AuthService``.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from shared.logger import get_logger
from schemas.auth_schema import UserRegistration, UserLogin, AuthResponse
from db.database import get_async_session
from services.auth_service import AuthService
from core.security import SecurityUtil
from core.config import setting


# Create Security Util instance
# SECURITY: the signing secret comes from settings (env) and is shared with the gateway.
security_util: SecurityUtil = SecurityUtil(
    jwt_secret_key = setting.JWT_SECRET_KEY,
    jwt_algorithm = setting.JWT_ALGORITHM,
    access_token_expire_minutes = setting.ACCESS_TOKEN_EXPIRE_MINUTES,
    refresh_token_expire_days = setting.REFRESH_TOKEN_EXPIRE_DAYS)

# Get configured logger
logger = get_logger("auth-service")

# Create Auth Service instance with SecurityUtil injected
auth_service: AuthService = AuthService(
    logger=logger,
    security_util=security_util
)

# Refresh-cookie lifetime in seconds. Must follow the REFRESH token lifetime, not the
# access token's ``expires_in``, or the browser drops the cookie after minutes.
REFRESH_COOKIE_MAX_AGE = setting.REFRESH_TOKEN_EXPIRE_DAYS * 86400

# Create router
router = APIRouter(prefix="/auth", tags=["auth"])


# Endpoints
@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user: UserRegistration,
    response: Response,
    session=Depends(get_async_session),
):
    """
    POST /api/v1/auth/register (gateway: POST /auth/register)

    Auth: public (no token required).
    Body: ``UserRegistration`` (email, password >= 8 chars, optional full_name).
    Response: 201 ``AuthResponse`` with user profile and token pair; the refresh token
    is also set as an HttpOnly cookie on the injected FastAPI ``Response``.
    Errors: 400 if validation fails or the email exists, 500 on unexpected failure.

    Register a new user with email and password.
    
    The request is delegated to AuthService which handles:
    - Email validation and uniqueness check
    - Password hashing using bcrypt
    - User model creation in database
    - Conversion from User model to UserProfile schema
    
    Args:
        user: UserRegistration schema with email, password, full_name
        response: FastAPI response object, used only to set the refresh cookie
        session: Async database session
        
    Returns:
        AuthResponse: Response with success status, message, and user profile
        
    Raises:
        HTTPException: 400 if email exists, 500 if registration fails
    """
    logger.info("Registering new user")
    try:
        # Delegate to AuthService for registration logic
        auth_result = await auth_service.register_user(user, session)

        # Set refresh token in cookie (on the injected Response, as login does).
        # SECURITY: HttpOnly; secure=False is dev-only (set True behind HTTPS).
        response.set_cookie(
            key="refresh_token",
            value=auth_result.token.refresh_token,
            httponly=True,
            secure=False,
            max_age=REFRESH_COOKIE_MAX_AGE,
            samesite="lax",
        )

        logger.info("User registered successfully")
        return auth_result
        
    except ValueError as ve:
        # Handle validation errors (email exists, etc.)
        error_msg = str(ve)
        logger.warning("Registration validation error: %s", error_msg)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    except HTTPException:
        # Let deliberate HTTP errors raised above pass through unchanged.
        raise
    except Exception as e:
        # Handle unexpected errors (logger.exception also records the traceback)
        logger.exception("Registration failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again later."
        )


@router.post("/login", response_model=AuthResponse)
async def login_user(user: UserLogin, response: Response, session=Depends(get_async_session)):
    """
    POST /api/v1/auth/login (gateway: POST /auth/login)

    Auth: public.
    Body: ``UserLogin`` (email, password).
    Response: 200 ``AuthResponse`` with profile and token pair; sets the HttpOnly
    ``refresh_token`` cookie.
    Errors: 401 for invalid credentials or a disabled account (any ValueError),
    500 on unexpected failure.

    Login a user with email and password.

    Delegates to AuthService which handles:
    - User lookup by email
    - Password verification
    - Access and refresh token creation
    - User profile and AuthResponse construction
    """
    logger.info("Login attempt")
    try:
        auth_result = await auth_service.login_user(user, session)
        logger.info("User logged in successfully")
        
        # SECURITY: HttpOnly cookie, SameSite=lax; secure=False is dev-only.
        # max_age follows the refresh-token lifetime (not the access token's expires_in).
        response.set_cookie(
            key="refresh_token",
            value=auth_result.token.refresh_token,
            httponly=True,
            secure=False,
            max_age=REFRESH_COOKIE_MAX_AGE,
            samesite="lax",
        )
        
        # Extract tokens from service result
        if not auth_result.token:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token not generated",
            )
        
        return auth_result
    except ValueError as ve:
        error_msg = str(ve)
        logger.warning("Login failed: %s", error_msg)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_msg,
        )
    except HTTPException:
        # Keep the explicit 500 "Token not generated" (and any other HTTPException) as is.
        raise
    except Exception as e:
        logger.exception("Login failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed. Please try again later.",
        )

@router.post("/refresh", response_model=AuthResponse)
async def refresh_token_endpoint(response: Response, request: Request, session=Depends(get_async_session)):
    """
    POST /api/v1/auth/refresh (gateway: POST /auth/refresh)

    Auth: public; identity is proven by the ``refresh_token`` HttpOnly cookie.
    Body: none.
    Response: 200 ``AuthResponse`` with a new access token; the rotated refresh token
    is set in the cookie and blanked in the JSON body.
    Errors: 401 if the cookie is missing or the token is invalid/revoked/expired,
    500 on unexpected failure.
    """
    # SECURITY: the token is read from the cookie only, never from the body or query string.
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Missing refresh token")
    try:
        # Use AuthService to generate new access + refresh tokens
        auth_result = await auth_service.refresh_token(refresh_token, session)

        # Set new refresh token in HTTP-only cookie (rotation)
        response.set_cookie(
            key="refresh_token",
            value=auth_result.token.refresh_token,
            httponly=True,
            secure=False,  # set True in production with HTTPS
            max_age=REFRESH_COOKIE_MAX_AGE,
            samesite="lax",
        )

        # SECURITY: don't expose the refresh token in the JSON body (JS must not see it).
        auth_result.token.refresh_token = None

        return auth_result
    except ValueError as ve:
        error_msg = str(ve)
        logger.warning("Refresh token verification failed: %s", error_msg)
        raise HTTPException(status_code=401, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Refresh token verification failed: %s", e)
        raise HTTPException(status_code=500, detail="Refresh token verification failed. Please try again later.")

@router.get("/me", response_model=AuthResponse)
async def get_me(request: Request, response: Response, session=Depends(get_async_session)):
    """
    GET /api/v1/auth/me (gateway: GET /auth/me)

    Auth: ``Authorization: Bearer <access JWT>``, verified in this handler because the
    gateway does not protect the ``/auth`` prefix.
    Body: none.
    Response: 200 ``AuthResponse`` with the user profile (``token`` is null).
    Errors: 401 for a missing/invalid/expired token or unknown user, 500 on
    unexpected failure.
    """
    # SECURITY: verify signature + expiry here; user_id comes from the verified claims.
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ", 1)[1]
    payload = security_util._verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User not found")
    logger.info("Getting user profile, user_id: %s", user_id)
    try:
        user = await auth_service.get_me(session, int(user_id))
        return AuthResponse(
            success=True,
            message="User profile fetched successfully",
            user=user,
            token=None
        )
    except ValueError as ve:
        error_msg = str(ve)
        logger.warning("User profile fetch failed: %s", error_msg)
        raise HTTPException(status_code=401, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("User profile fetch failed: %s", e)
        raise HTTPException(status_code=500, detail="User profile fetch failed. Please try again later.")
    