"""
User profile endpoints (router layer).

Mounted at ``/api/v1/user``; the gateway exposes it as ``/user/*`` with
``protected: true`` and injects the caller's id in the ``x-user-id`` header.
Logic lives in ``services/user_service.py``.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status, Header
from sqlalchemy.ext.asyncio import AsyncSession

from db.database import get_async_session
from shared.logger import get_logger
from services import user_service
from schemas.auth_schema import PasswordChangeRequest, UserPageResponse, UserProfileUpdate

# Get configured logger
logger = get_logger("auth-service")

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/me", status_code=status.HTTP_200_OK, response_model=UserPageResponse)
async def get_user_page_info(
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
):
    """
    GET /api/v1/user/me (gateway: GET /user/me)

    Auth: gateway JWT; caller id from the ``x-user-id`` header.
    Response: 200 ``UserPageResponse`` with the caller's profile.
    Errors: 404 if the user does not exist, 500 on unexpected failure.
    """
    try:
        user_profile = await user_service.get_user_profile_by_id(int(user_id), session)
        return UserPageResponse(
            success=True,
            message="User info fetched successfully",
            user=user_profile,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        # Log details server-side; the client only gets a generic message.
        logger.exception("User info fetch failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.patch("/me", status_code=status.HTTP_200_OK, response_model=UserPageResponse)
async def update_user_page_info(
    profile_update: UserProfileUpdate,
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
):
    """
    PATCH /api/v1/user/me (gateway: PATCH /user/me)

    Auth: gateway JWT; caller id from the ``x-user-id`` header. SECURITY: a user can
    only edit their own profile because the id comes from the header, not the body.
    Body: ``UserProfileUpdate`` (optional username, full_name).
    Response: 200 ``UserPageResponse`` with the updated profile.
    Errors: 400 if the user is missing or the username is taken, 500 on failure.
    """
    try:
        user_profile = await user_service.update_user_profile(
            int(user_id),
            profile_update,
            session,
        )
        return UserPageResponse(
            success=True,
            message="User profile updated successfully",
            user=user_profile,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Profile update failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/me/password", status_code=status.HTTP_200_OK)
async def change_user_password(
    password_request: PasswordChangeRequest,
    session: AsyncSession = Depends(get_async_session),
    user_id: int = Header(..., alias="x-user-id"),
):
    """
    POST /api/v1/user/me/password (gateway: POST /user/me/password)

    Auth: gateway JWT; caller id from the ``x-user-id`` header.
    Body: ``PasswordChangeRequest`` (current_password, new_password >= 8 chars).
    Response: 200 ``{"success": true, "message": ...}``.
    Errors: 400 if the current password is wrong or the user is missing, 500 on failure.
    """
    try:
        await user_service.change_user_password(int(user_id), password_request, session)
        return {"success": True, "message": "Password changed successfully"}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Password change failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/{profile_user_id}", status_code=status.HTTP_200_OK, response_model=UserPageResponse)
async def get_user_profile(
    profile_user_id: int,
    session: AsyncSession = Depends(get_async_session),
):
    """
    GET /api/v1/user/{profile_user_id} (gateway: GET /user/{profile_user_id})

    Auth: gateway JWT required, but any authenticated user may read any profile
    (no ownership check). NOTE: the profile includes the email address.
    Response: 200 ``UserPageResponse``.
    Errors: 404 if the user does not exist, 500 on unexpected failure.
    """
    try:
        user_profile = await user_service.get_user_profile_by_id(int(profile_user_id), session)
        return UserPageResponse(
            success=True,
            message="User profile fetched successfully",
            user=user_profile,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Profile fetch failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/matching-users", status_code=status.HTTP_200_OK)
async def get_matching_users(
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    ):
    """
    POST /api/v1/user/matching-users (gateway: POST /user/matching-users)

    Auth: gateway JWT (the handler itself does not read ``x-user-id``).
    Body: JSON ``{"q": "<search text>"}``.
    Response: 200 list of ``MatchingUsersResponse`` (id, username, email, full name).
    Errors: 400 if ``q`` is missing or blank, 500 on any other failure.
    """
    try:
        payload = await request.json()
        search_query = payload.get("q") if isinstance(payload, dict) else None
        if not search_query or not str(search_query).strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field 'q' is required",
            )

        users = await user_service.get_matching_users(str(search_query).strip(), session)
        # Return matching users
        return users
    except HTTPException:
        # Keep the 400 for a missing ``q`` instead of turning it into a 500.
        raise
    except Exception as e:
        logger.exception("User search failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
