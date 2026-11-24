"""User profile service (service layer).

Business logic for profile lookup/update, user search and password change. Called
by ``api/user.py``; talks to the DB only through ``UserRepository``. Raises
``ValueError`` for domain errors, which the router maps to HTTP status codes.
"""

from typing import Any, List
import hashlib
import bcrypt

from repositories.refresh_token_repository import RefreshTokenRepository
from repositories.user_repository import UserRepository
from models.user import User
from shared.logger import get_logger
from core.security import SecurityUtil

from schemas.auth_schema import MatchingUsersResponse, PasswordChangeRequest, UserProfile, UserProfileUpdate

logger = get_logger("auth-service")

async def get_matching_users(search_query: str, session: Any) -> List[MatchingUsersResponse]:
    """Search users by username, full name, email or numeric user id.

    Args:
        search_query: Non-empty, already stripped search text.
        session: Async DB session.

    Returns:
        Matching users (id, username, email, full name only).
    """
    try:
        user_repository = UserRepository(session)
        users = await user_repository.get_matching_users(search_query)
        return [MatchingUsersResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            full_name=user.full_name
        ) for user in users]
    except Exception as e:
        logger.error(f"Failed to get matching users: {str(e)}")
        raise


async def get_user_profile_by_id(user_id: int, session: Any) -> UserProfile:
    """Build the profile schema for one user.

    Args:
        user_id: Id of the user to load.
        session: Async DB session.

    Returns:
        The user's profile.

    Raises:
        ValueError: If the user does not exist.
    """
    try:
        user_repository = UserRepository(session)
        user = await user_repository.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        return UserProfile(
            user_id=str(user.user_id),
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login,
        )
    except Exception as e:
        logger.error(f"Failed to get user profile by id: {str(e)}")
        raise


async def update_user_profile(user_id: int, update_data: UserProfileUpdate, session: Any) -> UserProfile:
    """Update the editable profile fields (username, full_name) of a user.

    Blank/whitespace-only values are ignored. If nothing remains to update, the
    current profile is returned unchanged.

    Args:
        user_id: Id of the user being edited (comes from the gateway ``x-user-id``).
        update_data: Fields to change.
        session: Async DB session.

    Returns:
        The updated profile.

    Raises:
        ValueError: If the user does not exist or the username is taken by someone else.
    """
    user_repository = UserRepository(session)
    user = await user_repository.get_user_by_id(user_id)
    if not user:
        raise ValueError("User not found")

    payload = update_data.model_dump(exclude_unset=True)
    payload = {
        key: value.strip()
        for key, value in payload.items()
        if isinstance(value, str) and value.strip()
    }
    if not payload:
        return await get_user_profile_by_id(user_id, session)

    username = payload.get("username")
    if username and username != user.username:
        existing = await user_repository.get_user_by_username(username)
        if existing and existing.user_id != user_id:
            raise ValueError("Username is already in use")

    updated = await user_repository.update_user(user_id, payload)
    if not updated:
        raise ValueError("User not found")
    return await get_user_profile_by_id(user_id, session)


async def change_user_password(user_id: int, request: PasswordChangeRequest, session: Any) -> None:
    """Change a user's password after verifying the current one.

    SECURITY: the current password must match before the new hash is stored. After the
    update every refresh token of the user is revoked, so other sessions must log in
    again (already-issued access JWTs stay valid until they expire).

    Args:
        user_id: Id of the user (from the gateway ``x-user-id``).
        request: Current and new password.
        session: Async DB session.

    Raises:
        ValueError: If the user does not exist or the current password is wrong.
    """
    user_repository = UserRepository(session)
    user = await user_repository.get_user_by_id(user_id)
    if not user:
        raise ValueError("User not found")

    # SECURITY: same pre-hash scheme as SecurityUtil (SHA-256 digest, then bcrypt).
    sha256_hash = hashlib.sha256(request.current_password.encode("utf-8")).digest()
    if not bcrypt.checkpw(sha256_hash, user.hashed_password.encode("utf-8")):
        raise ValueError("Current password is incorrect")

    await user_repository.update_user(
        user_id,
        {"hashed_password": SecurityUtil.hash_password(request.new_password)},
    )

    # SECURITY: invalidate all sessions that could refresh with the old credentials.
    await RefreshTokenRepository(session).revoke_all_for_user(user_id)
