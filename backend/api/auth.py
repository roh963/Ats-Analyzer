import logging
import jwt   # library to encode/decode JSON Web Tokens (JWTs)
from fastapi import Depends, HTTPException, status                    # FastAPI tools for dependency injection and errors
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # extracts "Bearer <token>" from request headers
from backend.core.config import SUPABASE_JWT_SECRET, SUPABASE_URL      # secret key + URL from config

logger = logging.getLogger('ats_resume_scorer')

# scheme to read the Authorization header; auto_error=False means it won't crash if missing
_bearer_scheme = HTTPBearer(auto_error=False)

# supported algorithms for asymmetric (public/private key) JWT signing
_ASYMMETRIC_ALGS = ['ES256', 'RS256']

# will hold a JWKS client (fetches public keys), starts empty until initialized
_jwks_client: jwt.PyJWKClient | None = None




"""
This function creates (or reuses) a client that fetches public keys from Supabase, used to verify JWT tokens signed with asymmetric algorithms. It only creates the client once and reuses it afterward.
"""
def _get_jwks_client() -> jwt.PyJWKClient | None:
    global _jwks_client   # refer to the shared variable defined outside this function

    # if already created, reuse it instead of making a new one
    if _jwks_client is not None:
        return _jwks_client

    # can't build the client without a Supabase URL
    if not SUPABASE_URL:
        return None

    # build the standard JWKS endpoint URL for Supabase auth
    jwks_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"

    # create the client; cache_keys=True avoids refetching keys every request
    _jwks_client = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=3600)
    return _jwks_client



"""
This function checks if a login token (JWT) is genuine and untampered, using different verification methods depending on how it was signed. Used to confirm a user is really who they claim to be.
"""
def _verify_token(token: str) -> dict:
    # read the token's header without verifying it yet, just to check the algorithm used
    header = jwt.get_unverified_header(token)
    alg = header.get('alg')

    # case 1: token signed with public/private key (asymmetric)
    if alg in _ASYMMETRIC_ALGS:
        jwks_client = _get_jwks_client()
        if jwks_client is None:
            raise jwt.InvalidTokenError(
                'SUPABASE_URL not configured — cannot fetch JWKS to verify token'
            )
        # fetch the correct public key that matches this token
        signing_key = jwks_client.get_signing_key_from_jwt(token).key
        return jwt.decode(
            token,
            signing_key,
            algorithms=_ASYMMETRIC_ALGS,
            audience='authenticated',   # token must be meant for "authenticated" users
        )

    # case 2: token signed with a shared secret (symmetric)
    if alg == 'HS256':
        if not SUPABASE_JWT_SECRET:
            raise jwt.InvalidTokenError(
                'HS256 token received but SUPABASE_JWT_SECRET is not configured'
            )
        return jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=['HS256'],
            audience='authenticated',
        )

    # case 3: unknown/unsupported signing algorithm
    raise jwt.InvalidTokenError(f'Unsupported JWT algorithm: {alg}')



"""
This function is a FastAPI dependency that runs on protected routes — it extracts the login token from the request, verifies it's valid, and returns the logged-in user's ID. If anything is wrong (missing token, expired, invalid), it stops the request with an appropriate error.
"""
def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    # reject if no token was sent at all
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Missing Authorization: Bearer <token> header',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    # reject if the server itself isn't set up to verify tokens
    if not SUPABASE_URL and not SUPABASE_JWT_SECRET:
        logger.error('Neither SUPABASE_URL (for JWKS) nor SUPABASE_JWT_SECRET configured — cannot verify tokens')
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Auth not configured on the server',
        )

    try:
        payload = _verify_token(creds.credentials)   # actually verify the token's signature

    except jwt.ExpiredSignatureError:
        # token was valid but has expired
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Token expired — sign in again',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    except jwt.InvalidTokenError as exc:
        # token is malformed, tampered, or otherwise invalid
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f'Invalid token: {exc}',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    except Exception as exc:
        # catch network/other errors (e.g. fetching public keys failed)
        # treated as 401 so it looks like an auth problem, not a server crash
        logger.warning(f'JWT verification failed: {exc}')
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f'Token verification failed: {exc}',
            headers={'WWW-Authenticate': 'Bearer'},
        )

    # extract user ID from the token's "subject" claim
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Token missing subject claim',
        )

    return user_id   # this becomes available to any route using this dependency
