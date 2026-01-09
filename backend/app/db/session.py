"""
Database session management and initialization.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from typing import Generator, Optional
from app.core.config import settings

# Try to import redis, but make it optional
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️  Redis not available - running in demo mode without caching")


# Database Engine (SQLite for demo)
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    pool_pre_ping=True,  # Verify connections before using
    echo=settings.DEBUG
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting database session.
    Automatically closes session after request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Redis Clients (Optional - for production)
redis_client: Optional[redis.Redis] = None
redis_session_client: Optional[redis.Redis] = None
redis_cache_client: Optional[redis.Redis] = None
redis_queue_client: Optional[redis.Redis] = None

if REDIS_AVAILABLE:
    try:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        redis_session_client = redis.from_url(
            settings.REDIS_URL.replace("/0", f"/{settings.REDIS_SESSION_DB}"),
            decode_responses=True
        )
        redis_cache_client = redis.from_url(
            settings.REDIS_URL.replace("/0", f"/{settings.REDIS_CACHE_DB}"),
            decode_responses=True
        )
        redis_queue_client = redis.from_url(
            settings.REDIS_URL.replace("/0", f"/{settings.REDIS_QUEUE_DB}"),
            decode_responses=False  # For binary data
        )
        print("✓ Redis connected successfully")
    except Exception as e:
        print(f"⚠️  Redis connection failed: {e} - running without cache")
        redis_client = None
        redis_session_client = None
        redis_cache_client = None
        redis_queue_client = None


def get_redis() -> redis.Redis:
    """Get Redis client for general use."""
    return redis_client


def get_redis_cache() -> redis.Redis:
    """Get Redis client for caching."""
    return redis_cache_client


def get_redis_queue() -> redis.Redis:
    """Get Redis client for queue operations."""
    return redis_queue_client

