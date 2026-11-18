
from cachetools import TTLCache

# Store up to 500 customers' recommendations.
# Each cached result expires after 5 minutes.
recommendation_cache = TTLCache(
    maxsize=500,
    ttl=300,
)


def get_cached_recommendations(user_id, limit):
    """Retrieve cached recommendations."""

    cache_key = (user_id, limit)

    return recommendation_cache.get(cache_key)


def cache_recommendations(user_id, limit, recommendations):
    """Store recommendations in memory."""

    cache_key = (user_id, limit)

    recommendation_cache[cache_key] = recommendations


def clear_recommendation_cache():
    """Clear all cached recommendations."""

    recommendation_cache.clear()
