from config.settings import settings

try:
    if (
        getattr(settings, "langchain_api_key", None)
        and str(getattr(settings, "langchain_tracing_v2", "true")).lower() in {"1", "true", "yes"}
    ):
        from langsmith import traceable
    else:
        def traceable(*args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
except Exception:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
