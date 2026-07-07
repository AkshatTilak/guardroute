from fastapi import FastAPI
from common.config import settings

app = FastAPI(
    title="GuardRoute Gateway",
    description="Contract-Aware AI Gateway & MLOps Control Plane",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Health check endpoint to verify system status."""
    return {
        "status": "healthy",
        "environment": settings.app_env,
        "version": app.version,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.app_env == "development",
    )
