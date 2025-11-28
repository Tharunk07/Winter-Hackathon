from fastapi import FastAPI
import config as Config
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.requests import Request

logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title="MultiModel RAG",
    summary="MultiModel RAG Apis",
    version=Config.API_VERSION,
    redoc_url=f"{Config.API_PREFIX}/redoc",
    docs_url=f"{Config.API_PREFIX}/docs",
    openapi_url=f"{Config.API_PREFIX}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.allow_origins,
    allow_credentials=Config.allow_credentials,
    allow_methods=Config.allow_methods,
    allow_headers=Config.allow_headers,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"status": "error", "detail": exc.errors()},
    )


from app import routes
