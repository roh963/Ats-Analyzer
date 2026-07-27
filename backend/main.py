import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import (
    APP_TITLE,
    APP_VERSION,
    APP_DESCRIPTION,
    ALLOWED_ORIGINS,
    SPACY_MODEL_PRIMARY,
    SPACY_MODEL_SECONDARY,
    SENTENCE_TRANSFORMER_MODEL
)
from backend.api.routes import router 

logger = logging.getLogger('ats_resume_analyzer')


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the  Ats analyzerapplication...")
    logger.info(f"Loading spaCy model: {SPACY_MODEL_PRIMARY}...")
    import spacy
    try:
        app.state.nlp = spacy.load(SPACY_MODEL_PRIMARY)
        logger.info(f"Successfully loaded spaCy model: {SPACY_MODEL_PRIMARY}")
    except OSError:
        logger.warning(f"Primary spaCy model '{SPACY_MODEL_PRIMARY}' not found. Loading secondary model '{SPACY_MODEL_SECONDARY}'...")
        app.state.nlp = spacy.load(SPACY_MODEL_SECONDARY)
        logger.info(f"Successfully loaded secondary spaCy model: {SPACY_MODEL_SECONDARY}")
    logger.info(f"Loading Sentence Transformer model: {SENTENCE_TRANSFORMER_MODEL}...")
    from sentence_transformers import SentenceTransformer   
    app.state.sentence_transformer_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    logger.info(f"Successfully loaded Sentence Transformer model: {SENTENCE_TRANSFORMER_MODEL}")

    logger.info('all models loaded successfully. Application is ready to accept requests.')
    yield # yeld control back to FastAPI for handling requests
    logger.info("Shutting down the application...")

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[*ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run('backend.main:app', host="0.0.0.0", port=8000 , reload=True, log_level="info")

