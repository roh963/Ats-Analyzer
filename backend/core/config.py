import os 
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   


# API METADATA
APP_TITLE = 'ATS RESUME ANALYZER API'
APP_VERSION = '1.0.0'
APP_DESCRIPTION = 'An API for analyzing resumes and extracting relevant information for ATS (Applicant Tracking System) compatibility.'

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# File
MAX_FILE_SIZE_MB = 5  # Maximum file size in megabytes
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes

# Supported MIME types and their short Names
SUPPORTED_MIME_TYPES = {
    "application/pdf": "PDF",
    "application/msword": "DOC",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
    "text/plain": "TXT",
}

SUPPORTED_FILE_EXTENSIONS = {'.pdf', '.doc', '.docx', '.txt'}

SPACY_MODEL_PRIMARY = "en_core_web_md" # better accuracy
SPACY_MODEL_SECONDARY = "en_core_web_sm"
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")  # Default model if not set in .env

# score component weights
SCORE_WEIGHTS = {
    "formatting": 20,
    "keywords": 25,
    "content": 25,
    "skill_validation": 15,
    "ats_compatibility": 15
}   

JD_KEYWORDS_WEIGHTS = 0.6
JD_SKILLS_WEIGHTS = 0.4

SUPABASE_URL       = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY       = os.getenv('SUPABASE_KEY', '')          # service_role — DB writes (bypasses RLS)
SUPABASE_ANON_KEY  = os.getenv('SUPABASE_ANON_KEY', '')     # public anon — frontend auth calls
SUPABASE_JWT_SECRET= os.getenv('SUPABASE_JWT_SECRET', '')   # used by backend to verify access tokens
GROQ_API_KEY       = os.getenv('GROQ_API_KEY', '')