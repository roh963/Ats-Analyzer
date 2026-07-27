import logging 
import httpx
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger('ats_resume_scorer')

from backend.core.config import SUPABASE_URL, SUPABASE_KEY

def _get_supabase_headers() -> Dict[str, str]:
    """
    Returns the headers required for Supabase API requests.
    """
    if not SUPABASE_KEY or not SUPABASE_URL:
        raise ValueError("Supabase URL and Key must be set in the environment variables.")
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "prefer": "return=representation"
    }

async def save_analysis(user_id: str, filename: str, analysis_result: Dict) -> Optional[Dict]:
    """
    Saves the analysis result to the Supabase database.
    """
    headers = _get_supabase_headers()
    if not headers:
        logger.error("Supabase headers are not set. Cannot save analysis.")
        return None
    def _json_default(o):
        if hasattr(o, 'model_dump'):
            return o.model_dump()
        return str(o)
    
    serializable_result = json.loads(json.dumps(analysis_result, default=_json_default))

    doc ={
        "user_id": user_id,
        "filename": filename,
        "ats_score": serializable_result.get("ats_score",0),
        "keywords_match": serializable_result.get("keywords_match",0),
        "missing_keywords": serializable_result.get("missing_keywords",[]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_result": serializable_result
    }

    url = f"{SUPABASE_URL.strip('/')}/rest/v1/resume_analysis"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=doc)
            response.raise_for_status()
            data = response.json()
            if data and len(data) > 0:
                inserted_id = str(data[0].get("id"))
                logger.info(f"Analysis saved successfully with ID: {inserted_id}")
                return inserted_id
            return None
    except httpx.RequestError as e:
        logger.error(f"An error occurred while saving analysis to Supabase: {e}")
        return None



async def get_user_history(user_id: str) -> List[Dict]:
    """
    Retrieves the analysis history for a specific user from the Supabase database.
    """
    headers = _get_supabase_headers()
    if not headers:
        logger.error("Supabase headers are not set. Cannot retrieve user history.")
        return []
    
    url = f"{SUPABASE_URL.strip('/')}/rest/v1/analysis"
    params = {
        "user_id": f"eq.{user_id}",
        "order": "created_at.desc"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            docs = response.json()
            logger.info(f"Retrieved {len(docs)} records for user {user_id}.")
            results= []
            for doc in docs:
                results.append({
                        "id": str(doc.get("id")),
                        "filename": doc.get("filename", "resume"),
                        "resume_name": doc.get("filename", "resume"),
                        "job_title": "Software Engineer",
                        "ats_score": doc.get("ats_score", 0),
                        "keyword_match": doc.get("keyword_match", 0),
                        "missing_keywords": doc.get("missing_keywords", []),
                        "date": doc.get("created_at", ""),
                        "created_at": doc.get("created_at", ""),
                        "analysis_result": doc.get("analysis_result", {}),
                })
            return results
    except httpx.RequestError as e:
        logger.error(f"An error occurred while retrieving user history from Supabase: {e}")
        return []
    

async def delete_analysis(analysis_id: str, user_id: str) -> bool:
    """
    Deletes a specific analysis record from the Supabase database.
    """
    headers = _get_supabase_headers()
    if not headers:
        logger.error("Supabase headers are not set. Cannot delete analysis.")
        return False
    
    url = f"{SUPABASE_URL.strip('/')}/rest/v1/analysis?id=eq.{analysis_id}&user_id=eq.{user_id}"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(url, headers=headers, params={"id": f"eq.{analysis_id}", "user_id": f"eq.{user_id}"})
            response.raise_for_status()
            logger.info(f"Analysis with ID {analysis_id} deleted successfully.")
            return True
    except httpx.RequestError as e:
        logger.error(f"An error occurred while deleting analysis from Supabase: {e}")
        return False