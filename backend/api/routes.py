import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile  # FastAPI core tools

from backend.api.auth import get_current_user   # dependency to verify logged-in user

from backend.models.schemas import (
    AnalysisResponse, ComponentScores, JDComparison, SkillValidationDetails   # response data structures
)

from backend.utils.file_utils import (
    get_default_grammar_results,
    get_default_location_results,
    get_default_skill_validation_results,   # fallback values when analysis isn't run yet
)

logger = logging.getLogger('ats_resume_scorer')

# creates a group of routes, all starting with /api/v1, tagged as "Analysis" in API docs
router = APIRouter(prefix='/api/v1', tags=['Analysis'])


"""
This function removes leading emoji symbols (like ✅, ❌, ⚠️) from the start of a text string. Used to strip decorative icons before displaying plain text elsewhere.
"""

def _clean(text: str) -> str:
    # remove each emoji if it appears at the very start of the text
    for prefix in ('✅', '🌟', '❌', '⚠️', '📝', '🔴', '🟡', '🟢', '🟠', '👍'):
        text = text.lstrip(prefix)   # lstrip removes matching characters from the left side
    return text.strip()   # remove any leftover spaces at start/end

"""
This is the main API endpoint that handles a resume file upload. It reads the file, runs the full analysis pipeline, converts the results into the proper response format, saves the analysis to history, and sends the final response back to the user.
"""
@router.post('/analyze-resume', response_model=AnalysisResponse)
async def analyze_resume(
    request: Request,
    resume: UploadFile = File(..., description='Resume file — PDF or DOCX, max 5 MB'),
    job_description: str = Form('', description='Job description text (optional)'),
    user_id: str = Depends(get_current_user),   # requires user to be logged in
):
    warnings: List[str] = []
    nlp      = request.app.state.nlp        # shared NLP model loaded at app startup
    embedder = request.app.state.embedder   # shared embedding model loaded at app startup

    # ---------- Step 1: Read and parse the uploaded file ----------
    try:
        file_bytes = await resume.read()   # read raw file bytes (async, non-blocking)
        filename   = resume.filename or 'resume'
        from backend.services.resume_parser import (
            FileParsingError,
            FileValidationError,
            parse_resume_file,
        )
        resume_text, _metadata = parse_resume_file(file_bytes, filename)   # extract text from PDF/DOCX
        logger.info(f"Parsed '{filename}': {len(resume_text)} chars extracted")
    except Exception as exc:
        logger.error(f'File parsing failed: {exc}')
        raise HTTPException(
            status_code=422,   # "Unprocessable Entity" — file couldn't be read
            detail=f'Could not read or parse the resume: {exc}',
        )

    # ---------- Step 2: Run the full analysis pipeline ----------
    try:
        from backend.services.resume_analyzer import analyze_full_resume
        result = analyze_full_resume(
            resume_text=resume_text,
            nlp=nlp,
            embedder=embedder,
            job_description=job_description
        )
    except Exception as exc:
        logger.error(f'Full analysis pipeline failed: {exc}')
        raise HTTPException(status_code=500, detail=f'Analysis pipeline failed: {exc}')

    from backend.models.schemas import ComponentScores

    # ---------- Step 3: Build JD comparison object (if a job description was given) ----------
    jd_comparison_result = None
    if result.get('jd_comparison'):
        jd_comparison_result = JDComparison(
            match_percentage=round(float(result['jd_comparison'].get('match_percentage', 0.0)), 1),
            semantic_similarity=round(float(result['jd_comparison'].get('semantic_similarity', 0.0)), 3),
            matched_keywords=result['jd_comparison'].get('matched_keywords', [])[:20],   # limit to 20
            missing_keywords=result['jd_comparison'].get('missing_keywords', [])[:15],   # limit to 15
            skills_gap=result['jd_comparison'].get('skills_gap', [])[:10],               # limit to 10
        )

    detailed_fb = result.get('detailed_feedback', [])

    # ---------- Step 4: Build skill validation object ----------
    svd_raw = result.get('skill_validation_details') or {}
    skill_val_details = SkillValidationDetails(
        validated       = svd_raw.get('validated', []),
        unvalidated     = svd_raw.get('unvalidated', []),
        total           = svd_raw.get('total', 0),
        validated_count = svd_raw.get('validated_count', 0),
        validation_pct  = svd_raw.get('validation_pct', 0.0),
    )

    # ---------- Step 5: Build the final API response object ----------
    response = AnalysisResponse(
        ATS_score=result['ats_score'],
        component_scores=ComponentScores(**result['component_scores']),   # unpack dict into fields
        issues_summary=result['issues_summary'],
        detailed_feedback=detailed_fb,
        jd_match_analysis=jd_comparison_result,
        skill_validation_details=skill_val_details,
        # duplicate/legacy fields kept for backward compatibility with older clients
        ats_score=result['ats_score'],
        keyword_match=jd_comparison_result.match_percentage if jd_comparison_result else 0.0,
        missing_keywords=result.get('missing_keywords', []),
        matched_keywords=result.get('matched_keywords', []),
        skills=list(result.get('skills', [])[:20]),
        jd_comparison=jd_comparison_result,
        interpretation=result.get('interpretation', '')
    )

    # ---------- Step 6: Save this analysis to history (non-blocking) ----------
    try:
        from backend.database.supabase_db import save_analysis
        await save_analysis(user_id, filename, result)
    except Exception as exc:
        logger.warning(f'History save failed (non-blocking): {exc}')   # don't fail the request if saving fails

    return response



"""
This is a simple API endpoint used to check if the server is running properly and if the required AI models (NLP and embedder) have finished loading. Often used by monitoring tools or load balancers to confirm the app is ready.
"""
@router.get('/health')
async def health_check(request: Request):
    """Health check — confirms models are loaded and the API is ready."""
    return {
        'status':          'healthy',
        'nlp_loaded':      request.app.state.nlp is not None,       # True if NLP model is ready
        'embedder_loaded': request.app.state.embedder is not None,  # True if embedder model is ready
    }



"""
This API endpoint returns a logged-in user's past resume analyses. It identifies the user from their login token, not from any input the user provides, so people can only see their own history.
"""
@router.get('/history')
async def get_history(user_id: str = Depends(get_current_user)):
    """Return the signed-in user's past analyses (identity comes from the JWT)."""
    from backend.database.supabase_db import get_user_history
    try:
        return await get_user_history(user_id)   # fetch history tied to this user's ID
    except Exception as exc:
        logger.error(f'History fetch failed: {exc}')
        raise HTTPException(status_code=500, detail=f'Could not load history: {exc}')



"""
This API endpoint deletes one specific analysis from a user's history. It only deletes the record if it belongs to the logged-in user, preventing users from deleting someone else's data.
"""
@router.delete('/history/{analysis_id}')
async def delete_history_entry(
    analysis_id: str,   # comes from the URL path, e.g. /history/abc123
    user_id: str = Depends(get_current_user),   # identity from the verified login token
):
    """Delete one analysis from the signed-in user's history."""
    from backend.database.supabase_db import delete_analysis
    try:
        # only deletes if analysis_id belongs to this user_id
        success = await delete_analysis(analysis_id, user_id)

        if not success:
            # either the analysis doesn't exist, or it belongs to someone else
            raise HTTPException(status_code=404, detail='Analysis not found or not owned by this user.')

        return {'status': 'deleted', 'id': analysis_id}

    except HTTPException:
        raise   # re-raise HTTPExceptions as-is, don't treat them as unexpected errors

    except Exception as exc:
        logger.error(f'History delete failed: {exc}')
        raise HTTPException(status_code=500, detail=f'Could not delete: {exc}')



"""
This API endpoint takes an already-completed resume analysis and turns it into a downloadable PDF report. It builds HTML pages from the analysis data, converts them to PDF, and sends the file back to the user.
"""


@router.post('/generate-pdf')
async def generate_pdf(
    data: AnalysisResponse,   # analysis result sent from the frontend
    user_id: str = Depends(get_current_user),   # requires user to be logged in
):
    from backend.services.report_generator import generate_html_reports
    from backend.services.pdf_export import generate_combined_pdf
    from fastapi.responses import Response

    try:
        # convert analysis data into HTML report pages
        html_docs = generate_html_reports(data.model_dump())   # model_dump() turns Pydantic object into a dict

        # merge those HTML pages into one combined PDF
        pdf_bytes = generate_combined_pdf(html_docs)

        # send the PDF back as a downloadable file
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=ats_report.pdf"   # forces browser download
            }
        )
    except Exception as e:
        logger.error(f'Failed to generate PDF: {e}')
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {e}")

"""
This API endpoint generates a downloadable PDF for a specific past analysis stored in the user's history, instead of a freshly submitted one. It looks up the saved analysis by ID before converting it into a PDF.
"""
@router.get('/history/{analysis_id}/pdf')
async def generate_history_pdf(
    analysis_id: str,   # comes from the URL, identifies which past analysis to use
    user_id: str = Depends(get_current_user),   # requires user to be logged in
):
    from backend.database.supabase_db import get_user_history
    from backend.services.report_generator import generate_html_reports
    from backend.services.pdf_export import generate_combined_pdf
    from fastapi.responses import Response

    # fetch all of this user's saved analyses
    history = await get_user_history(user_id)

    # find the one matching the requested analysis_id (None if not found)
    analysis_data = next((item["analysis_result"] for item in history if item["id"] == analysis_id), None)

    if not analysis_data:
        raise HTTPException(status_code=404, detail="Analysis not found")

    try:
        # build HTML report pages from the saved analysis data
        html_docs = generate_html_reports(analysis_data)

        # merge HTML pages into one combined PDF
        pdf_bytes = generate_combined_pdf(html_docs)

        # send the PDF back as a downloadable file, named with the analysis ID
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=ats_report_{analysis_id}.pdf"
            }
        )
    except Exception as e:
        logger.error(f'Failed to generate PDF for history: {e}')
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {e}")