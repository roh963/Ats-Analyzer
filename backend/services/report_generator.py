"""
This part sets up Jinja2, a templating engine used to generate HTML pages from templates (like filling a form letter with real data). It finds the folder where HTML templates are stored and prepares the engine to load them.
"""
import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader   # tools to load and render HTML templates
from typing import Dict

# build path to the "templates" folder, one level above this file's location
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), '..', 'templates')

# create the template engine, pointing it to that folder
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

"""
This function converts a computer-style timestamp (like 2025-06-01T14:30:00Z) into an easy-to-read date (like "June 01, 2025 at 02:30 PM"). It's registered as a Jinja2 filter so it can be used directly inside HTML templates. 
"""
def format_date(value, fmt='%B %d, %Y at %I:%M %p'):
    """Convert ISO timestamp string → human-readable date string."""
    if not value:
        return ''   # nothing to format
    try:
        # replace 'Z' (UTC marker) with '+00:00' so Python can parse it
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return dt.strftime(fmt)   # format into readable string
    except Exception:
        return value   # if parsing fails, just return original value unchanged

env.filters['format_date'] = format_date   # register this function as a Jinja2 template filter


"""
This function takes all the raw analysis data (scores, feedback, skill validation, JD comparison) and prepares it into a clean, organized format. Then it renders 4 different HTML report pages using this shared data.           
"""
def generate_html_reports(analysis_data: Dict) -> Dict[str, str]:
    # get current timestamp for the report
    now = datetime.now().isoformat()

    # get overall score, checking both possible key names
    overall_score = analysis_data.get('ATS_score', 0) or analysis_data.get('ats_score', 0)
    interpretation = analysis_data.get('interpretation') or ''

    cs = analysis_data.get('component_scores') or {}
    if hasattr(cs, '__dict__'):      # convert Pydantic object to dict if needed
        cs = cs.__dict__

    # pull out each individual component score, default to 0 if missing
    component_scores = {
        'formatting':       float(cs.get('formatting', 0)),
        'keywords':         float(cs.get('keywords', 0)),
        'content':          float(cs.get('content', 0)),
        'skill_validation': float(cs.get('skill_validation', 0)),
        'ats_compatibility': float(cs.get('ats_compatibility', 0)),
    }

    # helper: convert a raw score into a percentage of its max, for progress bars
    def pct(score, max_score):
        return min(100, max(0, round(score / max_score * 100)))

    component_pct = {
        'formatting':       pct(component_scores['formatting'],       20),
        'keywords':         pct(component_scores['keywords'],         25),
        'content':          pct(component_scores['content'],          25),
        'skill_validation': pct(component_scores['skill_validation'], 15),
        'ats_compatibility': pct(component_scores['ats_compatibility'], 15),
    }

    raw_feedback = analysis_data.get('detailed_feedback', [])

    # helper: convert each feedback item into a plain dict, whether it's already a dict or a Pydantic model
    def to_dict(item):
        if isinstance(item, dict):
            return item
        return item.model_dump() if hasattr(item, 'model_dump') else item.__dict__

    detailed_feedback = [to_dict(fb) for fb in raw_feedback]

    # split feedback into severity groups for separate display sections
    high_priority   = [fb for fb in detailed_feedback
                      if fb.get('severity_level', '').lower() in ('high',)]

    medium_priority = [fb for fb in detailed_feedback
                       if fb.get('severity_level', '').lower() in ('moderate', 'medium')]

    low_priority    = [fb for fb in detailed_feedback
                       if fb.get('severity_level', '').lower() in ('low', 'info')]

    strengths = analysis_data.get('strengths', [])

    # get skill validation details, converting from Pydantic model if needed
    svd_raw = analysis_data.get('skill_validation_details') or {}
    if hasattr(svd_raw, 'model_dump'):
        svd_raw = svd_raw.model_dump()

    validated_skills   = svd_raw.get('validated', [])
    unvalidated_skills = svd_raw.get('unvalidated', [])
    total_skills       = svd_raw.get('total', len(validated_skills) + len(unvalidated_skills))
    validated_count    = svd_raw.get('validated_count', len(validated_skills))
    validation_pct     = svd_raw.get('validation_pct', 0.0)

    # get job description comparison data, checking both possible key names
    jd_raw = analysis_data.get('jd_match_analysis') or analysis_data.get('jd_comparison')
    if hasattr(jd_raw, 'model_dump'):
        jd_raw = jd_raw.model_dump()

    # pick a color for the score display based on how good it is
    if overall_score >= 80:
        score_color = '#16a34a'   # green = good
    elif overall_score >= 60:
        score_color = '#d97706'   # amber = okay
    else:
        score_color = '#dc2626'   # red = poor

    # build one shared dictionary of all data, passed into every template
    context = {
        'timestamp':          now,
        'overall_score':      overall_score,
        'score_color':        score_color,
        'interpretation':     interpretation,
        'component_scores':   component_scores,
        'component_pct':      component_pct,
        'strengths':          strengths,
        'high_priority':      high_priority,
        'medium_priority':    medium_priority,
        'low_priority':       low_priority,
        'all_feedback':       detailed_feedback,
        'validated_skills':   validated_skills,
        'unvalidated_skills': unvalidated_skills,
        'total_skills':       total_skills,
        'validated_count':    validated_count,
        'validation_pct':     validation_pct,
        'jd_analysis':        jd_raw,
    }

    # render each HTML template using the same shared context data
    return {
        'summary':         env.get_template('summary.html').render(**context),
        'skill_report':    env.get_template('action_items.html').render(**context),
        'jd_report':       env.get_template('quick_actions.html').render(**context),
        'recommendations': env.get_template('jd_comparison.html').render(**context),
    }


