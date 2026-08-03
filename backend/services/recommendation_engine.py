from dataclasses import dataclass   # simplifies creating classes that just hold data
from enum import Enum               # lets us define a fixed set of named values
from typing import Dict, List, Optional   # type hints for readability

class Priority(Enum):
    # fixed set of priority levels — only these 4 values are allowed
    CRITICAL = 'critical'   # must fix immediately
    HIGH     = 'high'       # fix soon
    MEDIUM   = 'medium'     # fix when possible
    LOW      = 'low'        # nice to have

# Priority.CRITICAL  → example of how to access a value from the Enum

@dataclass   # auto-generates __init__, __repr__, etc. for this class
class Recommendation:
    title:        str            # short name of the recommendation
    description:  str            # detailed explanation
    priority:     Priority       # how urgent this is (uses Priority enum above)
    impact_score: float          # numeric score showing how much this affects the resume
    category:     str            # type/group this recommendation belongs to
    action_items: List[str]      # list of specific steps to fix the issue


"""
This function checks how many of the resume's skills are unproven (not backed by projects/experience), and creates a single recommendation telling the user to fix this. The urgency (priority) depends on how bad the problem is.
"""
def generate_skill_recommendations(skill_validation_results: Dict) -> List[Recommendation]:
    recommendations = []   # empty list to collect recommendations

    unvalidated    = skill_validation_results.get('unvalidated_skills', [])
    validation_pct = skill_validation_results.get('validation_percentage', 0.0)

    # if every skill is validated, no recommendation needed
    if not unvalidated:
        return recommendations

    # decide urgency based on what % of skills are validated
    # lower validation % = more urgent problem
    if validation_pct < 0.4:
        priority, impact = Priority.CRITICAL, 8.0
    elif validation_pct < 0.6:
        priority, impact = Priority.HIGH, 6.0
    elif validation_pct < 0.8:
        priority, impact = Priority.MEDIUM, 4.0
    else:
        priority, impact = Priority.LOW, 2.0

    # create one action step per unvalidated skill (max 5 shown)
    action_items = [
        f"Add a project or experience demonstrating '{skill}', or remove it from skills"
        for skill in unvalidated[:5]
    ]
    # if more than 5 unvalidated skills, add a summary note for the rest
    if len(unvalidated) > 5:
        action_items.append(f'... and {len(unvalidated) - 5} more unvalidated skill(s)')

    # build and store the final recommendation
    recommendations.append(Recommendation(
        title        = 'Validate Your Listed Skills',
        description  = (
            f'{len(unvalidated)} skill(s) are not demonstrated in your projects or experience. '
            'ATS systems and recruiters look for evidence that you\'ve actually used the skills you claim.'
        ),
        priority     = priority,
        impact_score = impact,
        category     = 'skill_validation',
        action_items = action_items,
    ))

    return recommendations


#generator02: grammtical suggestions
"""
This function looks at grammar errors (critical, moderate, minor) and creates separate recommendations for each severity level. More serious errors get higher priority and more detailed fix instructions.
"""
def generate_grammar_recommendations(grammar_results: Dict) -> List[Recommendation]:
    recommendations = []   # empty list to collect recommendations

    critical_errors = grammar_results.get('critical_errors', [])
    moderate_errors = grammar_results.get('moderate_errors', [])
    minor_errors    = grammar_results.get('minor_errors', [])

    total = len(critical_errors) + len(moderate_errors) + len(minor_errors)

    # if there are no errors at all, skip everything
    if total == 0:
        return recommendations

    # ---------- Handle Critical Errors ----------
    if critical_errors:
        items = []
        for error in critical_errors[:5]:   # only show first 5 errors
            word    = error.get('error_text', 'unknown')      # the wrong word/phrase
            suggest = error.get('suggestions', [])             # possible corrections
            suffix  = f" → '{suggest[0]}'" if suggest else ''  # show first suggestion if available
            items.append(f"Fix '{word}'{suffix}: {error.get('message', '')}")

        # if more than 5 critical errors, add a summary line for the rest
        if len(critical_errors) > 5:
            items.append(f'... and {len(critical_errors) - 5} more critical error(s)')

        recommendations.append(Recommendation(
            title        = 'Fix Critical Spelling/Grammar Errors',
            description  = (
                f'{len(critical_errors)} critical error(s) found. These spelling mistakes or '
                'major grammar issues will make your resume look unprofessional.'
            ),
            priority     = Priority.CRITICAL,
            impact_score = min(10.0, len(critical_errors) * 2.0),   # cap impact score at 10
            category     = 'grammar',
            action_items = items,
        ))

    # ---------- Handle Moderate Errors ----------
    if moderate_errors:
        items = []
        for error in moderate_errors[:3]:   # only show first 3 errors
            word    = error.get('error_text', 'unknown')
            suggest = error.get('suggestions', [])
            suffix  = f" → '{suggest[0]}'" if suggest else ''
            items.append(f"Fix '{word}'{suffix}: {error.get('message', '')}")

        if len(moderate_errors) > 3:
            items.append(f'... and {len(moderate_errors) - 3} more moderate error(s)')

        recommendations.append(Recommendation(
            title        = 'Address Punctuation and Capitalization Issues',
            description  = (
                f'{len(moderate_errors)} moderate error(s) found. '
                'These punctuation or capitalization issues should be corrected.'
            ),
            priority     = Priority.HIGH,
            impact_score = min(6.0, len(moderate_errors) * 1.0),   # cap impact score at 6
            category     = 'grammar',
            action_items = items,
        ))

    # ---------- Handle Minor Errors ----------
    # only flag minor errors if there are at least 3 (not worth mentioning otherwise)
    if minor_errors and len(minor_errors) >= 3:
        recommendations.append(Recommendation(
            title        = 'Consider Style Improvements',
            description  = (
                f'{len(minor_errors)} minor style suggestion(s) found. '
                'These are optional improvements for better readability.'
            ),
            priority     = Priority.LOW,
            impact_score = 1.0,
            category     = 'grammar',
            action_items = [
                f'Review {len(minor_errors)} style suggestion(s) for improved readability',
                'Use consistent formatting throughout',
            ],
        ))

    return recommendations


#generator03: location recommendations
"""
This function checks if the resume has risky location details (like full addresses or zip codes) and creates one recommendation telling the user what to remove. The urgency depends on how serious the privacy risk is.
"""
def generate_location_recommendations(location_results: Dict) -> List[Recommendation]:
    recommendations    = []   # empty list to collect recommendations
    detected_locations = location_results.get('detected_locations', [])
    privacy_risk       = location_results.get('privacy_risk', 'none')

    # skip everything if there's no risk or no locations found
    if privacy_risk == 'none' or not detected_locations:
        return recommendations

    # separate detected locations into addresses and zip codes
    addresses = [loc for loc in detected_locations if loc.get('type') == 'address']
    zip_codes = [loc for loc in detected_locations if loc.get('type') == 'zip']

    action_items = []
    # list first 2 full addresses to remove
    for addr in addresses[:2]:
        action_items.append(f"Remove full address: '{addr.get('text', '')}'")
    # list first 2 zip codes to remove
    for z in zip_codes[:2]:
        action_items.append(f"Remove zip code: '{z.get('text', '')}'")
    # always suggest keeping just city/state
    action_items.append("Keep only 'City, State' in your contact header")

    # decide urgency and message based on risk level
    if privacy_risk == 'high':
        priority    = Priority.CRITICAL
        impact      = 5.0
        description = (
            'Your resume contains detailed location information that poses a privacy risk. '
            'Full addresses and zip codes are unnecessary and can be used to identify your location.'
        )
    elif privacy_risk == 'medium':
        priority    = Priority.HIGH
        impact      = 3.0
        description = (
            "Your resume contains multiple location mentions. Consider simplifying to just "
            "'City, State' in your contact header."
        )
    else:   # low risk
        priority    = Priority.MEDIUM
        impact      = 2.0
        description = (
            'Minor location information detected. Consider reviewing for unnecessary location details.'
        )

    # build and store the final recommendation
    recommendations.append(Recommendation(
        title        = 'Protect Your Location Privacy',
        description  = description,
        priority     = priority,
        impact_score = impact,
        category     = 'location',
        action_items = action_items,
    ))

    return recommendations



#generator04: keyword recommendations
"""
This function checks how well the resume's keywords match a job description (if provided), and creates recommendations to add missing keywords or close skill gaps. If no job description is given, it falls back to checking if the resume just has too few keywords overall.
"""
def generate_keyword_recommendations(
    keyword_analysis: Optional[Dict] = None,
    resume_keywords: Optional[List[str]] = None,
) -> List[Recommendation]:
    recommendations = []   # empty list to collect recommendations

    # ---------- Case 1: Job description comparison available ----------
    if keyword_analysis:
        missing   = keyword_analysis.get('missing_keywords', [])   # keywords in JD but not in resume
        gap       = keyword_analysis.get('skills_gap', [])          # skills mentioned in JD but missing
        match_pct = keyword_analysis.get('match_percentage', 0.0)   # how well resume matches JD

        # ---------- Sub-check: Missing keywords ----------
        if missing:
            # lower match % = more urgent to fix
            if match_pct < 40:
                priority, impact = Priority.CRITICAL, 8.0
            elif match_pct < 60:
                priority, impact = Priority.HIGH, 6.0
            else:
                priority, impact = Priority.MEDIUM, 4.0

            # list first 7 missing keywords as action items
            items = [f"Add '{kw}' to your resume in a relevant section" for kw in missing[:7]]
            if len(missing) > 7:
                items.append(f'... and {len(missing) - 7} more missing keyword(s)')

            recommendations.append(Recommendation(
                title        = 'Add Missing Job Description Keywords',
                description  = (
                    f'{len(missing)} keyword(s) from the job description are missing from '
                    f'your resume. Your current match is {match_pct:.0f}%.'
                ),
                priority     = priority,
                impact_score = impact,
                category     = 'keywords',
                action_items = items,
            ))

        # ---------- Sub-check: Skills gap ----------
        if gap:
            items = [f"Consider adding '{skill}' if you have this skill" for skill in gap[:5]]
            if len(gap) > 5:
                items.append(f'... and {len(gap) - 5} more skill(s) mentioned in the job')

            recommendations.append(Recommendation(
                title        = 'Address Skills Gap',
                description  = (
                    f'The job description mentions {len(gap)} skill(s) not found in your resume. '
                    'Add these skills if you have them, or consider gaining them.'
                ),
                priority     = Priority.HIGH,
                impact_score = 5.0,
                category     = 'keywords',
                action_items = items,
            ))

    # ---------- Case 2: No job description, just check keyword count ----------
    elif resume_keywords is not None:
        if len(resume_keywords) < 10:
            recommendations.append(Recommendation(
                title        = 'Increase Keyword Density',
                description  = (
                    f'Your resume contains only {len(resume_keywords)} keywords. '
                    'Adding more relevant keywords will improve ATS matching.'
                ),
                priority     = Priority.MEDIUM,
                impact_score = 4.0,
                category     = 'keywords',
                action_items = [
                    "Add more technical skills and tools you've used",
                    'Include industry-specific terminology',
                    'Mention relevant certifications and methodologies',
                ],
            ))

    return recommendations



#generator05: formatting and structure recommendations
"""
This function checks if the resume is missing key sections (like Experience, Education, Skills) or has poor overall formatting, then creates recommendations to fix these structural problems.
"""
def generate_formatting_recommendations(
    score_results: Dict,
    sections: Dict[str, str],
) -> List[Recommendation]:
    recommendations  = []   # empty list to collect recommendations
    formatting_score = score_results.get('formatting_score', 0.0)

    # suggestion message for each possible section
    section_recommendations = {
        'experience': "Add a clear 'Experience' or 'Work History' section",
        'education':  "Add an 'Education' section with your qualifications",
        'skills':     "Add a 'Skills' section listing your technical and soft skills",
        'summary':    "Consider adding a 'Summary' or 'Objective' section at the top",
        'projects':   "Consider adding a 'Projects' section to showcase your work",
    }

    # find sections that are missing OR too short (less than 20 characters)
    missing_sections = []
    for section_name, suggestion in section_recommendations.items():
        content = sections.get(section_name, '')
        if not content or len(content) < 20:
            missing_sections.append((section_name, suggestion))

    # split missing sections into "must-have" and "nice-to-have"
    core_missing     = [(n, s) for n, s in missing_sections if n in ['experience', 'education', 'skills']]
    optional_missing = [(n, s) for n, s in missing_sections if n in ['summary', 'projects']]

    # ---------- Recommendation 1: Missing core sections ----------
    if core_missing:
        recommendations.append(Recommendation(
            title        = 'Add Missing Core Sections',
            description  = (
                f'Your resume is missing {len(core_missing)} essential section(s). '
                'ATS systems expect standard resume sections.'
            ),
            priority     = Priority.CRITICAL,
            impact_score = 7.0,
            category     = 'formatting',
            action_items = [suggestion for _, suggestion in core_missing],   # get just the suggestion text
        ))

    # ---------- Recommendation 2: Missing optional sections ----------
    # only suggest if formatting is also weak overall (below 15/20)
    if optional_missing and formatting_score < 15:
        recommendations.append(Recommendation(
            title        = 'Consider Adding Optional Sections',
            description  = 'Adding a summary and projects section can strengthen your resume.',
            priority     = Priority.LOW,
            impact_score = 2.0,
            category     = 'formatting',
            action_items = [suggestion for _, suggestion in optional_missing],
        ))

    # ---------- Recommendation 3: Overall poor formatting ----------
    if formatting_score < 12:   # below 60% of the 20-point max
        recommendations.append(Recommendation(
            title        = 'Improve Resume Structure',
            description  = (
                f'Your formatting score is {formatting_score:.1f}/20. '
                'Better structure will improve ATS parsing and readability.'
            ),
            priority     = Priority.HIGH,
            impact_score = 5.0,
            category     = 'formatting',
            action_items = [
                'Use bullet points to list achievements and responsibilities',
                'Add clear section headers (Experience, Education, Skills)',
                'Ensure consistent formatting throughout',
                'Use a clean, single-column layout',
            ],
        ))

    return recommendations




"""
This function sorts all recommendations so the most urgent and impactful ones appear first. It's used to decide the order in which the user should work on fixing their resume.
"""
def _prioritize_recommendations(recommendations: List[Recommendation]) -> List[Recommendation]:
    # map each priority level to a number, lower number = more urgent
    priority_order = {
        Priority.CRITICAL: 0,
        Priority.HIGH:     1,
        Priority.MEDIUM:   2,
        Priority.LOW:      3,
    }
    # sort by priority first (critical → low), then by impact score (highest first)
    return sorted(
        recommendations,
        key=lambda r: (priority_order[r.priority], -r.impact_score)  # negative flips to descending order
    )



#orchestrator of this file
"""
This is the main function that gathers recommendations from all five checkers (skills, grammar, location, keywords, formatting), sorts them by urgency, groups them by priority level, and estimates how much the resume score could improve if all fixes are applied.
"""

def generate_all_recommendations(
    skill_validation_results: Dict,
    grammar_results: Dict,
    location_results: Dict,
    score_results: Dict,
    sections: Dict[str, str],
    keyword_analysis: Optional[Dict] = None,
    resume_keywords: Optional[List[str]] = None,
) -> Dict:
    all_recs = []   # empty list to collect every recommendation

    # collect recommendations from each of the 5 domain-specific generators
    all_recs.extend(generate_skill_recommendations(skill_validation_results))
    all_recs.extend(generate_grammar_recommendations(grammar_results))
    all_recs.extend(generate_location_recommendations(location_results))
    all_recs.extend(generate_keyword_recommendations(keyword_analysis, resume_keywords))
    all_recs.extend(generate_formatting_recommendations(score_results, sections))

    # sort so critical issues with highest impact come first
    prioritized = _prioritize_recommendations(all_recs)

    # split sorted list into separate groups by priority level
    critical = [r for r in prioritized if r.priority == Priority.CRITICAL]
    high     = [r for r in prioritized if r.priority == Priority.HIGH]
    medium   = [r for r in prioritized if r.priority == Priority.MEDIUM]
    low      = [r for r in prioritized if r.priority == Priority.LOW]

    # add up all impact scores to estimate total possible score improvement
    # capped at 30 so it never overstates the improvement
    estimated_improvement = min(30.0, sum(r.impact_score for r in prioritized))

    return {
        'all_recommendations':      prioritized,
        'critical_recommendations': critical,
        'high_recommendations':     high,
        'medium_recommendations':   medium,
        'low_recommendations':      low,
        'total_count':              len(prioritized),
        'estimated_improvement':    estimated_improvement,
    }


"""
This function converts the internal Recommendation objects into simple dictionaries that can be sent as an API response (like JSON). It adds visual icons and readable labels for each priority level.
"""
def format_recommendations_for_api(recommendations_result: Dict) -> List[Dict]:
    # emoji icon shown for each priority level
    priority_icons = {
        Priority.CRITICAL: '🔴',
        Priority.HIGH:     '🟠',
        Priority.MEDIUM:   '🟡',
        Priority.LOW:      '🟢',
    }
    # human-readable text shown for each priority level
    priority_labels = {
        Priority.CRITICAL: 'Critical',
        Priority.HIGH:     'High Priority',
        Priority.MEDIUM:   'Medium Priority',
        Priority.LOW:      'Low Priority',
    }

    # convert each Recommendation object into a plain dictionary
    return [
        {
            'title':          rec.title,
            'description':    rec.description,
            'priority_icon':  priority_icons[rec.priority],     # lookup icon by priority
            'priority_label': priority_labels[rec.priority],    # lookup label by priority
            'priority_value': rec.priority.value,                # raw string value (e.g. 'critical')
            'impact_score':   rec.impact_score,
            'category':       rec.category,
            'action_items':   rec.action_items,
        }
        for rec in recommendations_result.get('all_recommendations', [])
    ]



"""
This function writes one short summary sentence describing how many recommendations were found and how much the score could improve, based on the most serious issue level present.
"""
def get_recommendation_summary(recommendations_result: Dict) -> str:
    total       = recommendations_result.get('total_count', 0)
    critical    = len(recommendations_result.get('critical_recommendations', []))
    high        = len(recommendations_result.get('high_recommendations', []))
    improvement = recommendations_result.get('estimated_improvement', 0.0)

    # no issues at all, resume is already good
    if total == 0:
        return 'Excellent! No major recommendations. Your resume is well-optimized.'

    # mention critical issues first if any exist
    if critical > 0:
        return (
            f'Found {total} recommendation(s) including {critical} critical issue(s). '
            f'Addressing these could improve your score by up to {improvement:.0f} points.'
        )
    # otherwise mention high-priority issues if any exist
    elif high > 0:
        return (
            f'Found {total} recommendation(s) including {high} high-priority item(s). '
            f'Addressing these could improve your score by up to {improvement:.0f} points.'
        )
    # otherwise just give a general summary
    else:
        return (
            f'Found {total} recommendation(s) for improvement. '
            f'Addressing these could improve your score by up to {improvement:.0f} points.'
        )


    

