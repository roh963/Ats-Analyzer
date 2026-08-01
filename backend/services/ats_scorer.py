import re                          # regex module for pattern matching
import spacy                       # NLP library (tokenization, POS tagging, etc.)
import numpy as np                 # numerical operations, array handling
from sentence_transformers import SentenceTransformer  # generates text embeddings
from typing import Dict, List, Optional, Tuple          # type hints for readability

from backend.utils.file_utils import log_warning         # custom logging helper
from backend.core.config import SENTENCE_TRANSFORMER_MODEL  # model name from config
from backend.utils.matching import fuzzy_match_keywords     # custom fuzzy matching logic

ZIP_CODE_PATTERN = r'\b\d{5}(?:-\d{4})?\b'  
# Matches a 5-digit US ZIP code, optionally followed by "-XXXX" (ZIP+4 format)

STREET_ADDRESS_PATTERN = (
    r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+'  
    # Matches house number + one or more capitalized words (street name)
    r'(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way|Place|Pl)\b'
    # Requires the address to end with a known street-type suffix
)

"""
Checks a numeric value against a list of (threshold, points) pairs and returns the points for the first threshold it satisfies. Used to convert a raw number into a tiered score/penalty.
"""
def _tier_score(n: float, tiers: list) -> float:
    for threshold, pts in tiers:
        if n >= threshold:   # >= checks if n meets/exceeds this tier
            return pts       # first matching tier wins
    return 0.0                # no tier matched, default score


"""
Scans text for location-related information (place names, street addresses, ZIP codes) using both NLP and regex, then rates how risky it is for privacy and returns suggestions to remove that info. Likely used to flag sensitive personal data in resumes.
"""
def detect_location_info(text: str, nlp: spacy.Language) -> Dict:
    locations = []

    # method 1: spaCy Named Entity Recognition (detects place-like entities)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:   # GPE = country/city/state, LOC = other locations
            locations.append({'text': ent.text, 'type': ent.label_.lower(), 'start': ent.start_char})

    # method 2: regex for full street addresses
    for match in re.finditer(STREET_ADDRESS_PATTERN, text, re.IGNORECASE):
        locations.append({'text': match.group(), 'type': 'address', 'start': match.start()})

    # method 3: regex for ZIP/PIN codes
    for match in re.finditer(ZIP_CODE_PATTERN, text):
        locations.append({'text': match.group(), 'type': 'zip', 'start': match.start()})

    has_address = any(loc['type'] == 'address' for loc in locations)
    has_zip     = any(loc['type'] == 'zip'     for loc in locations)

    # risk level depends on how specific the leaked info is
    if has_address and has_zip:
        privacy_risk, penalty = 'high', 5.0
    elif has_address or has_zip:
        privacy_risk, penalty = 'high', 4.0
    elif len(locations) > 3:          # many general mentions still risky
        privacy_risk, penalty = 'medium', 3.0
    elif locations:
        privacy_risk, penalty = 'low', 2.0
    else:
        privacy_risk, penalty = 'none', 0.0

    recommendations = []
    if not locations:
        recommendations.append(" No privacy concerns detected.")
    if has_address:
        recommendations.append(" Remove full street addresses — ATS systems don't need this and it's a privacy risk.")
    if has_zip:
        recommendations.append(" Remove zip codes — this level of location detail is unnecessary.")
    if privacy_risk in ('low', 'medium') and not has_address and not has_zip:
        recommendations.append(" Consider reducing location mentions. 'City, State' in the contact header is sufficient.")

    return {
        'location_found':     len(locations) > 0,
        'detected_locations': locations,
        'privacy_risk':       privacy_risk,
        'recommendations':    recommendations,
        'penalty_applied':    penalty,
    }



"""
Measures how semantically similar a skill and a piece of text are, using sentence embeddings. Used to check if a skill is conceptually mentioned in text, even without exact word match.
"""
def _calculate_semantic_similarity(skill: str, text: str, embedder: SentenceTransformer) -> float:
    # cosine similarity formula: (A · B) / (|A| × |B|)
    if not skill or not text:
        return 0.0
    try:
        skill_vec = embedder.encode(skill, convert_to_tensor=False)   # convert skill to vector
        text_vec  = embedder.encode(text,  convert_to_tensor=False)   # convert text to vector

        similarity = np.dot(skill_vec, text_vec) / (
            np.linalg.norm(skill_vec) * np.linalg.norm(text_vec)      # normalize by vector lengths
        )
        return float(max(0.0, min(1.0, similarity)))  # clamp result between 0 and 1
    except Exception as e:
        log_warning(f"Similarity error for '{skill}': {e}", context='ats_scorer')
        return 0.0   # fail safe on embedding 


"""
Checks whether a skill is present in text, first by a quick exact substring check, then by semantic similarity if no direct match is found. Used to detect skills mentioned literally or conceptually.
"""
def _skill_matches(skill: str, text: str, embedder: SentenceTransformer, threshold: float) -> Tuple[bool, float]:
    # fast path: O(n) direct substring check, case-insensitive
    if skill.lower() in text.lower():
        return True, 1.0   # exact match, full confidence

    # slow path: semantic similarity via embeddings
    sim = _calculate_semantic_similarity(skill, text, embedder)
    return sim >= threshold, sim   # match if similarity meets threshold

"""
Checks each claimed skill against project descriptions and experience text to see if it's actually backed up. Used to score a resume's credibility by validating skills with real evidence.
"""
def validate_skills_with_projects(
    skills: List[str],
    projects: List[Dict],
    experience_entries: List[Dict],
    embedder: SentenceTransformer,
    threshold: float = 0.6,
) -> Dict:

    if not skills:   # nothing to validate
        return {
            'validated_skills':      [],
            'unvalidated_skills':    [],
            'validation_percentage': 0.0,
            'skill_project_mapping': {},
            'validation_score':      0.0,
        }

    # merge all experience entries into one searchable text block
    experience_text = ' '.join(
        f"{e.get('job_title', '')} {e.get('company', '')} {e.get('description', '')}"
        for e in experience_entries
        if isinstance(e, dict)
    ).strip()

    validated_skills      = []
    unvalidated_skills    = []
    skill_project_mapping = {}

    for skill in skills:
        matching_projects = []
        max_similarity    = 0.0

        # check skill against each project's text
        for project in projects:
            project_text = f"{project.get('title', '')} {project.get('description', '')}"
            matched, sim = _skill_matches(skill, project_text, embedder, threshold)
            max_similarity = max(max_similarity, sim)   # keep best match score
            if matched:
                matching_projects.append(project.get('title', 'Untitled Project'))

        # also check skill against combined experience text
        if experience_text:
            matched, sim = _skill_matches(skill, experience_text, embedder, threshold)
            max_similarity = max(max_similarity, sim)
            if matched and 'Experience Section' not in matching_projects:
                matching_projects.append('Experience Section')

        if matching_projects:
            validated_skills.append({'skill': skill, 'projects': matching_projects, 'similarity': max_similarity})
            skill_project_mapping[skill] = matching_projects
        else:
            unvalidated_skills.append(skill)
            skill_project_mapping[skill] = []   # no evidence found

    validation_percentage = len(validated_skills) / len(skills)
    validation_score      = validation_percentage * 15.0   # scale to 15-point score

    return {
        'validated_skills':      validated_skills,
        'unvalidated_skills':    unvalidated_skills,
        'validation_percentage': validation_percentage,
        'skill_project_mapping': skill_project_mapping,
        'validation_score':      validation_score,
    }

#01: formatting score
"""
This function checks a resume and gives it a score out of 20 based on how well it is organized. It checks if important sections exist (like experience, education, skills), and if bullet points are used properly.
"""
def _calc_formatting_score(parsed_resume: Dict, text: str) -> float:
    score = 0.0  # start score at zero

    # get list of experience entries (only keep valid dictionary items)
    exp_entries  = [e for e in parsed_resume.get('experience', []) if isinstance(e, dict)]

    # get list of education entries (only keep valid dictionary items)
    edu_entries  = [e for e in parsed_resume.get('education', [])  if isinstance(e, dict)]

    # get list of skills
    skills       = parsed_resume.get('skills', [])

    # get summary text (short intro about the person)
    summary      = parsed_resume.get('professional_summary', '')

    # get list of projects (only keep valid dictionary items)
    proj_entries = [p for p in parsed_resume.get('projects', [])   if isinstance(p, dict)]

    # if experience exists AND has a job title or description, add 3 points
    if exp_entries and any(e.get('job_title') or e.get('description') for e in exp_entries):
        score += 3.0

    # if education info exists, add 2 points
    if edu_entries:
        score += 2.0

    # if there are at least 3 skills listed, add 2 points
    if len(skills) >= 3:
        score += 2.0

    # if summary is longer than 30 characters, add 1.5 points
    if len(summary) > 30:
        score += 1.5

    # if projects exist, add 1.5 points
    if proj_entries:
        score += 1.5

    # count how many lines start with a bullet symbol or a number (like "1.")
    bullet_count = sum(
        1 for line in text.split('\n')
        if re.match(r'^\s*[•\-\*\◦]', line) or re.match(r'^\s*\d+\.', line)
    )

    # give extra points based on how many bullet points were found
    # more bullets = better formatting = more points
    score += _tier_score(bullet_count, [(15,5.0),(10,4.0),(5,3.0),(3,2.0),(1,1.0)])

    # count how many important sections are NOT empty
    # (experience, education, skills, summary, projects)
    filled = sum(1 for has_it in [
        bool(exp_entries), bool(edu_entries), bool(skills),
        bool(summary.strip()), bool(proj_entries),
    ] if has_it)

    # give extra points based on how many sections are filled
    score += _tier_score(filled, [(4,5.0),(3,4.0),(2,3.0),(1,2.0)])

    # make sure final score is not below 0 or above 20
    return min(20.0, max(0.0, score))

#02 keyword score
"""
This function gives a score out of 25 based on how many keywords and skills are in a resume, and how well they match a job description (if one is given.
"""
def _calc_keywords_score(
    resume_keywords: List[str],
    skills: List[str],
    jd_keywords: Optional[List[str]] = None,
) -> float:
    score = 0.0  # start score at zero

    # give points based on how many keywords the resume has
    # more keywords = higher score
    score += _tier_score(len(resume_keywords), [(20,10.0),(15,8.0),(10,6.0),(5,4.0),(3,2.0)])

    # give points based on how many skills the resume has
    score += _tier_score(len(skills),          [(15,10.0),(10,8.0),(7,6.0),(5,4.0),(3,2.0)])

    if jd_keywords:   # only run this if a job description was provided
        # combine resume keywords and skills into one list (no duplicates)
        all_resume_terms = list(set(resume_keywords + skills))

        # compare resume terms with job description keywords (allows small spelling differences)
        fuzzy_result = fuzzy_match_keywords(all_resume_terms, jd_keywords, threshold=80)

        # calculate what percent of job keywords were found in resume
        match_pct = len(fuzzy_result['matched']) / len(jd_keywords) if jd_keywords else 0

        # give bonus points based on how high the match percent is
        score += _tier_score(match_pct, [(0.7,5.0),(0.5,4.0),(0.3,3.0),(0.2,2.0),(0.1,1.0)])

    # if no job description given, but resume still has many keywords, give small bonus
    elif len(resume_keywords) >= 10:
        score += 3.0

    # keep final score between 0 and 25
    return min(25.0, max(0.0, score))


#3. CONTENT QUALITY SCORE
"""
This function checks the quality of resume content — how many strong action words it uses, how many measurable achievements (like numbers/percentages) it has, and how many grammar mistakes it contains. It gives a score out of 25.
"""
def _calc_content_score(
    text: str,
    action_verbs: List[str],
    grammar_results: Dict,
) -> float:

    score = 0.0  # start score at zero

    # give points based on how many action verbs are used (like "managed", "built")
    # more action verbs = stronger resume = more points
    score += _tier_score(len(action_verbs), [(15,10.0),(10,8.0),(7,6.0),(5,4.0),(3,2.0)])

    # patterns that show measurable achievements (numbers, money, growth words)
    number_patterns = [
        r'\d+%',                                    # matches percentages like "50%"
        r'\$\d+',                                   # matches money like "$500"
        r'\d+[kKmMbB]',                              # matches short forms like "10k", "2M"
        r'\d+\s*(?:users|customers|clients|projects|hours|days|months|years)',  # matches "100 users", "5 years" etc
        r'(?:increased|decreased|improved|reduced|grew|saved)\s+(?:by\s+)?\d+',  # matches "increased by 20"
    ]

    # count how many times these achievement patterns appear in the text
    achievement_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in number_patterns)

    # give points based on how many achievements were found
    score += _tier_score(achievement_count, [(10,5.0),(7,4.0),(5,3.0),(3,2.0),(1,1.0)])

    # get grammar penalty (how many points to subtract due to grammar mistakes)
    grammar_penalty = grammar_results.get('penalty_applied', 0.0)

    # convert grammar penalty into points, more mistakes = lower score
    # max 10 points if no mistakes, less if penalty is high
    score += max(0.0, 10.0 - grammar_penalty / 2.0)

    # keep final score between 0 and 25
    return min(25.0, max(0.0, score))

#4. SKILL VALIDATION SCORE
"""
This function simply takes the skill validation score (calculated elsewhere) and makes sure it stays between 0 and 15.
"""
def _calc_skill_validation_score(validation_results: Dict) -> float:
    # keep score safely between 0 and 15
    return min(15.0, max(0.0, validation_results.get('validation_score', 0.0)))



#5. ATS COMPATIBILITY SCORE
"""
This function checks how well a resume will work with ATS (Applicant Tracking Systems) — software that scans resumes before a human sees them. It starts with 15 points and subtracts points for things that confuse ATS, like weird symbols or very short sections.
"""
def _calc_ats_compatibility_score(
    text: str,
    location_results: Dict,
    parsed_resume: Dict,
) -> float:
    score = 15.0  # start with full score

    # deduction 1: subtract points if privacy-risky location info was found earlier
    score -= location_results.get('penalty_applied', 0.0)

    # deduction 2: count special table/border symbols (these confuse ATS parsers)
    special_chars = len(re.findall(r'[│┤├┼┴┬╔╗╚╝═║╠╣╦╩╬]', text))
    if special_chars > 20:    score -= 2.0   # too many symbols, big penalty
    elif special_chars > 10:  score -= 1.0   # some symbols, small penalty

    # get experience, education, and skills info
    exp_entries  = [e for e in parsed_resume.get('experience', []) if isinstance(e, dict)]
    edu_entries  = [e for e in parsed_resume.get('education', [])  if isinstance(e, dict)]
    skills_count = len(parsed_resume.get('skills', []))

    # measure how much text is written in experience descriptions
    exp_desc_len = sum(len(e.get('description', '')) for e in exp_entries)

    # measure how much text is written in education (degree + institution)
    # "or ''" prevents errors if degree/institution is None instead of a string
    edu_desc_len = sum(len((e.get('degree') or '') + (e.get('institution') or '')) for e in edu_entries)

    # deduction 3: check if sections exist but have very little detail (too short)
    short_sections = sum([
        bool(exp_entries) and exp_desc_len < 20,          # experience exists but barely any text
        bool(edu_entries) and edu_desc_len < 20,           # education exists but barely any text
        bool(parsed_resume.get('skills')) and skills_count < 2,  # skills exist but too few
    ])
    if short_sections >= 2:    score -= 2.0   # multiple thin sections, bigger penalty
    elif short_sections >= 1:  score -= 1.0   # one thin section, small penalty

    # small bonus if resume has real experience AND a good number of skills
    if exp_entries and skills_count > 5:
        score += 1.0

    # keep final score between 0 and 15
    return min(15.0, max(0.0, score))


"""
This function is the main "brain" that combines all the smaller scores (formatting, keywords, content, etc.) into one final resume score out of 100. It also adds bonuses for good work and subtracts penalties for problems, then explains what the score means.
"""

def calculate_overall_score(
    text: str,
    parsed_resume: Dict,
    skills: List[str],
    keywords: List[str],
    action_verbs: List[str],
    skill_validation_results: Dict,
    grammar_results: Dict,
    location_results: Dict,
    jd_keywords: Optional[List[str]] = None,
    experience_months: int = 0,
) -> Dict:

    # calculate each individual score using earlier functions
    formatting_score        = _calc_formatting_score(parsed_resume, text)
    keywords_score          = _calc_keywords_score(keywords, skills, jd_keywords)
    content_score           = _calc_content_score(text, action_verbs, grammar_results)
    skill_validation_score  = _calc_skill_validation_score(skill_validation_results)
    ats_compatibility_score = _calc_ats_compatibility_score(text, location_results, parsed_resume)

    # define the maximum possible points for each category
    COMPONENT_MAX = {
        'formatting': 20.0, 'keywords': 25.0, 'content': 25.0,
        'skill_validation': 15.0, 'ats_compatibility': 15.0,
    }

    # convert each score into a percentage (0-100) of its own max
    formatting_pct        = (formatting_score        / COMPONENT_MAX['formatting'])        * 100.0
    keywords_pct          = (keywords_score          / COMPONENT_MAX['keywords'])          * 100.0
    content_pct           = (content_score           / COMPONENT_MAX['content'])           * 100.0
    skill_validation_pct  = (skill_validation_score  / COMPONENT_MAX['skill_validation'])  * 100.0
    ats_compatibility_pct = (ats_compatibility_score / COMPONENT_MAX['ats_compatibility']) * 100.0

    # combine keywords and skill validation into one "skills" percentage
    # keywords count more (60%) than skill validation (40%)
    skills_keywords_pct = (keywords_pct * 0.6) + (skill_validation_pct * 0.4)

    # calculate base score using weighted average of all categories
    # content matters most (30%), skills/keywords matter most overall (40%)
    base_score = (
        skills_keywords_pct   * 0.40 +
        content_pct           * 0.30 +
        formatting_pct        * 0.15 +
        ats_compatibility_pct * 0.15
    )

    penalties = {}  # store reasons points were removed
    bonuses   = {}  # store reasons points were added
    score     = base_score  # start final score from base score

    # record grammar penalty if any mistakes were found
    if grammar_results.get('penalty_applied', 0.0) > 0:
        penalties['grammar'] = grammar_results['penalty_applied']

    # record location/privacy penalty if any risky info was found
    if location_results.get('penalty_applied', 0.0) > 0:
        penalties['location_privacy'] = location_results['penalty_applied']

    # give bonus points if skills were very well validated by projects/experience
    validation_pct = skill_validation_results.get('validation_percentage', 0.0)
    if validation_pct >= 0.9:
        bonuses['excellent_skill_validation'] = 2.0
        score += 2.0
    elif validation_pct >= 0.8:
        bonuses['good_skill_validation'] = 1.0
        score += 1.0

    # give bonus if there are zero grammar errors
    if grammar_results.get('total_errors', 0) == 0:
        bonuses['perfect_grammar'] = 1.0
        score += 1.0

    # if a job description was given, check how many of its keywords are missing
    if jd_keywords and len(jd_keywords) > 0:
        all_resume_terms = list(set((keywords or []) + (skills or [])))  # combine resume words, remove duplicates
        fuzzy_result     = fuzzy_match_keywords(all_resume_terms, jd_keywords, threshold=80)
        missing_pct      = len(fuzzy_result['missing']) / len(jd_keywords)  # % of JD keywords not found

        # bigger penalty for missing more job-required keywords
        if missing_pct > 0.7:
            penalties['missing_jd_keywords'] = 15.0
            score -= 15.0
        elif missing_pct > 0.5:
            penalties['missing_jd_keywords'] = 10.0
            score -= 10.0
        elif missing_pct > 0.3:
            penalties['missing_jd_keywords'] = 5.0
            score -= 5.0

    # keep final score between 0 and 100
    overall_score = min(100.0, max(0.0, score))

    # get a human-friendly explanation of what this score means
    interpretation = _generate_score_interpretation(overall_score)

    # return everything as one organized result
    return {
        'overall_score':           round(overall_score, 1),
        'formatting_score':        round(formatting_score, 1),
        'keywords_score':          round(keywords_score, 1),
        'content_score':           round(content_score, 1),
        'skill_validation_score':  round(skill_validation_score, 1),
        'ats_compatibility_score': round(ats_compatibility_score, 1),
        'overall_interpretation':  interpretation,
        'penalties':               penalties,
        'bonuses':                 bonuses,
    }


#Overall score calculation and interpretation
"""
This function looks at all the scores calculated earlier and writes a list of positive comments (strengths) about the resume. If nothing scored well enough, it gives a general encouraging message instead.
"""
def generate_strengths(
    score_results: Dict,
    skill_validation_results: Dict,
    grammar_results: Dict,
) -> List[str]:
    strengths = []  # empty list to collect positive comments

    # if formatting score is high, resume structure is good
    if score_results['formatting_score']       >= 16:
        strengths.append(' Well-structured with clear sections and bullet points')

    # if keywords score is high, resume has good keyword/skill presence
    if score_results['keywords_score']          >= 20:
        strengths.append(' Strong keyword optimization and skills presence')

    # if content score is high, resume uses strong action verbs and numbers
    if score_results['content_score']           >= 20:
        strengths.append(' Excellent use of action verbs and quantifiable achievements')

    # if skill validation score is high, mention what % of skills were proven
    if score_results['skill_validation_score']  >= 12:
        pct = skill_validation_results.get('validation_percentage', 0) * 100  # convert to percentage
        strengths.append(f' {pct:.0f}% of skills are validated by projects')  # :.0f = round to whole number

    # if ATS compatibility score is high, formatting won't confuse ATS software
    if score_results['ats_compatibility_score'] >= 13:
        strengths.append(' Excellent ATS compatibility with clean formatting')

    # if there are zero grammar errors, praise the writing quality
    if grammar_results.get('total_errors', 0)   == 0:
        strengths.append(' Error-free grammar and spelling')

    # if no strengths were found at all, give a fallback encouraging message
    if not strengths:
        strengths.append('Your resume has potential - focus on the recommendations below')

    return strengths


#Critical issues that could cause ATS rejection
"""
This function checks scores and results for serious problems in the resume (like grammar errors, privacy risks, or weak sections) and returns a list of warnings the user should fix.
"""
def generate_critical_issues(
    score_results: Dict,
    grammar_results: Dict,
    location_results: Dict,
) -> List[str]:
    issues = []  # empty list to collect problems

    # count how many critical (serious) grammar/spelling errors exist
    critical_errors = len(grammar_results.get('critical_errors', []))
    if critical_errors > 0:
        issues.append(f' {critical_errors} critical grammar/spelling error(s) detected')

    # warn if resume contains high-risk private info (like full address)
    if location_results.get('privacy_risk') == 'high':
        issues.append('High privacy risk: Remove detailed location information')

    # warn if formatting score is too low
    if score_results['formatting_score']       < 10:
        issues.append(' Poor formatting: Add clear sections and bullet points')

    # warn if keyword/skills score is too low
    if score_results['keywords_score']         < 12:
        issues.append(' Insufficient keywords and skills')

    # warn if most skills aren't backed by real project/experience evidence
    if score_results['skill_validation_score'] < 7:
        issues.append(' Most skills lack supporting evidence in projects')

    return issues


#Actionable improvements to enhance ATS performance
"""
This function looks at scores that are "okay but not great" (medium range) and suggests specific ways to improve them. It only gives advice for weak areas, not perfect or very poor ones.
"""
def generate_improvements(
    score_results: Dict,
    skill_validation_results: Dict,
) -> List[str]:
    improvements = []  # empty list to collect suggestions

    # if formatting is medium (not bad, not great), suggest improving structure
    if 12 <= score_results['formatting_score']       < 16:
        improvements.append('Add more bullet points and improve section organization')

    # if keywords score is medium, suggest adding more relevant terms
    if 14 <= score_results['keywords_score']          < 20:
        improvements.append('Include more relevant keywords and technical skills')

    # if content score is medium, suggest stronger achievements/verbs
    if 14 <= score_results['content_score']           < 20:
        improvements.append('Add more quantifiable achievements and action verbs')

    # if skill validation is medium, tell user how many skills still need proof
    if 7  <= score_results['skill_validation_score']  < 12:
        unvalidated_count = len(skill_validation_results.get('unvalidated_skills', []))
        improvements.append(f'Validate {unvalidated_count} skill(s) by adding relevant project details')

    # if ATS compatibility is medium, suggest simpler formatting
    if 9  <= score_results['ats_compatibility_score'] < 13:
        improvements.append('Simplify formatting for better ATS compatibility')

    return improvements

#Interpretation of overall score
"""
This function converts the final numeric score (0-100) into a simple text message that explains how good the resume is, using easy-to-understand categories.
"""
def _generate_score_interpretation(overall_score: float) -> str:
    # check score ranges from highest to lowest, return matching message
    if overall_score >= 90:    return 'Excellent! Your resume is highly optimized for ATS systems.'
    elif overall_score >= 80:  return 'Great! Your resume should perform well with most ATS systems.'
    elif overall_score >= 70:  return 'Good! Your resume is ATS-friendly with room for minor improvements.'
    elif overall_score >= 60:  return 'Fair. Your resume needs some improvements to be fully ATS-compatible.'
    elif overall_score >= 50:  return 'Below Average. Significant improvements needed for ATS compatibility.'
    else:                      return 'Poor. Your resume requires major revisions to pass ATS screening.'


