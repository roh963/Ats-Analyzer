# Import type hints
from typing import Dict, List
# Dict -> Dictionary type (key : value)
# List -> List type

# Import string matching function
from rapidfuzz import fuzz
# fuzz -> Provides functions to compare text similarity

# Dictionary to store skill aliases
SKILL_ALIASES: Dict[str, str] = {

    'reactjs': 'react',          # Convert "reactjs" -> "react"
    'react.js': 'react',         # Convert "react.js" -> "react"

    'angularjs': 'angular',      # Convert "angularjs" -> "angular"

    'vuejs': 'vue',              # Convert "vuejs" -> "vue"
    'vue.js': 'vue',             # Convert "vue.js" -> "vue"

    'nextjs': 'next.js',         # Convert "nextjs" -> "next.js"

    'nodejs': 'node.js',         # Convert "nodejs" -> "node.js"
    'node': 'node.js',           # Convert "node" -> "node.js"

    'expressjs': 'express',      # Convert "expressjs" -> "express"
    'express.js': 'express',     # Convert "express.js" -> "express"

    'springboot': 'spring boot', # Convert "springboot" -> "spring boot"

    'golang': 'go',              # Convert "golang" -> "go"

    'ml': 'machine learning',    # Convert short form -> full name
    'ai': 'artificial intelligence',
    'nlp': 'natural language processing',
    'cv': 'computer vision',

    'k8s': 'kubernetes',         # Convert abbreviation -> full name

    'sklearn': 'scikit-learn',   # Standard library name

    'postgres': 'postgresql',    # Standard database name

    'dotnet': '.net',            # Standard framework name

    'tailwindcss': 'tailwind',   # Standard CSS framework name

    'amazon web services': 'aws',# Convert full cloud name -> short name
    'google cloud': 'gcp',       # Convert full cloud name -> short name

    'pyspark': 'spark',          # Convert PySpark -> Spark

    'huggingface': 'hugging face'# Standard library name
}


def normalize_skill(skill: str) -> str:
    # skill: Input skill (Example: " ReactJS ")
    # -> str: Function returns a string

    cleaned = skill.strip().lower()
    # strip() -> Remove spaces from start and end
    # lower() -> Convert all letters to lowercase
    # Example: " ReactJS " -> "reactjs"

    return SKILL_ALIASES.get(cleaned, cleaned)
    # get(key, default)
    # Search 'cleaned' in SKILL_ALIASES
    # If found -> return standard skill
    # If not found -> return cleaned value



def fuzzy_match_keywords(
    resume_keywords: List[str],
    jd_keywords: List[str],
    threshold: int = 80,
):
    # resume_keywords -> Skills from resume
    # jd_keywords -> Skills from job description
    # threshold -> Minimum similarity score (default = 80)

    resume_normalized = {normalize_skill(kw): kw for kw in resume_keywords}
    # Normalize every resume skill
    # Key   -> Standard skill
    # Value -> Original skill

    jd_normalized = {normalize_skill(kw): kw for kw in jd_keywords}
    # Normalize every JD skill

    matched_jd_originals = []
    # Store matched JD skills

    missing_jd_originals = []
    # Store missing JD skills

    for jd_canon, jd_original in jd_normalized.items():
    # Loop through every JD skill.
    # jd_canon -> Normalized skill (python)
    # jd_original -> Original skill (Python)

        if jd_canon in resume_normalized:
        # Check exact match in resume.

            matched_jd_originals.append(jd_original)
            # Add skill to matched list.

            continue
            # Exact match found, skip fuzzy matching.

        best_score = 0
        # Store highest similarity score.

        for resume_canon in resume_normalized:
        # Compare with every resume skill.

            score = fuzz.token_sort_ratio(jd_canon, resume_canon)
            # Calculate similarity (0–100).

            best_score = max(best_score, score)
            # Keep the highest score.

        if best_score >= threshold:
        # If score is greater than or equal to threshold.

            matched_jd_originals.append(jd_original)
            # Add to matched list.

        else:
            missing_jd_originals.append(jd_original)
            # Add to missing list.

    return {
    # Return dictionary

    'matched': sorted(matched_jd_originals),
    # matched -> Key
    # sorted() -> Arrange list in alphabetical order
    # Value -> Matched skills

    'missing': missing_jd_originals,
    # missing -> Key
    # Value -> Missing skills
    }