from typing import List, Dict
# List -> Type for lists
# Dict -> Type for dictionaries

import numpy as np
# numpy -> Used for numerical operations
# np -> Short name for numpy

import spacy
# spacy -> Used for NLP (text processing)

from sentence_transformers import SentenceTransformer
# SentenceTransformer -> Generate text embeddings (vector representation)

from backend.utils.matching import fuzzy_match_keywords, normalize_skill
# fuzzy_match_keywords -> Compare resume and JD skills
# normalize_skill -> Convert skill names to a standard format

from rapidfuzz import fuzz
# fuzz -> Calculate text similarity score


def calculate_semantic_similarity(
    resume_text: str, jd_text: str, embedder: SentenceTransformer
) -> float:
    # resume_text -> Resume content as string
    # jd_text -> Job description content as string
    # embedder -> Model that converts text into vectors
    # -> float -> Function returns a float

    resume_emb = embedder.encode(resume_text[:5000], convert_to_tensor=False)
    # Take first 5000 characters of resume_text
    # encode() -> Converts text into a numeric vector (embedding)
    # convert_to_tensor=False -> Return NumPy array, not PyTorch tensor

    jd_emb     = embedder.encode(jd_text[:5000], convert_to_tensor=False)
    # Same as above, but for job description text

    similarity = np.dot(resume_emb, jd_emb) / (
        np.linalg.norm(resume_emb) * np.linalg.norm(jd_emb)
    )
    # np.dot() -> Multiplies matching elements of both vectors and sums them
    # np.linalg.norm() -> Calculates the length (magnitude) of a vector
    # Dividing dot product by product of norms -> Cosine similarity formula
    # Result range: -1 to 1 (1 = most similar)

    return float(np.clip(similarity, 0.0, 1.0))
    # np.clip() -> Forces similarity value to stay between 0.0 and 1.0
    # float() -> Converts NumPy value to plain Python float
    # return -> Sends this final similarity score back to caller


def identify_matched_keywords(
    resume_keywords: List[str], jd_keywords: List[str]
) -> List[str]:
    # resume_keywords -> List of skills from resume
    # jd_keywords -> List of skills from job description
    # -> List[str] -> Function returns a list of strings

    result = fuzzy_match_keywords(resume_keywords, jd_keywords, threshold=80)
    # Calls fuzzy_match_keywords() with default threshold 80
    # Stores returned dictionary (with 'matched' and 'missing') in result

    return result['matched']
    # Access 'matched' key from result dictionary
    # Returns only the list of matched skills


def identify_missing_keywords(
    resume_keywords: List[str], jd_keywords: List[str], top_n: int = 15
) -> List[str]:
    # resume_keywords -> List of skills from resume
    # jd_keywords -> List of skills from job description
    # top_n -> Max number of missing skills to return (default = 15)
    # -> List[str] -> Function returns a list of strings

    result = fuzzy_match_keywords(resume_keywords, jd_keywords, threshold=80)
    # Calls fuzzy_match_keywords() with default threshold 80
    # Stores returned dictionary (with 'matched' and 'missing') in result

    return result['missing'][:top_n]
    # Access 'missing' key from result dictionary
    # [:top_n] -> Slice list to keep only first top_n items
    # Returns limited list of missing skills


def analyze_skills_gap(
    resume_skills: List[str], jd_text: str, nlp: spacy.Language
) -> List[str]:
    # resume_skills -> List of skills from resume
    # jd_text -> Job description text
    # nlp -> spaCy language model object
    # -> List[str] -> Function returns a list of strings

    doc       = nlp(jd_text[:5000])
    # Take first 5000 characters of jd_text
    # nlp() -> Processes text and returns a spaCy Doc object (parsed text)

    jd_skills = set()
    # Create empty set to store unique JD skills

    for ent in doc.ents:
    # Loop through named entities found in the document
    # ent -> One entity (e.g., a product name, organization)

        if ent.label_ in ['PRODUCT', 'ORG', 'LANGUAGE']:
        # Check if entity type is Product, Organization, or Language

            jd_skills.add(ent.text.lower())
            # Add entity text (lowercased) to jd_skills set

    for chunk in doc.noun_chunks:
    # Loop through noun phrases in the document
    # chunk -> One noun phrase (e.g., "machine learning")

        ct = chunk.text.lower().strip()
        # Convert chunk text to lowercase and remove extra spaces

        if 1 <= len(ct.split()) <= 4:
        # Check if phrase has between 1 and 4 words

            jd_skills.add(ct)
            # Add phrase to jd_skills set

    resume_normalized = {normalize_skill(s) for s in resume_skills}
    # Normalize every resume skill
    # Store results in a set (unique values only)

    gap = []
    # Create empty list to store missing skills

    for jd_skill in jd_skills:
    # Loop through every skill/phrase found in JD

        jd_norm = normalize_skill(jd_skill)
        # Normalize current JD skill

        if jd_norm in resume_normalized:
        # Check exact match in resume

            continue
            # Exact match found, skip to next jd_skill

        best_score = max(
            (fuzz.token_sort_ratio(jd_norm, rs) for rs in resume_normalized),
            default=0,
        )
        # Compare jd_norm with every resume skill using fuzzy matching
        # Generator expression calculates score for each comparison
        # max() picks the highest score
        # default=0 -> If resume_normalized is empty, use 0

        if best_score < 75:
        # If highest similarity score is below 75

            gap.append(jd_skill)
            # Add original jd_skill text to gap list

    return sorted(gap)[:20]
    # sorted() -> Arrange gap list alphabetically
    # [:20] -> Keep only first 20 items
    # Return final list of missing skills


def calculate_match_percentage(
    resume_keywords: List[str],
    jd_keywords: List[str],
    semantic_similarity: float,
) -> float:
    # resume_keywords -> List of skills from resume
    # jd_keywords -> List of skills from job description
    # semantic_similarity -> Similarity score from embeddings (0 to 1)
    # -> float -> Function returns a float

    if not jd_keywords:
    # Check if jd_keywords list is empty
    # "not" -> Reverses truth value (empty list is False, so "not []" is True)

        return 0.0
        # If JD has no keywords, return 0% match immediately

    matched = identify_matched_keywords(resume_keywords, jd_keywords)
    # Get list of matched skills between resume and JD

    keyword_overlap = len(matched) / len(jd_keywords)
    # len() -> Counts number of items in a list
    # Divide matched count by total JD keywords -> Gives overlap ratio (0 to 1)

    match_pct = (keyword_overlap * 0.6 + semantic_similarity * 0.4) * 100
    # Combine keyword_overlap (60% weight) and semantic_similarity (40% weight)
    # Multiply by 100 to convert ratio into percentage

    return float(np.clip(match_pct, 0.0, 100.0))
    # np.clip() -> Keep match_pct between 0.0 and 100.0
    # float() -> Convert to plain Python float
    # Return final match percentage


def compare_resume_with_jd(
    resume_text: str,
    resume_keywords: List[str],
    resume_skills: List[str],
    jd_text: str,
    jd_keywords: List[str],
    embedder: SentenceTransformer,
    nlp: spacy.Language,
) -> Dict:
    # resume_text -> Full resume text
    # resume_keywords -> List of resume keywords
    # resume_skills -> List of resume skills
    # jd_text -> Full job description text
    # jd_keywords -> List of JD keywords
    # embedder -> SentenceTransformer model
    # nlp -> spaCy language model
    # -> Dict -> Function returns a dictionary

    semantic_similarity = calculate_semantic_similarity(resume_text, jd_text, embedder)
    # Call function to get similarity score (0 to 1) using embeddings

    matched_keywords    = identify_matched_keywords(resume_keywords, jd_keywords)
    # Call function to get list of matched keywords

    missing_keywords    = identify_missing_keywords(resume_keywords, jd_keywords)
    # Call function to get list of missing keywords (top 15 default)

    skills_gap          = analyze_skills_gap(resume_skills, jd_text, nlp)
    # Call function to get skill gap using NLP entity/phrase extraction

    match_percentage    = calculate_match_percentage(
        resume_keywords, jd_keywords, semantic_similarity
    )
    # Call function to get final combined match percentage

    return {
        'match_percentage':    match_percentage,
        'semantic_similarity': semantic_similarity,
        'matched_keywords':    matched_keywords,
        'missing_keywords':    missing_keywords,
        'skills_gap':          skills_gap,
    }
    # Return dictionary containing all computed results
    # Key -> Result name, Value -> Corresponding computed value


