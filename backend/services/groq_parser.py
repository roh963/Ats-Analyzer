# Import the os module to read environment variables.
import os

# Import the json module to work with JSON data.
import json

# Import the logging module to record logs (info, warning, error).
import logging

# Import Dict for type hints (used when a function returns a dictionary).
from typing import Dict

# Import the Groq client to communicate with the Groq API.
from groq import Groq


# Create a logger for this file.
# It is used to log messages for debugging and monitoring.
logger = logging.getLogger('ats_resume_scorer')


# Name of the Groq AI model that will process our prompts.
GROQ_MODEL = 'llama-3.3-70b-versatile'


# Create a global variable to store the Groq client.
# Initially, no client exists.
_client = None


def _get_client() -> Groq:
    # Use the global _client variable instead of creating a local one.
    global _client

    # Create the client only if it doesn't already exist.
    if _client is None:

        # Read the API key from the environment variables.
        api_key = os.getenv('GROQ_API_KEY')

        # If the API key is missing, stop the program.
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        # Create a new Groq client using the API key.
        _client = Groq(api_key=api_key)

    # Return the existing or newly created client.
    return _client


# System prompt:
# This tells the AI what role it should play.
# Here, we tell the AI to behave like a resume parser
# and return ONLY valid JSON (no explanation or markdown).
RESUME_SYSTEM_PROMPT = (
    "You are a resume parser. Extract information from the resume "
    "and return ONLY a valid JSON object. No explanation, no markdown."
)


# User prompt:
# This tells the AI exactly what information to extract
# from the resume and the format in which it should return it.

# Expected JSON format
RESUME_USER_PROMPT = """Extract the following from this resume and return as JSON:
{{
  "name": "full name",
  "email": "email address",
  "phone": "phone number",
  "linkedin": "LinkedIn URL if present, otherwise null",
  "github": "GitHub URL if present, otherwise null",
  "professional_summary": "the full text of the Summary, Profile, About Me, Objective, or Professional Summary section at the top of the resume. Copy the ENTIRE paragraph exactly as written. If no such section exists, return an empty string.",
  "skills": ["list", "of", "skills"],
  "experience": [
    {{
      "job_title": "",
      "company": "",
      "start_date": "",
      "end_date": "",
      "duration_months": 0,
      "description": ""
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "year": ""
    }}
  ],
  "certifications": ["list of certifications"],
  "projects": [
    {{
      "title": "project name",
      "description": "what the project does and how it was built",
      "technologies": ["tech", "used"]
    }}
  ],
  "action_verbs": ["strong action verbs used in bullet points, e.g. developed, implemented, designed"],
  "keywords": ["important keywords and phrases from the resume for ATS matching"]
}}

Important instructions:
- For duration_months, calculate the number of months between start_date and end_date. If end_date is "Present" or "Current", calculate from start_date to now.
- For skills, extract ALL technical and soft skills mentioned anywhere in the resume.
- For action_verbs, find verbs that start bullet points or describe achievements.
- For keywords, extract noun phrases and technical terms relevant to ATS matching.
- Return ONLY valid JSON. No markdown code fences, no explanation.

Resume Text:
{raw_text}"""


def _call_groq(client: Groq, system_prompt: str, user_prompt: str) -> str:

    # Send the prompts to the Groq AI model.
    response = client.chat.completions.create(

        # AI model to use.
        model=GROQ_MODEL,

        # Messages sent to the AI.
        # The system message defines the AI's role.
        # The user message contains the actual task and resume text.
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],

        # Make the response as consistent as possible.
        # 0.0 means almost no randomness.
        temperature=0.0,

        # Maximum number of tokens the AI can generate.
        max_tokens=4096
    )

    # Return only the AI's response as plain text.
    return response.choices[0].message.content.strip()


def _try_parse_json(text: str) -> dict | None:

    # Remove extra spaces from the beginning and end.
    cleaned = text.strip()

    # Sometimes the AI returns JSON inside markdown code blocks.
    # Example:
    # ```json
    # { ... }
    # ```
    # Remove those markdown fences before parsing.
    if cleaned.startswith("```"):

        # Find the end of the first line (```json).
        first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)

        # Remove the opening markdown fence.
        cleaned = cleaned[first_newline + 1:]

        # Remove the closing markdown fence if it exists.
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        # Remove any extra spaces again.
        cleaned = cleaned.strip()

    try:
        # Convert the JSON string into a Python dictionary.
        return json.loads(cleaned)

    except json.JSONDecodeError:
        # Return None if the response is not valid JSON.
        return None


def parse_resume(raw_text: str) -> Dict:

    # Get the Groq client.
    client = _get_client()

    # Insert the extracted resume text into the user prompt.
    prompt = RESUME_USER_PROMPT.format(raw_text=raw_text)

    # Send the prompt to Groq and get the response.
    raw_response = _call_groq(client, RESUME_SYSTEM_PROMPT, prompt)

    # Try converting the AI response into a Python dictionary.
    result = _try_parse_json(raw_response)

    # If the response is valid JSON,
    # validate the data and return it.
    if result is not None:
        return _validate_resume_result(result)

    # If the response is not valid JSON,
    # log a warning and try again with stricter instructions.
    logger.warning(
        "Groq resume parse: first attempt returned invalid JSON, retrying..."
    )

    # Tell the AI to return ONLY raw JSON.
    strict_prompt = (
        "Your previous response was not valid JSON. "
        "Return ONLY the raw JSON object, no markdown, "
        "no explanation, no code fences.\n\n"
        + prompt
    )

    # Send the stricter prompt to the AI.
    raw_response = _call_groq(client, RESUME_SYSTEM_PROMPT, strict_prompt)

    # Try parsing the new response.
    result = _try_parse_json(raw_response)

    # If parsing succeeds, validate and return the result.
    if result is not None:
        return _validate_resume_result(result)

    # If both attempts fail,
    # raise an error with part of the AI's response for debugging.
    raise ValueError(
        f"Groq returned unparseable response after retry. "
        f"Raw response:\n{raw_response[:500]}"
    )


# System prompt:
# This tells the AI to behave like a Job Description (JD) parser.
# It should extract information from the job description
# and return ONLY a valid JSON object.
# No explanations or markdown should be included.
JD_SYSTEM_PROMPT = (
    "You are a job description parser. Extract information and "
    "return ONLY a valid JSON object. No explanation, no markdown."
)

# User prompt:
# This tells the AI exactly what information to extract
# from the job description and the format in which
# the data should be returned.

JD_USER_PROMPT = """Extract the following from this job description and return as JSON:
{raw_text}"""



def parse_job_description(raw_text: str) -> Dict:
    # Get the Groq client.
    client = _get_client()

    # Insert the job description text into the prompt.
    prompt = JD_USER_PROMPT.format(raw_text=raw_text)

    # Send the prompt to Groq and get the AI response.
    raw_response = _call_groq(client, JD_SYSTEM_PROMPT, prompt)

    # Try converting the AI response into a Python dictionary.
    result = _try_parse_json(raw_response)

    # If the response is valid JSON,
    # validate the data and return it.
    if result is not None:
        return _validate_jd_result(result)

    # If the response is not valid JSON,
    # log a warning and try again with stricter instructions.
    logger.warning("Groq JD parse: first attempt returned invalid JSON, retrying...")

    # Tell the AI to return only raw JSON.
    strict_prompt = (
        "Your previous response was not valid JSON. "
        "Return ONLY the raw JSON object, no markdown, no explanation, no code fences.\n\n"
        + prompt
    )

    # Send the stricter prompt to Groq.
    raw_response = _call_groq(client, JD_SYSTEM_PROMPT, strict_prompt)

    # Try parsing the new response.
    result = _try_parse_json(raw_response)

    # If parsing succeeds, validate and return the result.
    if result is not None:
        return _validate_jd_result(result)

    # If both attempts fail, raise an error.
    raise ValueError(
        f"Groq returned unparseable response after retry. Raw response:\n{raw_response[:500]}"
    )


# Validate the parsed Job Description JSON.
# This ensures all expected fields exist.
def _validate_jd_result(result: dict) -> dict:

    # Default values for every expected field.
    defaults = {
        "job_title": "",
        "required_skills": [],
        "preferred_skills": [],
        "experience_required": "",
        "education_required": "",
        "key_responsibilities": [],
        "keywords": [],
    }

    # Check every expected field.
    for key, default in defaults.items():

        # If the field is missing or its value is None,
        # replace it with the default value.
        if key not in result or result[key] is None:
            result[key] = default

        # Make sure list fields are actually lists.
        if isinstance(default, list) and not isinstance(result[key], list):
            result[key] = default

    # Return the validated dictionary.
    return result


# Validate the parsed Resume JSON.
# This ensures every required field exists
# and has the correct data type.
def _validate_resume_result(result: dict) -> dict:

    # Default values for every expected resume field.
    defaults = {
        "name": "",
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "professional_summary": "",
        "skills": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "projects": [],
        "action_verbs": [],
        "keywords": [],
    }

    # Check every expected field.
    for key, default in defaults.items():

        # If the field is missing or None,
        # replace it with the default value.
        if key not in result or result[key] is None:
            result[key] = default

        # Make sure list fields are actually lists.
        if isinstance(default, list) and not isinstance(result[key], list):
            result[key] = default

    # Validate every work experience entry.
    for exp in result.get("experience", []):

        # Skip invalid experience entries.
        if not isinstance(exp, dict):
            continue

        # Add missing fields with default values.
        exp.setdefault("job_title", "")
        exp.setdefault("company", "")
        exp.setdefault("start_date", "")
        exp.setdefault("end_date", "")
        exp.setdefault("duration_months", 0)
        exp.setdefault("description", "")

        # Convert duration_months to an integer.
        # If conversion fails, use 0.
        try:
            exp["duration_months"] = int(exp["duration_months"])
        except (ValueError, TypeError):
            exp["duration_months"] = 0

    # Validate every project entry.
    for proj in result.get("projects", []):

        # Skip invalid project entries.
        if not isinstance(proj, dict):
            continue

        # Add missing project fields.
        proj.setdefault("title", "")
        proj.setdefault("description", "")
        proj.setdefault("technologies", [])

    # Return the validated resume dictionary.
    return result