"""
NEON Authentication Agent
=========================

A deterministic agent for the Neon Health hiring challenge.

Input:  { "type": "challenge", "message": [{"word": str, "timestamp": num}, ...] }
Output: { "type": "enter_digits", "digits": "..." } OR { "type": "speak_text", "text": "..." }

Checkpoints:
  - Signal Handshake (enter_digits)
  - Vessel Identification (enter_digits)
  - Computational Assessment (enter_digits)
  - Knowledge Archive Query (speak_text)
  - Crew Manifest Transmission (speak_text)
  - Transmission Verification (speak_text)
"""

import re
import json
import math
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# RESPONSE FORMATTERS (STRICT JSON ONLY)
# =============================================================================

def speak_text(text: str) -> Dict[str, str]:
    """Return speak_text response."""
    return {"type": "speak_text", "text": str(text)}


def enter_digits(digits: str) -> Dict[str, str]:
    """Return enter_digits response."""
    return {"type": "enter_digits", "digits": str(digits)}


# =============================================================================
# FRAGMENT RECONSTRUCTION
# =============================================================================

def reconstruct_message(fragments: List[Dict[str, Any]]) -> str:
    """
    Sort fragments by timestamp and join words.

    Input: [{"word": "2", "timestamp": 1}, {"word": "What's", "timestamp": 0}, ...]
    Output: "What's 2 plus 3?"
    """
    if not fragments:
        return ""

    sorted_fragments = sorted(fragments, key=lambda f: f.get("timestamp", 0))
    return " ".join(str(f.get("word", "")) for f in sorted_fragments)


# =============================================================================
# MATH EVALUATOR (Tool-based, not mental math)
# =============================================================================

class MathEvaluator:
    """
    Evaluate JavaScript-style arithmetic expressions.
    Supports: +, -, *, /, %, parentheses, Math.floor, Math.ceil, Math.round
    Also handles word numbers (one, two) and word operators (plus, minus)
    """

    WORD_TO_NUMBER = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20'
    }

    WORD_TO_OPERATOR = {
        'plus': '+', 'minus': '-', 'times': '*', 'multiplied by': '*',
        'divided by': '/', 'over': '/', 'mod': '%', 'modulo': '%'
    }

    def evaluate(self, expression: str) -> int:
        """Evaluate expression and return integer result."""
        # Clean expression
        expr = expression.strip().lower()

        # Convert word operators to symbols (do this first, before numbers)
        for word, op in self.WORD_TO_OPERATOR.items():
            expr = re.sub(r'\b' + word + r'\b', op, expr)

        # Convert word numbers to digits
        for word, num in self.WORD_TO_NUMBER.items():
            expr = re.sub(r'\b' + word + r'\b', num, expr)

        # Convert JavaScript Math functions to Python
        expr = re.sub(r'math\.floor\s*\(', 'math.floor(', expr)
        expr = re.sub(r'math\.ceil\s*\(', 'math.ceil(', expr)
        expr = re.sub(r'math\.round\s*\(', 'round(', expr)
        expr = re.sub(r'math\.abs\s*\(', 'abs(', expr)
        expr = re.sub(r'math\.sqrt\s*\(', 'math.sqrt(', expr)

        # Also handle capitalized versions
        expr = re.sub(r'Math\.floor\s*\(', 'math.floor(', expr)
        expr = re.sub(r'Math\.ceil\s*\(', 'math.ceil(', expr)
        expr = re.sub(r'Math\.round\s*\(', 'round(', expr)
        expr = re.sub(r'Math\.abs\s*\(', 'abs(', expr)
        expr = re.sub(r'Math\.sqrt\s*\(', 'math.sqrt(', expr)

        # Safe evaluation with only math operations
        allowed_names = {"math": math, "abs": abs, "round": round}

        try:
            result = eval(expr, {"__builtins__": {}}, allowed_names)
            return int(result)
        except Exception as e:
            raise ValueError(f"Failed to evaluate: {expression} - {e}")


# =============================================================================
# KNOWLEDGE ARCHIVE (Wikipedia API)
# =============================================================================

class KnowledgeArchive:
    """Fetch summaries from Wikipedia and extract specific words."""

    BASE_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    HEADERS = {"User-Agent": "NeonAgent/1.0"}

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def get_summary(self, title: str) -> str:
        """Fetch Wikipedia summary for a title."""
        title_key = title.strip().replace(" ", "_")

        if title_key in self._cache:
            return self._cache[title_key]

        url = self.BASE_URL.format(title=title_key)
        response = requests.get(url, headers=self.HEADERS, timeout=10)
        response.raise_for_status()

        summary = response.json().get("extract", "")
        self._cache[title_key] = summary
        return summary

    def get_nth_word(self, title: str, n: int) -> str:
        """
        Get the Nth word from a title's summary.
        n is 1-indexed (1st word, 2nd word, etc.)
        """
        summary = self.get_summary(title)
        words = summary.split()

        if n < 1 or n > len(words):
            raise ValueError(f"Word index {n} out of range (1-{len(words)})")

        return words[n - 1]


# =============================================================================
# RESUME / CREW MANIFEST
# =============================================================================

@dataclass
class Resume:
    """Structured resume data."""
    name: str
    education: str
    experience: str
    skills: str
    projects: str
    summary: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Resume':
        """Create Resume from dictionary."""
        # Handle education
        education = ""
        if "education" in data:
            edu_list = data["education"]
            if isinstance(edu_list, list):
                parts = []
                for edu in edu_list:
                    if isinstance(edu, dict):
                        degree = edu.get("degree", "")
                        field = edu.get("field", "")
                        school = edu.get("school", "")
                        year = edu.get("year", "")
                        parts.append(f"{degree} in {field} from {school} ({year})")
                education = ". ".join(parts)
            else:
                education = str(edu_list)

        # Handle experience
        experience = ""
        if "experience" in data:
            exp_list = data["experience"]
            if isinstance(exp_list, list):
                parts = []
                for exp in exp_list:
                    if isinstance(exp, dict):
                        title = exp.get("title", "")
                        company = exp.get("company", "")
                        desc = exp.get("description", "")
                        parts.append(f"{title} at {company}: {desc}")
                experience = ". ".join(parts)
            else:
                experience = str(exp_list)

        # Handle projects
        projects = ""
        if "projects" in data:
            proj_list = data["projects"]
            if isinstance(proj_list, list):
                parts = []
                for proj in proj_list:
                    if isinstance(proj, dict):
                        name = proj.get("name", "")
                        desc = proj.get("description", "")
                        parts.append(f"{name}: {desc}")
                projects = ". ".join(parts)
            else:
                projects = str(proj_list)

        # Handle skills
        skills = ""
        if "skills" in data:
            skill_list = data["skills"]
            if isinstance(skill_list, list):
                skills = ", ".join(skill_list)
            else:
                skills = str(skill_list)

        return cls(
            name=data.get("name", ""),
            education=education,
            experience=experience,
            skills=skills,
            projects=projects,
            summary=data.get("summary", "")
        )

    @classmethod
    def from_json_file(cls, path: str) -> 'Resume':
        """Load resume from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class CrewManifest:
    """Generate crew manifest responses with length constraints."""

    def __init__(self, resume: Resume):
        self.resume = resume

    def get_response(self, aspect: str, min_len: int = 0, max_len: int = 500) -> str:
        """
        Get crew manifest response for an aspect.
        Aspects: name, education, experience, skills, projects, background, deployment
        """
        if "name" in aspect:
            text = self.resume.name
        elif "education" in aspect or "background" in aspect or "qualification" in aspect:
            text = self.resume.education
        elif "experience" in aspect or "work" in aspect or "deployment" in aspect:
            text = self.resume.experience
        elif "project" in aspect:
            text = self.resume.projects
        elif "skill" in aspect:
            text = f"Skills: {self.resume.skills}"
        elif "summary" in aspect or "about" in aspect or "overview" in aspect:
            text = self.resume.summary or self.resume.experience
        else:
            # Unknown field - return not available
            text = "Information not available in crew manifest"

        # Apply length constraints
        if len(text) > max_len:
            text = text[:max_len]
            # Truncate at word boundary
            last_space = text.rfind(' ')
            if last_space > max_len * 0.7:
                text = text[:last_space]
            text = text.rstrip('.,;: ')

        return text


# =============================================================================
# MEMORY STORE (For Transmission Verification)
# =============================================================================

class MemoryStore:
    """Track all crew manifest transmissions for verification."""

    def __init__(self):
        self.transmissions: List[str] = []  # Crew manifest responses only
        self.all_responses: List[Dict[str, Any]] = []  # All responses

    def store_crew_manifest(self, text: str):
        """Store a crew manifest transmission."""
        self.transmissions.append(text)

    def store_response(self, response: Dict[str, str], prompt: str):
        """Store any response."""
        self.all_responses.append({
            "prompt": prompt,
            "response": response,
            "text": response.get("text", response.get("digits", ""))
        })

    def recall_word(self, transmission_index: int, word_index: int) -> str:
        """
        Recall a word from an earlier crew manifest transmission.
        Both indices are 1-indexed.
        """
        if transmission_index < 1 or transmission_index > len(self.transmissions):
            raise ValueError(f"Transmission {transmission_index} not found")

        text = self.transmissions[transmission_index - 1]
        words = text.split()

        if word_index < 1 or word_index > len(words):
            raise ValueError(f"Word {word_index} not found in transmission")

        return words[word_index - 1]

    def clear(self):
        """Clear all memory."""
        self.transmissions.clear()
        self.all_responses.clear()


# =============================================================================
# PROMPT CLASSIFIER
# =============================================================================

class PromptClassifier:
    """Classify reconstructed prompts into checkpoint types."""

    # Checkpoint types
    SIGNAL_HANDSHAKE = "signal_handshake"
    VESSEL_ID = "vessel_id"
    MATH = "math"
    KNOWLEDGE = "knowledge"
    CREW_MANIFEST = "crew_manifest"
    VERIFICATION = "verification"

    def classify(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """
        Classify prompt and extract parameters.
        Returns: (checkpoint_type, params)
        """
        prompt_lower = prompt.lower().strip()

        # Check for verification/recall (must check first - it mentions "earlier transmission")
        if self._is_verification(prompt_lower):
            return self.VERIFICATION, self._extract_recall_params(prompt_lower)

        # Check for signal frequency
        if self._is_signal_handshake(prompt_lower):
            return self.SIGNAL_HANDSHAKE, self._extract_frequency(prompt_lower)

        # Check for vessel ID / Neon code
        if self._is_vessel_id(prompt_lower):
            return self.VESSEL_ID, self._extract_vessel_params(prompt_lower)

        # Check for knowledge archive query
        if self._is_knowledge_query(prompt_lower):
            return self.KNOWLEDGE, self._extract_knowledge_params(prompt_lower)

        # Check for math expression
        if self._is_math(prompt_lower):
            return self.MATH, self._extract_math_params(prompt_lower, prompt)

        # Check for crew manifest
        if self._is_crew_manifest(prompt_lower):
            return self.CREW_MANIFEST, self._extract_crew_params(prompt_lower)

        # Default to crew manifest for unknown prompts
        return self.CREW_MANIFEST, {"aspect": "general", "min_len": 0, "max_len": 500}

    def _is_verification(self, prompt: str) -> bool:
        patterns = [
            r'recall.*word.*(?:earlier|previous).*transmission',
            r'what.*word.*(?:earlier|previous).*transmission',
            r'recall.*transmission.*\d+',
            r'(?:earlier|previous).*crew.*manifest.*transmission',
        ]
        return any(re.search(p, prompt) for p in patterns)

    def _is_signal_handshake(self, prompt: str) -> bool:
        patterns = [
            r'respond.*frequency',
            r'tune.*(?:comms?|frequency)',
            r'signal.*frequency',
            r'transmit.*frequency',
        ]
        return any(re.search(p, prompt) for p in patterns)

    def _is_vessel_id(self, prompt: str) -> bool:
        patterns = [
            r'vessel.*(?:authorization|auth|code|id)',
            r'neon.*code',
            r'authorization.*code',
            r'enter.*(?:your|the).*code',
        ]
        return any(re.search(p, prompt) for p in patterns)

    def _is_knowledge_query(self, prompt: str) -> bool:
        patterns = [
            r'knowledge.*archive',
            r'\d+(?:st|nd|rd|th)?\s*word.*(?:entry|archive)',
            r'speak.*word.*(?:for|entry)',
        ]
        return any(re.search(p, prompt) for p in patterns)

    def _is_math(self, prompt: str) -> bool:
        # Word numbers to detect
        word_nums = r'(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)'
        patterns = [
            r'calculate',
            r'compute',
            r'evaluate',
            r'math\.floor',
            r'math\.ceil',
            r'what.*(?:is|\'s)\s*' + word_nums + r'.*(?:\+|\-|\*|\/|plus|minus|times|divided)',
        ]
        return any(re.search(p, prompt) for p in patterns)

    def _is_crew_manifest(self, prompt: str) -> bool:
        patterns = [
            r'crew.*member',
            r'transmit.*(?:education|experience|skills?|projects?|background|deployment)',
            r'manifest',
        ]
        return any(re.search(p, prompt) for p in patterns)

    def _extract_frequency(self, prompt: str) -> Dict[str, Any]:
        match = re.search(r'frequency\s*(\d+)', prompt)
        frequency = match.group(1) if match else ""
        append_hash = "pound" in prompt or "#" in prompt
        return {"frequency": frequency, "append_hash": append_hash}

    def _extract_vessel_params(self, prompt: str) -> Dict[str, Any]:
        append_hash = "pound" in prompt or "#" in prompt
        return {"append_hash": append_hash}

    def _extract_knowledge_params(self, prompt: str) -> Dict[str, Any]:
        # Extract word position
        word_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*word', prompt)
        word_index = int(word_match.group(1)) if word_match else 1

        # Extract title - look for "entry for X" pattern more specifically
        # Remove everything before and including "entry for"
        title = ""
        patterns = [
            r'(?:knowledge\s+archive\s+)?entry\s+for\s+(.+?)(?:\s*$|\s*\.)',
            r'archive\s+for\s+(.+?)(?:\s*$|\s*\.)',
            r'(?:about|of)\s+(.+?)(?:\s*$|\s*\.)',
        ]
        for pattern in patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                break

        return {"word_index": word_index, "title": title}

    def _extract_math_params(self, prompt_lower: str, prompt_original: str) -> Dict[str, Any]:
        # Extract the expression
        expr = prompt_original

        # Remove common prefixes
        expr = re.sub(r'^.*?(?:calculate|compute|evaluate)\s*', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'^.*?(?:what\'s|what is)\s*', '', expr, flags=re.IGNORECASE)

        # Remove suffixes like "and transmit..."
        expr = re.sub(r'\s+and\s+transmit.*$', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\s+followed\s+by.*$', '', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\s*[\.?]$', '', expr)

        # Check for pound key
        append_hash = "pound" in prompt_lower

        return {"expression": expr.strip(), "append_hash": append_hash}

    def _extract_crew_params(self, prompt: str) -> Dict[str, Any]:
        # Determine aspect
        if "name" in prompt:
            aspect = "name"
        elif "education" in prompt or "background" in prompt or "degree" in prompt or "school" in prompt:
            aspect = "education"
        elif "experience" in prompt or "deployment" in prompt or "work" in prompt or "job" in prompt:
            aspect = "experience"
        elif "project" in prompt:
            aspect = "projects"
        elif "skill" in prompt or "technolog" in prompt or "expertise" in prompt:
            aspect = "skills"
        elif "summary" in prompt or "about" in prompt or "overview" in prompt or "introduce" in prompt:
            aspect = "summary"
        else:
            # Pass the actual query term to check if it's an unknown field
            aspect = prompt

        # Extract length constraints
        min_len = 0
        max_len = 500

        len_match = re.search(r'between\s+(\d+)\s+and\s+(\d+)\s*(?:character|char|letter)?', prompt)
        if len_match:
            min_len = int(len_match.group(1))
            max_len = int(len_match.group(2))

        max_match = re.search(r'(?:max|at most|under|less than)\s*(\d+)', prompt)
        if max_match:
            max_len = int(max_match.group(1))

        return {"aspect": aspect, "min_len": min_len, "max_len": max_len}

    def _extract_recall_params(self, prompt: str) -> Dict[str, Any]:
        # Extract word position
        word_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*word', prompt)
        word_index = int(word_match.group(1)) if word_match else 1

        # Extract transmission number
        trans_match = re.search(r'transmission\s*(\d+)', prompt)
        transmission_index = int(trans_match.group(1)) if trans_match else 1

        return {"word_index": word_index, "transmission_index": transmission_index}


# =============================================================================
# MAIN AGENT
# =============================================================================

class NeonAgent:
    """
    Main agent for the Neon Health challenge.

    Processes challenge messages and returns strict JSON responses.
    """

    def __init__(self, resume_path: Optional[str] = None, neon_code: Optional[str] = None):
        self.classifier = PromptClassifier()
        self.math_eval = MathEvaluator()
        self.knowledge = KnowledgeArchive()
        self.memory = MemoryStore()
        self.neon_code = neon_code

        # Load resume
        self.resume = None
        self.crew_manifest = None
        if resume_path:
            self.load_resume(resume_path)

    def load_resume(self, path: str):
        """Load resume from JSON file."""
        self.resume = Resume.from_json_file(path)
        self.crew_manifest = CrewManifest(self.resume)

    def load_resume_dict(self, data: Dict[str, Any]):
        """Load resume from dictionary."""
        self.resume = Resume.from_dict(data)
        self.crew_manifest = CrewManifest(self.resume)

    def set_neon_code(self, code: str):
        """Set the Neon authorization code."""
        self.neon_code = code

    def process(self, challenge: Dict[str, Any]) -> Dict[str, str]:
        """
        Process a challenge message.

        Input: {"type": "challenge", "message": [{"word": str, "timestamp": num}, ...]}
        Output: {"type": "enter_digits", "digits": "..."} or {"type": "speak_text", "text": "..."}
        """
        # Extract fragments from challenge
        fragments = challenge.get("message", [])

        # Reconstruct the prompt
        prompt = reconstruct_message(fragments)

        # Classify and handle
        checkpoint_type, params = self.classifier.classify(prompt)

        try:
            if checkpoint_type == PromptClassifier.SIGNAL_HANDSHAKE:
                response = self._handle_signal(params)
            elif checkpoint_type == PromptClassifier.VESSEL_ID:
                response = self._handle_vessel_id(params)
            elif checkpoint_type == PromptClassifier.MATH:
                response = self._handle_math(params)
            elif checkpoint_type == PromptClassifier.KNOWLEDGE:
                response = self._handle_knowledge(params)
            elif checkpoint_type == PromptClassifier.CREW_MANIFEST:
                response = self._handle_crew_manifest(params)
            elif checkpoint_type == PromptClassifier.VERIFICATION:
                response = self._handle_verification(params)
            else:
                response = speak_text("Unknown checkpoint type")
        except Exception as e:
            response = speak_text(f"Error: {str(e)}")

        # Store response
        self.memory.store_response(response, prompt)

        return response

    def process_raw(self, prompt: str) -> Dict[str, str]:
        """Process a raw prompt string (for testing)."""
        return self.process({"type": "challenge", "message": [{"word": prompt, "timestamp": 0}]})

    def _handle_signal(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Handle signal handshake checkpoint."""
        frequency = params.get("frequency", "")
        if params.get("append_hash"):
            frequency += "#"
        return enter_digits(frequency)

    def _handle_vessel_id(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Handle vessel identification checkpoint."""
        if not self.neon_code:
            raise ValueError("Neon code not set")
        code = self.neon_code
        if params.get("append_hash"):
            code += "#"
        return enter_digits(code)

    def _handle_math(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Handle computational assessment checkpoint."""
        expression = params.get("expression", "")
        result = self.math_eval.evaluate(expression)
        digits = str(result)
        if params.get("append_hash"):
            digits += "#"
        return enter_digits(digits)

    def _handle_knowledge(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Handle knowledge archive query checkpoint."""
        title = params.get("title", "")
        word_index = params.get("word_index", 1)
        word = self.knowledge.get_nth_word(title, word_index)
        return speak_text(word)

    def _handle_crew_manifest(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Handle crew manifest transmission checkpoint."""
        if not self.crew_manifest:
            raise ValueError("Resume not loaded")

        text = self.crew_manifest.get_response(
            params.get("aspect", "general"),
            params.get("min_len", 0),
            params.get("max_len", 500)
        )

        # Store for verification
        self.memory.store_crew_manifest(text)

        return speak_text(text)

    def _handle_verification(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Handle transmission verification checkpoint."""
        transmission_index = params.get("transmission_index", 1)
        word_index = params.get("word_index", 1)

        word = self.memory.recall_word(transmission_index, word_index)
        return speak_text(word)

    def reset(self):
        """Reset agent memory."""
        self.memory.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_agent(resume_path: str, neon_code: str) -> NeonAgent:
    """Create an agent with resume and Neon code."""
    agent = NeonAgent(resume_path=resume_path, neon_code=neon_code)
    return agent


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys
    import os

    # Load resume
    resume_path = os.path.join(os.path.dirname(__file__), "nithin_resume.json")

    agent = NeonAgent(resume_path=resume_path)

    print("NEON Agent CLI")
    print("=" * 50)
    print("Enter prompts or JSON challenges. Type 'quit' to exit.")
    print()

    # First, get Neon code
    neon_code = input("Enter your Neon Code: ").strip()
    agent.set_neon_code(neon_code)
    print(f"Neon code set: {neon_code}")
    print()

    while True:
        try:
            prompt = input("> ").strip()

            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'reset':
                agent.reset()
                print("Memory cleared.")
                continue
            elif prompt.lower() == 'memory':
                print(f"Crew manifests: {len(agent.memory.transmissions)}")
                for i, t in enumerate(agent.memory.transmissions, 1):
                    print(f"  [{i}] {t[:50]}...")
                continue

            # Try to parse as JSON challenge
            try:
                challenge = json.loads(prompt)
                if "type" in challenge and "message" in challenge:
                    response = agent.process(challenge)
                else:
                    response = agent.process_raw(prompt)
            except json.JSONDecodeError:
                response = agent.process_raw(prompt)

            print(json.dumps(response))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")
