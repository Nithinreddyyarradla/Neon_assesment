"""
Neon Health / Nerdsnipe Hiring Challenge - Stateful AI Agent
============================================================

A deterministic, tool-using, memory-aware agent with strict IO contracts.

Features:
- Transmission reconstruction (fragment sorting by timestamp)
- Math evaluation with safe eval
- Wikipedia knowledge queries
- Resume-based crew manifest transmissions
- Stateful memory with recall capability
- Strict JSON output formatting
"""

import re
import json
import math
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ast
import operator


# =============================================================================
# OUTPUT FORMATTERS (STRICT - NEVER RETURN PLAIN TEXT)
# =============================================================================

def speak(text: str) -> Dict[str, str]:
    """Format a speak_text response."""
    return {"type": "speak_text", "text": str(text)}


def digits(value: Union[int, float, str]) -> Dict[str, str]:
    """Format an enter_digits response."""
    # Convert to string, handle floats by rounding if needed
    if isinstance(value, float):
        if value == int(value):
            value = int(value)
    return {"type": "enter_digits", "digits": str(value)}


# =============================================================================
# TASK TYPES
# =============================================================================

class TaskType(Enum):
    MATH = "math"
    KNOWLEDGE = "knowledge"
    RESUME = "resume"
    RECALL = "recall"
    FRAGMENT_RECONSTRUCTION = "fragment"
    SIGNAL_FREQUENCY = "signal_frequency"
    VESSEL_AUTH = "vessel_auth"
    UNKNOWN = "unknown"


# =============================================================================
# NEON CODE CONFIGURATION
# =============================================================================

# Your Neon authorization code (will be validated on first use)
NEON_CODE = None  # Set via agent.set_neon_code() or during authentication

# Authentication state
class AuthState(Enum):
    AWAITING_CODE = "awaiting_code"
    AUTHENTICATED = "authenticated"


# =============================================================================
# TRANSMISSION RECONSTRUCTION (Fragment Sorting)
# =============================================================================

@dataclass
class Fragment:
    """A transmission fragment with timestamp and content."""
    timestamp: float
    word: str


def reconstruct_transmission(fragments: List[Dict[str, Any]]) -> str:
    """
    Reconstruct a transmission from fragments.

    Input: [{"timestamp": 1.0, "word": "Hello"}, {"timestamp": 0.5, "word": "World"}]
    Output: "World Hello" (sorted by timestamp)
    """
    if not fragments:
        return ""

    # Parse fragments
    parsed = []
    for frag in fragments:
        ts = frag.get("timestamp", frag.get("ts", frag.get("t", 0)))
        word = frag.get("word", frag.get("text", frag.get("w", "")))
        parsed.append(Fragment(timestamp=float(ts), word=str(word)))

    # Sort by timestamp
    parsed.sort(key=lambda f: f.timestamp)

    # Join words
    return " ".join(f.word for f in parsed)


def parse_fragment_input(raw_input: str) -> Optional[List[Dict[str, Any]]]:
    """Try to parse input as fragments."""
    try:
        data = json.loads(raw_input)
        if isinstance(data, list) and len(data) > 0:
            if all(isinstance(item, dict) and
                   any(k in item for k in ["timestamp", "ts", "t"]) and
                   any(k in item for k in ["word", "text", "w"])
                   for item in data):
                return data
    except:
        pass
    return None


# =============================================================================
# SAFE MATH EVALUATOR
# =============================================================================

class SafeMathEvaluator:
    """
    Safely evaluate mathematical expressions without using eval().
    Supports: +, -, *, /, //, %, **, parentheses, and common functions.
    """

    # Allowed operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Allowed functions (Python and JavaScript-style)
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'int': int,
        'float': float,
        # JavaScript-style Math functions
        'floor': math.floor,
        'ceil': math.ceil,
        'sqrt': math.sqrt,
        'trunc': math.trunc,
    }

    def evaluate(self, expression: str) -> Union[int, float]:
        """Safely evaluate a math expression."""
        # Clean the expression
        expression = expression.strip()

        # Parse the expression
        try:
            tree = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid expression: {e}")

        return self._eval_node(tree.body)

    def _eval_node(self, node) -> Union[int, float]:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")

        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(operand)

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name not in self.FUNCTIONS:
                    raise ValueError(f"Unsupported function: {func_name}")
                args = [self._eval_node(arg) for arg in node.args]
                return self.FUNCTIONS[func_name](*args)
            raise ValueError("Unsupported function call")

        elif isinstance(node, ast.List):
            return [self._eval_node(elem) for elem in node.elts]

        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elem) for elem in node.elts)

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")


# =============================================================================
# WIKIPEDIA KNOWLEDGE HANDLER
# =============================================================================

class KnowledgeHandler:
    """Fetch summaries from Wikipedia API and extract specific words."""

    BASE_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"

    # Required by Wikipedia API
    HEADERS = {
        "User-Agent": "NeonAgent/1.0 (https://github.com/neon-health; contact@example.com) python-requests"
    }

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self._cache: Dict[str, str] = {}

    def fetch_summary(self, topic: str) -> str:
        """Fetch the summary for a given topic."""
        # Normalize topic
        topic_key = topic.strip().replace(" ", "_")

        # Check cache
        if topic_key in self._cache:
            return self._cache[topic_key]

        # Fetch from API
        url = self.BASE_URL.format(topic=topic_key)
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            summary = data.get("extract", "")
            self._cache[topic_key] = summary
            return summary
        except requests.RequestException as e:
            raise ValueError(f"Failed to fetch knowledge for '{topic}': {e}")

    def get_nth_word(self, topic: str, n: int) -> str:
        """
        Get the Nth word from a topic's summary.
        Note: n is 1-indexed (1st word, 2nd word, etc.)
        """
        summary = self.fetch_summary(topic)

        # Split into words (handle punctuation)
        words = re.findall(r'\b\w+\b', summary)

        if not words:
            raise ValueError(f"No words found in summary for '{topic}'")

        # Convert to 0-indexed
        index = n - 1

        if index < 0 or index >= len(words):
            raise ValueError(f"Word index {n} out of range (1-{len(words)})")

        return words[index]


# =============================================================================
# RESUME / CREW MANIFEST HANDLER
# =============================================================================

@dataclass
class Resume:
    """Structured resume data."""
    name: str = ""
    education: List[Dict[str, str]] = field(default_factory=list)
    experience: List[Dict[str, str]] = field(default_factory=list)
    projects: List[Dict[str, str]] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Resume':
        """Create a Resume from a dictionary."""
        return cls(
            name=data.get("name", ""),
            education=data.get("education", []),
            experience=data.get("experience", []),
            projects=data.get("projects", []),
            skills=data.get("skills", []),
            summary=data.get("summary", ""),
        )

    @classmethod
    def from_json_file(cls, path: str) -> 'Resume':
        """Load resume from a JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


class ResumeHandler:
    """Generate resume summaries with strict length constraints."""

    def __init__(self, resume: Resume):
        self.resume = resume

    def get_education_summary(self, min_len: int = 0, max_len: int = 500) -> str:
        """Generate education summary within length constraints."""
        parts = []
        for edu in self.resume.education:
            degree = edu.get("degree", "")
            school = edu.get("school", edu.get("institution", ""))
            year = edu.get("year", edu.get("graduation_year", ""))
            field_of_study = edu.get("field", edu.get("major", ""))

            if degree and school:
                entry = f"{degree} in {field_of_study} from {school}" if field_of_study else f"{degree} from {school}"
                if year:
                    entry += f" ({year})"
                parts.append(entry)

        text = ". ".join(parts) if parts else self.resume.summary
        return self._constrain_length(text, min_len, max_len)

    def get_experience_summary(self, min_len: int = 0, max_len: int = 500) -> str:
        """Generate work experience summary within length constraints."""
        parts = []
        for exp in self.resume.experience:
            title = exp.get("title", exp.get("role", ""))
            company = exp.get("company", exp.get("organization", ""))
            description = exp.get("description", exp.get("summary", ""))

            if title and company:
                entry = f"{title} at {company}"
                if description:
                    entry += f": {description}"
                parts.append(entry)

        text = ". ".join(parts) if parts else self.resume.summary
        return self._constrain_length(text, min_len, max_len)

    def get_projects_summary(self, min_len: int = 0, max_len: int = 500) -> str:
        """Generate projects summary within length constraints."""
        parts = []
        for proj in self.resume.projects:
            name = proj.get("name", proj.get("title", ""))
            description = proj.get("description", proj.get("summary", ""))
            tech = proj.get("technologies", proj.get("tech", []))

            if name:
                entry = name
                if description:
                    entry += f": {description}"
                if tech:
                    tech_str = ", ".join(tech) if isinstance(tech, list) else tech
                    entry += f" (Tech: {tech_str})"
                parts.append(entry)

        text = ". ".join(parts) if parts else self.resume.summary
        return self._constrain_length(text, min_len, max_len)

    def get_skills_summary(self, min_len: int = 0, max_len: int = 500) -> str:
        """Generate skills summary within length constraints."""
        if self.resume.skills:
            text = "Skills: " + ", ".join(self.resume.skills)
        else:
            text = self.resume.summary
        return self._constrain_length(text, min_len, max_len)

    def get_full_summary(self, min_len: int = 0, max_len: int = 500) -> str:
        """Generate full resume summary within length constraints."""
        text = self.resume.summary
        if not text:
            parts = []
            if self.resume.name:
                parts.append(self.resume.name)
            parts.append(self.get_education_summary(0, max_len // 3))
            parts.append(self.get_experience_summary(0, max_len // 3))
            text = ". ".join(filter(None, parts))
        return self._constrain_length(text, min_len, max_len)

    def _constrain_length(self, text: str, min_len: int, max_len: int) -> str:
        """Constrain text to be within length bounds."""
        text = text.strip()

        # If too long, truncate intelligently
        if len(text) > max_len:
            # Try to truncate at word boundary
            truncated = text[:max_len]
            last_space = truncated.rfind(' ')
            if last_space > max_len * 0.8:  # Only if we don't lose too much
                truncated = truncated[:last_space]
            # Remove trailing punctuation that might look odd
            truncated = truncated.rstrip('.,;:')
            text = truncated

        # If too short, pad with relevant info (or return as-is if we can't)
        if len(text) < min_len:
            # This is a constraint violation - we don't have enough content
            # In a real scenario, we might need to add more detail
            pass

        return text


# =============================================================================
# MEMORY SYSTEM
# =============================================================================

@dataclass
class MemoryEntry:
    """A single memory entry."""
    index: int
    task_type: TaskType
    prompt: str
    response_text: str
    response_json: Dict[str, str]


class MemoryStore:
    """
    Stateful memory that tracks all responses.
    Critical for the final recall checkpoint.
    """

    def __init__(self):
        self.entries: List[MemoryEntry] = []
        self._crew_manifest_indices: List[int] = []  # Track crew manifest responses specifically

    def store(self, task_type: TaskType, prompt: str, response_text: str,
              response_json: Dict[str, str]) -> int:
        """Store a response and return its index."""
        index = len(self.entries)
        entry = MemoryEntry(
            index=index,
            task_type=task_type,
            prompt=prompt,
            response_text=response_text,
            response_json=response_json
        )
        self.entries.append(entry)

        # Track crew manifest entries separately
        if task_type == TaskType.RESUME:
            self._crew_manifest_indices.append(index)

        return index

    def recall_word(self, response_index: int, word_index: int) -> str:
        """
        Recall a specific word from a previous response.

        Args:
            response_index: Which response (0-indexed or 1-indexed depending on prompt)
            word_index: Which word in that response (typically 1-indexed)

        Returns:
            The requested word
        """
        if response_index < 0 or response_index >= len(self.entries):
            raise ValueError(f"Response index {response_index} out of range (0-{len(self.entries)-1})")

        text = self.entries[response_index].response_text
        words = text.split()

        # Assume 1-indexed word position (convert to 0-indexed)
        idx = word_index - 1

        if idx < 0 or idx >= len(words):
            raise ValueError(f"Word index {word_index} out of range (1-{len(words)})")

        return words[idx]

    def recall_crew_manifest_word(self, manifest_index: int, word_index: int) -> str:
        """
        Recall a word specifically from crew manifest transmissions.

        Args:
            manifest_index: Which crew manifest response (1-indexed typically)
            word_index: Which word (1-indexed)
        """
        # Convert to 0-indexed
        idx = manifest_index - 1

        if idx < 0 or idx >= len(self._crew_manifest_indices):
            raise ValueError(f"Crew manifest index {manifest_index} out of range")

        actual_index = self._crew_manifest_indices[idx]
        return self.recall_word(actual_index, word_index)

    def get_all_responses(self) -> List[str]:
        """Get all response texts."""
        return [e.response_text for e in self.entries]

    def get_response(self, index: int) -> Optional[MemoryEntry]:
        """Get a specific response entry."""
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def clear(self):
        """Clear all memory."""
        self.entries.clear()
        self._crew_manifest_indices.clear()


# =============================================================================
# TASK ROUTER
# =============================================================================

class TaskRouter:
    """Classify prompts into task types."""

    # Patterns for each task type
    MATH_PATTERNS = [
        r'(?:calculate|compute|evaluate|solve|what is|what\'s)\s*[:.]?\s*[\d\+\-\*\/\(\)\.\s\^%]+',
        r'^\s*[\d\+\-\*\/\(\)\.\s\^%]+\s*[=?]?\s*$',
        r'math\s*[:.]?\s*',
        r'expression\s*[:.]?\s*',
        # Word-based math: "What's 2 plus 3?"
        r'(?:what\'s|what is|calculate|compute)\s+\d+\s+(?:plus|minus|times|multiplied by|divided by|over)\s+\d+',
    ]

    # Word to operator mapping
    WORD_TO_OPERATOR = {
        'plus': '+',
        'minus': '-',
        'times': '*',
        'multiplied by': '*',
        'divided by': '/',
        'over': '/',
        'mod': '%',
        'modulo': '%',
        'to the power of': '**',
        'power': '**',
    }

    KNOWLEDGE_PATTERNS = [
        # "Speak the 8th word in the knowledge archive entry for Saturn"
        r'(?:speak|get|tell|fetch)\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+word\s+(?:in\s+)?(?:the\s+)?(?:knowledge\s+)?(?:archive\s+)?(?:entry\s+)?(?:for|of|from|in|about)\s+(.+)',
        # "Speak the 1st word entry for Python" or "Speak the 2nd word for Saturn"
        r'(?:speak|get|tell|fetch)\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+word\s+(?:entry\s+)?(?:for|of|from|in)\s+(.+)',
        # "What is the 3rd word of the entry for Mars"
        r'(?:what|speak|tell|get|fetch|retrieve)\s+(?:is\s+)?(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+word\s+(?:of|for|from|in)\s+(?:the\s+)?(?:entry\s+(?:for|of|about)\s+)?(.+)',
        # "knowledge archive for Python"
        r'(?:knowledge|wiki|wikipedia)\s+(?:archive|entry|summary)\s*[:.]?\s*(.+)',
        # "entry for Python"
        r'entry\s+(?:for|about)\s+(.+)',
    ]

    RECALL_PATTERNS = [
        # "Recall the 1st word from crew manifest transmission 1"
        r'recall\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+word\s+(?:from|of|in)\s+(?:your\s+)?(?:crew\s+)?(?:manifest\s+)?(?:transmission|response)\s*(\d+)?',
        # "Recall the 1st word from your earlier crew manifest transmission 1"
        r'recall\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+word\s+(?:from|of|in)\s+(?:your\s+)?(?:earlier\s+)?(?:crew\s+)?(?:manifest\s+)?(?:transmission|response)\s*(\d+)?',
        # "What was the 3rd word of your earlier transmission"
        r'what\s+(?:was|is)\s+(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+word\s+(?:of|in|from)\s+(?:your\s+)?(?:earlier|previous|last)?\s*(?:crew\s+)?(?:manifest\s+)?(?:transmission|response)\s*(\d+)?',
        # "Remember word 2 from manifest 1"
        r'(?:remember|recall)\s+(?:word\s+)?(\d+)\s+(?:from|of)\s+(?:crew\s+)?(?:manifest\s+)?(?:transmission\s+)?(\d+)?',
    ]

    RESUME_PATTERNS = [
        r'(?:crew|manifest|transmission|transmit)',
        r'(?:background|education|experience|skills|projects?)',
        r'(?:resume|cv|profile)',
        r'(?:summary|describe|tell\s+(?:me\s+)?about)',
        r'(?:deployment|qualification|work\s+history)',
    ]

    # Signal frequency patterns
    SIGNAL_PATTERNS = [
        r'(?:respond|tune|set|enter)\s+(?:on\s+)?(?:frequency|signal|comms?)\s*(?:to\s+)?(\d+)',
        r'frequency\s+(\d+)',
        r'signal\s+frequency\s*[:.]?\s*(\d+)',
    ]

    # Vessel authentication patterns
    VESSEL_AUTH_PATTERNS = [
        r'(?:enter|provide|transmit)\s+(?:your\s+)?(?:vessel\s+)?(?:authorization|auth|neon)\s*(?:code)?',
        r'authenticate\s+(?:your\s+)?vessel',
        r'vessel\s+(?:identification|id|code)',
        r'neon\s+code',
    ]

    def classify(self, prompt: str) -> Tuple[TaskType, Dict[str, Any]]:
        """
        Classify a prompt and extract relevant parameters.

        Returns:
            Tuple of (TaskType, extracted_params)
        """
        prompt_lower = prompt.lower().strip()

        # Check for signal frequency (highest priority)
        for pattern in self.SIGNAL_PATTERNS:
            match = re.search(pattern, prompt_lower)
            if match:
                frequency = match.group(1)
                # Check if # should be appended
                append_hash = '#' in prompt or 'hash' in prompt_lower or 'pound' in prompt_lower
                return TaskType.SIGNAL_FREQUENCY, {"frequency": frequency, "append_hash": append_hash}

        # Check for vessel authentication
        for pattern in self.VESSEL_AUTH_PATTERNS:
            if re.search(pattern, prompt_lower):
                append_hash = '#' in prompt or 'hash' in prompt_lower or 'pound' in prompt_lower
                return TaskType.VESSEL_AUTH, {"append_hash": append_hash}

        # Check for recall (high priority)
        for pattern in self.RECALL_PATTERNS:
            match = re.search(pattern, prompt_lower)
            if match:
                groups = match.groups()
                word_idx = int(groups[0]) if groups[0] else 1
                response_idx = int(groups[1]) if len(groups) > 1 and groups[1] else 1
                return TaskType.RECALL, {"word_index": word_idx, "response_index": response_idx}

        # Check for knowledge queries
        for pattern in self.KNOWLEDGE_PATTERNS:
            match = re.search(pattern, prompt_lower)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    word_idx = int(groups[0])
                    topic = groups[1].strip().rstrip('.')
                    return TaskType.KNOWLEDGE, {"word_index": word_idx, "topic": topic}
                elif len(groups) == 1:
                    topic = groups[0].strip().rstrip('.')
                    return TaskType.KNOWLEDGE, {"topic": topic, "word_index": None}

        # Check for math expressions
        for pattern in self.MATH_PATTERNS:
            if re.search(pattern, prompt_lower):
                # Extract the expression
                expr = re.sub(r'^.*?(?:calculate|compute|evaluate|solve|what is|what\'s|math|expression)\s*[:.]?\s*', '', prompt_lower)
                expr = re.sub(r'\s*[=?]\s*$', '', expr)
                expr = expr.strip()

                # Convert JavaScript Math.xxx to Python functions
                expr = re.sub(r'math\.floor', 'floor', expr)
                expr = re.sub(r'math\.ceil', 'ceil', expr)
                expr = re.sub(r'math\.round', 'round', expr)
                expr = re.sub(r'math\.sqrt', 'sqrt', expr)
                expr = re.sub(r'math\.trunc', 'trunc', expr)
                expr = re.sub(r'math\.abs', 'abs', expr)

                # Convert word operators to symbols
                for word, symbol in self.WORD_TO_OPERATOR.items():
                    expr = expr.replace(word, symbol)

                # Clean up spaces around operators
                expr = re.sub(r'\s*([+\-*/^%])\s*', r'\1', expr)

                # Check if it's a valid math expression (including function calls)
                if expr and re.match(r'^[\d\+\-\*\/\(\)\.\s\^%a-z_,]+$', expr):
                    return TaskType.MATH, {"expression": expr}

        # Check for pure math expression
        if re.match(r'^[\d\+\-\*\/\(\)\.\s\^%]+$', prompt_lower):
            return TaskType.MATH, {"expression": prompt_lower}

        # Check for resume/crew manifest
        for pattern in self.RESUME_PATTERNS:
            if re.search(pattern, prompt_lower):
                # Extract length constraints if present
                min_len = 0
                max_len = 500
                len_match = re.search(r'(?:between\s+)?(\d+)\s*(?:and|to|-)\s*(\d+)\s*(?:characters?|chars?|letters?)?', prompt_lower)
                if len_match:
                    min_len = int(len_match.group(1))
                    max_len = int(len_match.group(2))

                max_match = re.search(r'(?:max(?:imum)?|at most|under|less than)\s*(\d+)\s*(?:characters?|chars?|letters?)?', prompt_lower)
                if max_match:
                    max_len = int(max_match.group(1))

                # Determine which aspect of resume
                aspect = "full"
                if "education" in prompt_lower or "background" in prompt_lower or "qualification" in prompt_lower:
                    aspect = "education"
                elif "experience" in prompt_lower or "work" in prompt_lower or "deployment" in prompt_lower:
                    aspect = "experience"
                elif "project" in prompt_lower:
                    aspect = "projects"
                elif "skill" in prompt_lower:
                    aspect = "skills"

                return TaskType.RESUME, {"aspect": aspect, "min_len": min_len, "max_len": max_len}

        return TaskType.UNKNOWN, {}


# =============================================================================
# MAIN AGENT
# =============================================================================

class NeonAgent:
    """
    The main stateful AI agent for the Neon Health challenge.

    This agent:
    - Handles authentication with Neon code
    - Reconstructs fragmented transmissions
    - Routes tasks to appropriate handlers
    - Maintains memory across the session
    - Returns strictly formatted JSON responses
    """

    def __init__(self, resume: Optional[Resume] = None, neon_code: Optional[str] = None):
        self.memory = MemoryStore()
        self.router = TaskRouter()
        self.math_eval = SafeMathEvaluator()
        self.knowledge = KnowledgeHandler()
        self.resume_handler = ResumeHandler(resume) if resume else None
        self.resume = resume
        self.neon_code = neon_code  # Will be set during authentication
        self.auth_state = AuthState.AWAITING_CODE if neon_code is None else AuthState.AUTHENTICATED

    def set_neon_code(self, code: str) -> bool:
        """Set and validate the Neon code."""
        if code and len(code) > 0:
            self.neon_code = code.strip()
            self.auth_state = AuthState.AUTHENTICATED
            return True
        return False

    def is_authenticated(self) -> bool:
        """Check if the agent is authenticated."""
        return self.auth_state == AuthState.AUTHENTICATED

    def load_resume(self, resume: Resume):
        """Load or update the resume."""
        self.resume = resume
        self.resume_handler = ResumeHandler(resume)

    def load_resume_from_dict(self, data: Dict[str, Any]):
        """Load resume from a dictionary."""
        self.load_resume(Resume.from_dict(data))

    def load_resume_from_json(self, path: str):
        """Load resume from a JSON file."""
        self.load_resume(Resume.from_json_file(path))

    def process(self, prompt: str) -> Dict[str, str]:
        """
        Process a prompt and return a strictly formatted JSON response.

        This is the main entry point for the agent.
        """
        # Step 0: Check authentication state
        if self.auth_state == AuthState.AWAITING_CODE:
            # Check if this is a Neon code submission
            code = prompt.strip()
            # Accept the code if it looks like a hex string or alphanumeric code
            if re.match(r'^[a-fA-F0-9]{8,}$', code) or re.match(r'^[a-zA-Z0-9_-]{8,}$', code):
                self.set_neon_code(code)
                return speak(f"Neon code accepted. Authentication successful. Awaiting signal frequency...")
            else:
                return speak("Please enter your Neon authorization code to begin.")

        # Step 1: Check if input is fragmented transmission
        fragments = parse_fragment_input(prompt)
        if fragments:
            prompt = reconstruct_transmission(fragments)

        # Step 2: Classify the task
        task_type, params = self.router.classify(prompt)

        # Step 3: Handle the task
        try:
            if task_type == TaskType.SIGNAL_FREQUENCY:
                response = self._handle_signal_frequency(params)
            elif task_type == TaskType.VESSEL_AUTH:
                response = self._handle_vessel_auth(params)
            elif task_type == TaskType.MATH:
                response = self._handle_math(params)
            elif task_type == TaskType.KNOWLEDGE:
                response = self._handle_knowledge(params)
            elif task_type == TaskType.RESUME:
                response = self._handle_resume(params)
            elif task_type == TaskType.RECALL:
                response = self._handle_recall(params)
            else:
                # Default: try to interpret as resume question
                response = self._handle_resume({"aspect": "full", "min_len": 0, "max_len": 500})
        except Exception as e:
            # Even errors must be formatted correctly
            response = speak(f"Error: {str(e)}")

        # Step 4: Store in memory (extract text from response)
        response_text = response.get("text", response.get("digits", ""))
        self.memory.store(task_type, prompt, response_text, response)

        return response

    def _handle_signal_frequency(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Handle signal frequency handshake."""
        frequency = params.get("frequency", "")
        append_hash = params.get("append_hash", False)
        if append_hash:
            frequency += "#"
        return digits(frequency)

    def _handle_vessel_auth(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Handle vessel authentication with Neon code."""
        append_hash = params.get("append_hash", False)
        code = self.neon_code
        if append_hash:
            code += "#"
        return digits(code)

    def _handle_math(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Handle math evaluation tasks."""
        expression = params.get("expression", "")

        # Clean the expression (replace ^ with ** for Python)
        expression = expression.replace("^", "**")

        result = self.math_eval.evaluate(expression)
        return digits(result)

    def _handle_knowledge(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Handle Wikipedia knowledge queries."""
        topic = params.get("topic", "")
        word_index = params.get("word_index")

        if word_index:
            word = self.knowledge.get_nth_word(topic, word_index)
            return speak(word)
        else:
            summary = self.knowledge.fetch_summary(topic)
            return speak(summary)

    def _handle_resume(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Handle resume/crew manifest queries."""
        if not self.resume_handler:
            return speak("No resume loaded")

        aspect = params.get("aspect", "full")
        min_len = params.get("min_len", 0)
        max_len = params.get("max_len", 500)

        if aspect == "education":
            text = self.resume_handler.get_education_summary(min_len, max_len)
        elif aspect == "experience":
            text = self.resume_handler.get_experience_summary(min_len, max_len)
        elif aspect == "projects":
            text = self.resume_handler.get_projects_summary(min_len, max_len)
        elif aspect == "skills":
            text = self.resume_handler.get_skills_summary(min_len, max_len)
        else:
            text = self.resume_handler.get_full_summary(min_len, max_len)

        return speak(text)

    def _handle_recall(self, params: Dict[str, Any]) -> Dict[str, str]:
        """Handle recall queries from memory."""
        word_index = params.get("word_index", 1)
        response_index = params.get("response_index", 1)

        # Try crew manifest first, then fall back to all responses
        try:
            word = self.memory.recall_crew_manifest_word(response_index, word_index)
        except ValueError:
            # Fall back to general response index (0-indexed internally)
            word = self.memory.recall_word(response_index - 1, word_index)

        return speak(word)

    def reset_memory(self):
        """Reset the agent's memory."""
        self.memory.clear()

    def get_memory_dump(self) -> List[Dict[str, Any]]:
        """Get a dump of all memory entries for debugging."""
        return [
            {
                "index": e.index,
                "task_type": e.task_type.value,
                "prompt": e.prompt,
                "response_text": e.response_text,
                "response_json": e.response_json,
            }
            for e in self.memory.entries
        ]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_agent_with_resume(resume_data: Dict[str, Any]) -> NeonAgent:
    """Create an agent with a resume loaded."""
    agent = NeonAgent()
    agent.load_resume_from_dict(resume_data)
    return agent


def create_agent_from_resume_file(path: str) -> NeonAgent:
    """Create an agent with a resume loaded from a JSON file."""
    agent = NeonAgent()
    agent.load_resume_from_json(path)
    return agent


# =============================================================================
# CLI INTERFACE (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys

    # Sample resume for testing
    sample_resume = {
        "name": "Test Candidate",
        "summary": "Experienced software engineer with expertise in distributed systems and machine learning.",
        "education": [
            {
                "degree": "Master of Science",
                "field": "Computer Science",
                "school": "Stanford University",
                "year": "2020"
            },
            {
                "degree": "Bachelor of Science",
                "field": "Computer Engineering",
                "school": "MIT",
                "year": "2018"
            }
        ],
        "experience": [
            {
                "title": "Senior Software Engineer",
                "company": "Google",
                "description": "Led development of distributed data processing pipelines handling petabytes of data"
            },
            {
                "title": "Software Engineer",
                "company": "Meta",
                "description": "Built real-time ML inference systems serving millions of requests per second"
            }
        ],
        "projects": [
            {
                "name": "DistributedDB",
                "description": "Open-source distributed database with strong consistency guarantees",
                "technologies": ["Go", "Raft", "gRPC"]
            }
        ],
        "skills": ["Python", "Go", "Kubernetes", "Machine Learning", "Distributed Systems"]
    }

    # Create agent
    agent = create_agent_with_resume(sample_resume)

    print("Neon Agent CLI - Type 'quit' to exit, 'memory' to see history")
    print("-" * 50)

    while True:
        try:
            prompt = input("\nPrompt> ").strip()

            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'memory':
                print("\nMemory Dump:")
                for entry in agent.get_memory_dump():
                    print(f"  [{entry['index']}] {entry['task_type']}: {entry['response_text'][:50]}...")
                continue
            elif not prompt:
                continue

            response = agent.process(prompt)
            print(f"\nResponse: {json.dumps(response, indent=2)}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nGoodbye!")
