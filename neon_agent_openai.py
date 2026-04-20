"""
NEON Agent with OpenAI Integration
===================================

Uses OpenAI GPT to handle prompt classification and response generation.
Maintains strict output format: enter_digits or speak_text.

Requires: OPENAI_API_KEY environment variable
"""

import os
import re
import json
import math
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from openai import OpenAI

# =============================================================================
# RESPONSE FORMATTERS
# =============================================================================

def speak_text(text: str) -> Dict[str, str]:
    return {"type": "speak_text", "text": str(text)}

def enter_digits(digits: str) -> Dict[str, str]:
    return {"type": "enter_digits", "digits": str(digits)}


# =============================================================================
# FRAGMENT RECONSTRUCTION
# =============================================================================

def reconstruct_message(fragments: List[Dict[str, Any]]) -> str:
    if not fragments:
        return ""
    sorted_fragments = sorted(fragments, key=lambda f: f.get("timestamp", 0))
    return " ".join(str(f.get("word", "")) for f in sorted_fragments)


# =============================================================================
# TOOLS
# =============================================================================

class MathTool:
    """Safe math evaluation tool."""

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
        expr = expression.strip().lower()

        for word, op in self.WORD_TO_OPERATOR.items():
            expr = re.sub(r'\b' + word + r'\b', op, expr)
        for word, num in self.WORD_TO_NUMBER.items():
            expr = re.sub(r'\b' + word + r'\b', num, expr)

        expr = re.sub(r'math\.floor\s*\(', 'math.floor(', expr, flags=re.IGNORECASE)
        expr = re.sub(r'math\.ceil\s*\(', 'math.ceil(', expr, flags=re.IGNORECASE)
        expr = re.sub(r'math\.round\s*\(', 'round(', expr, flags=re.IGNORECASE)

        allowed = {"math": math, "abs": abs, "round": round}
        result = eval(expr, {"__builtins__": {}}, allowed)
        return int(result)


class WikipediaTool:
    """Wikipedia summary lookup tool."""

    BASE_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    HEADERS = {"User-Agent": "NeonAgent/1.0"}

    def __init__(self):
        self._cache = {}

    def get_summary(self, title: str) -> str:
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
        summary = self.get_summary(title)
        words = summary.split()
        if n < 1 or n > len(words):
            raise ValueError(f"Word index {n} out of range")
        return words[n - 1]


# =============================================================================
# RESUME
# =============================================================================

@dataclass
class Resume:
    name: str
    education: str
    experience: str
    skills: str
    projects: str
    summary: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Resume':
        education = ""
        if "education" in data and isinstance(data["education"], list):
            parts = []
            for edu in data["education"]:
                if isinstance(edu, dict):
                    parts.append(f"{edu.get('degree', '')} in {edu.get('field', '')} from {edu.get('school', '')} ({edu.get('year', '')})")
            education = ". ".join(parts)

        experience = ""
        if "experience" in data and isinstance(data["experience"], list):
            parts = []
            for exp in data["experience"]:
                if isinstance(exp, dict):
                    parts.append(f"{exp.get('title', '')} at {exp.get('company', '')}: {exp.get('description', '')}")
            experience = ". ".join(parts)

        projects = ""
        if "projects" in data and isinstance(data["projects"], list):
            parts = []
            for proj in data["projects"]:
                if isinstance(proj, dict):
                    parts.append(f"{proj.get('name', '')}: {proj.get('description', '')}")
            projects = ". ".join(parts)

        skills = ""
        if "skills" in data:
            if isinstance(data["skills"], list):
                skills = ", ".join(data["skills"])
            else:
                skills = str(data["skills"])

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
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# =============================================================================
# MEMORY
# =============================================================================

class MemoryStore:
    def __init__(self):
        self.transmissions: List[str] = []
        self.all_responses: List[Dict[str, Any]] = []

    def store_crew_manifest(self, text: str):
        self.transmissions.append(text)

    def store_response(self, response: Dict[str, str], prompt: str):
        self.all_responses.append({
            "prompt": prompt,
            "response": response,
            "text": response.get("text", response.get("digits", ""))
        })

    def recall_word(self, transmission_index: int, word_index: int) -> str:
        if transmission_index < 1 or transmission_index > len(self.transmissions):
            raise ValueError(f"Transmission {transmission_index} not found")
        text = self.transmissions[transmission_index - 1]
        words = text.split()
        if word_index < 1 or word_index > len(words):
            raise ValueError(f"Word {word_index} not found")
        return words[word_index - 1]

    def clear(self):
        self.transmissions.clear()
        self.all_responses.clear()


# =============================================================================
# OPENAI AGENT
# =============================================================================

class NeonAgent:
    """OpenAI-powered agent for Neon Health challenge."""

    SYSTEM_PROMPT = """You are an AI agent handling the Neon Health authentication challenge.
You must analyze prompts and respond with ONLY a JSON object in one of these formats:
- {"type": "enter_digits", "digits": "..."} for numeric responses
- {"type": "speak_text", "text": "..."} for text responses

CHECKPOINT TYPES:
1. SIGNAL HANDSHAKE: "Respond on frequency X" → enter_digits with the frequency number
2. VESSEL ID: "Enter your authorization code" → enter_digits with the neon_code
3. MATH: "Calculate/Compute/What is X" → enter_digits with the result (use calculate tool)
4. KNOWLEDGE: "Speak the Nth word of entry for X" → speak_text with the word (use wikipedia tool)
5. CREW MANIFEST: Questions about crew member → speak_text with info from resume
6. VERIFICATION: "Recall word N from transmission M" → speak_text with the recalled word

CRITICAL RULES:
- ONLY append "#" to digits if the prompt explicitly says "pound key" or "followed by pound" or "#"
- Do NOT add "#" unless explicitly requested
- For math, ALWAYS use the calculate tool - never compute mentally
- For knowledge queries, ALWAYS use the wikipedia_lookup tool
- For crew manifest, ONLY answer with data from the resume. If the info is not in the resume (like age, phone, address), respond with: {"type": "speak_text", "text": "Information not available in crew manifest"}
- For recall/verification, use the recall_transmission tool
- Return ONLY the JSON object, no explanation or other text"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.math_tool = MathTool()
        self.wiki_tool = WikipediaTool()
        self.memory = MemoryStore()
        self.resume: Optional[Resume] = None
        self.neon_code: Optional[str] = None

    def load_resume(self, path: str):
        self.resume = Resume.from_json_file(path)

    def load_resume_dict(self, data: Dict[str, Any]):
        self.resume = Resume.from_dict(data)

    def set_neon_code(self, code: str):
        self.neon_code = code

    def _get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a mathematical expression. Supports +, -, *, /, %, Math.floor(), Math.ceil(), and word numbers/operators.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The math expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "wikipedia_lookup",
                    "description": "Get a specific word from a Wikipedia article summary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "The Wikipedia article title"
                            },
                            "word_index": {
                                "type": "integer",
                                "description": "The word position (1-indexed)"
                            }
                        },
                        "required": ["title", "word_index"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "recall_transmission",
                    "description": "Recall a word from a previous crew manifest transmission.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "transmission_index": {
                                "type": "integer",
                                "description": "Which transmission (1-indexed)"
                            },
                            "word_index": {
                                "type": "integer",
                                "description": "Which word (1-indexed)"
                            }
                        },
                        "required": ["transmission_index", "word_index"]
                    }
                }
            }
        ]

    def _execute_tool(self, name: str, args: dict) -> str:
        try:
            if name == "calculate":
                result = self.math_tool.evaluate(args["expression"])
                return str(result)
            elif name == "wikipedia_lookup":
                word = self.wiki_tool.get_nth_word(args["title"], args["word_index"])
                return word
            elif name == "recall_transmission":
                word = self.memory.recall_word(args["transmission_index"], args["word_index"])
                return word
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _build_context(self, prompt: str) -> str:
        context = f"PROMPT: {prompt}\n\n"

        if self.neon_code:
            context += f"NEON_CODE: {self.neon_code}\n"

        if self.resume:
            context += f"""
CREW MANIFEST DATA:
- Name: {self.resume.name}
- Education: {self.resume.education}
- Experience: {self.resume.experience}
- Skills: {self.resume.skills}
- Projects: {self.resume.projects}
- Summary: {self.resume.summary}
"""

        if self.memory.transmissions:
            context += f"\nPREVIOUS CREW MANIFEST TRANSMISSIONS:\n"
            for i, t in enumerate(self.memory.transmissions, 1):
                context += f"  Transmission {i}: {t[:100]}{'...' if len(t) > 100 else ''}\n"

        return context

    def process(self, challenge: Dict[str, Any]) -> Dict[str, str]:
        fragments = challenge.get("message", [])
        prompt = reconstruct_message(fragments)

        context = self._build_context(prompt)

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self._get_tools(),
                tool_choice="auto",
                temperature=0
            )

            message = response.choices[0].message

            # Handle tool calls
            while message.tool_calls:
                messages.append(message)

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    result = self._execute_tool(fn_name, fn_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=self._get_tools(),
                    tool_choice="auto",
                    temperature=0
                )
                message = response.choices[0].message

            # Parse the final response
            content = message.content.strip()

            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(content)

            # Store crew manifest transmissions
            if result.get("type") == "speak_text":
                text = result.get("text", "")
                # Check if this was a crew manifest response (has resume-like content)
                if self.resume and any(x in text.lower() for x in [
                    self.resume.name.lower(),
                    "masters", "bachelors", "engineer", "amazon", "skills"
                ]):
                    self.memory.store_crew_manifest(text)

            self.memory.store_response(result, prompt)
            return result

        except Exception as e:
            return speak_text(f"Error: {str(e)}")

    def process_raw(self, prompt: str) -> Dict[str, str]:
        return self.process({"type": "challenge", "message": [{"word": prompt, "timestamp": 0}]})

    def reset(self):
        self.memory.clear()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    agent = NeonAgent()

    resume_path = os.path.join(os.path.dirname(__file__), "nithin_resume.json")
    if os.path.exists(resume_path):
        agent.load_resume(resume_path)
        print("Resume loaded.")

    print("\nNEON Agent (OpenAI)")
    print("=" * 50)

    code = input("Enter Neon Code: ").strip()
    agent.set_neon_code(code)

    while True:
        try:
            prompt = input("\n> ").strip()
            if prompt.lower() == 'quit':
                break
            if prompt.lower() == 'reset':
                agent.reset()
                print("Memory cleared.")
                continue

            response = agent.process_raw(prompt)
            print(json.dumps(response))
        except KeyboardInterrupt:
            break

    print("\nGoodbye!")
