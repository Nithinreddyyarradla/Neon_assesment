"""
NEON Agent with OpenAI Integration
===================================

Uses OpenAI GPT to handle all prompt classification and response generation.
No hardcoded responses - OpenAI handles everything.

Requires: OPENAI_API_KEY environment variable
"""

import os
import re
import json
import math
import requests
from typing import Dict, List, Any, Optional
from openai import OpenAI


# =============================================================================
# FRAGMENT RECONSTRUCTION
# =============================================================================

def reconstruct_message(fragments: List[Dict[str, Any]]) -> str:
    """Sort fragments by timestamp and join words."""
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
# MEMORY STORE
# =============================================================================

class MemoryStore:
    def __init__(self):
        self.transmissions: List[str] = []
        self.all_responses: List[Dict[str, Any]] = []

    def store_transmission(self, text: str):
        self.transmissions.append(text)

    def store_response(self, response: Dict[str, str], prompt: str):
        self.all_responses.append({"prompt": prompt, "response": response})

    def recall_word(self, transmission_index: int, word_index: int) -> str:
        if transmission_index < 1 or transmission_index > len(self.transmissions):
            raise ValueError(f"Transmission {transmission_index} not found")
        text = self.transmissions[transmission_index - 1]
        words = text.split()
        if word_index < 1 or word_index > len(words):
            raise ValueError(f"Word {word_index} not found in transmission")
        return words[word_index - 1]

    def clear(self):
        self.transmissions.clear()
        self.all_responses.clear()


# =============================================================================
# OPENAI AGENT
# =============================================================================

class NeonAgent:
    """OpenAI-powered agent for Neon Health challenge."""

    SYSTEM_PROMPT = """You are NEON, an AI assistant for the Neon Health authentication challenge.

You MUST respond with ONLY a valid JSON object in one of these two formats:
1. {"type": "enter_digits", "digits": "..."} - for numeric/code responses
2. {"type": "speak_text", "text": "..."} - for text responses

TASK TYPES AND HOW TO HANDLE THEM:

1. SIGNAL/FREQUENCY: If asked to "respond on frequency X" or similar
   → Return {"type": "enter_digits", "digits": "X"}
   → Only add "#" at the end if they explicitly say "pound key" or "followed by #"

2. AUTHORIZATION CODE: If asked for "authorization code" or "neon code" or "vessel code"
   → Return {"type": "enter_digits", "digits": "<the neon_code provided in context>"}
   → Only add "#" if explicitly requested

3. MATH/CALCULATIONS: If asked to calculate, compute, or "what is X + Y"
   → Use the calculate tool to get the answer
   → Return {"type": "enter_digits", "digits": "<result>"}
   → Only add "#" if explicitly requested

4. KNOWLEDGE/WIKIPEDIA: If asked about the "Nth word" of a "knowledge archive entry" for something
   → Use the wikipedia_lookup tool
   → Return {"type": "speak_text", "text": "<the word>"}

5. CREW MANIFEST/RESUME: If asked about the crew member/mate (who, name, email, phone, location, linkedin, education, experience, skills, projects, etc.)
   → Look at the CREW MANIFEST DATA JSON provided in context
   → Extract and return the EXACT value from the JSON for the requested field
   → For email: return the "email" field value directly (e.g., "reddy.nithin2026@gmail.com")
   → For phone: return the "phone" field value directly
   → For location: return the "location" field value directly
   → For linkedin: return the "linkedin" field value directly
   → For name: return the "name" field value directly
   → Return {"type": "speak_text", "text": "<the exact value from the JSON>"}
   → IMPORTANT: The email, phone, location, linkedin fields ARE in the JSON - always check and return them

6. RECALL/VERIFICATION: If asked to recall a word from a previous transmission
   → Use the recall_transmission tool
   → Return {"type": "speak_text", "text": "<the word>"}

IMPORTANT RULES:
- NEVER add "#" unless the prompt explicitly mentions "pound key" or "followed by #"
- For math, ALWAYS use the calculate tool - do not compute mentally
- For Wikipedia queries, ALWAYS use the wikipedia_lookup tool
- For recall queries, ALWAYS use the recall_transmission tool
- Return ONLY the JSON object, no additional text or explanation
- If a crew manifest field is empty or not provided, mention what information IS available"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.math_tool = MathTool()
        self.wiki_tool = WikipediaTool()
        self.memory = MemoryStore()
        self.resume_data: Optional[Dict[str, Any]] = None
        self.neon_code: Optional[str] = None

    def load_resume(self, path: str):
        with open(path, 'r') as f:
            self.resume_data = json.load(f)

    def load_resume_dict(self, data: Dict[str, Any]):
        self.resume_data = data

    def set_neon_code(self, code: str):
        self.neon_code = code

    @property
    def resume(self):
        return self.resume_data

    def _get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a mathematical expression. Use this for ANY math calculation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "The math expression"}
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
                            "title": {"type": "string", "description": "The Wikipedia article title"},
                            "word_index": {"type": "integer", "description": "Which word (1-indexed)"}
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
                            "transmission_index": {"type": "integer", "description": "Which transmission (1-indexed)"},
                            "word_index": {"type": "integer", "description": "Which word (1-indexed)"}
                        },
                        "required": ["transmission_index", "word_index"]
                    }
                }
            }
        ]

    def _execute_tool(self, name: str, args: dict) -> str:
        try:
            if name == "calculate":
                return str(self.math_tool.evaluate(args["expression"]))
            elif name == "wikipedia_lookup":
                return self.wiki_tool.get_nth_word(args["title"], args["word_index"])
            elif name == "recall_transmission":
                return self.memory.recall_word(args["transmission_index"], args["word_index"])
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _build_context(self, prompt: str) -> str:
        context = f"USER PROMPT: {prompt}\n\n"

        if self.neon_code:
            context += f"NEON_CODE (use this for authorization requests): {self.neon_code}\n\n"

        if self.resume_data:
            context += "CREW MANIFEST DATA:\n"
            context += json.dumps(self.resume_data, indent=2)
            context += "\n\n"

        if self.memory.transmissions:
            context += "PREVIOUS TRANSMISSIONS (for recall queries):\n"
            for i, t in enumerate(self.memory.transmissions, 1):
                context += f"  Transmission {i}: {t}\n"

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

            # Parse response
            content = message.content.strip()
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(content)

            # Store transmission if it's a crew manifest response
            if result.get("type") == "speak_text" and self.resume_data:
                text = result.get("text", "")
                # Store if it contains resume-like content
                name = self.resume_data.get("name", "").lower()
                if name and name in text.lower():
                    self.memory.store_transmission(text)

            self.memory.store_response(result, prompt)
            return result

        except Exception as e:
            return {"type": "speak_text", "text": f"Error: {str(e)}"}

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
