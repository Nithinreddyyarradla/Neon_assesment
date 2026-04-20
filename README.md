# Neon Health AI Agent

A stateful AI agent built for the Neon Health hiring challenge. This agent processes fragmented transmissions, handles multiple checkpoint types, and maintains memory for verification.

## Features

- **OpenAI GPT-4o-mini Integration** - Intelligent prompt classification and response generation
- **Fragment Reconstruction** - Sorts and joins timestamped message fragments
- **Multiple Checkpoint Types**:
  - Signal Handshake (frequency responses)
  - Vessel Identification (authorization codes)
  - Computational Assessment (math with word numbers/operators)
  - Knowledge Archive (Wikipedia lookups)
  - Crew Manifest (resume-based responses)
  - Transmission Verification (memory recall)
- **Strict Output Format** - Returns only `enter_digits` or `speak_text` JSON responses
- **Memory Store** - Tracks transmissions for later recall
- **Web UI** - Terminal-style frontend interface

## Installation

```bash
# Clone the repository
git clone https://github.com/Nithinreddyyarradla/Neon_assesment.git
cd Neon_assesment

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Option 1: Run with Python

```bash
python neon_server.py
```

### Option 2: Run with Docker

```bash
# Build and run with docker-compose
OPENAI_API_KEY=your-api-key docker-compose up --build

# Or build manually
docker build -t neon-agent .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-api-key neon-agent
```

The server will start at `http://localhost:8000`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/challenge` | POST | Process a challenge message |
| `/set-code` | POST | Set the Neon authorization code |
| `/reset` | POST | Reset agent memory |
| `/reset-full` | POST | Full reset (memory + auth) |
| `/health` | GET | Health check |
| `/memory` | GET | Get stored transmissions |

### Challenge Format

```json
{
  "type": "challenge",
  "message": [
    {"word": "Respond", "timestamp": 0},
    {"word": "on", "timestamp": 1},
    {"word": "frequency", "timestamp": 2},
    {"word": "1234", "timestamp": 3}
  ]
}
```

### Response Format

```json
{"type": "enter_digits", "digits": "1234"}
```
or
```json
{"type": "speak_text", "text": "Response text here"}
```

## Examples

### Signal Handshake
```
Input: "Respond on frequency 4567"
Output: {"type": "enter_digits", "digits": "4567"}
```

### Math with Word Numbers
```
Input: "Calculate five plus three"
Output: {"type": "enter_digits", "digits": "8"}
```

### Crew Manifest
```
Input: "name of the crew member"
Output: {"type": "speak_text", "text": "Nithin R"}
```

### With Pound Key
```
Input: "Respond on frequency 1234 followed by pound key"
Output: {"type": "enter_digits", "digits": "1234#"}
```

## Project Structure

```
├── neon_agent_openai.py   # OpenAI-powered agent
├── neon_agent.py          # Rule-based fallback agent
├── neon_server.py         # FastAPI server
├── static/
│   └── index.html         # Frontend UI
├── nithin_resume.json     # Resume data for crew manifest
├── test_neon_agent.py     # Test suite
└── requirements.txt       # Dependencies
```

## Configuration

The server automatically uses the OpenAI agent if `OPENAI_API_KEY` is set, otherwise falls back to the rule-based agent.

## Tech Stack

- **Backend**: Python, FastAPI, OpenAI API
- **Frontend**: HTML, CSS, JavaScript
- **APIs**: Wikipedia REST API for knowledge queries

## Author

Nithin Reddy Yarradla 
## License

MIT License
