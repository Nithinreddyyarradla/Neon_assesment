"""
Test Suite for NEON Agent
=========================

Tests all checkpoint types with exact challenge format.
"""

import json
from neon_agent import NeonAgent, reconstruct_message


def create_challenge(prompt: str) -> dict:
    """Create a challenge from a prompt string."""
    words = prompt.split()
    fragments = [{"word": w, "timestamp": i} for i, w in enumerate(words)]
    return {"type": "challenge", "message": fragments}


def test_fragment_reconstruction():
    """Test fragment sorting and joining."""
    print("\n" + "=" * 60)
    print("TEST: Fragment Reconstruction")
    print("=" * 60)

    # Out of order fragments
    fragments = [
        {"word": "world", "timestamp": 2},
        {"word": "Hello", "timestamp": 0},
        {"word": "beautiful", "timestamp": 1}
    ]

    result = reconstruct_message(fragments)
    expected = "Hello beautiful world"

    print(f"Fragments: {fragments}")
    print(f"Result:    {result}")
    print(f"Expected:  {expected}")
    print(f"PASS: {result == expected}")
    return result == expected


def test_signal_handshake():
    """Test signal handshake checkpoint."""
    print("\n" + "=" * 60)
    print("TEST: Signal Handshake")
    print("=" * 60)

    agent = NeonAgent()

    # Test basic frequency
    challenge = create_challenge("Respond on frequency 1234")
    response = agent.process(challenge)

    print(f"Prompt:   'Respond on frequency 1234'")
    print(f"Response: {response}")

    assert response["type"] == "enter_digits"
    assert response["digits"] == "1234"
    print("PASS")

    # Test with pound key
    challenge2 = create_challenge("Respond on frequency 5678 followed by the pound key")
    response2 = agent.process(challenge2)

    print(f"\nPrompt:   'Respond on frequency 5678 followed by the pound key'")
    print(f"Response: {response2}")

    assert response2["type"] == "enter_digits"
    assert response2["digits"] == "5678#"
    print("PASS")

    return True


def test_vessel_id():
    """Test vessel identification checkpoint."""
    print("\n" + "=" * 60)
    print("TEST: Vessel Identification")
    print("=" * 60)

    agent = NeonAgent()
    agent.set_neon_code("SECRET123")

    challenge = create_challenge("Enter your vessel authorization code")
    response = agent.process(challenge)

    print(f"Neon Code: SECRET123")
    print(f"Prompt:    'Enter your vessel authorization code'")
    print(f"Response:  {response}")

    assert response["type"] == "enter_digits"
    assert response["digits"] == "SECRET123"
    print("PASS")
    return True


def test_math():
    """Test computational assessment checkpoint."""
    print("\n" + "=" * 60)
    print("TEST: Computational Assessment")
    print("=" * 60)

    agent = NeonAgent()

    test_cases = [
        ("Calculate 2 + 3", "5"),
        ("What's 10 * 5", "50"),
        ("Calculate Math.floor(7.8)", "7"),
        ("Calculate Math.ceil(3.2)", "4"),
        ("Calculate 10 % 3", "1"),
        ("Calculate (5 + 3) * 2", "16"),
    ]

    all_pass = True
    for prompt, expected in test_cases:
        challenge = create_challenge(prompt)
        response = agent.process(challenge)

        status = "PASS" if response["digits"] == expected else "FAIL"
        print(f"Prompt: '{prompt}' -> {response['digits']} (expected: {expected}) [{status}]")

        if response["digits"] != expected:
            all_pass = False

    return all_pass


def test_knowledge():
    """Test knowledge archive query checkpoint."""
    print("\n" + "=" * 60)
    print("TEST: Knowledge Archive Query")
    print("=" * 60)

    agent = NeonAgent()

    # Test getting first word of a Wikipedia article
    challenge = create_challenge("Speak the 1st word of the knowledge archive entry for Python programming language")
    response = agent.process(challenge)

    print(f"Prompt:   'Speak the 1st word of the knowledge archive entry for Python programming language'")
    print(f"Response: {response}")

    assert response["type"] == "speak_text"
    assert len(response["text"]) > 0
    print(f"First word: '{response['text']}'")
    print("PASS")
    return True


def test_crew_manifest():
    """Test crew manifest transmission checkpoint."""
    print("\n" + "=" * 60)
    print("TEST: Crew Manifest Transmission")
    print("=" * 60)

    agent = NeonAgent()

    # Load resume
    import os
    resume_path = os.path.join(os.path.dirname(__file__), "nithin_resume.json")
    if os.path.exists(resume_path):
        agent.load_resume(resume_path)
    else:
        agent.load_resume_dict({
            "name": "Test User",
            "education": [{"degree": "Masters", "field": "CS", "school": "MIT", "year": "2024"}],
            "experience": [{"title": "Engineer", "company": "Tech Corp", "description": "Built systems"}],
            "skills": ["Python", "AI"],
            "projects": [{"name": "Project A", "description": "A cool project"}],
            "summary": "Software engineer"
        })

    # Test education
    challenge = create_challenge("Transmit crew member education background between 50 and 200 characters")
    response = agent.process(challenge)

    print(f"Prompt:   'Transmit crew member education background between 50 and 200 characters'")
    print(f"Response: {response}")
    print(f"Length:   {len(response['text'])} chars")

    assert response["type"] == "speak_text"
    assert len(response["text"]) > 0
    print("PASS")

    # Verify it was stored in memory
    assert len(agent.memory.transmissions) == 1
    print(f"Stored in memory: transmission 1 = '{response['text'][:50]}...'")

    return True


def test_verification():
    """Test transmission verification checkpoint."""
    print("\n" + "=" * 60)
    print("TEST: Transmission Verification (Recall)")
    print("=" * 60)

    agent = NeonAgent()

    # Load resume
    agent.load_resume_dict({
        "name": "Test User",
        "education": [{"degree": "Masters", "field": "CS", "school": "MIT", "year": "2024"}],
        "experience": [{"title": "Engineer", "company": "Tech Corp", "description": "Built amazing systems"}],
        "skills": ["Python", "AI"],
        "projects": [{"name": "Project A", "description": "A cool project"}],
        "summary": "Software engineer"
    })

    # First, create a crew manifest transmission
    manifest_challenge = create_challenge("Transmit crew member experience")
    manifest_response = agent.process(manifest_challenge)

    print(f"Created transmission 1: '{manifest_response['text']}'")
    words = manifest_response["text"].split()
    print(f"Words: {words[:10]}...")

    # Now test recall
    recall_challenge = create_challenge("Recall the 3rd word of the earlier crew manifest transmission 1")
    recall_response = agent.process(recall_challenge)

    expected_word = words[2]  # 3rd word (0-indexed as 2)

    print(f"\nRecall prompt: 'Recall the 3rd word of the earlier crew manifest transmission 1'")
    print(f"Response:      {recall_response}")
    print(f"Expected:      '{expected_word}'")

    assert recall_response["type"] == "speak_text"
    assert recall_response["text"] == expected_word
    print("PASS")
    return True


def test_full_sequence():
    """Test a full challenge sequence."""
    print("\n" + "=" * 60)
    print("TEST: Full Challenge Sequence")
    print("=" * 60)

    agent = NeonAgent()
    agent.set_neon_code("NEON42")

    # Load resume
    agent.load_resume_dict({
        "name": "Nithin Reddy",
        "education": [{"degree": "Masters", "field": "Computer Science", "school": "CMU", "year": "2024"}],
        "experience": [{"title": "SDE", "company": "Amazon", "description": "Built vector search systems"}],
        "skills": ["Python", "ML", "AWS"],
        "projects": [{"name": "RAG Agent", "description": "Built RAG chatbot"}],
        "summary": "ML Engineer at Amazon"
    })

    # Simulate full sequence
    sequence = [
        ("Signal Handshake", "Respond on frequency 7890"),
        ("Vessel ID", "Enter your Neon authorization code"),
        ("Math", "Calculate Math.floor(15.7) + 3"),
        ("Crew Manifest", "Transmit crew member experience"),
        ("Verification", "Recall the 1st word of the earlier crew manifest transmission 1"),
    ]

    print("\nRunning challenge sequence:\n")

    all_pass = True
    for name, prompt in sequence:
        challenge = create_challenge(prompt)
        response = agent.process(challenge)
        print(f"[{name}]")
        print(f"  Prompt:   {prompt}")
        print(f"  Response: {json.dumps(response)}")
        print()

    print("Sequence complete!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("NEON AGENT TEST SUITE")
    print("=" * 60)

    tests = [
        ("Fragment Reconstruction", test_fragment_reconstruction),
        ("Signal Handshake", test_signal_handshake),
        ("Vessel ID", test_vessel_id),
        ("Math", test_math),
        ("Knowledge Archive", test_knowledge),
        ("Crew Manifest", test_crew_manifest),
        ("Verification", test_verification),
        ("Full Sequence", test_full_sequence),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed_count}/{len(results)} tests passed")


if __name__ == "__main__":
    run_all_tests()
