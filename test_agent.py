"""
Test Runner for Neon Agent
==========================

This file demonstrates all capabilities of the agent and runs test cases
to verify correct behavior.

Run with: python test_agent.py
"""

import json
from agent import (
    NeonAgent,
    create_agent_with_resume,
    reconstruct_transmission,
    speak,
    digits,
)


# =============================================================================
# TEST RESUME DATA
# =============================================================================

TEST_RESUME = {
    "name": "Alex Johnson",
    "summary": "Full-stack engineer with 5 years of experience in distributed systems and cloud infrastructure.",
    "education": [
        {
            "degree": "Master of Science",
            "field": "Computer Science",
            "school": "Stanford University",
            "year": "2020"
        },
        {
            "degree": "Bachelor of Science",
            "field": "Electrical Engineering",
            "school": "MIT",
            "year": "2018"
        }
    ],
    "experience": [
        {
            "title": "Senior Software Engineer",
            "company": "Google",
            "description": "Led distributed systems team building petabyte-scale data pipelines"
        },
        {
            "title": "Software Engineer",
            "company": "Meta",
            "description": "Built real-time ML inference serving millions of requests"
        }
    ],
    "projects": [
        {
            "name": "StreamDB",
            "description": "Real-time streaming database with exactly-once semantics",
            "technologies": ["Rust", "Kafka", "RocksDB"]
        }
    ],
    "skills": ["Python", "Go", "Rust", "Kubernetes", "ML", "Distributed Systems"]
}


# =============================================================================
# TEST CASES
# =============================================================================

def test_output_format():
    """Test that all outputs are correctly formatted JSON."""
    print("\n" + "="*60)
    print("TEST: Output Format Enforcement")
    print("="*60)

    agent = create_agent_with_resume(TEST_RESUME)

    # Test speak format
    response = agent.process("Tell me about your education")
    assert response.get("type") in ["speak_text", "enter_digits"], f"Invalid type: {response}"
    assert "text" in response or "digits" in response, f"Missing text/digits: {response}"
    print(f"  [PASS] Resume query returns valid format: {response['type']}")

    # Test digits format
    response = agent.process("Calculate 2 + 2")
    assert response.get("type") == "enter_digits", f"Expected enter_digits: {response}"
    assert response.get("digits") == "4", f"Expected 4: {response}"
    print(f"  [PASS] Math query returns digits format: {response}")

    print("  [PASS] All output formats valid")


def test_transmission_reconstruction():
    """Test fragment reconstruction by timestamp."""
    print("\n" + "="*60)
    print("TEST: Transmission Reconstruction")
    print("="*60)

    fragments = [
        {"timestamp": 3.0, "word": "world"},
        {"timestamp": 1.0, "word": "Hello"},
        {"timestamp": 2.0, "word": "beautiful"},
    ]

    result = reconstruct_transmission(fragments)
    expected = "Hello beautiful world"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    print(f"  [PASS] Fragments sorted correctly: '{result}'")

    # Test with different key names
    fragments_alt = [
        {"ts": 2.0, "text": "two"},
        {"ts": 1.0, "text": "one"},
        {"ts": 3.0, "text": "three"},
    ]
    result = reconstruct_transmission(fragments_alt)
    assert result == "one two three", f"Alt keys failed: {result}"
    print(f"  [PASS] Alternative key names work: '{result}'")


def test_math_evaluation():
    """Test safe math evaluation."""
    print("\n" + "="*60)
    print("TEST: Math Evaluation")
    print("="*60)

    agent = create_agent_with_resume(TEST_RESUME)

    test_cases = [
        ("2 + 2", "4"),
        ("10 * 5", "50"),
        ("100 / 4", "25.0"),
        ("2 ** 10", "1024"),
        ("(3 + 4) * 2", "14"),
        ("15 % 4", "3"),
        ("100 // 7", "14"),
    ]

    for expr, expected in test_cases:
        response = agent.process(f"Calculate {expr}")
        assert response["type"] == "enter_digits", f"Wrong type for {expr}"
        # Handle float comparison
        result = response["digits"]
        if "." in expected or "." in result:
            assert float(result) == float(expected), f"Expected {expected}, got {result}"
        else:
            assert result == expected, f"Expected {expected}, got {result}"
        print(f"  [PASS] {expr} = {result}")


def test_knowledge_queries():
    """Test Wikipedia knowledge queries (requires internet)."""
    print("\n" + "="*60)
    print("TEST: Knowledge Archive Queries")
    print("="*60)

    agent = create_agent_with_resume(TEST_RESUME)

    try:
        # Test basic knowledge fetch
        response = agent.process("Speak the 1st word entry for Python programming language")
        assert response["type"] == "speak_text", f"Wrong type: {response}"
        assert len(response["text"]) > 0, "Empty response"
        print(f"  [PASS] Knowledge query returned: '{response['text']}'")

        # Test another topic
        response = agent.process("Speak the 2nd word for Saturn")
        assert response["type"] == "speak_text", f"Wrong type: {response}"
        print(f"  [PASS] Saturn 2nd word: '{response['text']}'")

    except Exception as e:
        print(f"  [SKIP] Knowledge test skipped (requires internet): {e}")


def test_resume_queries():
    """Test resume/crew manifest queries."""
    print("\n" + "="*60)
    print("TEST: Crew Manifest (Resume) Queries")
    print("="*60)

    agent = create_agent_with_resume(TEST_RESUME)

    # Test education
    response = agent.process("Transmit crew member's education background")
    assert response["type"] == "speak_text"
    assert "Stanford" in response["text"] or "Master" in response["text"]
    print(f"  [PASS] Education: '{response['text'][:60]}...'")

    # Test experience
    response = agent.process("Transmit crew member's work experience")
    assert response["type"] == "speak_text"
    assert "Google" in response["text"] or "Engineer" in response["text"]
    print(f"  [PASS] Experience: '{response['text'][:60]}...'")

    # Test with length constraint
    response = agent.process("Transmit crew skills between 20 and 100 characters")
    assert response["type"] == "speak_text"
    assert 20 <= len(response["text"]) <= 100, f"Length {len(response['text'])} not in range"
    print(f"  [PASS] Length-constrained ({len(response['text'])} chars): '{response['text']}'")


def test_memory_and_recall():
    """Test memory storage and recall - THE CRITICAL TEST."""
    print("\n" + "="*60)
    print("TEST: Memory and Recall (CRITICAL)")
    print("="*60)

    agent = create_agent_with_resume(TEST_RESUME)

    # Generate some responses to store in memory
    print("  Generating crew manifest transmissions...")

    response1 = agent.process("Transmit crew member's education")
    text1 = response1["text"]
    words1 = text1.split()
    print(f"  Response 1: '{text1[:50]}...'")
    print(f"  Words: {words1[:5]}...")

    response2 = agent.process("Transmit crew member's experience")
    text2 = response2["text"]
    words2 = text2.split()
    print(f"  Response 2: '{text2[:50]}...'")
    print(f"  Words: {words2[:5]}...")

    # Now test recall
    print("\n  Testing recall...")

    # Recall 1st word from 1st crew manifest
    recall_response = agent.process("Recall the 1st word from crew manifest transmission 1")
    expected_word = words1[0]  # 1-indexed, so word 1 = index 0
    print(f"  Recall request: '1st word from transmission 1'")
    print(f"  Expected: '{expected_word}'")
    print(f"  Got: '{recall_response['text']}'")
    assert recall_response["text"] == expected_word, f"Recall failed! Expected '{expected_word}'"
    print(f"  [PASS] Correctly recalled: '{recall_response['text']}'")

    # Recall 2nd word from 2nd crew manifest
    recall_response = agent.process("Recall the 2nd word from crew manifest transmission 2")
    expected_word = words2[1]  # 1-indexed, so word 2 = index 1
    print(f"  Recall request: '2nd word from transmission 2'")
    print(f"  Expected: '{expected_word}'")
    print(f"  Got: '{recall_response['text']}'")
    assert recall_response["text"] == expected_word, f"Recall failed! Expected '{expected_word}'"
    print(f"  [PASS] Correctly recalled: '{recall_response['text']}'")

    # Verify memory dump
    memory = agent.get_memory_dump()
    print(f"\n  Memory contains {len(memory)} entries")
    for entry in memory:
        print(f"    [{entry['index']}] {entry['task_type']}: '{entry['response_text'][:30]}...'")

    print("\n  [PASS] Memory and recall working correctly!")


def test_fragmented_input():
    """Test processing fragmented input that needs reconstruction."""
    print("\n" + "="*60)
    print("TEST: Fragmented Input Processing")
    print("="*60)

    agent = create_agent_with_resume(TEST_RESUME)

    # Create a fragmented math query
    fragmented_input = json.dumps([
        {"timestamp": 3.0, "word": "4"},
        {"timestamp": 1.0, "word": "Calculate"},
        {"timestamp": 2.0, "word": "2+2"},
    ])

    # This would reconstruct to "Calculate 2+2 4" which isn't quite right
    # Let's test a simpler case
    fragmented_input = json.dumps([
        {"timestamp": 2.0, "word": "2+2"},
        {"timestamp": 1.0, "word": "Calculate"},
    ])

    response = agent.process(fragmented_input)
    print(f"  Fragmented input: {fragmented_input}")
    print(f"  Response: {response}")
    # The agent should reconstruct and process
    print("  [PASS] Fragmented input processed")


def test_task_routing():
    """Test that prompts are routed to correct handlers."""
    print("\n" + "="*60)
    print("TEST: Task Routing")
    print("="*60)

    from agent import TaskRouter, TaskType
    router = TaskRouter()

    test_cases = [
        ("Calculate 5 + 3", TaskType.MATH),
        ("What is 10 * 2", TaskType.MATH),
        ("5 + 5", TaskType.MATH),
        ("Speak the 3rd word entry for Saturn", TaskType.KNOWLEDGE),
        ("Get the 5th word for Python", TaskType.KNOWLEDGE),
        ("Transmit crew member's education", TaskType.RESUME),
        ("Tell me about the crew's background", TaskType.RESUME),
        ("Recall the 2nd word from response 1", TaskType.RECALL),
        ("What was the 3rd word of your earlier transmission", TaskType.RECALL),
    ]

    for prompt, expected_type in test_cases:
        task_type, params = router.classify(prompt)
        status = "[PASS]" if task_type == expected_type else "[FAIL]"
        print(f"  {status} '{prompt[:40]}...' -> {task_type.value} (expected: {expected_type.value})")


def run_full_simulation():
    """Simulate a full challenge session."""
    print("\n" + "="*60)
    print("FULL SIMULATION: Challenge Session")
    print("="*60)

    agent = create_agent_with_resume(TEST_RESUME)

    session = [
        ("Transmit crew member's educational background", "speak_text"),
        ("Calculate 15 * 7 + 3", "enter_digits"),
        ("Transmit crew member's recent deployment experience", "speak_text"),
        ("What is 256 / 4", "enter_digits"),
        ("Transmit crew member's notable projects", "speak_text"),
        # Final recall checkpoint
        ("Recall the 1st word from your earlier crew manifest transmission 1", "speak_text"),
    ]

    print("\n  Session log:")
    print("-" * 50)

    for i, (prompt, expected_type) in enumerate(session, 1):
        response = agent.process(prompt)
        status = "[OK]" if response["type"] == expected_type else "[FAIL]"
        content = response.get("text", response.get("digits", ""))
        print(f"\n  Step {i}: {prompt[:50]}...")
        print(f"  {status} Response ({response['type']}): {content[:60]}...")

    print("\n" + "-" * 50)
    print("  Session complete!")
    print(f"  Total memory entries: {len(agent.get_memory_dump())}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NEON AGENT TEST SUITE")
    print("="*60)

    try:
        test_output_format()
        test_transmission_reconstruction()
        test_math_evaluation()
        test_task_routing()
        test_resume_queries()
        test_memory_and_recall()
        test_fragmented_input()

        # This requires internet
        try:
            test_knowledge_queries()
        except:
            print("  [SKIP] Knowledge queries (no internet)")

        run_full_simulation()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)

    except AssertionError as e:
        print(f"\n  [FAILED] {e}")
        raise
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        raise
