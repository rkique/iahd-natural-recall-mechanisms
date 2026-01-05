#!/usr/bin/env python3
"""
Test script for LLMContextNeedleTester
"""

from retrieval_head_detection import LLMContextNeedleTester


def test_small_sample():
    """Test with a small sample of 5 contexts."""
    tester = LLMContextNeedleTester()
    results = tester.run_evaluation(max_rows=5)
    tester.save_results(results)
    
    return results


if __name__ == "__main__":
    results = test_small_sample()
    print("Sample results:")
    for r in results:
        print(f"Index {r['index']}: State={r['inc_state']}, First Token={r['first_token']}, Match={r['match']}")