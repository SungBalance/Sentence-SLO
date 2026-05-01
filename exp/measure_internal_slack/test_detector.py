import sys
sys.path.insert(0, '/workspace/mlsys/vllm')
from vllm.sslo.slo_state import SentenceChunkDetector, RequestSLOState

det = SentenceChunkDetector()

assert det.find_boundary("Hello world.") == 12
assert det.find_boundary("Hello world. Next sentence") == 12  # boundary after ".", space carried forward
assert det.find_boundary("Hello world") is None
assert det.find_boundary("3.14 is pi") is None
assert det.find_boundary("It costs $3.14") is None
assert det.find_boundary("Wait... then what?") == 7
assert det.find_boundary("Good? Yes!") == 5

# Multi-sentence: should return first boundary only
text = "First sentence. Second sentence. Third."
pos = det.find_boundary(text)
assert pos == 15, f"Expected 15, got {pos}: {repr(text[:pos])}"

# End-of-text boundary
assert det.find_boundary("Hello.") == 6

# Multiple chunks via RequestSLOState
state = RequestSLOState()
state.on_text_delta("Hello world. ", 0.0)
state.on_text_delta("Second sentence. ", 1.0)
state.on_finish(2.0)
records = state.chunk_records
print(f"Chunks: {len(records)}")
for r in records:
    print(f"  idx={r['chunk_idx']} wc={r['word_count']} text={repr(r['text'])}")
assert len(records) == 3, f"Expected 3 chunks, got {len(records)}"
assert records[0]["word_count"] == 2, f"chunk0 wc={records[0]['word_count']}"
assert records[1]["word_count"] == 2, f"chunk1 wc={records[1]['word_count']}"

# Two sentences in one delta — while loop must flush both
state2 = RequestSLOState()
state2.on_text_delta("Sentence one. Sentence two. Incomplete", 0.0)
state2.on_finish(1.0)
r2 = state2.chunk_records
print(f"Two-in-one delta chunks: {len(r2)}")
for r in r2:
    print(f"  idx={r['chunk_idx']} wc={r['word_count']} text={repr(r['text'])}")
assert len(r2) == 3, f"Expected 3, got {len(r2)}"

print("All assertions passed.")
