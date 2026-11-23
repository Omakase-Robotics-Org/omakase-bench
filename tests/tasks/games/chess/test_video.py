"""The chess video composer: the cut's timeline from a trace (no video is read or written)."""
import pytest

from omakase_bench.tasks.games.chess import video


def _episode():
    def model(robot, t, seconds, text, thinking):
        return {"kind": "model", "robot": robot, "sim_time": t, "seconds": seconds, "text": text, "thinking": thinking,
                "prompt": "t00 | qL ... ->"}
    trace = [
        {"kind": "turn_start", "robot": "a", "sim_time": 10.0, "turn": 1},
        {"kind": "turn_ready", "robot": "a", "sim_time": 13.0, "turn": 1},
        model("a", 13.0, 40.0, "move: e4\na: L hold | R hold", "The king's pawn opens lines. It is free to move two squares."),
        {"kind": "say", "robot": "a", "sim_time": 13.0, "keyword": "move", "text": "e4", "answer": "Accepted: e4."},
        model("a", 15.0, 5.0, "a: L hold | R 315 -23 800 45 w27", "Lowering."),
        {"kind": "turn_end", "robot": "a", "sim_time": 33.0, "turn": 1},
        {"kind": "referee", "robot": "a", "sim_time": 40.0, "turn": 1, "note": "turn 1: white played e4"},
        {"kind": "turn_start", "robot": "b", "sim_time": 45.0, "turn": 2},
        {"kind": "turn_ready", "robot": "b", "sim_time": 48.0, "turn": 2},
        model("b", 48.0, 20.0, "move: d5", ""),
        {"kind": "say", "robot": "b", "sim_time": 48.0, "keyword": "move", "text": "d5", "answer": "Accepted: d5."},
        {"kind": "turn_end", "robot": "b", "sim_time": 70.0, "turn": 2},
        {"kind": "referee", "robot": "b", "sim_time": 76.0, "turn": 2,
         "note": "turn 2: black played d5 (manipulation fault, the referee restored black pawn); hand-over between the arms"},
    ]
    result = {"robots": {
        "a": {"subject": "anthropic/claude-opus-5-5:ee", "metrics": {"colour": "white", "move_times": [
            {"san": "e4", "think_s": 45.0, "exec_s": 20.0, "clock_s": 3565.0}]}},
        "b": {"subject": "openai/gpt-6.2-astra:ee", "metrics": {"colour": "black", "move_times": [
            {"san": "d5", "think_s": 20.0, "exec_s": 22.0, "clock_s": 3588.0}]}}}}
    return trace, result


def test_model_calls_get_their_real_seconds_back_in_the_cut():
    trace, result = _episode()
    turns = video.timeline(trace, result)
    assert [(t.n, t.robot, t.san, t.ready, t.t1) for t in turns] == [(1, "a", "e4", 13.0, 33.0), (2, "b", "d5", 48.0, 70.0)]
    assert turns[0].move_call.seconds == 40.0 and (turns[0].think_s, turns[0].exec_s) == (45.0, 20.0)
    assert turns[0].clock_before == 3600.0 and turns[1].clock_after == 3588.0
    assert turns[1].tags == ["HAND-OVER", "FAULT"] and turns[0].fen_after.startswith("rnbqkbnr/pppppppp/8/8/4P3")
    segs = video.segments(turns, 0.0, 80.0)
    think = [s for s in segs if s.kind == "think"]
    assert [(s.a, s.dur) for s in think] == [(13.0, 40.0), (15.0, 5.0), (48.0, 20.0)]      # no simulated time passes
    assert sum(s.dur for s in segs if s.kind != "think") == pytest.approx(80.0)
    assert [s.kind for s in segs] == ["transfer", "transfer", "think", "exec", "think", "exec", "transfer", "transfer",
                                      "think", "exec", "transfer"]
    # review: everything at one speed; game: compressed to the length asked for, execution least
    assert video.plan(segs, "review", 0, 2.0, None) == 2.0 and sum(s.out for s in segs) == pytest.approx((80 + 65) / 2)
    k = video.plan(segs, "game", 10.0, 1.0, None)
    assert sum(s.out for s in segs) == pytest.approx(10.0) and k > video.FULL_DETAIL_SPEED
    per_real = {kind: next(s.out / s.dur for s in segs if s.kind == kind) for kind in ("exec", "transfer", "think")}
    assert per_real["exec"] > per_real["transfer"] > per_real["think"] > 0
    video.plan(segs, "highlights", 6.0, 1.0, {2})
    assert all((s.out > 0) == (s.turn.n == 2 and s.kind != "transfer") for s in segs)


def test_key_line_highlights_and_formats():
    trace, result = _episode()
    turns = video.timeline(trace, result)
    assert video.key_line(turns[0]) == "The king's pawn opens lines. It is free to move two squares."
    assert video.key_line(turns[0], 40) == "The king's pawn opens lines."
    assert video.key_line(turns[1]) == ""                           # no summary and nothing but commands in the reply
    assert video.choose_highlights(turns, 1) == {2} and video.choose_highlights(turns, 5) == {1, 2}
    assert (video.clock_text(3605), video.clock_text(59.9), video.secs(42.4), video.secs(125)) == ("1:00:05", "0:59", "42 s", "2:05")
    assert video.material("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1") == (34, 39)
    assert video.model_name("anthropic/claude-opus-5-5:ee") == "claude-opus-5-5"


def test_markdown_of_a_summary_is_parsed_not_shown():
    head, body = video.summary_parts("**Considering chess moves**\n\nI'm reviewing a **series** of moves.\n\n**Next**\n\nThen Be6.")
    assert head == "Considering chess moves" and "*" not in body
    assert [ln for ln in body.splitlines() if ln] == ["I'm reviewing a series of moves.", "Next:", "Then Be6."]
    assert video.summary_parts("No header here.") == ("", "No header here.") and video.summary_parts("") == ("", "")
    trace, result = _episode()
    trace[2]["thinking"] = "**Opening choice**\n\nThe king's pawn opens lines."
    assert video.key_line(video.timeline(trace, result)[0]) == "The king's pawn opens lines."


def test_highlights_prefer_eval_swings_and_queens_moved_into_attack():
    def turn(n, robot, t, san, note_side):
        return [{"kind": "turn_start", "robot": robot, "sim_time": t, "turn": n},
                {"kind": "say", "robot": robot, "sim_time": t + 1, "keyword": "move", "text": san, "answer": f"Accepted: {san}."},
                {"kind": "turn_end", "robot": robot, "sim_time": t + 5, "turn": n},
                {"kind": "referee", "robot": robot, "sim_time": t + 6, "turn": n, "note": f"turn {n}: {note_side} played {san}"}]
    line = ["e4", "e5", "Qh5", "Nc6", "Bc4", "g6", "Qf3", "Nf6", "Qxf6", "Qxf6", "d3", "d6", "a3", "a6"]
    trace = [e for k, san in enumerate(line) for e in turn(k + 1, "ab"[k % 2], 10.0 * k, san, ("white", "black")[k % 2])]
    result = {"robots": {"a": {"subject": "x/white:ee", "metrics": {"colour": "white"}},
                         "b": {"subject": "x/black:ee", "metrics": {"colour": "black"}}}}
    plain = video.timeline(trace, result)
    offers = [t.san for t in plain if t.queen_offer]
    assert offers == ["Qxf6"]                              # the queen takes on a square Black's queen attacks
    assert "QUEEN OFFER" in plain[8].tags and "EVAL SWING" not in plain[8].tags
    assert not plain[2].queen_offer and not plain[6].queen_offer          # Qh5 and Qf3 stand on unattacked squares
    cpl = {"moves": [{"move": "3. Bc4", "loss": 30}, {"move": "5. Qxf6", "loss": 800}],
           "black": {"worst": [{"move": "6... d6", "loss": 250}]}}
    swung = video.timeline(trace, result, cpl)
    assert [(t.san, t.swing) for t in swung if t.swing] == [("Bc4", 30), ("Qxf6", 800), ("d6", 250)]
    assert "EVAL SWING" in swung[8].tags and "EVAL SWING" in swung[11].tags and "EVAL SWING" not in swung[4].tags
    # with room for four turns: the last two, then the swings; without the engine's figures a quiet d6 is not chosen
    assert video.choose_highlights(swung, 4) == {9, 12, 13, 14}
    assert 12 not in video.choose_highlights(plain, 4)
