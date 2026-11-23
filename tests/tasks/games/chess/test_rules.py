"""The chess rules of games/chess: perft on the standard positions, recorded games, SAN, FEN, PGN, endings."""

import pytest

from omakase_bench.tasks.games.chess.rules import (STARTING_FEN, AmbiguousMoveError, Board, IllegalMoveError,
                                             InvalidMoveError, perft, read_pgn, to_pgn)

# Perft results from the chessprogramming wiki "Perft Results" page (positions 1-6)
PERFT = [
    ("start", STARTING_FEN, [20, 400, 8902, 197281, 4865609]),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862]),
    ("position 3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("position 4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("position 4 mirrored", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", [6, 264, 9467]),
    ("position 5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
    ("position 6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079, 89890]),
]


@pytest.mark.parametrize("name,fen,counts", PERFT, ids=[p[0] for p in PERFT])
def test_perft(name, fen, counts):
    board = Board(fen)
    assert [perft(board, d + 1) for d in range(len(counts))] == counts
    assert board.fen() == Board(fen).fen()          # make and unmake leave the position as it was


GAMES = {  # recorded games, SAN as published
"opera": "1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0",
"immortal": "1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 6. Nf3 Qh6 7. d3 Nh5 8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 13. h5 Qg5 14. Qf3 Ng8 15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2 Na6 21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# 1-0",
"evergreen": "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O d3 8. Qb3 Qf6 9. e5 Qg6 10. Re1 Nge7 11. Ba3 b5 12. Qxb5 Rb8 13. Qa4 Bb6 14. Nbd2 Bb7 15. Ne4 Qf5 16. Bxd3 Qh5 17. Nf6+ gxf6 18. exf6 Rg8 19. Rad1 Qxf3 20. Rxe7+ Nxe7 21. Qxd7+ Kxd7 22. Bf5+ Ke8 23. Bd7+ Kf8 24. Bxe7# 1-0",
"century": "1. Nf3 Nf6 2. c4 g6 3. Nc3 Bg7 4. d4 O-O 5. Bf4 d5 6. Qb3 dxc4 7. Qxc4 c6 8. e4 Nbd7 9. Rd1 Nb6 10. Qc5 Bg4 11. Bg5 Na4 12. Qa3 Nxc3 13. bxc3 Nxe4 14. Bxe7 Qb6 15. Bc4 Nxc3 16. Bc5 Rfe8+ 17. Kf1 Be6 18. Bxb6 Bxc4+ 19. Kg1 Ne2+ 20. Kf1 Nxd4+ 21. Kg1 Ne2+ 22. Kf1 Nc3+ 23. Kg1 axb6 24. Qb4 Ra4 25. Qxb6 Nxd1 26. h3 Rxa2 27. Kh2 Nxf2 28. Re1 Rxe1 29. Qd8+ Bf8 30. Nxe1 Bd5 31. Nf3 Ne4 32. Qb8 b5 33. h4 h5 34. Ne5 Kg7 35. Kg1 Bc5+ 36. Kf1 Ng3+ 37. Ke1 Bb4+ 38. Kd1 Bb3+ 39. Kc1 Ne2+ 40. Kb1 Nc3+ 41. Kc1 Rc2# 0-1",
}

FINAL = {
    "opera": ("1n1Rkb1r/p4ppp/4q3/4p1B1/4P3/8/PPP2PPP/2K5 b k - 1 17", 33, "1-0"),
    "immortal": ("r1bk3r/p2pBpNp/n4n2/1p1NP2P/6P1/3P4/P1P1K3/q5b1 b - - 1 23", 45, "1-0"),
    "evergreen": ("1r3kr1/pbpBBp1p/1b3P2/8/8/2P2q2/P4PPP/3R2K1 b - - 0 24", 47, "1-0"),
    "century": ("1Q6/5pk1/2p3p1/1p2N2p/1b5P/1bn5/2r3P1/2K5 w - - 16 42", 82, "0-1"),
}


@pytest.mark.parametrize("name", sorted(GAMES))
def test_recorded_games_replay_to_their_final_position(name):
    headers, board = read_pgn(GAMES[name])
    fen, plies, result = FINAL[name]
    assert board.fen() == fen and len(board.move_stack) == plies
    out = board.outcome()
    assert out.termination == "checkmate" and out.result() == result
    # SAN out, SAN in: the exported PGN replays to the same position
    _, again = read_pgn(to_pgn(board, {"White": "w", "Black": "b"}, result))
    assert again.fen() == fen


def test_special_moves_and_their_san():
    b = Board()
    for san in "e4 Nf6 e5 d5".split():
        b.push_san(san)
    assert b.fen() == "rnbqkb1r/ppp1pppp/5n2/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    ep = b.parse_san("exd6")
    assert b.is_en_passant(ep) and b.is_capture(ep) and b.san(ep) == "exd6"
    b.push(ep)
    assert b.piece_at(35) is None                                   # the pawn on d5 is gone
    for san in "Bf5 dxc7 e6".split():
        b.push_san(san)
    promo = b.parse_san("cxb8=Q")
    assert promo.promotion == 5 and b.san(promo) == "cxb8=Q" and b.parse_san("cxb8Q") == promo
    assert b.parse_uci("c7b8n").promotion == 2
    b = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert b.san(b.parse_san("O-O")) == "O-O" and b.parse_san("0-0-0").to_square == 2
    b = Board("r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1")     # f1 and the king's path are attacked
    with pytest.raises(IllegalMoveError):
        b.parse_san("O-O")


def test_san_errors_disambiguation_and_check_marks():
    b = Board("4k3/R7/8/8/8/8/8/R3K3 w - - 0 1")
    with pytest.raises(AmbiguousMoveError):
        b.parse_san("Ra4")
    assert b.san(b.parse_san("R1a4")) == "R1a4" and b.san(b.parse_san("R7a4")) == "R7a4"
    with pytest.raises(IllegalMoveError):
        b.parse_san("Rb3")
    with pytest.raises(InvalidMoveError):
        b.parse_san("hello")
    assert b.san(b.parse_san("Ra8")) == "Ra8+"
    b = Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert b.san(b.parse_san("Ra8")) == "Ra8#"


def test_draws():
    b = Board()
    for san in "Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1".split():
        b.push_san(san)
    assert b.outcome(claim_draw=True) is None
    b.push_san("Ng8")                                               # the start position a third time
    assert b.outcome(claim_draw=True).termination == "threefold_repetition" and b.outcome() is None
    assert Board("8/8/8/4k3/8/8/2K5/8 w - - 0 1").outcome().termination == "insufficient_material"
    assert Board("8/8/8/4k3/8/8/2KN4/8 w - - 0 1").is_insufficient_material()
    assert not Board("8/8/8/4k3/8/8/2KNN3/8 w - - 0 1").is_insufficient_material()
    assert Board("8/8/2b5/4k3/8/8/2K1B3/8 w - - 0 1").is_insufficient_material()     # bishops on one colour
    assert not Board("8/8/3b4/4k3/8/8/2K1B3/8 w - - 0 1").is_insufficient_material()
    assert Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").outcome().termination == "stalemate"
    b = Board("8/8/8/4k3/8/8/2K5/R7 w - - 99 80")
    assert b.outcome(claim_draw=True) is None
    b.push_san("Ra2")
    assert b.outcome(claim_draw=True).termination == "fifty_moves" and b.outcome() is None


def test_fen_round_trip_and_errors():
    for _, fen, _ in PERFT:
        assert Board(fen).fen() == fen
    with pytest.raises(ValueError):
        Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1")
    b = Board()
    b.push_san("e4")
    assert b.fen() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"   # no legal capture: no e3


def test_cpl_tool_with_a_stand_in_uci_engine(tmp_path):
    """``omakase-bench chess cpl`` drives any UCI engine; a stand-in that scores every position 0 gives 0 loss."""
    import sys
    fake = tmp_path / "engine.py"
    fake.write_text("import sys\nfor line in sys.stdin:\n    w = line.split()\n    if not w: continue\n"
                    "    if w[0] == 'uci': print('uciok', flush=True)\n"
                    "    elif w[0] == 'go': print('info depth 1 score cp 0', flush=True); print('bestmove a1a1', flush=True)\n"
                    "    elif w[0] == 'quit': break\n")
    runner = tmp_path / "engine"
    runner.write_text(f"#!/bin/sh\nexec {sys.executable} {fake}\n")
    runner.chmod(0o755)
    from omakase_bench.tasks.games.chess import cpl as mod
    _, board = read_pgn("1. e4 e5 2. Nf3 Nc6 *")
    eng = mod.Engine(str(runner))
    try:
        assert mod.acpl(board, eng, 1) == {"white": {"moves": 2, "acpl": 0.0}, "black": {"moves": 2, "acpl": 0.0}}
        report = mod.analyse(board, eng, 1)
        assert report["white"] == {"moves": 2, "acpl": 0.0, "inaccuracies": 0, "mistakes": 0, "blunders": 0, "worst": []}
        assert [m["move"] for m in report["moves"]] == ["1. e4", "1... e5", "2. Nf3", "2... Nc6"]
        # a stand-in that scores +300 for whoever is to move: every move loses 600 by its figures
        eng.evaluate = lambda fen, depth: (300, "a2a3" if " w " in fen else "a7a6")
        report = mod.analyse(board, eng, 1, worst=1)
        assert report["black"]["blunders"] == 2 and report["black"]["acpl"] == 600.0
        assert report["white"]["worst"] == [{"side": "white", "move": "1. e4", "eval_before": 300, "eval_after": -300,
                                             "loss": 600, "engine_move": "a3"}]
    finally:
        eng.close()


def test_a_pawn_move_without_a_file_is_a_push_never_a_capture():
    b = Board("4k3/8/8/8/4p3/3P4/8/4K3 w - - 0 1")          # d3 can take on e4; no pawn can push there
    with pytest.raises(IllegalMoveError):
        b.parse_san("e4")
    assert b.parse_san("dxe4").uci() == "d3e4" and b.parse_san("de4").uci() == "d3e4"
    ep = Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    with pytest.raises(IllegalMoveError):
        ep.parse_san("d6")
    assert ep.parse_san("exd6").uci() == "e5d6" and ep.parse_san("e6").uci() == "e5e6"


def test_rule_corners_that_decide_games():
    # en passant that would uncover a rook's check along the rank is not legal, and is not in the FEN
    pin = Board("8/8/8/8/k2Pp2R/8/8/3K4 b - d3 0 1")
    assert "e4d3" not in {m.uci() for m in pin.legal_moves} and pin.fen().split()[3] == "-"
    # no castling out of, through or into check; a rook passing an attacked square is fine
    assert {m.uci() for m in Board("r3k2r/8/8/8/4R3/8/8/4K3 b kq - 0 1").legal_moves} & {"e8g8", "e8c8"} == set()
    through = {m.uci() for m in Board("r3k2r/8/8/8/5R2/8/8/4K3 b kq - 0 1").legal_moves}
    assert "e8g8" not in through and "e8c8" in through
    assert "e8c8" in {m.uci() for m in Board("r3k2r/8/8/8/1R6/8/8/4K3 b kq - 0 1").legal_moves}
    # capturing a rook on its home square removes that castling right
    b = Board("r3k2r/8/8/8/8/8/6B1/4K3 w kq - 0 1")
    b.push_san("Bxa8")
    assert b.castling == "k"
    # under-promotion, promotion with capture and with mate
    p = Board("1n5k/P7/8/8/8/8/8/K7 w - - 0 1")
    assert {m.uci() for m in p.legal_moves if m.from_square == 48} == {
        f"a7{to}{k}" for to in ("a8", "b8") for k in "qrbn"}
    assert Board("6k1/5ppp/8/8/8/8/P7/K7 w - - 0 1").outcome() is None
    assert p.san(p.parse_san("axb8=N")) == "axb8=N"
    q = Board("k7/2P5/1K6/8/8/8/8/8 w - - 0 1")
    assert q.san(q.parse_san("c8Q")) == "c8=Q#" and q.san(q.parse_san("c8=R")) == "c8=R#"
    # insufficient material: K v K, one minor piece, bishops of one colour; not two knights or opposite bishops
    for fen, dead in (("8/8/8/8/8/8/8/K6k w - - 0 1", True), ("8/8/8/8/8/8/8/KN5k w - - 0 1", True),
                      ("8/8/8/8/8/8/8/KB5k w - - 0 1", True), ("8/8/8/8/8/8/8/KB3b1k w - - 0 1", True),
                      ("8/8/8/8/8/8/8/KB4bk w - - 0 1", False), ("8/8/8/8/8/8/8/KNN4k w - - 0 1", False),
                      ("8/8/8/8/8/8/P7/K6k w - - 0 1", False)):
        assert Board(fen).is_insufficient_material() is dead, fen
    # the fifty-move rule counts from the last capture or pawn move, and mate on the hundredth half-move wins
    assert Board("7k/8/8/8/8/8/8/KR6 w - - 99 80").outcome(claim_draw=True) is None
    assert Board("7k/8/8/8/8/8/8/KR6 w - - 100 80").outcome(claim_draw=True).termination == "fifty_moves"
    mate = Board("7k/R7/6K1/8/8/8/8/8 w - - 99 80")
    mate.push_san("Ra8#")
    assert mate.outcome(claim_draw=True).termination == "checkmate" and mate.outcome(claim_draw=True).winner is True
    # threefold: the third occurrence of the position, same side to move and rights, ends the game
    r = Board()
    for k, san in enumerate("Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8".split()):
        assert r.outcome(claim_draw=True) is None, k
        r.push_san(san)
    assert r.outcome(claim_draw=True).termination == "threefold_repetition" and r.outcome() is None
    # a lost castling right makes the position a different one
    k = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    for san in "Kd1 Kd8 Ke1 Ke8 Kd1 Kd8 Ke1 Ke8".split():
        k.push_san(san)
    assert k._repetitions() == 2
