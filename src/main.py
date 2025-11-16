# File: src/main.py
from __future__ import annotations

import os, time, math, random
from typing import Dict, List, Optional, Tuple

import chess
from chess import Move

from .utils import chess_manager, GameContext  # ChessHacks runtime

# ---------- env ----------
MAX_DEPTH = int(os.getenv("ENGINE_DEPTH", "1"))      # keep fast & predictable
TIME_MS   = int(os.getenv("ENGINE_TIME_MS", "150"))  # strict per-move budget (ms)
TEMP      = float(os.getenv("ENGINE_TEMP", "1.0"))   # softmax temperature

# ---------- material tables ----------
MAT_P    = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}
PIECE_CP = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:0}
CENTER   = (chess.D4, chess.E4, chess.D5, chess.E5)
WHITE_MINOR_START = {chess.B1, chess.G1, chess.C1, chess.F1}
BLACK_MINOR_START = {chess.B8, chess.G8, chess.C8, chess.F8}

# ---------- helpers (ALWAYS use board copies that you can push/pop) ----------
def _least_attacker_value_cp(b: chess.Board, color: chess.Color, square: chess.Square) -> Optional[int]:
    attackers = list(b.attackers(color, square))
    if not attackers:
        return None
    best = None
    for sq in attackers:
        p = b.piece_at(sq)
        if p:
            v = PIECE_CP[p.piece_type]
            best = v if best is None else min(best, v)
    return best

def _is_hanging_after(pre: chess.Board, m: Move) -> Tuple[bool, int]:
    """Check on a COPY that we are allowed to mutate."""
    pre.push(m)
    try:
        to_sq = m.to_square
        moved = pre.piece_at(to_sq)
        if moved is None:
            return (False, 0)
        mover = not pre.turn        # mover is the side that just moved
        opp   = pre.turn
        attacked = pre.is_attacked_by(opp, to_sq)
        defended = pre.is_attacked_by(mover, to_sq)
        if attacked and not defended:
            return (True, PIECE_CP[moved.piece_type] + 120)  # strong penalty
        lav = _least_attacker_value_cp(pre, opp, to_sq)
        if lav is not None and lav + 50 < PIECE_CP[moved.piece_type]:
            return (True, PIECE_CP[moved.piece_type] - lav)
        return (False, 0)
    finally:
        pre.pop()

def _center_control(after: chess.Board, mover_white: bool) -> int:
    color = chess.WHITE if mover_white else chess.BLACK
    return sum(1 for sq in CENTER if after.is_attacked_by(color, sq))

def _capture_score(pre: chess.Board, m: Move) -> int:
    """MVV-LVA minus suicidal penalty. Uses a COPY `pre` (will push/pop inside helpers)."""
    if not pre.is_capture(m):
        return 0
    attacker = pre.piece_at(m.from_square)
    if attacker is None:
        return 0
    if pre.is_en_passant(m):
        victim_type = chess.PAWN
    else:
        v_piece = pre.piece_at(m.to_square)
        victim_type = v_piece.piece_type if v_piece else None
    if victim_type is None:
        return 0
    v = PIECE_CP[victim_type]
    a = PIECE_CP[attacker.piece_type]
    base = (10 * v - a)
    hang, pen = _is_hanging_after(pre, m)  # safe (pre is a copy)
    if hang:
        base -= pen
    return base

def _positional(pre: chess.Board, m: Move, after: chess.Board) -> int:
    """Uses pre (copy) + after (copy)."""
    s = 0
    opening = pre.fullmove_number <= 12
    piece = pre.piece_at(m.from_square)

    if piece:
        if opening and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            starts = WHITE_MINOR_START if piece.color == chess.WHITE else BLACK_MINOR_START
            if m.from_square in starts:
                s += 40
        if opening and piece.piece_type in (chess.ROOK, chess.QUEEN):
            s -= 35
        if pre.is_castling(m):
            s += 90
        if piece.piece_type == chess.PAWN and m.to_square in CENTER:
            s += 28

    if after.is_check():
        s += 45

    mover_white = not after.turn
    s += 12 * _center_control(after, mover_white)

    # mobility pressure (mild)
    s -= 1 * sum(1 for _ in after.legal_moves)

    # avoid undefended landings
    hang, pen = _is_hanging_after(pre, m)  # safe (pre is a copy)
    if hang:
        s -= pen

    # deterministic (no random jitter)
    return s

# ---------- move scoring (NEVER mutate the original board) ----------
def _score_move(orig: chess.Board, m: Move) -> Tuple[int, bool]:
    # copies for each push/pop path:
    pre_for_cap   = orig.copy(stack=False)
    pre_for_safe  = orig.copy(stack=False)
    pre_for_pos   = orig.copy(stack=False)
    after         = orig.copy(stack=False); after.push(m)

    cap = _capture_score(pre_for_cap, m)
    pos = _positional(pre_for_pos, m, after)
    hanging, _ = _is_hanging_after(pre_for_safe, m)

    # checkmate bonus (after is already pushed)
    if after.is_checkmate():
        return (100_000, True)

    total = cap + pos
    return (total, not hanging)

# ---------- selection ----------
def _softmax(scores: Dict[Move, int], temp: float = TEMP) -> Dict[Move, float]:
    if not scores:
        return {}
    t = max(temp, 1e-3)
    mx = max(scores.values())
    exps = {m: math.exp((s - mx) / (150.0 * t)) for m, s in scores.items()}
    z = sum(exps.values())
    return {m: (v / z if z > 0 else 1.0 / len(exps)) for m, v in exps.items()}

def pick_move(board: chess.Board, time_ms: int) -> Tuple[Move, Dict[Move, float]]:
    start = time.time()
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves")

    # sanity: all moves belong to current side
    turn = board.turn
    legal = [m for m in legal if board.piece_at(m.from_square) and board.piece_at(m.from_square).color == turn]
    if not legal:
        # fallback: if something odd, rebuild legal
        legal = list(board.legal_moves)

    scored: List[Tuple[int, Move, bool]] = []
    for m in legal:
        sc, safe = _score_move(board, m)
        scored.append((sc, m, safe))
        if (time.time() - start) * 1000.0 >= time_ms:  # strict budget
            break

    # prefer safe first
    safe_pool = [t for t in scored if t[2]]
    pool = safe_pool if safe_pool else scored
    best_sc, best_mv, _ = max(pool, key=lambda t: t[0])

    # extra guard: must be legal for current side
    if best_mv not in board.legal_moves:
        best_mv = list(board.legal_moves)[0]

    probs = _softmax({m: s for s, m, _ in pool})
    return best_mv, probs

# ---------- entrypoints ----------
@chess_manager.entrypoint
def test_func(ctx: GameContext):
    try:
        b = ctx.board
        if b.is_game_over(claim_draw=True):
            ctx.logProbabilities({})
            raise ValueError("Game over")

        mv, probs = pick_move(b, TIME_MS)

        # Try native Move keys, fallback to UCI if platform requires
        try:
            ctx.logProbabilities(probs)
        except Exception:
            ctx.logProbabilities({m.uci(): p for m, p in probs.items()})

        return mv
    except Exception as e:
        # Never 500 — always return a legal move
        print(f"[engine] exception: {e!r}; fallback to first legal")
        legals = list(ctx.board.legal_moves)
        if legals:
            fb = legals[0]
            try:
                ctx.logProbabilities({fb: 1.0})
            except Exception:
                ctx.logProbabilities({fb.uci(): 1.0})
            return fb
        ctx.logProbabilities({})
        return Move.null()

@chess_manager.reset
def reset_func(ctx: GameContext):
    # stateless; nothing persistent to clear
    return