# File: src/main.py
from __future__ import annotations

# --- stdlib ---
import os
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- third-party (runtime) ---
import numpy as np
import torch
import torch.nn as nn
import chess
from chess import Move

# --- chesshacks runtime ---
from .utils import chess_manager, GameContext  # provided by ChessHacks

# ========= config (env) =========
# Why: allow quick tuning without code edits.
_ENV_MODEL = os.getenv("CHESS_EVAL_MODEL", "")  # absolute or relative path to .pt
_ENV_DEVICE = os.getenv("CHESS_EVAL_DEVICE", "cpu")  # auto|cpu|mps|cuda
_MAX_DEPTH = int(os.getenv("ENGINE_DEPTH", "3"))
_TIME_MS = int(os.getenv("ENGINE_TIME_MS", "150"))
_TEMP = float(os.getenv("ENGINE_TEMP", "1.0"))

# ========= features =========
# 12x64 bitboards + side + castling (4) + en passant (64) = 837
def fen_to_features(fen: str) -> np.ndarray:
    b = chess.Board(fen)
    planes: List[np.ndarray] = []
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            m = np.zeros(64, dtype=np.float32)
            for sq in b.pieces(pt, color):
                m[sq] = 1.0
            planes.append(m)
    side = np.array([1.0 if b.turn == chess.WHITE else 0.0], dtype=np.float32)
    cast = np.array([
        1.0 if b.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if b.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if b.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if b.has_queenside_castling_rights(chess.BLACK) else 0.0,
    ], dtype=np.float32)
    ep = np.zeros(64, dtype=np.float32)
    if b.ep_square is not None:
        ep[b.ep_square] = 1.0
    return np.concatenate([*planes, side, cast, ep], dtype=np.float32)

# ========= model (pure Torch) =========
class EvalMLPtorch(nn.Module):
    def __init__(self, input_dim: int = 837, hidden: int = 1024, layers: int = 4):
        super().__init__()
        blocks: List[nn.Module] = []
        d = input_dim
        for _ in range(layers - 1):
            blocks += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        blocks += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ========= scorer (loads .pt + .json in src/) =========
def _pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(name)

class EvalScorer:
    """
    Loads eval_mlp.pt (plus eval_mlp.json) and returns eval in pawns (White POV).
    Falls back to simple material if files are missing or invalid.
    """
    _MAT = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}

    def __init__(self, model_path: Optional[str] = None, device_name: str = "auto") -> None:
        self.device = _pick_device(device_name)
        self.model: Optional[nn.Module] = None
        self.using_fallback = True

        # Resolve default to src/eval_mlp.pt
        if not model_path:
            here = Path(__file__).resolve().parent
            model_path = str(here / "eval_mlp.pt")

        p = Path(model_path)
        cfg_path = p.with_suffix(".json")

        if p.exists() and cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                self.model = EvalMLPtorch(**cfg).to(self.device).eval()
                sd = torch.load(p, map_location="cpu")
                self.model.load_state_dict(sd, strict=True)
                self.using_fallback = False
                print(f"[eval] loaded {p.name} on {self.device} (cfg={cfg})")
            except Exception as e:
                print(f"[eval] failed to load model, falling back to material: {e}")
                self.using_fallback = True
        else:
            print(f"[eval] missing model/config ({p}, {cfg_path}); using material fallback")
            self.using_fallback = True

    def eval_board_white_pawns(self, board: chess.Board) -> float:
        if self.using_fallback or self.model is None:
            # Why: guarantees a stable baseline even without the NN.
            s = 0.0
            for _, piece in board.piece_map().items():
                s += self._MAT[piece.piece_type] if piece.color == chess.WHITE else -self._MAT[piece.piece_type]
            return s
        with torch.no_grad():
            x = torch.from_numpy(fen_to_features(board.fen())).unsqueeze(0).to(self.device)
            y = self.model(x).squeeze(0).item()
            return float(y)

# ========= search (NN-guided) =========
_PIECE_CP = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:0}

def _stm_cp_from_white(cp_white: int, board: chess.Board) -> int:
    return cp_white if board.turn == chess.WHITE else -cp_white

def _blended_eval_cp(board: chess.Board, scorer: EvalScorer) -> int:
    # Why: anchor NN with material for stability on shallow nodes.
    nn_pawns = scorer.eval_board_white_pawns(board)
    nn_cp_white = int(nn_pawns * 100)
    mat = 0
    for _, p in board.piece_map().items():
        mat += _PIECE_CP[p.piece_type] if p.color == chess.WHITE else -_PIECE_CP[p.piece_type]
    white_cp = int(0.75 * nn_cp_white + 0.25 * mat)
    return _stm_cp_from_white(white_cp, board)

def _capture_heuristic(board: chess.Board, m: Move) -> int:
    s = 0
    if board.is_capture(m):
        victim = board.piece_at(m.to_square)
        attacker = board.piece_at(m.from_square)
        if victim and attacker:
            s += 1000 + 10 * victim.piece_type - attacker.piece_type
    if m.promotion:
        s += 400 + m.promotion
    return s

def _order_root(board: chess.Board, moves: List[Move], scorer: EvalScorer) -> List[Move]:
    scored: List[Tuple[float, Move]] = []
    for m in moves:
        base = _capture_heuristic(board, m)
        board.push(m)
        prior = _blended_eval_cp(board, scorer)  # stm pov after move
        board.pop()
        scored.append((base + prior, m))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored]

def _order_inner(board: chess.Board, moves: List[Move]) -> List[Move]:
    return sorted(moves, key=lambda m: (_capture_heuristic(board, m), m.uci()), reverse=True)

def _quiescence(board: chess.Board, alpha: int, beta: int, scorer: EvalScorer) -> int:
    stand = _blended_eval_cp(board, scorer)
    if stand >= beta:
        return beta
    if alpha < stand:
        alpha = stand
    for m in _order_inner(board, list(board.legal_moves)):
        if not board.is_capture(m) and not m.promotion:
            continue
        board.push(m)
        score = -_quiescence(board, -beta, -alpha, scorer)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

def _negamax(board: chess.Board, depth: int, alpha: int, beta: int, dl: float, scorer: EvalScorer) -> int:
    if time.time() >= dl:
        return _blended_eval_cp(board, scorer)  # Why: strict time budget.
    if depth == 0:
        return _quiescence(board, alpha, beta, scorer)
    if board.is_game_over(claim_draw=True):
        if board.is_checkmate():
            return -100_000
        return 0
    best = -10**9
    for m in _order_inner(board, list(board.legal_moves)):
        board.push(m)
        val = -_negamax(board, depth - 1, -beta, -alpha, dl, scorer)
        board.pop()
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best

def _softmax(scores: Dict[Move, int], t: float) -> Dict[Move, float]:
    if not scores:
        return {}
    t = max(t, 1e-3)
    mx = max(scores.values())
    exps = {m: math.exp((s - mx) / (150.0 * t)) for m, s in scores.items()}  # 150cp temp scale
    z = sum(exps.values())
    return {m: (v / z if z > 0 else 1.0 / len(exps)) for m, v in exps.items()}

def search_move(board: chess.Board, scorer: EvalScorer, max_depth: int, time_ms: int) -> Tuple[Move, Dict[Move, float]]:
    start = time.time()
    deadline = start + time_ms / 1000.0
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves")
    order = _order_root(board, legal, scorer)
    best_move: Optional[Move] = None
    root_scores: Dict[Move, int] = {m: 0 for m in legal}

    for depth in range(1, max_depth + 1):
        if time.time() >= deadline:
            break
        current: Dict[Move, int] = {}
        cur_best: Optional[Move] = None
        for m in order:
            if time.time() >= deadline:
                break
            board.push(m)
            sc = -_negamax(board, depth - 1, -10**9, 10**9, deadline, scorer)
            board.pop()
            current[m] = sc
            if cur_best is None or sc > current[cur_best]:
                cur_best = m
        if current:
            root_scores = current
            best_move = cur_best
        order = sorted(order, key=lambda k: root_scores.get(k, -10**9), reverse=True)

    if best_move is None:
        best_move = order[0]
    probs = _softmax(root_scores, _TEMP)
    return best_move, probs

# ========= ChessHacks entrypoints =========
_SCORER: Optional[EvalScorer] = None

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """Loads the NN once and returns a legal Move using NN-guided search."""
    global _SCORER
    if _SCORER is None:
        # Default model path is src/eval_mlp.pt unless CHESS_EVAL_MODEL is set.
        model_path = _ENV_MODEL if _ENV_MODEL else None
        _SCORER = EvalScorer(model_path, device_name=_ENV_DEVICE)
        print(f"[engine] scorer ready (fallback={_SCORER.using_fallback}) depth={_MAX_DEPTH} budget={_TIME_MS}ms")

    b = ctx.board
    if b.is_game_over(claim_draw=True):
        ctx.logProbabilities({})
        raise ValueError("Game over")

    mv, probs = search_move(b, _SCORER, _MAX_DEPTH, _TIME_MS)
    ctx.logProbabilities(probs)
    return mv

@chess_manager.reset
def reset_func(ctx: GameContext):
    """Clears global state between games (prevents stale devices)."""
    global _SCORER
    _SCORER = None
    return