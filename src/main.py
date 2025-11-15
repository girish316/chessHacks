# File: src/main.py
from __future__ import annotations

# stdlib
import os, json, time, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# third-party (light)
import chess
from chess import Move

# chesshacks runtime
from .utils import chess_manager, GameContext

# ---------- config ----------
ENV_MODEL = os.getenv("CHESS_EVAL_MODEL", "")
ENV_DEVICE = os.getenv("CHESS_EVAL_DEVICE", "cpu")      # default CPU
ENV_DISABLE_NN = os.getenv("CHESS_EVAL_DISABLE_NN", "1") in ("1", "true", "True")  # default: NN off for fast boot
MAX_DEPTH = int(os.getenv("ENGINE_DEPTH", "3"))
TIME_MS = int(os.getenv("ENGINE_TIME_MS", "250"))
TEMP = float(os.getenv("ENGINE_TEMP", "1.0"))


# ---------- features (no numpy dependency; pure Python lists) ----------
# 12x64 bitboards + side + castling(4) + en passant(64) = 837 floats
def fen_to_features_py(fen: str) -> List[float]:
    b = chess.Board(fen)
    out: List[float] = []
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            m = [0.0] * 64
            for sq in b.pieces(pt, color):
                m[sq] = 1.0
            out.extend(m)
    out.append(1.0 if b.turn == chess.WHITE else 0.0)  # side to move
    out.extend([
        1.0 if b.has_kingside_castling_rights(chess.WHITE) else 0.0,
        1.0 if b.has_queenside_castling_rights(chess.WHITE) else 0.0,
        1.0 if b.has_kingside_castling_rights(chess.BLACK) else 0.0,
        1.0 if b.has_queenside_castling_rights(chess.BLACK) else 0.0,
    ])
    ep = [0.0] * 64
    if b.ep_square is not None:
        ep[b.ep_square] = 1.0
    out.extend(ep)
    return out  # len=837

# ---------- material fallback (no torch) ----------
MAT_PAWNS = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}
PIECE_CP  = {chess.PAWN:100, chess.KNIGHT:320, chess.BISHOP:330, chess.ROOK:500, chess.QUEEN:900, chess.KING:0}

def material_eval_pawns(board: chess.Board) -> float:
    s = 0.0
    for _, p in board.piece_map().items():
        s += MAT_PAWNS[p.piece_type] if p.color == chess.WHITE else -MAT_PAWNS[p.piece_type]
    return s

def blended_eval_cp_material(board: chess.Board) -> int:
    # cp anchor from material only
    mat = 0
    for _, p in board.piece_map().items():
        mat += PIECE_CP[p.piece_type] if p.color == chess.WHITE else -PIECE_CP[p.piece_type]
    # 0.75*0 + 0.25*mat ≈ material-only cp
    white_cp = int(0.25 * mat)
    return white_cp if board.turn == chess.WHITE else -white_cp

# ---------- lazy NN loader ----------
class _TorchBundle:
    torch = None
    nn = None

def _device_from(env: str) -> "object":
    # returns torch.device or a sentinel string when torch is absent
    if _TorchBundle.torch is None:
        return "cpu"
    t = _TorchBundle.torch
    if env == "cpu":
        return t.device("cpu")
    if env == "cuda" and t.cuda.is_available():
        return t.device("cuda")
    if env == "mps" and getattr(t.backends, "mps", None) and t.backends.mps.is_available():
        return t.device("mps")
    return t.device("cpu")

class EvalMLPtorch:
    def __init__(self, input_dim: int, hidden: int, layers: int, device):
        t = _TorchBundle.torch; nn = _TorchBundle.nn
        blocks: List[object] = []
        d = input_dim
        for _ in range(layers - 1):
            blocks += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        blocks += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*blocks).to(device)
        self.device = device

    def forward(self, x_list: List[float]) -> float:
        t = _TorchBundle.torch
        x = t.tensor([x_list], dtype=t.float32, device=self.device)
        with t.no_grad():
            y = self.net(x).squeeze(0).item()
        return float(y)

class EvalScorer:
    """Uses NN if available and not disabled; otherwise material fallback."""
    def __init__(self, model_path: Optional[str], device_pref: str, disable_nn: bool) -> None:
        self.disable_nn = disable_nn
        self.using_fallback = True
        self.device_pref = device_pref
        self.device = "cpu"
        self.model: Optional[EvalMLPtorch] = None

        # 1) Fast exit if disabled
        if self.disable_nn:
            print("[eval] NN disabled via CHESS_EVAL_DISABLE_NN; using material fallback")
            return

        # 2) Lazy import torch
        try:
            import torch as _t
            import torch.nn as _nn
            _TorchBundle.torch = _t
            _TorchBundle.nn = _nn
        except Exception as e:
            print(f"[eval] torch import failed ({e}); using material fallback")
            return

        # 3) Device and paths
        self.device = _device_from(device_pref)
        p = Path(model_path) if model_path else Path(__file__).resolve().parent / "eval_mlp.pt"
        cfg_p = p.with_suffix(".json")
        if not (p.exists() and cfg_p.exists()):
            print(f"[eval] missing model/config ({p}, {cfg_p}); using material fallback")
            return

        # 4) Build model + robust state_dict load
        try:
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            self.model = EvalMLPtorch(
                input_dim=int(cfg.get("input_dim", 837)),
                hidden=int(cfg.get("hidden", 1024)),
                layers=int(cfg.get("layers", 4)),
                device=self.device,
            )
            sd = _TorchBundle.torch.load(str(p), map_location="cpu")

            def _try_load_whole(state) -> bool:
                try:
                    # prefer loading into the WHOLE model (expects keys like "net.0.weight")
                    self.model.load_state_dict(state, strict=True)
                    return True
                except Exception:
                    return False

            loaded = _try_load_whole(sd)
            if not loaded:
                # If keys have "net.", strip and load into inner Sequential
                if all(k.startswith("net.") for k in sd.keys()):
                    sd2 = {k.split("net.", 1)[1]: v for k, v in sd.items()}  # "0.weight" ...
                    self.model.net.load_state_dict(sd2, strict=True)
                    loaded = True
                else:
                    # If keys lack "net.", add it and load to whole model
                    sd2 = {f"net.{k}": v for k, v in sd.items()}
                    loaded = _try_load_whole(sd2)

            if not loaded:
                raise RuntimeError("state_dict key mismatch (even after prefix adjust)")

            self.model.net.eval()
            self.using_fallback = False
            print(f"[eval] loaded {p.name} on {self.device} (cfg={cfg})")
        except Exception as e:
            print(f"[eval] failed to load model ({e}); using material fallback")
            self.model = None
            self.using_fallback = True

    def eval_white_pawns(self, board: chess.Board) -> float:
        if self.using_fallback or self.model is None:
            return material_eval_pawns(board)
        feats = fen_to_features_py(board.fen())
        return self.model.forward(feats)

# ---------- search ----------
def _stm_cp_from_white(cp_white: int, board: chess.Board) -> int:
    return cp_white if board.turn == chess.WHITE else -cp_white

def _blended_eval_cp(board: chess.Board, scorer: EvalScorer) -> int:
    if scorer.using_fallback or scorer.model is None:
        return _stm_cp_from_white(blended_eval_cp_material(board), board)
    nn_pawns = scorer.eval_white_pawns(board)
    nn_cp_white = int(nn_pawns * 100)
    mat = 0
    for _, p in board.piece_map().items():
        mat += PIECE_CP[p.piece_type] if p.color == chess.WHITE else -PIECE_CP[p.piece_type]
    white_cp = int(0.75 * nn_cp_white + 0.25 * mat)
    return _stm_cp_from_white(white_cp, board)

def _capture_score(board: chess.Board, m: Move) -> int:
    s = 0
    if board.is_capture(m):
        v = board.piece_at(m.to_square)
        a = board.piece_at(m.from_square)
        if v and a:
            s += 1000 + 10 * v.piece_type - a.piece_type
    if m.promotion:
        s += 400 + m.promotion
    return s

def _order_root(board: chess.Board, moves: List[Move], scorer: EvalScorer) -> List[Move]:
    scored = []
    for m in moves:
        base = _capture_score(board, m)
        board.push(m)
        prior = _blended_eval_cp(board, scorer)
        board.pop()
        scored.append((base + prior, m))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [m for _, m in scored]

def _order_inner(board: chess.Board, moves: List[Move]) -> List[Move]:
    return sorted(moves, key=lambda m: (_capture_score(board, m), m.uci()), reverse=True)

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

def _negamax(board: chess.Board, depth: int, alpha: int, beta: int, deadline: float, scorer: EvalScorer) -> int:
    if time.time() >= deadline:
        return _blended_eval_cp(board, scorer)
    if depth == 0:
        return _quiescence(board, alpha, beta, scorer)
    if board.is_game_over(claim_draw=True):
        if board.is_checkmate(): return -100_000
        return 0
    best = -10**9
    for m in _order_inner(board, list(board.legal_moves)):
        board.push(m)
        val = -_negamax(board, depth - 1, -beta, -alpha, deadline, scorer)
        board.pop()
        if val > best: best = val
        if best > alpha: alpha = best
        if alpha >= beta: break
    return best

def _softmax(scores: Dict[Move, int], t: float) -> Dict[Move, float]:
    if not scores: return {}
    t = max(t, 1e-3)
    mx = max(scores.values())
    exps = {m: math.exp((s - mx) / (150.0 * t)) for m, s in scores.items()}
    z = sum(exps.values())
    return {m: (v / z if z > 0 else 1.0 / len(exps)) for m, v in exps.items()}

def search_move(board: chess.Board, scorer: EvalScorer, max_depth: int, time_ms: int) -> Tuple[Move, Dict[Move, float]]:
    start = time.time(); deadline = start + time_ms / 1000.0
    legal = list(board.legal_moves)
    if not legal: raise ValueError("No legal moves")
    order = _order_root(board, legal, scorer)
    best_move: Optional[Move] = None
    root_scores: Dict[Move, int] = {m: 0 for m in legal}
    for depth in range(1, max_depth + 1):
        if time.time() >= deadline: break
        current: Dict[Move, int] = {}; cur_best: Optional[Move] = None
        for m in order:
            if time.time() >= deadline: break
            board.push(m)
            sc = -_negamax(board, depth - 1, -10**9, 10**9, deadline, scorer)
            board.pop()
            current[m] = sc
            if cur_best is None or sc > current[cur_best]: cur_best = m
        if current:
            root_scores = current; best_move = cur_best
        order = sorted(order, key=lambda k: root_scores.get(k, -10**9), reverse=True)
    if best_move is None: best_move = order[0]
    return best_move, _softmax(root_scores, TEMP)

# ---------- ChessHacks entrypoints ----------
_SCORER: Optional[EvalScorer] = None

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    global _SCORER
    if _SCORER is None:
        model_path = ENV_MODEL if ENV_MODEL else None
        _SCORER = EvalScorer(model_path, device_pref=ENV_DEVICE, disable_nn=ENV_DISABLE_NN)
        print(f"[engine] ready (fallback={_SCORER.using_fallback}) depth={MAX_DEPTH} budget={TIME_MS}ms device={ENV_DEVICE}")
    b = ctx.board
    if b.is_game_over(claim_draw=True):
        ctx.logProbabilities({})
        raise ValueError("Game over")
    mv, probs = search_move(b, _SCORER, MAX_DEPTH, TIME_MS)
    ctx.logProbabilities(probs)
    return mv

@chess_manager.reset
def reset_func(ctx: GameContext):
    global _SCORER
    _SCORER = None
    return