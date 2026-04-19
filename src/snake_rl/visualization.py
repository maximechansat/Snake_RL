from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from src.snake_rl.agent import SnakeAgent

# Palette
BACKGROUND = np.array([15, 17, 26], dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  Apple sprite  (6 × 7, RGB — [0,0,0] = transparent)
# ══════════════════════════════════════════════════════════════════════════════

_AT = [0, 0, 0]  # transparent
_AS = [139, 69, 19]  # stem brown
_AL = [45, 138, 62]  # leaf green
_AR = [232, 34, 58]  # red body
_AD = [192, 57, 43]  # dark red
_AH = [255, 107, 122]  # highlight pink
_AB = [169, 50, 38]  # bottom shadow

APPLE_SPRITE: np.ndarray = np.array(
    [
        [_AT, _AT, _AB, _AB, _AB, _AT, _AT],
        [_AT, _AR, _AR, _AD, _AR, _AD, _AT],
        [_AR, _AR, _AR, _AD, _AR, _AD, _AR],
        [_AR, _AH, _AR, _AD, _AR, _AD, _AR],
        [_AT, _AR, _AR, _AR, _AR, _AR, _AT],
        [_AT, _AT, _AL, _AS, _AT, _AT, _AT],
    ],
    dtype=np.uint8,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Snake sprites  (all 8 × 8, RGB — [0,0,0] = transparent)
# ══════════════════════════════════════════════════════════════════════════════

_ST = np.array([0, 0, 0], dtype=np.uint8)  # transparent
_SE = np.array([20, 20, 20], dtype=np.uint8)  # eye
_SH = np.array([80, 220, 120], dtype=np.uint8)  # head green (bright)
_SB = np.array([50, 180, 90], dtype=np.uint8)  # body mid green
_SD = np.array([30, 130, 65], dtype=np.uint8)  # body dark green
_SX = np.array([20, 90, 45], dtype=np.uint8)  # tail tip

# Head facing RIGHT — eyes near the front (right) side
_HEAD_R: np.ndarray = np.array(
    [
        [_ST, _ST, _ST, _ST, _ST, _ST, _ST, _ST],
        [_ST, _SB, _SH, _SH, _SH, _SH, _SH, _ST],
        [_ST, _SH, _SH, _SE, _SH, _SH, _SH, _ST],
        [_ST, _SH, _SH, _SH, _SH, _SH, _SH, _ST],
        [_ST, _SH, _SH, _SE, _SH, _SH, _SH, _ST],
        [_ST, _SB, _SH, _SH, _SH, _SH, _SH, _ST],
        [_ST, _ST, _ST, _ST, _ST, _ST, _ST, _ST],
        [_ST, _ST, _ST, _ST, _ST, _ST, _ST, _ST],
    ],
    dtype=np.uint8,
)

# Body straight HORIZONTAL
_BODY_H: np.ndarray = np.array(
    [
        [_ST, _ST, _ST, _ST, _ST, _ST, _ST, _ST],
        [_SB, _SB, _SB, _SB, _SB, _SB, _SB, _SB],
        [_SB, _SD, _SD, _SD, _SD, _SD, _SD, _SB],
        [_SB, _SD, _SB, _SB, _SB, _SB, _SD, _SB],
        [_SB, _SD, _SB, _SB, _SB, _SB, _SD, _SB],
        [_SB, _SD, _SD, _SD, _SD, _SD, _SD, _SB],
        [_SB, _SB, _SB, _SB, _SB, _SB, _SB, _SB],
        [_ST, _ST, _ST, _ST, _ST, _ST, _ST, _ST],
    ],
    dtype=np.uint8,
)

# Body corner — incoming from LEFT, turning UP
_BODY_CORNER_LU: np.ndarray = np.array(
    [
        [_ST, _ST, _ST, _ST, _ST, _ST, _ST, _ST],
        [_SB, _SB, _SB, _SB, _SB, _ST, _ST, _ST],
        [_SB, _SD, _SD, _SD, _SD, _SB, _ST, _ST],
        [_SB, _SD, _SB, _SB, _SB, _SD, _SB, _ST],
        [_SB, _SD, _SB, _SB, _SB, _SD, _SB, _ST],
        [_SB, _SB, _SD, _SD, _SD, _SD, _SB, _ST],
        [_ST, _ST, _SB, _SB, _SB, _SB, _SB, _ST],
        [_ST, _ST, _ST, _ST, _ST, _ST, _ST, _ST],
    ],
    dtype=np.uint8,
)

# Tail tip pointing LEFT, connecting to body on the right
_TAIL_L: np.ndarray = np.array(
    [
        [_ST, _ST, _ST, _ST, _ST, _ST, _ST, _ST],
        [_ST, _ST, _ST, _SX, _SB, _SB, _SB, _ST],
        [_ST, _ST, _SX, _SD, _SD, _SD, _SB, _ST],
        [_ST, _SX, _SD, _SB, _SB, _SD, _SB, _ST],
        [_ST, _SX, _SD, _SB, _SB, _SD, _SB, _ST],
        [_ST, _ST, _SX, _SD, _SD, _SD, _SB, _ST],
        [_ST, _ST, _ST, _SX, _SB, _SB, _SB, _ST],
        [_ST, _ST, _ST, _ST, _ST, _ST, _ST, _ST],
    ],
    dtype=np.uint8,
)


def _rot(sprite: np.ndarray, k: int) -> np.ndarray:
    """Rotate sprite 90° counter-clockwise k times."""
    return np.rot90(sprite, k=k)


def _flip(sprite: np.ndarray) -> np.ndarray:
    return np.fliplr(sprite)


# Head sprites keyed by movement direction (dx, dy)
HEAD_SPRITES: dict[tuple[int, int], np.ndarray] = {
    (1, 0): _HEAD_R,
    (-1, 0): _flip(_HEAD_R),
    (0, 1): _rot(_HEAD_R, 1),
    (0, -1): _rot(_HEAD_R, 3),
}

# Body straight sprites keyed by axis
BODY_STRAIGHT: dict[str, np.ndarray] = {
    "H": _BODY_H,
    "V": _rot(_BODY_H, 1),
}

# Body corner sprites keyed by (incoming_dir, outgoing_dir).
# incoming = direction FROM previous segment TO this one
# outgoing = direction FROM this segment TO next one
BODY_CORNERS: dict[tuple[tuple[int, int], tuple[int, int]], np.ndarray] = {
    ((1, 0), (0, 1)): _BODY_CORNER_LU,
    ((0, -1), (-1, 0)): _BODY_CORNER_LU,
    ((1, 0), (0, -1)): _flip(_BODY_CORNER_LU),
    ((0, 1), (-1, 0)): _flip(_BODY_CORNER_LU),
    ((-1, 0), (0, 1)): _flip(_rot(_BODY_CORNER_LU, 2)),
    ((0, -1), (1, 0)): _flip(_rot(_BODY_CORNER_LU, 2)),
    ((-1, 0), (0, -1)): _rot(_BODY_CORNER_LU, 2),
    ((0, 1), (1, 0)): _rot(_BODY_CORNER_LU, 2),
}

# Tail sprites keyed by direction FROM tail tip TOWARD body
TAIL_SPRITES: dict[tuple[int, int], np.ndarray] = {
    (1, 0): _TAIL_L,
    (-1, 0): _flip(_TAIL_L),
    (0, 1): _rot(_TAIL_L, 1),
    (0, -1): _rot(_TAIL_L, 3),
}

_TRANSPARENT = np.array([0, 0, 0], dtype=np.uint8)


def _unit(a: tuple, b: tuple) -> tuple[int, int]:
    """Unit direction vector from cell a to cell b."""
    dx = int(b[0]) - int(a[0])
    dy = int(b[1]) - int(a[1])
    return (max(-1, min(1, dx)), max(-1, min(1, dy)))


def _blit(frame: np.ndarray, sprite: np.ndarray, cx: int, cy: int, scale: int, grid: int) -> None:
    """Blit a sprite centered on grid cell (cx, cy) into frame, skipping transparent pixels."""
    sh, sw = sprite.shape[:2]
    pad_y = (scale - sh) // 2
    pad_x = (scale - sw) // 2
    limit = grid * scale
    for sy in range(sh):
        for sx in range(sw):
            color = sprite[sy, sx]
            if np.array_equal(color, _TRANSPARENT):
                continue
            fy = cy * scale + pad_y + sy
            fx = cx * scale + pad_x + sx
            if 0 <= fy < limit and 0 <= fx < limit:
                frame[fy, fx] = color


def snake_to_frame(env: Any, scale: int = 12) -> np.ndarray:
    """Convert the current game state to an RGB frame using pixel-art sprites."""
    e = env.unwrapped
    grid = e.size
    frame = np.tile(BACKGROUND, (grid * scale, grid * scale, 1)).astype(np.uint8)

    # Apple
    if getattr(e, "apple", None) is not None:
        _blit(frame, APPLE_SPRITE, int(e.apple[0]), int(e.apple[1]), scale, grid)

    # Snake
    body = list(e.snake)
    n = len(body)
    if n == 0:
        return frame[::-1]

    for i, (x, y) in enumerate(body):
        if i == 0:
            d = _unit(body[1], body[0]) if n > 1 else (1, 0)
            sprite = HEAD_SPRITES.get(d, _HEAD_R)

        elif i == n - 1:
            d = _unit(body[i], body[i - 1])
            sprite = TAIL_SPRITES.get(d, _TAIL_L)

        else:
            d_in = _unit(body[i - 1], body[i])
            d_out = _unit(body[i], body[i + 1])
            if d_in == d_out:
                sprite = BODY_STRAIGHT["H"] if d_in[0] != 0 else BODY_STRAIGHT["V"]
            else:
                sprite = BODY_CORNERS.get((d_in, d_out), _BODY_H)

        _blit(frame, sprite, x, y, scale, grid)

    return frame[::-1]


def rollout_frames(
    agent: SnakeAgent,
    env: Any,
    max_steps: int = 100_000,
    greedy: bool = True,
    scale: int = 12,
) -> list[np.ndarray]:
    """Run one episode and return a list of rendered frames."""
    old_eps = agent.epsilon
    if greedy:
        agent.epsilon = 0.0

    obs, _ = env.reset()
    frames = [snake_to_frame(env, scale=scale)]
    terminated = truncated = dead = False
    steps = 0

    while not (terminated or truncated or dead) and steps < max_steps:
        action = agent.get_action(obs)
        obs, _, terminated, truncated, info = env.step(action)
        dead = bool(info.get("dead", False))
        frames.append(snake_to_frame(env, scale=scale))
        steps += 1

    agent.epsilon = old_eps
    return frames


def build_animation_html(frames: list[np.ndarray], interval_ms: int = 120) -> str:
    """Wrap a list of frames into a self-contained HTML animation."""
    # frame shape is (grid*scale, grid*scale, 3) — grid lines at every `scale` pixels
    frame_px = frames[0].shape[0]
    scale = 12  # must match snake_to_frame default; consider threading via argument
    grid = frame_px // scale

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0f111a")
    ax.set_facecolor("#0f111a")
    ax.axis("off")

    # Grid lines — one per cell boundary, in pixel-image coordinates
    for i in range(grid + 1):
        ax.axhline(i * scale - 0.5, color="#191c28", linewidth=0.4)
        ax.axvline(i * scale - 0.5, color="#191c28", linewidth=0.4)

    img_artist = ax.imshow(
        frames[0],
        interpolation="nearest",
        vmin=0,
        vmax=255,
    )
    fig.tight_layout(pad=0)

    def _update(idx: int):
        img_artist.set_data(frames[idx])
        return (img_artist,)

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), interval=interval_ms, blit=True)
    html = ani.to_jshtml()
    plt.close(fig)
    return html


def visualize_episode(
    agent: SnakeAgent,
    env: Any,
    max_steps: int = 100_000,
    greedy: bool = True,
    interval_ms: int = 120,
    scale: int = 12,
) -> str:
    """Run one greedy episode and return an HTML animation string."""
    frames = rollout_frames(agent=agent, env=env, max_steps=max_steps, greedy=greedy, scale=scale)
    return build_animation_html(frames=frames, interval_ms=interval_ms)
