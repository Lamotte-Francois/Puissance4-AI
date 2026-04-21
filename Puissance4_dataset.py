import connect_four_ai as Solver
import random
import json
import numpy as np
import os

ROWS, COLS = 6, 7
BASE = ROWS + 1

# =========================
# 🧠 BITBOARD CORE
# =========================

def idx(col, row):
    return col * BASE + row


def mirror_bitboard(bb):
    res = 0
    for c in range(COLS):
        for r in range(ROWS):
            bit = 1 << idx(c, r)
            if bb & bit:
                mc = COLS - 1 - c
                res |= 1 << idx(mc, r)
    return res


# ✔ KEY SANS turn
def board_key(p1, p2):
    return min(
        (p1, p2),
        (mirror_bitboard(p1), mirror_bitboard(p2))
    )


# =========================
# 🎯 VALID MOVES
# =========================

def valid_moves(heights):
    return [c for c in range(COLS) if heights[c] < ROWS]


# =========================
# 🔥 SOFTMAX
# =========================

def softmax_scores(scores, valid):
    x = np.array(scores, dtype=float)

    mask = np.zeros_like(x, dtype=bool)
    mask[valid] = True
    x[~mask] = -np.inf

    if valid:
        x = x - np.max(x[mask])

    exp_x = np.exp(x)
    probs = exp_x / np.sum(exp_x)

    probs = np.round(probs, 3)
    probs = [float(f"{p:.3f}") for p in probs]

    return probs


def choose_move(probs, valid):
    r = random.random()
    c = 0.0

    for m in valid:
        c += probs[m]
        if r < c:
            return m

    return valid[-1]


# =========================
# 💾 DATASET
# =========================

def load_dataset(file):
    if os.path.exists(file):
        with open(file, "r") as f:
            return json.load(f)
    return {}


def save_dataset(dataset, file):
    with open(file, "w") as f:
        json.dump(dataset, f)


def store(dataset, key, probs, turn):
    k = str(key)

    probs = [float(f"{p:.3f}") for p in probs]

    # ✔ FORMAT DEMANDÉ
    value = [probs, turn]

    if k not in dataset:
        dataset[k] = value
        return True
    return False


# =========================
# 🎮 GAME GENERATION
# =========================

def generate_game(dataset):
    player = Solver.Position()
    solver = Solver.Solver()

    p1 = 0
    p2 = 0
    heights = [0] * COLS
    turn = 1

    new_positions = 0

    while True:

        valid = valid_moves(heights)
        if not valid:
            break

        scores = solver.get_all_move_scores(player)
        probs = softmax_scores(scores, valid)

        key = board_key(p1, p2)

        if store(dataset, key, probs, turn):
            new_positions += 1

        if new_positions > 0 and new_positions % 50 == 0:
            print(f"💾 autosave ({len(dataset)})")
            save_dataset(dataset, "dataset.json")

        move = choose_move(probs, valid)

        player.play(move)

        row = heights[move]
        bit = 1 << idx(move, row)

        if turn == 1:
            p1 ^= bit
        else:
            p2 ^= bit

        heights[move] += 1
        turn = 2 if turn == 1 else 1

        if player.is_won_position():
            break


# =========================
# 🚀 DATASET GENERATION
# =========================

def generate_dataset(n_games=200, file="dataset.json"):
    dataset = load_dataset(file)

    print(f"📂 loaded: {len(dataset)} positions")

    try:
        for i in range(n_games):
            generate_game(dataset)
            print(f"✅ game {i+1}/{n_games} | {len(dataset)} positions")

            if i % 5 == 0:
                save_dataset(dataset, file)

    except KeyboardInterrupt:
        print("\n⚠️ Interruption détectée (Ctrl+C)")
        print("💾 sauvegarde en cours...")

    finally:
        save_dataset(dataset, file)
        print("💾 dataset sauvegardé avec succès")


# =========================
# ▶️ MAIN
# =========================

if __name__ == "__main__":
    generate_dataset(5)