import os
import json

def load_game_metadata(path):
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None

def count_game_outcomes(logs_dir):
    villager_wins = 0
    werewolf_wins = 0

    for folder in os.listdir(logs_dir):
        folder_path = os.path.join(logs_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        meta_path = os.path.join(folder_path, "game_complete.json")
        if not os.path.isfile(meta_path):
            continue

        meta = load_game_metadata(meta_path)
        if not meta or not isinstance(meta, dict):
            continue

        winner = meta.get("winner", "")
        if winner == "Villagers":
            villager_wins += 1
        elif winner == "Werewolves":
            werewolf_wins += 1

    print(f"🏆 Villagers won {villager_wins} games")
    print(f"🐺 Werewolves won {werewolf_wins} games")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", required=True, help="Directory containing game folders")
    args = parser.parse_args()
    count_game_outcomes(args.logs_dir)
