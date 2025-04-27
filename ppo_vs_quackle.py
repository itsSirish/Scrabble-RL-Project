# # ppo_vs_quackle_manual.py
# import torch
# from scrabble_rl.model import ActorCritic
# from scrabble_rl.utils import mapBoardToState

# def main():
#     model = ActorCritic(action_dim=10)
#     model.load_state_dict(torch.load("scrabble_ppo.pt"))
#     model.eval()

#     # Initialize board manually
#     current_board = YOUR_BOARD_OBJECT   # create empty board
#     current_rack = ['A', 'E', 'I', 'N', 'R', 'S', 'T']  # example starting rack

#     while True:
#         obs = mapBoardToState(current_board, current_rack)
#         obs_tensor = {k: torch.tensor(v).float() for k, v in obs.items()}
#         logits, _ = model(obs_tensor)
#         action = torch.argmax(logits).item()

#         # Generate moves (top 10)
#         moves = current_board.find_best_moves(current_rack, 10)
#         if not moves:
#             print("No valid moves left. Ending game.")
#             break

#         move = moves[min(action, len(moves)-1)]
#         print(f"🤖 Agent suggests playing: {move[0]} at {move[2]} {'HORIZ' if move[3] == 0 else 'VERT'} (Score {move[1]})")

#         input("➡️ Play this move manually in Quackle, press Enter when done...")

#         # Here you must manually update:
#         # - current_board (set the new word in your board object)
#         # - current_rack (remove used letters, draw new ones)

#         # (We can make helper functions to make this easier.)

#         continue

# if __name__ == "__main__":
#     main()
