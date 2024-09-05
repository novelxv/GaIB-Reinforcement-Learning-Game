import tkinter as tk
from tkinter import ttk
from rl_agent import ReinforcementLearningAgent
import time

class Game:
    def __init__(self, root):
        self.root = root
        self.root.title("Reinforcement Learning Game")

        # Setup dropdown to select algorithm
        self.algo_label = tk.Label(self.root, text="Select Algorithm:")
        self.algo_label.pack()
        
        self.algo = tk.StringVar()
        self.algo_dropdown = ttk.Combobox(self.root, textvariable=self.algo)
        self.algo_dropdown['values'] = ("q-learning", "sarsa")
        self.algo_dropdown.pack()
        self.algo_dropdown.current(0)  

        # Setup canvas for game visualization
        self.canvas = tk.Canvas(self.root, width=700, height=100)
        self.canvas.pack()
        self.rects = []
        self.hole = 0
        self.apple = 9
        self.board_size = 10
        self.player_position = 2
        self.player_rect = None

        # Create board and items (hole and apple)
        for i in range(self.board_size):
            color = "white"
            if i == self.hole:
                color = "black"  # Hole
            elif i == self.apple:
                color = "green"  # Apple
            rect = self.canvas.create_rectangle(i * 50, 10, (i + 1) * 50, 60, fill=color)
            self.rects.append(rect)

        # Player setup
        self.player_rect = self.canvas.create_oval(self.player_position * 50 + 15, 20, (self.player_position + 1) * 50 - 15, 50, fill="blue")

        # Game status display
        self.status_label = tk.Label(self.root, text="Episode: 0, Points: 0")
        self.status_label.pack()

        # Setup start button
        self.start_button = tk.Button(self.root, text="Start Game", command=self.start_game)
        self.start_button.pack()

        # Setup result area for displaying game results
        self.result_label = tk.Label(self.root, text="Game Results:")
        self.result_label.pack()
        self.result_text = tk.Text(self.root, height=30, width=100)
        self.result_text.pack()

    def update_gui(self):
        """
        Update the player's position on the board.
        """
        self.canvas.coords(self.player_rect, self.player_position * 50 + 15, 20, (self.player_position + 1) * 50 - 15, 50)
        self.root.update()

    def move(self, action: int) -> None:
        """
        Move player based on action. 0 = left, 1 = right.
        """
        if action == 0:
            self.player_position = max(0, self.player_position - 1)
        elif action == 1:
            self.player_position = min(self.board_size - 1, self.player_position + 1)

    def get_reward(self) -> float:
        """
        Get reward based on the player's current position.
        """
        if self.player_position == self.hole:
            return -100  # Fell in the hole
        elif self.player_position == self.apple:
            return 100  # Got the apple
        else:
            return -1 

    def reset_position(self) -> None:
        """
        Reset the player's position if they reach an end state (hole or apple).
        """
        self.player_position = 3

    def play_game(self, algo: str, max_episodes: int = 10):
        """
        Play the game for a number of episodes using the RL agent.
        """
        agent = ReinforcementLearningAgent(n_states=self.board_size, n_actions=2, algo=algo)
        for episode in range(max_episodes):
            points = 0
            self.player_position = 2
            state = self.player_position
            n_steps = 0
            path = [self.player_position]

            while True:
                self.update_gui()
                time.sleep(0.1)  # Delay for visualizing movement

                action = agent.choose_action(state)
                self.move(action)
                n_steps += 1
                reward = self.get_reward()
                next_state = self.player_position

                path.append(next_state)

                if agent.algo == "sarsa":
                    next_action = agent.choose_action(next_state)
                    agent.update_q_value(state, action, reward, next_state, next_action)
                else:  # q-learning
                    agent.update_q_value(state, action, reward, next_state)

                points += reward
                state = next_state

                self.status_label.config(text=f"Episode {episode+1}: Points = {points}, Position = {self.player_position}")

                # Check win/lose conditions
                if points >= 500:
                    self.result_text.insert(tk.END, f"Player won in episode {episode+1} with {points} points. Steps taken: {n_steps}\n")
                    self.result_text.insert(tk.END, f"Path taken: {' - '.join(map(str, path))}\n")
                    break
                elif points <= -200:
                    self.result_text.insert(tk.END, f"Player lost in episode {episode+1} with {points} points. Steps taken: {n_steps}\n")
                    self.result_text.insert(tk.END, f"Path taken: {' - '.join(map(str, path))}\n")
                    break

                if self.player_position == self.hole or self.player_position == self.apple:
                    self.reset_position()

        # Final Q-Table
        self.result_text.insert(tk.END, f"Final Q-Table:\n{agent.get_q_table()}\n")

    def start_game(self):
        """
        Start the game by retrieving the selected algorithm and calling play_game.
        """
        algo = self.algo.get()
        self.result_text.delete(1.0, tk.END)  
        self.play_game(algo)

if __name__ == "__main__":
    root = tk.Tk()
    game_gui = Game(root)
    root.mainloop()
