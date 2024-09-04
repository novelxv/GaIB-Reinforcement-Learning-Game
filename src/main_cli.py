from rl_agent import ReinforcementLearningAgent
import numpy as np

class Game_CLI:
    def __init__(self, algo: str = "q-learning"):
        """
        Initialize the game with rules specified.
        """
        self.board = 10  
        self.player_position = 2  # Start position
        self.hole = 0
        self.apple = 9
        self.points = 0  
        self.agent = ReinforcementLearningAgent(n_states=self.board, n_actions=2, algo=algo)

    def move(self, action: int) -> None:
        """
        Move player based on action. 0 = left, 1 = right.
        """
        if action == 0:
            self.player_position = max(0, self.player_position - 1)  
        elif action == 1:
            self.player_position = min(self.board - 1, self.player_position + 1)  

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
    
    def win(self) -> bool:
        """
        Check if the player wins the game.
        """
        return self.points >= 500
    
    def lose(self) -> bool:
        """
        Check if the player loses the game.
        """
        return self.points <= -200

    def play(self, max_episodes: int = 10) -> None:
        """
        Play the game for a number of episodes using the RL agent.
        """
        for episode in range(max_episodes):
            self.points = 0
            self.player_position = 2
            state = self.player_position  
            n_steps = 0

            while not self.win() and not self.lose():
                action = self.agent.choose_action(state) 
                self.move(action)  
                n_steps += 1
                reward = self.get_reward()  
                next_state = self.player_position  

                if self.agent.algo == "sarsa":
                    next_action = self.agent.choose_action(next_state)
                    self.agent.update_q_value(state, action, reward, next_state, next_action)
                else: # q-learning
                    self.agent.update_q_value(state, action, reward, next_state)

                self.points += reward
                state = next_state  

                print(f"Episode {episode+1}: Points = {self.points}, Position = {self.player_position}")

                if self.win():
                    print(f"Player won in episode {episode+1} with {self.points} points. Steps taken: {n_steps}")
                    break
                elif self.lose():
                    print(f"Player lost in episode {episode+1} with {self.points} points. Steps taken: {n_steps}")
                    break

                if self.player_position == self.hole or self.player_position == self.apple:
                    self.reset_position()

        print("Final Q-Table:")
        print(self.agent.get_q_table())

if __name__ == "__main__":
    algo = input("Enter the algorithm (q-learning/sarsa): ")
    game = Game_CLI(algo=algo)  
    game.play()
