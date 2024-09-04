import numpy as np

class ReinforcementLearningAgent:
    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1, algo: str = "q-learning"):
        """
        Initialize the agent.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
        self.algo = algo  # q-learning or sarsa

    def choose_action(self, state: int) -> int:
        """
        Epsilon-greedy policy to choose an action.
        """
        if np.random.uniform(0, 1) < self.epsilon:  # Explore
            return np.random.choice(self.n_actions)
        else:  # Exploit
            return np.argmax(self.q_table[state])

    def update_q_value(self, state: int, action: int, reward: float, next_state: int, next_action: int = None) -> None:
        """
        Update Q-value using either Q-Learning or SARSA based on the algorithm.
        """
        if self.algo == "q-learning":
            # Q-Learning update rule
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        elif self.algo == "sarsa":
            # SARSA update rule
            td_target = reward + self.gamma * self.q_table[next_state][next_action]

        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def set_algo(self, algo: str) -> None:
        """
        Set the agent's learning algo.
        """
        if algo in ["q-learning", "sarsa"]:
            self.algo = algo
        else:
            raise ValueError("Invalid algorithm. Choose either q-learning or sarsa.")

    def get_q_table(self) -> np.ndarray:
        """
        Return the Q-table for inspection.
        """
        return self.q_table
