from collections import deque
import os
import numpy as np
from .training_agent import Training_Agent, state_to_features, state_to_features_2
import torch

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

def setup(self):
    """
    Setup your agent. This is called once when loading each agent.
    """
    self.logger.info("Setting up model from scratch.")
    self.learning_rate = 0.001  # Adjusted learning rate
    self.gamma = 0.99
    self.epsilon = 1
    self.eps_decay = 0.99975
    self.eps_min = 0.01
    self.coord_history = deque([], maxlen=30)
    self.bomb_history = deque([], maxlen=30)
    self.training_agent = Training_Agent(
        learning_rate=self.learning_rate,
        gamma=self.gamma,
        epsilon=self.epsilon,
        eps_decay=self.eps_decay,
        eps_min=self.eps_min,
        n_actions=len(ACTIONS),
        batch_size=64
    )
    '''
    # Load the model if not training or model_weights.pt file exists
    if self.train:
        self.logger.info("Loading model from saved state.")
        checkpoint = torch.load("coin_model_weights.pt", map_location=self.training_agent.device)
        
        # Load model weights and optimizer state
        self.training_agent.q_eval.load_state_dict(checkpoint['q_eval_state_dict'])
        self.training_agent.q_next.load_state_dict(checkpoint['q_next_state_dict'])
        self.training_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_agent.q_eval.eval()
        self.training_agent.q_next.eval()

        # Load epsilon and step counter if stored
        if 'epsilon' in checkpoint:
            self.training_agent.epsilon = checkpoint['epsilon']
        if 'learn_step_counter' in checkpoint:
            self.training_agent.learn_step_counter = checkpoint['learn_step_counter']'''


def act(self, game_state: dict) -> str:
    """
    Decide what action to take based on the current game state.
    """
    if game_state['step'] == 1:
        self.coord_history = deque([], maxlen=30)
        self.bomb_history = deque([], maxlen=30)

    current_state = state_to_features(game_state)
    if current_state is None:
        return 'WAIT'  # Default action if state is invalid

    # Choose action using the training agent
    action_idx = self.training_agent.choose_action(current_state)
    chosen_action = ACTIONS[action_idx]

    # Log the chosen action
    self.training_agent.writer.add_scalar('Action/Chosen', action_idx, self.training_agent.learn_step_counter)

    return chosen_action

#--------------------
# :param agent_position: Current position of the agent
# :param field: In order to read the walls for the game
# :param game_state: Info to see if a bomb is nearby, used later for the training
#                   ,but not in the first stage of the training
# :output: vector for all possible moves
#--------------------
def is_valid_move(game_state, position):
    """
    Check if a move is valid.
    A move is valid if it is within bounds and the position is a free tile (not a wall or crate).
    """
    x, y = position
    field = game_state['field']
    
    # Check if the position is within bounds
    if not (0 <= x < len(field) and 0 <= y < len(field[0])):
        return False
    
    # Check if the position is a free tile (not a wall or crate)
    if field[x][y] == -1:  # Walls are marked as -1
        return False
    if field[x][y] == 1:  # Crates are marked as 1
        return False
    
    return True



def save_model(self):
    """
    Save the model to a .pt file. This includes both the model's state dict and other important variables.
    """
    self.logger.info("Saving model weights to coin_model_weights.pt.")
    torch.save({
        'q_eval_state_dict': self.training_agent.q_eval.state_dict(),
        'q_next_state_dict': self.training_agent.q_next.state_dict(),
        'optimizer_state_dict': self.training_agent.optimizer.state_dict(),
        'epsilon': self.training_agent.epsilon,
        'learn_step_counter': self.training_agent.learn_step_counter,
    }, "coin_model_weights.pt")


def load_model(self):
    """
    Load the model from a .pt file.
    """
    if os.path.isfile("coin_model_weights.pt"):
        self.logger.info("Loading model weights from coin_model_weights.pt.")
        checkpoint = torch.load("coin_model_weights.pt", map_location=self.training_agent.device)
        
        # Load model weights and optimizer state
        self.training_agent.q_eval.load_state_dict(checkpoint['q_eval_state_dict'])
        self.training_agent.q_next.load_state_dict(checkpoint['q_next_state_dict'])
        self.training_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_agent.q_eval.eval()
        self.training_agent.q_next.eval()

        # Load epsilon and step counter if stored
        if 'epsilon' in checkpoint:
            self.training_agent.epsilon = checkpoint['epsilon']
        if 'learn_step_counter' in checkpoint:
            self.training_agent.learn_step_counter = checkpoint['learn_step_counter']
    else:
        self.logger.info("No coin_model_weights.pt file found. Starting training from scratch.")
