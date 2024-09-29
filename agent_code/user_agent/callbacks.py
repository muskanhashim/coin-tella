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
    #the parameters that seemed to work best for us after hyperparameter tuning:
    self.logger.info("setting up model from scratch!")
    self.learning_rate = 0.001  
    self.gamma = 0.99
    self.epsilon = 0.75
    self.epsdecay = 0.9999 
    self.eps_min = 0.01
    self.coord_history = deque([], maxlen=30)
    self.bombhistory = deque([], maxlen=30)
    self.training_agent = Training_Agent(
        lr_rate=self.learning_rate,
        gamma=self.gamma,
        epsilon=self.epsilon,
        epsdecay=self.epsdecay,
        eps_min=self.eps_min,
        n_actions=len(ACTIONS),
        batch_size=32
    )

    '''if self.train:
        self.logger.info("Loading model from saved state")
        checkpoint = torch.load("coin_model_weights.pt", map_location=self.training_agent.device)
#loading our saved model weights
        self.training_agent.q_eval.load_state_dict(checkpoint['q_eval_state_dict'])
        self.training_agent.q_next.load_state_dict(checkpoint['q_next_state_dict'])
        self.training_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_agent.q_eval.eval()
        self.training_agent.q_next.eval()
#for step counter/epsilon
        if 'epsilon' in checkpoint:
            self.training_agent.epsilon = checkpoint['epsilon']
        if 'learn_step_counter' in checkpoint:
            self.training_agent.learn_step_counter = checkpoint['learn_step_counter']'''


def act(self, game_state: dict) -> str:
    """
    Deciding what the best action would be based on the state-of-the-game
    """
    if game_state['step'] == 1:
        self.coord_history = deque([], maxlen=30)
        self.bombhistory = deque([], maxlen=30)

    current_state = state_to_features(game_state)
    if current_state is None:
        return 'WAIT'  # for any state that is: invalid

    action_idx = self.training_agent.choose_action(current_state)
    chosen_action = ACTIONS[action_idx]

    # necessary action-logging
    self.training_agent.writer.add_scalar('Action/Chosen', action_idx, self.training_agent.learn_step_counter)

    return chosen_action




def is_valid_move(game_state, position):
    """
    With this function, we aim to check if a move is valid
    We define a move as valid if: (i) it is within bounds (board walls) &
    (ii) position is a free tile (not a wall oder crate)
    """
    x, y = position
    field = game_state['field']
    
    # checking condition (i)
    if not (0 <= x < len(field) and 0 <= y < len(field[0])):
        return False
    
    # to check if tile is free (walls=-1, crates=1)
    if field[x][y] == -1:  
        return False
    if field[x][y] == 1:  
        return False
    
    return True



def save_model(self):
    """
    saving model as .pt file
    """
    self.logger.info("saving model weights to :- coin_model_weights.pt")
    torch.save({
        'q_eval_state_dict': self.training_agent.q_eval.state_dict(),
        'q_next_state_dict': self.training_agent.q_next.state_dict(),
        'optimizer_state_dict': self.training_agent.optimizer.state_dict(),

        'epsilon': self.training_agent.epsilon,
        'learn_step_counter': self.training_agent.learn_step_counter,
    }, "coin_model_weights.pt")


def load_model(self):
    """
    loading model from .pt file
    """
    if os.path.isfile("coin_model_weights.pt"):
        self.logger.info("loading model weights from coin_model_weights.pt.")
        checkpoint = torch.load("coin_model_weights.pt", map_location=self.training_agent.device)
        self.training_agent.q_eval.load_state_dict(checkpoint['q_eval_state_dict'])
        self.training_agent.q_next.load_state_dict(checkpoint['q_next_state_dict'])

        self.training_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_agent.q_eval.eval()

        self.training_agent.q_next.eval()
        if 'epsilon' in checkpoint:
            self.training_agent.epsilon = checkpoint['epsilon']
        if 'learn_step_counter' in checkpoint:
            self.training_agent.learn_step_counter = checkpoint['learn_step_counter']
    else:
        self.logger.info("No coin_model_weights.pt file was found... Starting training from scratch!")
