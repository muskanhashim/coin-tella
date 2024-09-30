from collections import deque
import numpy as np
import torch
from .training_agent import Training_Agent, state_to_features, state_to_features_2
from .callbacks import ACTIONS
import events as e
import os

def setup_training(self):
    """
    Initialize training parameters.
    """
    self.logger.info("Setting up training variables.")
    self.epsilon_history = []
    self.coord_history = deque([], maxlen=30)
    self.bomb_history = deque([], maxlen=30)
    self.reward_history = deque([], maxlen = 50)
    self.transitions = deque([], maxlen=1000)
    self.learning_rate = 0.001
    self.gamma = 0.99
    self.epsilon = 0.75
    self.eps_decay = 0.9999
    self.eps_min = 0.01
    if not hasattr(self, 'training_agent'):
        self.training_agent = Training_Agent(
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            epsilon=self.epsilon,
            eps_decay=self.eps_decay,
            eps_min=self.eps_min,
            n_actions=len(ACTIONS),
            batch_size=64
        )
    #-----------------------------
    #This commented out is what you want to do, when in loading a certain model that you want to train upon
    #POSSIBLE CHANGES: Also saving epsilon, depending on whether you want to change it for encouraging it to learn
    #                  a new task
    #-----------------------------
    model_path = "coin_model_weights.pt"  # Updated to load from coin_model_weights.pt
    
    self.logger.info(f"Loading pre-trained model from {model_path}.")
    checkpoint = torch.load(model_path)
        
        # Load model weights
    self.training_agent.q_eval.load_state_dict(checkpoint['q_eval_state_dict'])
    self.training_agent.q_next.load_state_dict(checkpoint['q_next_state_dict'])
        
        # Load optimizer state
    self.training_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    self.logger.info("Pre-trained model loaded successfully.")

    self.memory = self.training_agent.memory

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Called once per step to allow intermediate rewards based on game events.
    """
    self.logger.debug(f'Events: {", ".join(events)} in step {new_game_state["step"]}')
    
    if old_game_state is None or new_game_state is None:
        return
    
    player_pos = old_game_state['self'][3]

    # Update position history and check for bomb/explosion danger
    self.coord_history.append(player_pos)  # Append the player's current position

    if len(self.coord_history) >= 3:
        recent_positions = list(self.coord_history)[-3:]
        if all(pos == recent_positions[0] for pos in recent_positions):
            events.append('WAITED_TOO_LONG') # The agent has been waiting too long

    if len(new_game_state['coins'])>0: 
        best_directions = bfs_find_nearest_coin(new_game_state, player_pos)
        if self_action in best_directions:
            events.append('WALKING_TOWARDS_COIN')
        elif not self_action == 'WAIT' :
            events.append('WALKING_AWAY_FROM_COIN')

    # Calculate and apply rewards based on events
    reward = calculate_reward_based_on_events(self, events)
    state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    action = ACTIONS.index(self_action)

    self.reward_history.append(reward)
    avg_reward = np.mean(self.reward_history)

    # Save the model if the average reward is above a threshold
    if avg_reward > 150:  # Set your performance threshold here
        self.logger.info(f"Average reward over last 200 steps: {avg_reward}. Saving model.")
        torch.save({
            'q_eval_state_dict': self.training_agent.q_eval.state_dict(),
            'q_next_state_dict': self.training_agent.q_next.state_dict(),
            'optimizer_state_dict': self.training_agent.optimizer.state_dict(),
        }, "coin_model_weights.pt")

    # Store transition and train the agent
    self.training_agent.store_transition(state, action, reward, new_state, False)
    self.training_agent.learn()

def end_of_round(self, last_game_state, last_action, events):
    """
    Called at the end of each game to hand out final rewards and do training.
    """
    coins_left = last_game_state['coins']
    if len(last_game_state['coins']) > 0:
        for i in range(len(last_game_state['coins'])):
            events.append('COINS_LEFT_TO_COLLECT')

    steps_left = last_game_state['step']
    if steps_left < 400:
        for i in range(400 - steps_left):
            events.append('COLLECTED_ALL_COINS')
    #-----------------------------
    #POSSIBLE IDEA:
    #Save the model, if the performance significantly improved, otherwise,
    #discard what the agent tried
    #-----------------------------
    self.logger.debug(f'End of round reached with events: {", ".join(events)}')
    reward = calculate_reward_based_on_events(self, events)
    state = state_to_features(last_game_state)
    action = ACTIONS.index(last_action)
    self.training_agent.store_transition(state, action, reward, None, False)
    self.training_agent.learn()
    torch.save({
        'q_eval_state_dict': self.training_agent.q_eval.state_dict(),
        'q_next_state_dict': self.training_agent.q_next.state_dict(),
        'optimizer_state_dict': self.training_agent.optimizer.state_dict(),
    }, "coin_model_weights.pt")

def learn(self):
    """
    Trigger the agent to learn from its experiences.
    """
    self.training_agent.learn()

# ------------------------------
# Helper Functions
# ------------------------------
def calculate_reward_based_on_events(self, events, old_game_state=None):
    """
    Calculate the total reward based on the events that occurred.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1000,
        e.KILLED_OPPONENT: 5000,
        e.AGENT_LOOPING_3: -1000,
        e.KILLED_SELF: -5000,
        e.SURVIVED_ROUND: 500,
        e.WALKING_TOWARDS_COIN: 500,
        e.WALKING_AWAY_FROM_COIN: -100,
        e.CRATE_DESTROYED: 100,
        e.INVALID_ACTION: -100,
        e.WAITED_TOO_LONG: -50,
        e.COINS_LEFT_TO_COLLECT: -100,
        e.COLLECTED_ALL_COINS: 100,
    }

    reward_sum = sum(game_rewards.get(event, 0) for event in events)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def manhattan_dist(pos1, pos2):
    """
    Calculate the Manhattan distance between two positions.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def bfs_find_nearest_coin(game_state, position):
    """
    Use BFS to find the shortest paths to the nearest coins.
    Returns a list of all possible first moves towards the nearest coins.
    """

    start_pos = position  # Player's position
    coins = game_state['coins']  # List of coins
    if not coins:
        return None  # No coins available

    if start_pos in coins:
        return []

    # Directions mapping: action name to movement (dx, dy)
    directions = {'UP': (0, 1), 'DOWN': (0, -1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
    reverse_directions = {(0, 1): 'UP', (0, -1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}

    queue = deque([(start_pos, [])])  # Queue stores (current position, path)
    visited = set([start_pos])
    nearest_coins_moves = []
    shortest_distance = None

    while queue:
        current_pos, path = queue.popleft()
        
        # If we found a coin, check if it's the first one or another coin at the same shortest distance
        if current_pos in coins:
            if shortest_distance is None:
                shortest_distance = len(path)  # Set the shortest distance
            if len(path) == shortest_distance:  # Only consider coins at the same shortest distance
                if path:  # Collect the first move if a path exists
                    first_move = reverse_directions[(path[0][0] - start_pos[0], path[0][1] - start_pos[1])]
                    nearest_coins_moves.append(first_move)
            continue  # No need to explore further once coins are found at this distance
        
        # Explore neighbors (UP, DOWN, LEFT, RIGHT)
        for direction, (dx, dy) in directions.items():
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)

            if is_valid_move(game_state, new_pos) and new_pos not in visited:
                visited.add(new_pos)
                queue.append((new_pos, path + [new_pos]))  # Append new position to path

    return nearest_coins_moves if nearest_coins_moves else None  # Return all possible first moves

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
