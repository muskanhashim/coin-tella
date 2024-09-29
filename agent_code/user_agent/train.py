from collections import deque
import numpy as np
import torch
from .training_agent import Training_Agent, state_to_features, state_to_features_2
from .callbacks import ACTIONS
import events as e
import os

#the hyperparameters that seemed to work best for us!
def setup_training(self):
    """
    function to initialize the parameters
    """
    self.logger.info("setting up training variables!")
    self.epsilon_history = []
    self.coord_history = deque([], maxlen=30)
    self.bombhistory = deque([], maxlen=30)
    self.reward_history = deque([], maxlen = 50)
    self.transitions = deque([], maxlen=1000)
    self.learning_rate = 0.001 #note
    self.gamma = 0.99 #test
    self.epsilon = 0.75
    self.epsdecay = 0.9999 #initially tried out the 0.995 but changing this made a huge difference: note for future training 
    self.eps_min = 0.01
    if not hasattr(self, 'training_agent'):
        self.training_agent = Training_Agent(
            lr_rate=self.learning_rate,
            gamma=self.gamma,
            epsilon=self.epsilon,
            eps_decay=self.epsdecay,
            eps_min=self.eps_min,
            n_actions=len(ACTIONS),
            batch_size=64 #earlier:32?
        )
#possible: also saving epsilon if we'd want to change it for encouraging learning of new tasks
  
    ''' model_path = "coin_model_weights.pt"  #name of file where our model weights are saved!
    
    self.logger.info(f"please wait: loading the pre-trained model from {model_path}")
    checkpoint = torch.load(model_path)
    self.training_agent.q_eval.load_state_dict(checkpoint['q_eval_state_dict'])
    self.training_agent.q_next.load_state_dict(checkpoint['q_next_state_dict'])
    self.training_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.logger.info("pre-trained model loaded!")'''
    self.memory = self.training_agent.memory

#Collecting and defining events, in order to give rewards for certain behaviour of our agent
def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    intermediate rewards for game events like coin collection or walking away from a coin.
    """
    self.logger.debug(f'Events: {", ".join(events)} in step {new_game_state["step"]}')
    if old_game_state is None or new_game_state is None:
        return
    player_pos = old_game_state['self'][3] #as mentioned in project guidelines
    self.coord_history.append(player_pos)  
    if len(self.coord_history) >= 3:
        recent_positions = list(self.coord_history)[-3:]
        if all(pos == recent_positions[0] for pos in recent_positions):
            events.append('WAITED_TOO_LONG') #felt the need to penalize excessive waiting.

    if len(new_game_state['coins'])>0: 

        best_directions = bfs_find_nearest_coin(new_game_state, player_pos) #classic bfs algo

        if self_action in best_directions:

            events.append('WALKING_TOWARDS_COIN')
        elif not self_action == 'WAIT' :

            events.append('WALKING_AWAY_FROM_COIN')

    reward = calculate_reward_based_on_events(self, events) #reward calc
    state = state_to_features(old_game_state) #self-explanatory
    new_state = state_to_features(new_game_state) #self-explanatory

    action = ACTIONS.index(self_action)
    self.reward_history.append(reward)
    avg_reward = np.mean(self.reward_history) #& good for visualizations but

    # we only save the model if avg reward is above a "certain" threshold
    if avg_reward > 150: #experimentation
        self.logger.info(f"the avg reward over last 200 steps: {avg_reward}. Model will be saved...")
        torch.save({
            'q_eval_state_dict': self.training_agent.q_eval.state_dict(),
            'q_next_state_dict': self.training_agent.q_next.state_dict(),
            'optimizer_state_dict': self.training_agent.optimizer.state_dict(),
        }, "coin_model_weights.pt")
    self.training_agent.storetransition(state, action, reward, new_state, False) #now: we train our agent based on the transitions stored
    self.training_agent.learn()

# So reaching step 400 results in the game finishing and saving all needed values for the
# Q-learning algorithm
def end_of_round(self, last_game_state, last_action, events):
    """
    this function gets called end of every game for final rewards calc + training
    """
    coins_left = last_game_state['coins']
    if len(last_game_state['coins']) > 0:
        for i in range(len(last_game_state['coins'])):
            events.append('COINS_LEFT_TO_COLLECT')
    steps_left = last_game_state['step']
    if steps_left < 400:

        for i in range(400 - steps_left):
            events.append('COLLECTED_ALL_COINS') 
#possible implementation idea: save only if perf improves based on what the agent tried to do.
    self.logger.debug(f'end of round reached with events: {", ".join(events)}')
    reward = calculate_reward_based_on_events(self, events)
    state = state_to_features(last_game_state)
    action = ACTIONS.index(last_action)
    self.training_agent.storetransition(state, action, reward, None, False)
    self.training_agent.learn()
    torch.save({
        'q_eval_state_dict': self.training_agent.q_eval.state_dict(),
        'q_next_state_dict': self.training_agent.q_next.state_dict(),
        'optimizer_state_dict': self.training_agent.optimizer.state_dict(),
    }, "coin_model_weights.pt")

def learn(self):
    """
    function to define learning - from it's experiences
    """
    self.training_agent.learn()

#
#
# 
#
#
#space for added implementation function
#
#
#
#
#
def calculate_reward_based_on_events(self, events, old_game_state=None):
    """
    function to calculate total reward for all events of the game
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
    self.logger.info(f"the agent gets awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def manhatten_dist(pos1, pos2):
    """
    function that csalculate the manhatten distance between 2 positions
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# :param game_state: state of the game
# :param position: current position of the player or any other coordinate
# output: gives a list of possible directions, to get to the nearest coin. We give a list
#         because there is the possibility, that two coins are the closest, so going into each direction
#         should result in getting points for WALKING_TOWARDS_COIN
def bfs_find_nearest_coin(game_state, position):
    """
    function using breadth-first-search : finding shortest paths - to nearest coins, it returns list of all possible 1st moves towards nearest coins
    """

    start_pos = position #same as defined earlier
    coins = game_state['coins'] 
    if not coins: #no-coin case
        return None 
    if start_pos in coins:
        return []
    #if there are coins on the board, only then will it spend 
    #precious computing resources for computing the first moves
    # we want directions mapping: action name to movement (dx, dy)
    directions = {'UP': (0, 1), 'DOWN': (0, -1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
    reverse_directions = {(0, 1): 'UP', (0, -1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}
    queue = deque([(start_pos, [])])  # to store current position as well as path)
    visited = set([start_pos]) #marking visited
    nearest_coins_moves = []
    shortest_distance = None
    while queue:
        current_pos, path = queue.popleft()
        if current_pos in coins:
            if shortest_distance is None:# for every coin it finds, we must check if its the first coin or if we have another one at the same shortest distance?
                shortest_distance = len(path)  
            if len(path) == shortest_distance:  
                if path:  # if a path exists, only then 
                    first_move = reverse_directions[(path[0][0] - start_pos[0], path[0][1] - start_pos[1])]
                    nearest_coins_moves.append(first_move)
            continue  # no need to explore more
        for direction, (dx, dy) in directions.items():   # "neighbourhood" exploration
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)

            if is_valid_move(game_state, new_pos) and new_pos not in visited:
                visited.add(new_pos)
                queue.append((new_pos, path + [new_pos]))  
    return nearest_coins_moves if nearest_coins_moves else None 


# :param game_state: state of the game
# :param position: current position of the player or any other coordinate
# output: checks, whether there is a wall or crate blokcing the way, resulting in an invalid action
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
