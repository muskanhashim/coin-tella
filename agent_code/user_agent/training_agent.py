import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

#we want to apply transformations to input tensor and use Pytorch for the same.
def mirror(input_tensor):
    """function that would mirror input tensor : horizontally..."""
    return torch.flip(input_tensor, dims=[-1])
def rotate_90(input_tensor):
    """function to rotate input tensor by : 90 degrees..."""
    return torch.rot90(input_tensor, 1, dims=[-2, -1])

def rotate_180(input_tensor):
    """function to rotate input tensor by : 180 degrees..."""
    return torch.rot90(input_tensor, 2, dims=[-2, -1])
def rotate_270(input_tensor):
    """function thst would rotate input tensor by : 270 degrees..."""
    return torch.rot90(input_tensor, 3, dims=[-2, -1])
#now we define functions for state preprocessing.
def state_to_features(game_state: dict) -> torch.Tensor:
    """
    function that helps convert the game state to input format for our neural network
    """
    if game_state is None:
        return None

    field = game_state['field']
    player_pos = game_state['self'][3]
    player_board = np.zeros((17, 17))
    player_board[player_pos[0]][player_pos[1]] = 1

    coin_board = np.zeros((17, 17))
    if len(game_state['coins']):

        for coin in game_state['coins']:

            coin_board[coin] = 1
            #here first, we stack along the 1st axis, then normalize state, convert it to pytorch tensor and then add a batch dimension
            #fornmat is NCHW is ensured.
    state = np.stack([player_board, coin_board, field], axis=0).astype(np.float32) 
    state = (state - state.mean()) / (state.std() + 1e-8)
    state = torch.tensor(state)
    state = state.unsqueeze(0)
    return state

def state_to_features_2(game_state: dict) -> torch.Tensor: #experimenting with difference in implemetations
    """
    An alternative state preprocessing function?
    """
    if game_state is None:
        return None


    field = game_state['field']
    player_pos = game_state['self'][3]
    board = np.zeros((17, 17))
    board[player_pos[0]][player_pos[1]] = 1
    board[field == -1] = -1
    for coin in game_state['coins']:
        board[coin] = 2
    if len(game_state['bombs']) > 0:
        for bomb in game_state['bombs']:
            bomb_pos, _ = bomb
            board[bomb_pos] = 5

    # same as earlier - stacking along the 1st axis but in (1, 17, 17) format, normalizing->pytorchtensor->adding bacth dim+format
    state = np.stack([board], axis=0).astype(np.float32)
    state = (state - state.mean()) / (state.std() + 1e-8)

    state = torch.tensor(state)


    state = state.unsqueeze(0)

    return state

#for replay buffer class
'''class ReplayBuffer:
    def __init__(self, maxsize):
            self.mem_size = maxsize
            self.mem_counter = 0
                self.actions_memory = np.zeros(self.mem_size, dtype=np.int32)
            self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
            self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
            self.state_memory = np.zeros((self.mem_size, 7, 17, 17), dtype=np.float32)
            self.new_state_memory = np.zeros((self.mem_size, 7, 17, 17), dtype=np.float32)
def storetransitions(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state.cpu().numpy()
        self.actions_memory[index] = action
        self.reward_memory[index] = reward

        if new_state is not None:
        
            self.new_state_memory[index] = new_state.cpu().numpy()
        self.terminal_memory[index] = done
        self.mem_counter += 1
def samp_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = torch.tensor(self.state_memory[batch]).float()
                actions = torch.tensor(self.actions_memory[batch]).long()
                rewards = torch.tensor(self.reward_memory[batch]).float()
                new_states = torch.tensor(self.new_state_memory[batch]).float()

                dones = torch.tensor(self.terminal_memory[batch]).bool()
                return states, actions, rewards, new_states, dones'''

#for prioritized replay buffer?
# why we need it: it would initialize buffer with a max size to simulate experiences 
# from random actions in environment
#
#

class prioritized_replaybuffer: # we initialize the PER with given size and parameters for alpha und beta

    def __init__(self, max_size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, 3, 17, 17), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 3, 17, 17), dtype=np.float32)
        self.actions_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
#here we set priorities for the PER
        self.priorities = np.zeros((self.mem_size,), dtype=np.float32)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def storetransitions(self, state, action, reward, new_state, done): #we store a transition in the buffer and update priority for it

        
        index = self.mem_counter % self.mem_size
        self.actions_memory[index] = action
        self.reward_memory[index] = reward
        self.state_memory[index] = state.cpu().numpy()
#for new transitions we need to set the maximum prioriety
        if new_state is not None:
            self.new_state_memory[index] = new_state.cpu().numpy()
        self.terminal_memory[index] = done

        
        self.priorities[index] = self.priorities.max() if self.mem_counter > 0 else 1.0
        self.mem_counter += 1

    def _getbeta(self): 
        """this function will calculate current beta value for IS weights"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def samp_buffer(self, batch_size): # we get priorities and sampling probabilities. we sample based on the priorities we set.
        #then we retreive the samples.
        #we also calculate IS weights which refer to importance-sampling
        #later we will need to also update
        max_mem = min(self.mem_counter, self.mem_size)
        if self.mem_counter == 0:
            return None
        priorities = self.priorities[:max_mem] ** self.alpha
        probabilities = priorities / priorities.sum()
        batch_indices = np.random.choice(max_mem, batch_size, p=probabilities)
        states = torch.tensor(self.state_memory[batch_indices]).float()
        actions = torch.tensor(self.actions_memory[batch_indices]).long()
        rewards = torch.tensor(self.reward_memory[batch_indices]).float()
        new_states = torch.tensor(self.new_state_memory[batch_indices]).float()
        dones = torch.tensor(self.terminal_memory[batch_indices]).bool()
        beta = self._getbeta()
        self.frame += 1
        weights = (max_mem * probabilities[batch_indices]) ** (-beta)
        weights /= weights.max()  

        return states, actions, rewards, new_states, dones, torch.tensor(weights).float(), batch_indices

    def updatepriorities(self, indices, priorities): #based on new td errors
#for sampled transitions
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority



class residualblock(nn.Module):
#now we define residualblocknetwork. in case our dims dont match we apply convolution for matching the input sizes with the output
# and batch normalization 
    def __init__(self, in_channels, out_channels, stride=1):

        super(residualblock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)


        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(


                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),

                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
#with possible shortcut connection
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  ##
        out = F.relu(out)
        return out


#add and work on dueling-deep-q-network
# now we set up the dueling deep Q-Net with residual blocks, LSTM layers (#experimented and tested), separate value and advantage streams"""
class DuelingDeepQNetwork(nn.Module):
#explanation for our choices: we use residual blocks for the conv part, lstm layers and fully connected layers after. V & A streams
    def __init__(self, fc1_dims, fc2_dims, n_actions=5, input_shape=(3, 17, 17), lstm_hidden_dim=64, lstm_layers=1):
        super(DuelingDeepQNetwork, self).__init__()

        self.residual_block1 = residualblock(input_shape[0], 32)
        self.residual_block2 = residualblock(32, 64)
        '''self.residual_block3 = ResidualBlock(64, 64)'''
        
        conv_output_size = self._get_conv_output_size(input_shape) # so we compute size of flattened feature map after convolution
        # LSTM layers apply
        self.lstm = nn.LSTM(input_size=conv_output_size, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
    
        self.fc1 = nn.Linear(lstm_hidden_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions) # <- a-stream & ^ v-stream

#define fn for size of conv output for LSTM input

    def _get_conv_output_size(self, input_shape): 
        """ func to calculate output size after residual blocks"""
        dummy_input = torch.zeros(1, *input_shape)

        out = self.residual_block1(dummy_input)

        out = self.residual_block2(out)

        #out = self.residual_block3(out)

        out = out.view(1, -1)

        return out.size(1)

    def forward(self, state):

        # we pass state thu RBs
        # flatten output
        # reshape for LSTM with (batch_size, seq_len=1, input_dim)
         # fully connected layers
         # to compute V and A
         # to compute Q -values
        x = self.residual_block1(state)
        x = self.residual_block2(x)
        '''x = self.residual_block3(x)'''

        
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        
        x = x.unsqueeze(1)

        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  #  last output

       
        x = torch.relu(self.fc1(lstm_out))
        x = torch.relu(self.fc2(x))

        
        V = self.V(x)
        A = self.A(x)

        
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q

#symmetric Q-Net with data augmentation mappings for different transformations
class SymmetricQNetwork(nn.Module):
    def __init__(self, fc1_dims, fc2_dims, n_actions, input_shape=(3, 17, 17)):
        super(SymmetricQNetwork, self).__init__()
        self.q_network = DuelingDeepQNetwork(fc1_dims, fc2_dims, n_actions, input_shape)

        # action mapping will help us handleing rotations/mirroring for each action
        # here we assume 8 transformations: original, mirror, rotate_90, mirror+rotate_90, rotate_180, mirror+rotate_180, rotate_270, mirror+rotate_270
        self.mapping = torch.tensor([
            [0, 0, 3, 3, 2, 2, 1, 1],  # for action 0, 1, 2, 3, ,4 ....
            [1, 3, 0, 2, 3, 1, 2, 0],  
            [2, 2, 1, 1, 0, 0, 3, 3],  
            [3, 1, 2, 0, 1, 3, 0, 2], 
            [4, 4, 4, 4, 4, 4, 4, 4]  
            #[5, 5, 5, 5, 5, 5, 5, 5]   
        ])

    def forward(self, state):
        '''Forward pass applies all transformations to state-> gets Q-values-> averages them considering action mappings'''
        states = [
            state,  # original
            mirror(state),  # mirrored
            rotate_90(state),  # 90degree rotation
            mirror(rotate_90(state)),  # mirrored after 90degree rotation
            rotate_180(state),  # 180degree rotation
            mirror(rotate_180(state)),  # mirrored after 180degree rotation
            rotate_270(state),  # 270-degree rotation
            mirror(rotate_270(state))  #  mirrored after 270degree rotation
        ]

  #we stack transformations, pass through the q-network, and sepeate q-values for each
        states = torch.cat(states, dim=0)
        q_values = self.q_network(states)
        q_values = q_values.view(8, state.size(0), -1)  #(8, batch_size, n_actions)
        q_output = []
        for action_idx in range(self.q_network.A.out_features):
            q_values_for_action = [q_values[i][:, self.mapping[action_idx, i]] for i in range(8)]
            q_mean_for_action = torch.mean(torch.stack(q_values_for_action, dim=-1), dim=-1, keepdim=True)
            q_output.append(q_mean_for_action)
        # back into final Q-value predictions
        q_output = torch.cat(q_output, dim=-1)
        return q_output
class Training_Agent:
    '''to initialize training agent with networks, optimizer, 
    replay buffer... & other hyperparameters'''
    def __init__(self, lr_rate, gamma, epsilon, batch_size,
                 n_actions, epsdecay, eps_min, mem_size=100000,
                 fc1_dims=32, fc2_dims=32, replace=1000, tau=0.001):
        self.n_actions = n_actions
        self.action_space = list(range(n_actions))
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsdecay = epsdecay
        self.eps_min = eps_min
        self.replace = replace
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #cudahelps
        self.q_eval = SymmetricQNetwork(fc1_dims, fc2_dims, n_actions).to(self.device)
        self.q_next = SymmetricQNetwork(fc1_dims, fc2_dims, n_actions).to(self.device)
        self.q_next.load_state_dict(self.q_eval.state_dict())
        self.q_next.eval()


        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=lr_rate)
        self.loss = nn.MSELoss()


        self.memory = prioritized_replaybuffer(mem_size)
#tensorboard integration for better monitoring!

        self.writer = SummaryWriter()

        self.reward_history = []

#for logging info- please ignore
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler('training_agent.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def storetransition(self, state, action, reward, new_state, done):
        '''store transition in replay buffer & keep track of recently received rewards'''
        self.memory.storetransitions(state, action, reward, new_state, done)
        #so as to save rewards for training 
        self.reward_history.append(reward)
        if len(self.reward_history) > 20:
            self.reward_history.pop(0)

    def choose_action(self, state):
        '''choose action based on epsilon greedy policy. so we choose randomly or one with best q-value'''
        state = state.to(self.device)
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            with torch.no_grad():
                q_values = self.q_eval(state)
                action = torch.argmax(q_values, dim=1).item()
        return action

    def learn(self):
        """to  perform a learning step. we sample from buffer, compute loss, backpropagate then update networks"""
        if self.memory.mem_counter < self.batch_size:
            return
        if self.learn_step_counter % self.replace == 0:
            self.soft_update(self.q_eval, self.q_next)
        states, actions, rewards, new_states, dones, *_ = self.memory.samp_buffer(self.batch_size)

        # converting to tensors...
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        new_states = new_states.to(self.device)
        dones = dones.to(self.device)

        # current val
        q_pred = self.q_eval(states)
        q_pred = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():# using double DQN :
            q_eval_next = self.q_eval(new_states)
            max_actions = torch.argmax(q_eval_next, dim=1)
            q_next = self.q_next(new_states).gather(1, max_actions.unsqueeze(1)).squeeze(1)
            q_target = rewards + self.gamma * q_next * (~dones)

        loss  = self.loss(q_pred, q_target) #for loss computing
        self.optimizer.zero_grad()
        loss.backward() #"""backpropagation"""
        torch.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 1.0)  #"""for gradient clipping"""
        self.optimizer.step()
        self.epsilon = max(self.eps_min, self.epsilon * self.epsdecay)# """and updating epsilon"""

        # code for tensorboard integration...
        self.writer.add_scalar('Loss/train', loss.item(), self.learn_step_counter)
        self.writer.add_scalar('Q-values/max', q_pred.max().item(), self.learn_step_counter)
        self.writer.add_scalar('Q-values/mean', q_pred.mean().item(), self.learn_step_counter)
        self.writer.add_scalar('Epsilon', self.epsilon, self.learn_step_counter)
        
        avg_reward = calculate_average_reward(self.reward_history, 50)
        self.writer.add_scalar('Avg. reward over 20 steps', avg_reward, self.learn_step_counter)
        self.learn_step_counter += 1

    def soft_update(self, source, target):
        """so as to soft update model parameters - stablility."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

def calculate_average_reward(reward_list, N):
    """to calculate avg reward from last N rewards in the reward list"""
    if len(reward_list) >= N:
        last_20_rewards = reward_list[-N:]
    else:
        last_20_rewards = reward_list  #we take the whole list if there are < than 20 rewards
    average_reward = sum(last_20_rewards) / len(last_20_rewards) #mean ^
    return average_reward
