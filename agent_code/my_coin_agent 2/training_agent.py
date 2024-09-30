import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# ------------------------------
# Transformation Functions
# ------------------------------
def mirror(input_tensor):
    """Mirror the input tensor horizontally."""
    return torch.flip(input_tensor, dims=[-1])

def rotate_90(input_tensor):
    """Rotate the input tensor by 90 degrees."""
    return torch.rot90(input_tensor, 1, dims=[-2, -1])

def rotate_180(input_tensor):
    """Rotate the input tensor by 180 degrees."""
    return torch.rot90(input_tensor, 2, dims=[-2, -1])

def rotate_270(input_tensor):
    """Rotate the input tensor by 270 degrees."""
    return torch.rot90(input_tensor, 3, dims=[-2, -1])

# ------------------------------
# State Preprocessing Functions
# ------------------------------
def state_to_features(game_state: dict) -> torch.Tensor:
    """
    Convert the game state to the input format for the neural network.
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

    # Stack along the first axis for (3, 17, 17) format
    state = np.stack([player_board, coin_board, field], axis=0).astype(np.float32)

    # Normalize the state
    state = (state - state.mean()) / (state.std() + 1e-8)

    # Convert to PyTorch tensor
    state = torch.tensor(state)

    # Add a batch dimension and ensure the format is (NCHW)
    state = state.unsqueeze(0)

    return state

def state_to_features_2(game_state: dict) -> torch.Tensor:
    """
    An alternative state preprocessing function.
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

    # Stack along the first axis for (1, 17, 17) format
    state = np.stack([board], axis=0).astype(np.float32)

    # Normalize the state
    state = (state - state.mean()) / (state.std() + 1e-8)

    # Convert to PyTorch tensor
    state = torch.tensor(state)

    # Add a batch dimension and ensure the format is (NCHW)
    state = state.unsqueeze(0)

    return state

# ------------------------------
# Replay Buffer
# ------------------------------
'''class ReplayBuffer:
    def __init__(self, max_size):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, 7, 17, 17), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 7, 17, 17), dtype=np.float32)
        self.actions_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transitions(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state.cpu().numpy()
        self.actions_memory[index] = action
        self.reward_memory[index] = reward
        if new_state is not None:
            self.new_state_memory[index] = new_state.cpu().numpy()
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = torch.tensor(self.state_memory[batch]).float()
        actions = torch.tensor(self.actions_memory[batch]).long()
        rewards = torch.tensor(self.reward_memory[batch]).float()
        new_states = torch.tensor(self.new_state_memory[batch]).float()
        dones = torch.tensor(self.terminal_memory[batch]).bool()
        return states, actions, rewards, new_states, dones'''

# ------------------------------
# Prioritized Replay Buffer (PER)
# ------------------------------

class PrioritizedReplayBuffer:
    def __init__(self, max_size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, 3, 17, 17), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, 3, 17, 17), dtype=np.float32)
        self.actions_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        # Priorities and settings for PER
        self.priorities = np.zeros((self.mem_size,), dtype=np.float32)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def store_transitions(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state.cpu().numpy()
        self.actions_memory[index] = action
        self.reward_memory[index] = reward
        if new_state is not None:
            self.new_state_memory[index] = new_state.cpu().numpy()
        self.terminal_memory[index] = done

        # Set max priority for new transitions
        self.priorities[index] = self.priorities.max() if self.mem_counter > 0 else 1.0
        self.mem_counter += 1

    def _get_beta(self):
        # Annealing beta over time
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        if self.mem_counter == 0:
            return None
        
        # Get priorities and sampling probabilities
        priorities = self.priorities[:max_mem] ** self.alpha
        probabilities = priorities / priorities.sum()

        # Sample based on priorities
        batch_indices = np.random.choice(max_mem, batch_size, p=probabilities)

        # Retrieve samples
        states = torch.tensor(self.state_memory[batch_indices]).float()
        actions = torch.tensor(self.actions_memory[batch_indices]).long()
        rewards = torch.tensor(self.reward_memory[batch_indices]).float()
        new_states = torch.tensor(self.new_state_memory[batch_indices]).float()
        dones = torch.tensor(self.terminal_memory[batch_indices]).bool()

        # Compute importance-sampling (IS) weights
        beta = self._get_beta()
        self.frame += 1
        weights = (max_mem * probabilities[batch_indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights

        return states, actions, rewards, new_states, dones, torch.tensor(weights).float(), batch_indices

    def update_priorities(self, indices, priorities):
        # Update priorities for the sampled transitions
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


# ------------------------------
# ResidualBlockNetwork
# ------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If dimensions don't match, apply a 1x1 convolution to match input and output sizes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Residual block forward pass
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Adding the residual (identity shortcut)
        out = F.relu(out)
        return out


# ------------------------------
# Dueling Deep Q-Network with LSTM
# ------------------------------
class DuelingDeepQNetwork(nn.Module):
    def __init__(self, fc1_dims, fc2_dims, n_actions=5, input_shape=(3, 17, 17), lstm_hidden_dim=64, lstm_layers=1):
        super(DuelingDeepQNetwork, self).__init__()

        # Use Residual Blocks for the convolutional part
        self.residual_block1 = ResidualBlock(input_shape[0], 32)
        self.residual_block2 = ResidualBlock(32, 64)
        #self.residual_block3 = ResidualBlock(64, 64)

        # Compute the size of the flattened feature map after convolution
        conv_output_size = self._get_conv_output_size(input_shape)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=conv_output_size, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)

        # Fully connected layers after LSTM
        self.fc1 = nn.Linear(lstm_hidden_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)

        # Streams for state value (V) and advantage (A)
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)

    def _get_conv_output_size(self, input_shape):
        """Helper function to calculate the output size after residual blocks."""
        dummy_input = torch.zeros(1, *input_shape)
        out = self.residual_block1(dummy_input)
        out = self.residual_block2(out)
        #out = self.residual_block3(out)
        out = out.view(1, -1)
        return out.size(1)

    def forward(self, state):
        # Pass state through residual blocks
        x = self.residual_block1(state)
        x = self.residual_block2(x)
        #x = self.residual_block3(x)

        # Flatten the output
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Reshape for LSTM: (batch_size, seq_len=1, input_dim)
        x = x.unsqueeze(1)

        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Get the last output

        # Fully connected layers
        x = torch.relu(self.fc1(lstm_out))
        x = torch.relu(self.fc2(x))

        # Compute V and A
        V = self.V(x)
        A = self.A(x)

        # Compute Q values
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


# ------------------------------
# Symmetric Q-Network for Data Augmentation
# ------------------------------
class SymmetricQNetwork(nn.Module):
    def __init__(self, fc1_dims, fc2_dims, n_actions, input_shape=(3, 17, 17)):
        super(SymmetricQNetwork, self).__init__()
        self.q_network = DuelingDeepQNetwork(fc1_dims, fc2_dims, n_actions, input_shape)

        # Action mapping to handle rotations/mirroring for each action
        # Assuming 8 transformations: original, mirror, rotate_90, mirror+rotate_90, rotate_180, mirror+rotate_180, rotate_270, mirror+rotate_270
        self.mapping = torch.tensor([
            [0, 0, 3, 3, 2, 2, 1, 1],  # Action 0
            [1, 3, 0, 2, 3, 1, 2, 0],  # Action 1
            [2, 2, 1, 1, 0, 0, 3, 3],  # Action 2
            [3, 1, 2, 0, 1, 3, 0, 2],  # Action 3
            [4, 4, 4, 4, 4, 4, 4, 4]  # Action 4 (Invariant)
            #[5, 5, 5, 5, 5, 5, 5, 5]   # Action 5 (Invariant)
        ])

    def forward(self, state):
        # Apply transformations
        states = [
            state,  # Original
            mirror(state),  # Mirrored
            rotate_90(state),  # 90-degree rotation
            mirror(rotate_90(state)),  # Mirrored after 90-degree rotation
            rotate_180(state),  # 180-degree rotation
            mirror(rotate_180(state)),  # Mirrored after 180-degree rotation
            rotate_270(state),  # 270-degree rotation
            mirror(rotate_270(state))  # Mirrored after 270-degree rotation
        ]

        # Stack transformations
        states = torch.cat(states, dim=0)

        # Pass through shared Q-network
        q_values = self.q_network(states)

        # Separate q-values for each transformation
        q_values = q_values.view(8, state.size(0), -1)  # (8, batch_size, n_actions)

        # Average the Q-values, considering action rotations
        q_output = []
        for action_idx in range(self.q_network.A.out_features):
            q_values_for_action = [q_values[i][:, self.mapping[action_idx, i]] for i in range(8)]
            q_mean_for_action = torch.mean(torch.stack(q_values_for_action, dim=-1), dim=-1, keepdim=True)
            q_output.append(q_mean_for_action)

        # Concatenate the results back into the final Q-value predictions
        q_output = torch.cat(q_output, dim=-1)
        return q_output

# ------------------------------
# Training Agent Class
# ------------------------------
class Training_Agent:
    def __init__(self, learning_rate, gamma, epsilon, batch_size,
                 n_actions, eps_decay, eps_min, mem_size=100000,
                 fc1_dims=32, fc2_dims=32, replace=1000, tau=0.001):
        self.n_actions = n_actions
        self.action_space = list(range(n_actions))
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.replace = replace
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.tau = tau

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.q_eval = SymmetricQNetwork(fc1_dims, fc2_dims, n_actions).to(self.device)
        self.q_next = SymmetricQNetwork(fc1_dims, fc2_dims, n_actions).to(self.device)
        self.q_next.load_state_dict(self.q_eval.state_dict())
        self.q_next.eval()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        # Replay buffer
        self.memory = PrioritizedReplayBuffer(mem_size)

        # TensorBoard writer
        self.writer = SummaryWriter()

        self.reward_history = []

        # Logger setup
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler('training_agent.log')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transitions(state, action, reward, new_state, done)
        #Save rewards for training purpose
        self.reward_history.append(reward)
        if len(self.reward_history) > 20:
            self.reward_history.pop(0)

    def choose_action(self, state):
        state = state.to(self.device)
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            with torch.no_grad():
                q_values = self.q_eval(state)
                action = torch.argmax(q_values, dim=1).item()
        return action

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        # Replace target network
        if self.learn_step_counter % self.replace == 0:
            self.soft_update(self.q_eval, self.q_next)

        # Sample batch
        states, actions, rewards, new_states, dones, *_ = self.memory.sample_buffer(self.batch_size)

        # Convert to tensors
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        new_states = new_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values
        q_pred = self.q_eval(states)
        q_pred = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q-values using Double DQN
        with torch.no_grad():
            q_eval_next = self.q_eval(new_states)
            max_actions = torch.argmax(q_eval_next, dim=1)
            q_next = self.q_next(new_states).gather(1, max_actions.unsqueeze(1)).squeeze(1)
            q_target = rewards + self.gamma * q_next * (~dones)

        # Compute loss
        loss = self.loss(q_pred, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss/train', loss.item(), self.learn_step_counter)
        self.writer.add_scalar('Q-values/max', q_pred.max().item(), self.learn_step_counter)
        self.writer.add_scalar('Q-values/mean', q_pred.mean().item(), self.learn_step_counter)
        self.writer.add_scalar('Epsilon', self.epsilon, self.learn_step_counter)
        
        avg_reward = calculate_average_reward(self.reward_history, 50)
        self.writer.add_scalar('Avg. reward over 20 steps', avg_reward, self.learn_step_counter)

        # Increment step counter
        self.learn_step_counter += 1

    def soft_update(self, source, target):
        """Soft update model parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

def calculate_average_reward(reward_list, N):
    # Check if the list has at least 20 elements
    if len(reward_list) >= N:
        last_20_rewards = reward_list[-N:]  # Get the last 20 rewards
    else:
        last_20_rewards = reward_list  # If less than 20 rewards, take the whole list
    
    # Calculate the mean of the last 20 rewards
    average_reward = sum(last_20_rewards) / len(last_20_rewards)
    
    return average_reward
