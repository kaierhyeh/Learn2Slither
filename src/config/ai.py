"""AI and Q-learning configuration hyperparameters."""

# Learning hyperparameters
DEFAULT_ALPHA = 0.2     # Learning rate (step size)
DEFAULT_GAMMA = 0.9     # Discount factor for future rewards
DEFAULT_EPSILON = 0.2   # Exploration probability (epsilon-greedy)

# Reward definitions
REWARD_GREEN_APPLE = 10.0
REWARD_RED_APPLE = -10.0
REWARD_STEP = -0.1
REWARD_COLLISION = -50.0
