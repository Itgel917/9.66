from environment import Environment
from cooperative import cooperative_value_iteration, individual_action_chooser, pi_value_finder
from agent import Agent
from state import State

gamma = 0.9
beta = 4
w = 0.5

# #Environment 1

# cell_positions = [(4, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (4, 2)]
# reward_positions = {1: {(4, 0): 10, (0, 1):20}, 2: {(8, 1):10, (4, 2):10}}
# env = Environment(cell_positions, reward_positions)

# agent_1 = Agent((3, 1), 1)
# agent_2 = Agent((5, 1), 2)

# Environment 2
cell_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1), (2, 2), (3, 0), (3, 1), (4, 1), (4, 2)]
reward_positions = {1: {(1, 0): 10}, 2: {(3, 0): 10}}
env = Environment(cell_positions, reward_positions)

agent_1 = Agent((3, 1), 1)
agent_2 = Agent((1, 1), 2)

V, pi = cooperative_value_iteration(env, w)
current_state = State(agent_1, agent_2) #initial state

# print(V.values())

while True:    
    print(current_state.agent_1.agent_position, current_state.agent_2.agent_position)
    
    if env.is_end_state(current_state):
        break  
      
    action_1, action_2 = individual_action_chooser(current_state, env, V, gamma, beta, w)

    new_positions = current_state.apply_action_pairs(action_1, action_2)
    current_state.agent_1.update_position(new_positions[0])
    current_state.agent_2.update_position(new_positions[1])