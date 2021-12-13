from environment import Environment
from level_1_agent import lvl_1_value_iteration, lvl_1_action_chooser
from level_0_agent import lvl_0_value_iteration, lvl_0_action_chooser
from agent import Agent
from state import State

gamma = 0.9
beta = 4

# Environment 1
cell_positions = [(4, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (4, 2)]
reward_positions = {1: {(4, 0): 10, (0, 1): 10}, 2: {(8, 1): 10, (4, 2): 10}}
env = Environment(cell_positions, reward_positions)

agent_1 = Agent((3, 1), 1)
agent_2 = Agent((5, 1), 2)

# # Environment 2
# cell_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1), (2, 2), (3, 0), (3, 1), (4, 1), (4, 2)]
# reward_positions = {1: {(1, 0): 10}, 2: {(3, 0): 10}}
# env = Environment(cell_positions, reward_positions)

# agent_1 = Agent((3, 1), 1)
# agent_2 = Agent((1, 1), 2)

# # Environment 3
# cell_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
# reward_positions = {1: {(1, 1): 10}, 2: {(0, 1): 10}}
# env = Environment(cell_positions, reward_positions)

# agent_1 = Agent((0, 1), 1)
# agent_2 = Agent((1, 1), 2)

V_agent_1= lvl_1_value_iteration(env, agent_1, agent_2, beta)
V_agent_2 = lvl_1_value_iteration(env, agent_2, agent_1, beta)

# V_agent_1= lvl_0_value_iteration(env, agent_1)
# V_agent_2 = lvl_0_value_iteration(env, agent_2)

current_state = State(agent_1, agent_2) #initial state

# print(V_agent_1)
# print(V_agent_2)

while True:    
    print(current_state.agent_1.agent_position, current_state.agent_2.agent_position)
    
    if env.is_end_state(current_state):
        break  
      
    action_1 = lvl_1_action_chooser(agent_1, env, V_agent_1, gamma, beta, agent_2)
    action_2 = lvl_1_action_chooser(agent_2, env, V_agent_2, gamma, beta, agent_1)

    # action_1 = lvl_0_action_chooser(agent_1, env, V_agent_1, gamma, beta)
    # action_2 = lvl_0_action_chooser(agent_2, env, V_agent_2, gamma, beta)
    
    agent_1_new_pos, agent_2_new_pos = current_state.apply_action_pairs(action_1, action_2)
    
    current_state.agent_1.update_position(agent_1_new_pos)
    current_state.agent_2.update_position(agent_2_new_pos)