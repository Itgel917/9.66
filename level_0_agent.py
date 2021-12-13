import math
from state import State
from agent import Agent

## HELPER FUNCTIONS ----------------------------------------------------------------------------
def lvl_0_result_of_action(env, action, agent):
    '''
    env - an instance of Environment
    action - action of the given agent
    Returns a list of (new_position, prob, reward) where prob is the probability of being in that state and reward is the reward received from executing the action
    Note: assuming that action_1 and action_2 are allowed actions, meaning that when executed, the resulting state guaranteed to be in the environment
    '''
    rewards = env.reward_positions[agent.agent_number]
    
    new_pos = agent.result_of_action(action)
    reward = 0
    
    if new_pos in rewards:
        reward += rewards[new_pos]
    return (new_pos, 1, reward)

def lvl_0_Q(V, env, agent, action, gamma):
    new_position, prob, reward = lvl_0_result_of_action(env, action, agent)
    result = prob * (reward + gamma * V[new_position])
    return result

def lvl_0_prob(agent, action, beta, V, env, gamma):
    '''
    Computing P(a | s)
    '''
    return math.exp(beta * lvl_0_Q(V, env, agent, action, gamma))

def lvl_0_scale(agent, action, beta, V, env, gamma):
    all_actions = agent.get_all_possible_actions(env)
    total = 0
    for each in all_actions:
        current = lvl_0_prob(agent, each, beta, V, env, gamma)
        total += current
        if each == action:
            action_prob = current
    return action_prob/total

## FINDING INDIVIDUAL POLICIES ----------------------------------------------------------------------------
def lvl_0_action_chooser(agent, env, V, gamma, beta):
    '''
    Given current state and environment of the game, returns a tuple containing the best actions that should be taken by each player using the group agent.
    '''
    possible_actions = agent.get_all_possible_actions(env)
    
    best = float("-inf")
    best_action = None
    for each in possible_actions:
        current = lvl_0_scale(agent, each, beta, V, env, gamma)
        # print("action chooser", each, current)
        if current > best:
            best = current
            best_action = each
            
    return best_action

## VALUE ITERATION ----------------------------------------------------------------------------
def lvl_0_value_iteration(env, agent):
    '''
    env - an instance of environment
    '''
    gamma = 0.9 #discount
    
    #initialize V values - should be 0 for each state initially
    all_states = env.possible_positions_single_agent()
    V = {state: 0 for state in all_states}
    
    while True:
        new_V = {}
        for state in all_states:
            new_agent = Agent(state, agent.agent_number)
            if env.is_end_state_single_agent(state, new_agent):
                new_V[state] = env.get_end_state_reward_single_agent(state, new_agent)
            else:
                new_V[state] = max(lvl_0_Q(V, env, new_agent, action, gamma) for action in new_agent.get_all_possible_actions(env))
                
        if max(abs(V[state] - new_V[state]) for state in all_states) < 1e-10:
            break
        
        V = new_V
    
    return V