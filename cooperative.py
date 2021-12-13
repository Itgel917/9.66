import math
from state import State
from agent import Agent

## HELPER FUNCTIONS ----------------------------------------------------------------------------
def result_of_actions(state, env, action_1, action_2, w):
    '''
    env - an instance of Environment
    action_1 - action of the agent_1
    action_2 - action of the agent_2
    Returns a list of (new_state, prob, reward) where prob is the probability of being in that state and reward is the reward received from executing the action
    Note: assuming that action_1 and action_2 are allowed actions, meaning that when executed, the resulting state guaranteed to be in the environment
    '''
    rewards = env.reward_positions
    
    old_pos_1 = state.agent_1.agent_position
    old_pos_2 = state.agent_2.agent_position
    
    new_pos_1 = state.agent_1.result_of_action(action_1)
    new_pos_2 = state.agent_2.result_of_action(action_2)
    result = []
    
    #unless the agent wants to stay at its current positions, the action costs -1
    moving_cost = 0 
    if action_1 != (0, 0): 
        moving_cost -= 1 * w
    if action_2 != (0, 0):
        moving_cost -= 1 * (1-w)
    
    if new_pos_1 != new_pos_2:    
        reward = moving_cost
        if new_pos_1 in rewards[1]:
            reward += rewards[1][new_pos_1] * w
        if new_pos_2 in rewards[2]:
            reward += rewards[2][new_pos_2] * (1-w)
            
        result.append(((new_pos_1, new_pos_2), 1, reward))
    
    else:
        reward_1 = moving_cost
        reward_2 = moving_cost
        
        if new_pos_1 in rewards[1]:
            reward_1 += rewards[1][new_pos_1] * w
        if new_pos_2 in rewards[2]:
            reward_2 += rewards[2][new_pos_2] * (1-w)
            
        if new_pos_1 == old_pos_2:
            result.append(((old_pos_1, old_pos_2), 1, moving_cost))
        
        elif new_pos_2 == old_pos_1:
            result.append(((old_pos_1, old_pos_2), 1, moving_cost))
        
        else: 
            result.append(((new_pos_1, old_pos_2), 0.5, reward_1))
            result.append(((old_pos_1, new_pos_2), 0.5, reward_2))
    
    return result

def V_value_finder(V, state):
    '''
    Given state, returns V[state]
    '''
    for each in V:
        # print("each inside V", isinstance(each, dict))
        # if isinstance(each, dict):
            # print("V", type(V))
        if each.check_equal_state(state):
            return V[each]
        
def pi_value_finder(pi, state):
    '''
    Given state, returns pi[state]
    '''
    for each in pi:
        if each.check_equal_state(state):
            return pi[each]

def Q(V, env, state, actions, gamma, w):
    action_1 = actions[0]
    action_2 = actions[1]
    
    # print("state inside Q", state)
            
    initial_V = V
    result = 0
    # print(state.agent_1.agent_position, state.agent_2.agent_position)
    # if isinstance(V, tuple):
        # print("V", V[0] == V[1])
    for new_positions, prob, reward in result_of_actions(state, env, action_1, action_2, w):
        agent_1 = Agent(new_positions[0], 1)
        agent_2 = Agent(new_positions[1], 2)
        new_state = State(agent_1, agent_2)
        result += prob * (reward + gamma * V_value_finder(V, new_state))
    # print("V inside Q", initial_V == V)
    return result

def prob(state, actions, beta, V, env, gamma, w):
    '''
    Computing P(a_1, a_2 | s)
    '''
    return math.exp(beta * Q(V, env, state, actions, gamma, w))

## FINDING INDIVIDUAL POLICIES ----------------------------------------------------------------------------
def individual_policy(state, action, agent_number, env, V, gamma, beta, w):
    '''
    state - current state, an instance of State
    action - actions being considered by the agent with agent_number
    env - an instance of Environment
    Returns pi(s, a)
    '''
    initial_V = V
    agent_1 = state.agent_1
    agent_2 = state.agent_2
    
    # print("state inside individual policy", state)
    
    if agent_number == 1:
        possible_actions = agent_2.get_all_possible_actions(env)
    else:
        possible_actions = agent_1.get_all_possible_actions(env)
    
    result = 0
    for each in possible_actions:
        if agent_number == 1:
            actions = (action, each)
        else:
            actions = (each, action)
        result += prob(state, actions, beta, V, env, gamma, w)
    # print("individual_policy", initial_V == V)
    return result

def individual_action_chooser(state, env, V, gamma, beta, w, return_all=False):
    '''
    Given current state and environment of the game, returns a tuple containing the best actions that should be taken by each player using the group agent.
    return_all = True: returns all the actions that can be taken if there's more than one
    '''
    initial_V = V
    agent_1 = state.agent_1
    agent_2 = state.agent_2
    
    # print("state inside individual action chooser", state)
    
    possible_actions_1 = agent_1.get_all_possible_actions(env)
    possible_actions_2 = agent_2.get_all_possible_actions(env)
    
    best_1 = float("-inf")
    best_action_1 = None
    best_actions_agent_1 = []
    
    for each_1 in possible_actions_1:
        current_1 = individual_policy(state, each_1, 1, env, V, gamma, beta, w)
        if current_1 > best_1:
            best_1 = current_1
            best_action_1 = each_1
            
    for each_1 in possible_actions_1:
        current_1 = individual_policy(state, each_1, 1, env, V, gamma, beta, w)
        if abs(current_1 - best_1) < 1e-3:
            best_actions_agent_1.append(each_1)
    
    best_2 = float("-inf")
    best_action_2 = None
    best_actions_agent_2 = []
    
    for each_2 in possible_actions_2:
        current_2 = individual_policy(state, each_2, 2, env, V, gamma, beta, w)
        if current_2 > best_2:
            best_2 = current_2
            best_action_2 = each_2
    
    for each_2 in possible_actions_2:
        current_2 = individual_policy(state, each_2, 2, env, V, gamma, beta, w)
        if abs(current_2 - best_2) < 1e-3:
            best_actions_agent_2.append(each_2)
    
    if return_all:
        return (best_actions_agent_1, best_actions_agent_2)
    
    # print("individual_action_chooser", initial_V == V)
    return (best_action_1, best_action_2)

## VALUE ITERATION ----------------------------------------------------------------------------
def cooperative_value_iteration(env, w):
    '''
    env - an instance of environment
    w - weight of the agent 1 in the group agent's decision (initially 0.5)
    '''
    gamma = 0.9 #discount
    
    #initialize V values - should be 0 for each state initially
    all_states = env.possible_states()
    V = {state: 0 for state in all_states}
    
    while True:
        new_V = {}
        for state in all_states:
            if env.is_end_state(state):
                new_V[state] = env.get_end_state_reward(state)
            else:
                new_V[state] = max(Q(V, env, state, actions, gamma, w) for actions in state.possible_action_pairs(env))
                
        if max(abs(V[state] - new_V[state]) for state in all_states) < 1e-10:
            break
        
        V = new_V
        
        #Policies
        pi = {} 
        for state in all_states:
            if env.is_end_state(state):
                pi[state] = 'None'
            else:
                pi[state] = max((Q(V, env, state, actions, gamma, w), actions) for actions in state.possible_action_pairs(env))[1]
    
    return V, pi