import math
from agent import Agent
from level_0_agent import lvl_0_value_iteration, lvl_0_Q, lvl_0_scale


## HELPER FUNCTIONS ----------------------------------------------------------------------------
def prob(agent, action, other_agent, other_action):
    '''
    P(s' | s, a_i, a_{-i})
    '''
    agent_new_pos = agent.result_of_action(action)
    other_agent_new_pos = other_agent.result_of_action(other_action)
    
    if agent_new_pos == other_agent_new_pos:
        if other_action == (0, 0):
            return 0
        elif action == (0, 0):
            return 1
        else:
            return 0.5
    else:
        return 1

def prob_iteration(agent, action, other_agent, beta, gamma, env):
    '''
    agent - an instance of Agent, representing the main agent being considered
    action - the action of the main agent
    other_agent - an instance of Agent, representing the agent in the game
    P(s' | s, a) for Level 1 Agent using Level 0 Agent 
    '''
    result = 0
    other_agent_pos_actions = other_agent.get_all_possible_actions(env)
    V = lvl_0_value_iteration(env, other_agent)
    for other_action in other_agent_pos_actions:
        prob_1 = lvl_0_scale(other_agent, other_action, beta, V, env, gamma) #P(a_{-i} | s, k = 0)
        prob_2 = prob(agent, action, other_agent, other_action)
        # print("checking", prob_1 < 1)
        result += prob_1 * prob_2
    # print("result", result)
    return result
    
def lvl_1_Q(V, env, agent, action, gamma, other_agent, beta):
    return lvl_0_Q(V, env, agent, action, gamma) * prob_iteration(agent, action, other_agent, beta, gamma, env)

def lvl_1_prob(agent, action, beta, V, env, gamma, other_agent):
    '''
    Computing P(a | s)
    '''
    return math.exp(beta * lvl_1_Q(V, env, agent, action, gamma, other_agent, beta))

def lvl_1_scale(agent, action, beta, V, env, gamma, other_agent):
    all_actions = agent.get_all_possible_actions(env)
    total = 0
    for each in all_actions:
        current = lvl_1_prob(agent, each, beta, V, env, gamma, other_agent)
        total += current
        if each == action:
            action_prob = current
    return action_prob/total

## FINDING INDIVIDUAL POLICIES ----------------------------------------------------------------------------
def lvl_1_action_chooser(agent, env, V, gamma, beta, other_agent, return_all=False):
    '''
    Given current state and environment of the game, returns a tuple containing the best actions that should be taken by each player using the group agent.
    '''
    possible_actions = agent.get_all_possible_actions(env)
    
    best = float("-inf")
    best_action = None
    best_actions = []
    
    for each in possible_actions:
        current = lvl_1_scale(agent, each, beta, V, env, gamma, other_agent)
        if current > best:
            best = current
            best_action = each
            
    for each in possible_actions:
        current = lvl_1_scale(agent, each, beta, V, env, gamma, other_agent)
        if abs(current - best) < 1e-3:
            best_actions.append(each)
    
    if return_all:
        return best_actions
            
    return best_action

## VALUE ITERATION ----------------------------------------------------------------------------
def lvl_1_value_iteration(env, agent, other_agent, beta):
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
                new_V[state] = max(lvl_1_Q(V, env, new_agent, action, gamma, other_agent, beta) for action in new_agent.get_all_possible_actions(env))
                
        if max(abs(V[state] - new_V[state]) for state in all_states) < 1e-10:
            break
        
        V = new_V
    
    return V