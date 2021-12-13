from environment import Environment
from agent import Agent
from state import State
from cooperative import cooperative_value_iteration, individual_action_chooser, pi_value_finder, result_of_actions
from level_1_agent import lvl_1_action_chooser, lvl_1_value_iteration


def prob_observation_given_intention(intention, observation, state, agent, env, w, gamma, beta):
    '''
    intention - string, either COOPERATIVE or COMPETITIVE
    observation - actions executed by the agents
    agent - an instance of Agent
    state - an instance of State, current state of the game
    env - an instance of Environment, representing the game field
    Returns the probability of the observation happening given the intention
    '''    
    agent_1 = state.agent_1
    agent_2 = state.agent_2
    
    # print("state inside obs given int", state)
    
    coop_V, pi = cooperative_value_iteration(env, w)
    if agent.agent_number == 1:
        comp_V = lvl_1_value_iteration(env, agent_1, agent_2, beta)
        other_agent = agent_2
    else:
        comp_V = lvl_1_value_iteration(env, agent_2, agent_1, beta)
        other_agent = agent_1
    
    # best_actions_1, best_actions_2 = individual_action_chooser(state, env, coop_V, gamma, beta, w, True)
    
    # print(best_actions_1)
    
    best_actions_comp = lvl_1_action_chooser(agent, env, comp_V, gamma, beta, other_agent, True)
    
    if intention == "COOPERATIVE":
        best_actions = [pi_value_finder(pi, state)]
        # if agent.agent_number == 1:
        #     best_actions = best_actions_1
        # else:
        #     best_actions = best_actions_2
        
        if observation in best_actions:
            return 1/len(best_actions)
        else:
            return 0    
        
    else:
        if observation in best_actions_comp:
            return 1/len(best_actions_comp)
        else:
            return 0

def prob_intention_given_observation(observations, env, w, gamma, beta, state):
    '''
    observations - actions of the agents
    Returns the probability of the intentions (being either COMPETITIVE or COOPERATIVE) of the agents given observations
    '''
    agent_1 = state.agent_1
    agent_2 = state.agent_2
    
    # print("state inside int given obs", state)
    
    prob_agent_1_coop= 0.5 * prob_observation_given_intention("COOPERATIVE", observations[0], state, agent_1, env, w, gamma, beta)
    
    prob_agent_1_comp = 0.5 * prob_observation_given_intention("COMPETITIVE", observations[0], state, agent_1, env, w, gamma, beta)
    
    prob_agent_2_coop = 0.5 * prob_observation_given_intention("COOPERATIVE", observations[1], state, agent_2, env, w, gamma, beta)
    
    prob_agent_2_comp = 0.5 * prob_observation_given_intention("COMPETITIVE", observations[1], state, agent_2, env, w, gamma, beta)
    
    return (prob_agent_1_coop, prob_agent_1_comp, prob_agent_2_coop, prob_agent_2_comp)


## TESTING ----------------------------------------------------------------------
if __name__ == '__main__':
    cell_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1), (2, 2), (3, 0), (3, 1), (4, 1), (4, 2)]
    reward_positions = {1: {(1, 0): 10}, 2: {(3, 0): 10}}
    env = Environment(cell_positions, reward_positions)

    agent_1 = Agent((3, 1), 1)
    agent_2 = Agent((1, 1), 2)
    
    state = State(agent_1, agent_2)
    observations = [(-1, 0), (0, 0)]
    
    # print("state inside main", state)
    
    w = 0.5
    beta = 4
    gamma = 0.9
    
    result = prob_intention_given_observation(observations, env, w, gamma, beta, state)
    print(result)