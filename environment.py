from agent import Agent
from state import State


class Environment():
    def __init__(self, cell_positions, reward_positions):
        '''
        cell_positions - list of tuples, each representing coordinate a cell in the environment
        agent_positions - dictionary mapping the agent number to its initial location
        reward_positions - dictionary mapping the agent number to another dictionary, which maps the position of the reward to the amount of the reward
        '''
        self.all_positions = cell_positions
        self.reward_positions = reward_positions
    
    def is_valid_cell(self, coordinate):
        '''
        coordinate - a tuple representing a random coordinate
        Returns True if the coordinate is in the environment, False otherwise
        '''
        all_cells = self.all_positions
        return coordinate in all_cells
    
    def possible_states(self):
        '''
        Returns all the possible states in the environment
        '''
        all_cells = self.all_positions
        result = []
        for pos_1 in all_cells:
            for pos_2 in all_cells:
                if pos_1 != pos_2:
                    agent_1 = Agent(pos_1, 1)
                    agent_2 = Agent(pos_2, 2)
                    state = State(agent_1, agent_2)
                    result.append(state)
        return result
    
    def possible_positions_single_agent(self):
        '''
        Returns all the possible states for level-0 Agent in the environment
        '''
        return self.all_positions
    
    def is_end_state(self, state):
        '''
        Returns True if the state is end state, False otherwise.
        '''
        rewards_1 = self.reward_positions[1]
        rewards_2 = self.reward_positions[2]
        
        state_agent_1 = state.agent_1.agent_position
        state_agent_2 = state.agent_2.agent_position
        
        if state_agent_1 in rewards_1 or state_agent_2 in rewards_2:
            return True
        return False
    
    def is_end_state_single_agent(self, state, agent):
        '''
        state - tuple representing position of the given agent
        Returns True if the state is end state, False otherwise
        '''
        rewards = self.reward_positions[agent.agent_number]
        if state in rewards:
            return True
        return False
    
    def get_end_state_reward(self, end_state):
        '''
        Returns the reward associated with the given end state
        '''
        rewards_1 = self.reward_positions[1]
        rewards_2 = self.reward_positions[2]
        
        state_agent_1 = end_state.agent_1.agent_position
        state_agent_2 = end_state.agent_2.agent_position
        
        reward = 0
        
        if state_agent_1 in rewards_1:
            reward += rewards_1[state_agent_1]
        
        if state_agent_2 in rewards_2:
            reward += rewards_2[state_agent_2]
        
        return reward
    
    def get_end_state_reward_single_agent(self, end_state, agent):
        '''
        end_state - tuple representing the position of the given agent
        Returns the reward associated with the given end state
        '''
        rewards = self.reward_positions[agent.agent_number]
        if end_state in rewards:
            return rewards[end_state]
        else:
            return 0