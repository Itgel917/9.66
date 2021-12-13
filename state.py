import random

class State():
    def __init__(self, agent_1, agent_2):
        '''
        agent_1, agent_2 - instances of Agent
        '''
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        
    def check_equal_state(self, other):
        '''
        Checks equality of two states
        '''
        if self.agent_1.check_equal_agent(other.agent_1) and self.agent_2.check_equal_agent(other.agent_2):
            return True
        return False
        
    def possible_action_pairs(self, env):
        '''
        env - an instance of Environment
        Returns a list of all the possible action pairs the two agents can take from the current state
        '''
        agent_1_actions = self.agent_1.get_all_possible_actions(env)  
        agent_2_actions = self.agent_2.get_all_possible_actions(env)
        
        result = []
        for action_1 in agent_1_actions:
            for action_2 in agent_2_actions:
                result.append((action_1, action_2))
        
        return result
    
    def apply_action_pairs(self, action_1, action_2):
        '''
        env - an instance of Environment
        action_1 - action of the agent_1
        action_2 - action of the agent_2
        Returns the positions of the agents after the actions
        '''
        old_pos_1 = self.agent_1.agent_position
        old_pos_2 = self.agent_2.agent_position
        
        new_pos_1 = self.agent_1.result_of_action(action_1)
        new_pos_2 = self.agent_2.result_of_action(action_2)
        
        if new_pos_1 != new_pos_2:
            return (new_pos_1, new_pos_2)
    
        else:
            should_one_move = random.choice([True, False])
            if should_one_move:
                return (new_pos_1, old_pos_2)
            else:
                return (old_pos_1, new_pos_2)   
    