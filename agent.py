class Agent():
    def __init__(self, agent_position, agent_number):
        '''
        agent_possible - tuple, representing the coordinate of the agent
        agent_number - int, representing the number correspondes to the agent
        '''
        self.agent_position = agent_position
        self.agent_number = agent_number
    
    def update_position(self, new_position):
        self.agent_position = new_position
        
    def check_equal_agent(self, other):
        '''
        Checks equality of two Agents
        '''
        if self.agent_position == other.agent_position and self.agent_number == other.agent_number:
            return True
        return False
        
    def get_all_possible_actions(self, env):
        '''
        env - an instance of Environment
        Returns a list of all the possible action the agent can take in the environment from its current position
        '''
        actions = [(1, 0), (0, 1), (0, 0), (-1, 0), (0, -1)]
        current_x_pos = self.agent_position[0]
        current_y_pos = self.agent_position[1]
        
        result = []
        for action in actions:
            new_x_pos = current_x_pos + action[0]
            new_y_pos = current_y_pos + action[1]
            new_pos = (new_x_pos, new_y_pos)
            
            if env.is_valid_cell(new_pos):
                result.append(action)
        
        return result
            
    def result_of_action(self, action):
        '''
        Given action, returns the new position of the agent after the action is done.
        Note: assuming that the action is from the allowed actions list
        '''     
        current_x_pos = self.agent_position[0]
        current_y_pos = self.agent_position[1]
        
        shift_x = action[0]
        shift_y = action[1]
        
        new_pos = (current_x_pos + shift_x, current_y_pos + shift_y)
        return new_pos