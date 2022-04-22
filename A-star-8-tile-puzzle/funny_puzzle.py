import heapq

#################################################
# Helper Functions
#################################################

def swap_item(state, s1, s2):
    tmp_s = state.copy()
    t1 = tmp_s[s1]    
    tmp_s[s1] = state[s2]
    tmp_s[s2] = t1
    return tmp_s


def state_to_dict(states):
    res = dict()
    row = col = 0
    for i in states:
        if col > 2:
            col = 0
            row = row + 1  
        res[i] = [row, col]
        col = col + 1
    return res


def h_sum(s, goal=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    s_dict = state_to_dict(s)
    g_dict = state_to_dict(goal)
    
    # skip zero index
    s_dict[0] = [0, 0]
    g_dict[0] = [0, 0]
    
    # get heuristic sum
    summ = 0
    for k in s_dict:
        summ += get_manhattan_distance(s_dict[k], g_dict[k])
    return summ


def get_parent(dic, s):
    for k, v in dic.items():
        if v[0] == s:
            return k
    return -1


def show_step(t_dic, goal = [1, 2, 3, 4, 5, 6, 7, 0, 0]):    
    steps = []
    p_idx = -1
    # get parent
    for k, v in t_dic.items():
        if v[0] == goal:
            p_idx = k      
               
    # track
    while p_idx != -1:
        steps.append(t_dic[p_idx][0])
        p_idx = t_dic[p_idx][1]
    
    # print steps
    moves = 0
    for i in steps[::-1]:
        print("{} h={} moves: {}".format(i, h_sum(i, goal), moves))
        moves += 1


def contain(pq, s):
    for idx in pq:
        if idx[1] == s:
            return True
    return False


#################################################
# Required Functions
#################################################

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    return abs(from_state[0] - to_state[0]) + abs(from_state[1] - to_state[1])


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)    
    
    for succ_state in succ_states:
        print(succ_state, "h={}".format(h_sum(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    zeros = list(filter(lambda x: state[x] == 0, range(len(state))))
    succ_states = []
    
    for zero_idx in zeros:
        if zero_idx == 4: # cetner
            if state[1] != 0:
                succ_states.append(swap_item(state, 1, 4))
            if state[3] != 0:
                succ_states.append(swap_item(state, 3, 4))
            if state[5] != 0:
                succ_states.append(swap_item(state, 5, 4))
            if state[7] != 0:
                succ_states.append(swap_item(state, 7, 4))       
        elif zero_idx % 2 == 0: # corner
            if zero_idx == 0:
                if state[1] != 0:
                    succ_states.append(swap_item(state, 0, 1))
                if state[3] != 0:
                    succ_states.append(swap_item(state, 0, 3))
            if zero_idx == 2:
                if state[1] != 0:
                    succ_states.append(swap_item(state, 2, 1))
                if state[5] != 0:
                    succ_states.append(swap_item(state, 2, 5))
            if zero_idx == 6:
                if state[3] != 0:
                    succ_states.append(swap_item(state, 6, 3))
                if state[7] != 0:
                    succ_states.append(swap_item(state, 6, 7))
            if zero_idx == 8:
                if state[7] != 0:
                    succ_states.append(swap_item(state, 8, 7))
                if state[5] != 0:
                    succ_states.append(swap_item(state, 8, 5))    
        else: # middle of boundary
            if zero_idx == 1:
                if state[0] != 0:
                    succ_states.append(swap_item(state, 1, 0))
                if state[2] != 0:
                    succ_states.append(swap_item(state, 1, 2))
                if state[4] != 0:
                    succ_states.append(swap_item(state, 1, 4))
            if zero_idx == 3:
                if state[0] != 0:
                    succ_states.append(swap_item(state, 3, 0))
                if state[4] != 0:
                    succ_states.append(swap_item(state, 3, 4))
                if state[6] != 0:
                    succ_states.append(swap_item(state, 3, 6))
            if zero_idx == 5:
                if state[2] != 0:
                    succ_states.append(swap_item(state, 5, 2))
                if state[4] != 0:
                    succ_states.append(swap_item(state, 5, 4))
                if state[8] != 0:
                    succ_states.append(swap_item(state, 5, 8))
            if zero_idx == 7:
                if state[4] != 0:
                    succ_states.append(swap_item(state, 7, 4))
                if state[6] != 0:
                    succ_states.append(swap_item(state, 7, 6))
                if state[8] != 0:
                    succ_states.append(swap_item(state, 7, 8))
               
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    pq = []
    heapq.heappush(pq, (0 + h_sum(state, goal_state), state, (0, h_sum(state, goal_state), -1)))

    track_dic = {0: [state, -1]}
    visited = []
    curr_idx = 1
    max_len = 0    
    
    while True:        
        popped = heapq.heappop(pq)        
        popped_s = popped[1]              
        # end
        if popped_s == goal_state:                   
            show_step(track_dic, goal_state)
            print("Max queue length: {}".format(max_len))            
            break
        
        popped_i = popped[2]
        if popped_s not in visited:   
            visited.append(popped_s)        
        p_idx = get_parent(track_dic, popped_s)
        succ_states = get_succ(popped_s)            
        for succ_state in succ_states:
            if succ_state not in visited and not contain(pq, succ_state):               
                g = popped_i[0] + 1
                h = h_sum(succ_state, goal_state)                                         
                heapq.heappush(pq, (g + h, succ_state, (g, h, popped_i[2] + 1)))                
                if len(pq) > max_len:
                    max_len = len(pq)
                track_dic[curr_idx] = [succ_state, p_idx]
                curr_idx += 1

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    test = [4,3,0,5,1,6,7,2,0]
    
    print_succ(test)
    print()

    print(get_manhattan_distance(test, [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve(test)
    print()
