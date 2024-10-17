from collections import deque
import heapq

class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[0] * cols for _ in range(rows)]  

    def add_wall(self, x, y, w, h):
        for i in range(x, min(x + w, self.rows)):
            for j in range(y, min(y + h, self.cols)):
                self.grid[i][j] = 1  

    def is_valid(self, x, y): 
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x][y] != 1
    
    def set_grid(self, grid):
        self.grid = grid

class Node:
    def __init__(self, x, y, parent=None, path_cost=0):
        self.x = x
        self.y = y
        self.parent = parent
        self.path_cost = path_cost
        self.depth = 0

    def __lt__(self, other):
        if isinstance(other, Node): 
            return self.x + self.y < other.x + other.y
        return False
    

    def __eq__(self, other):
        if isinstance(other, Node): 
            return self.x == other.x and self.y == other.y
        return False

# Utility function for finding neighbors based on the specified direction order
def get_neighbors(pos, grid, reverse=False, from_goal=False):
    neighbors = []
    rows, cols = len(grid.grid), len(grid.grid[0])
    
    # Direction order: UP, LEFT, DOWN, RIGHT
    moves = {
        'UP': (0, -1),
        'LEFT': (-1, 0),
        'DOWN': (0, 1),
        'RIGHT': (1, 0),
    }
    if reverse:
        moves = {k: v for k, v in reversed(moves.items())}
    if from_goal:
        moves = {
            'UP': (0, 1),
            'LEFT': (1, 0),
            'DOWN': (0, -1),
            'RIGHT': (-1, 0),
    }

    for direction, move in moves.items():
        new_row, new_col = pos[0] + move[0], pos[1] + move[1]
        if 0 <= new_row < rows and 0 <= new_col < cols and grid.grid[new_row][new_col] == 0:
            neighbors.append(((new_row, new_col), direction))
    return neighbors

# BFS Search
def bfs(initial_state, goal_states, grid):
    frontier = deque([(initial_state, None)])  # Add None for the initial state (no direction)
    came_from = {initial_state: (None, None)}  # Store both parent and direction 
    total_nodes = 0
    traversed = []

    while frontier:
        current, _ = frontier.popleft()
        traversed.append(current)
        total_nodes += 1
        
        if current in goal_states:
            return reconstruct_path(came_from, current, grid), total_nodes, traversed
        
        for neighbor, direction in get_neighbors(current, grid):
            if neighbor not in came_from:
                frontier.append((neighbor, direction))
                came_from[neighbor] = (current, direction)
            
    return [], total_nodes, traversed

# DFS Search
def dfs(initial_state, goal_states, grid):
    frontier = [(initial_state, None)]  # Add None for the initial state (no direction)
    came_from = {initial_state: (None, None)}  # Store both parent and direction
    total_nodes = 0
    traversed = []
    
    while frontier:
        current, _ = frontier.pop()
        total_nodes += 1
        traversed.append(current)
        if current in goal_states:
            return reconstruct_path(came_from, current, grid), total_nodes, traversed
        
        for neighbor, direction in get_neighbors(current, grid, reverse=True):
            if neighbor not in traversed:
                frontier.append((neighbor, direction))
                came_from[neighbor] = (current, direction)
    
    return [], total_nodes, traversed

def astar(initial_state, goal_states, grid):
    goal = goal_states[0]  # Assuming one goal
    frontier = []
    heapq.heappush(frontier, (0, initial_state))  # Priority queue with initial state
    came_from = {initial_state: (None, None)}  # Keep track of the parent node and direction
    cost_so_far = {initial_state: 0}  # Cost of reaching each node from the start
    total_nodes = 0  # Total nodes expanded
    traversed = []  # Keep track of nodes traversed for visualization/debugging

    while frontier:
        _, current = heapq.heappop(frontier) 
        total_nodes += 1  # Increment the number of nodes expanded
        traversed.append(current)

        # Goal check
        if current == goal:
            return reconstruct_path(came_from, current, grid), total_nodes, traversed
        
        # Explore the neighbors of the current node
        for neighbor, direction in get_neighbors(current, grid):
            new_cost = cost_so_far[current] + 1  # Uniform cost for each step

            # If neighbor hasn't been visited or we've found a cheaper path to it
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost  # Update the cost
                priority = new_cost + heuristic(neighbor, goal) 
                heapq.heappush(frontier, (priority, neighbor))  # Add/update in the priority queue
                came_from[neighbor] = (current, direction)  # Keep track of the path and direction
    
    return [], total_nodes, traversed  # No solution found

# Utility function to reconstruct the path once goal is reached
def reconstruct_path(came_from, current, grid):
    path = []
    while current is not None:
        parent, direction = came_from[current]
        if parent is not None:
            grid.grid[current[0]][current[1]] = 3  # Mark the path in the grid (for visualization)
        if direction:
            path.append

# Greedy Best-First Search (GBFS)
def gbfs(initial_state, goal_states, grid):
    goal = goal_states[0]  # Assuming one goal
    frontier = []
    heapq.heappush(frontier, (0, initial_state))
    came_from = {initial_state: (None, None)}
    total_nodes = 0
    traversed = []
    
    while frontier:
        _, current = heapq.heappop(frontier)
        total_nodes += 1
        traversed.append(current)
        
        if current == goal:
            return reconstruct_path(came_from, current, grid), total_nodes, traversed
        
        for neighbor, direction in get_neighbors(current, grid):
            if neighbor not in traversed:
                priority = heuristic(neighbor, goal)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = (current, direction)
    
    return [], total_nodes, traversed
def bidirectional_dfs(initial_state, goal_states, grid):
    goal = goal_states[0]  # Assuming one goal
    frontier_start = [(initial_state, None)]
    frontier_goal = [(goal, None)]
    
    came_from_start = {initial_state: (None, None)}
    came_from_goal = {goal: (None, None)}
    
    total_nodes = 0
    traversed = []
    
    while frontier_start and frontier_goal:
        # Alternate between expanding from start and goal
        
        if frontier_start:  # Expand from start
            current_start, _ = frontier_start.pop()
            total_nodes += 1
            traversed.append(current_start)
            
            if current_start in came_from_goal:
                # Meet point
                path_from_start = reconstruct_path(came_from_start, current_start, grid)
                path_from_goal = reconstruct_path(came_from_goal, current_start, grid)
                # Merge the two paths into one history
                return path_from_start + path_from_goal[::-1], total_nodes, traversed
            
            for neighbor, direction in get_neighbors(current_start, grid, reverse=True):
                if neighbor not in came_from_start:
                    frontier_start.append((neighbor, direction))
                    came_from_start[neighbor] = (current_start, direction)
        
        if frontier_goal:  # Expand from goal
            current_goal, _ = frontier_goal.pop()
            traversed.append(current_goal)
            
            if current_goal in came_from_start:
                # Meet point
                path_from_start = reconstruct_path(came_from_start, current_goal, grid)
                path_from_goal = reconstruct_path(came_from_goal, current_goal, grid)
                # Merge the two paths into one history
                return path_from_start + path_from_goal[::-1], total_nodes, traversed
            
            for neighbor, direction in get_neighbors(current_goal, grid, from_goal=True):
                if neighbor not in came_from_goal:
                    frontier_goal.append((neighbor, direction))
                    came_from_goal[neighbor] = (current_goal, direction)
    return [], total_nodes, traversed

def ida_star(initial_state, goal_states, grid):
    goal = goal_states[0]
    threshold = heuristic(initial_state, goal)
    traversed = []
    
    while True:
        came_from = {}
        current_threshold = threshold
        stack = [(initial_state, 0)]
        came_from[initial_state] = (None, None)
        temp_min_cost = float('inf')

        while stack:
            current, g = stack.pop()
            traversed.append(current)
            f = g + heuristic(current, goal)  # f = g + h
            
            if f > current_threshold:  # If f exceeds threshold, we track minimum cost for next iteration
                temp_min_cost = min(temp_min_cost, f)
                continue

            if current == goal:  # If the goal is reached
                return reconstruct_path(came_from, current, grid), len(traversed), traversed

            # Explore neighbors
            for neighbor, direction in get_neighbors(current, grid):
                if neighbor not in came_from:
                    came_from[neighbor] = (current, direction)
                    stack.append((neighbor, g + 1))

        if temp_min_cost == float('inf'):  # If no valid path was found, return failure
            return [], len(traversed), traversed
        threshold = temp_min_cost

# Heuristic function (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Reconstruct path from the search tree with directions
def reconstruct_path(came_from, current, grid):
    path = []
    while current is not None:
        parent, direction = came_from[current]
        if parent is not None:
            grid.grid[current[0]][current[1]] = 3
        if direction:
            path.append(direction)
        current = parent
    path.reverse()
    return path

def get_path(directions, start=(0, 0)):
    path = []
    position = (start[0], start[1])
    path.append(position)
    for direction in directions:
        if direction == 'UP':
            position = (position[0], position[1] - 1)
        elif direction == 'LEFT':
            position = (position[0] - 1, position[1])
        elif direction == 'DOWN':
            position = (position[0], position[1] + 1)
        elif direction == 'RIGHT':
            position = (position[0] + 1, position[1])
        path.append(position)
    return path

def append_unique(lst, items):
    for item in items:
        if item not in lst:
            lst.append(item)
