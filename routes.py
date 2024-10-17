from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import bfs, dfs, astar, gbfs, bidirectional_dfs, ida_star, Grid, append_unique
from pydantic import BaseModel

app = FastAPI()

# CORS configuration (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

class getResult(BaseModel):
    algorithm: str
    initialstate: list[int]
    goalstate: list[list[int]]
    grid: list[list[int]]

@app.post("/getResult")
def get_result(data: getResult):
    algorithm = data.algorithm
    initialstate = (data.initialstate[0], data.initialstate[1])
    goalstate = [(x[0], x[1]) for x in data.goalstate]
    rows = len(data.grid)
    cols = len(data.grid[0])
    grid = Grid(rows, cols)
    grid.set_grid(data.grid)
    
    # Initialize variables
    results = []
    total_nodes = 0
    traversed = []

    for goal in goalstate:
        if(goal[0] > grid.rows or goal[1] > grid.cols):
            continue
        temp_grid = Grid(grid.rows, grid.cols)
        temp_grid.set_grid([row[:] for row in grid.grid])  # Deep copy of grid for each search
        temp_result, temp_total_nodes, temp_traversed = None, 0, []

        # Choose algorithm
        if algorithm == "bfs":
            temp_result, temp_total_nodes, temp_traversed = bfs(initialstate, [goal], temp_grid)
        elif algorithm == "dfs":
            temp_result, temp_total_nodes, temp_traversed = dfs(initialstate, [goal], temp_grid)
        elif algorithm == "astar":
            temp_result, temp_total_nodes, temp_traversed = astar(initialstate, [goal], temp_grid)
        elif algorithm == "gbfs":
            temp_result, temp_total_nodes, temp_traversed = gbfs(initialstate, [goal], temp_grid)
        elif algorithm == "bdfs":  # Bidirectional DFS
            temp_result, temp_total_nodes, temp_traversed = bidirectional_dfs(initialstate, [goal], temp_grid)
        elif algorithm == "ida":  # IDA* Search
            temp_result, temp_total_nodes, temp_traversed = ida_star(initialstate, [goal], temp_grid)
        else:
            return {"error": "Invalid algorithm"}
        
        results.append({
            "goal": goal,
            "path": temp_result,
            "total_nodes": temp_total_nodes,
            "traversed": temp_traversed,
            "reachable": bool(temp_result)
        })
        total_nodes += temp_total_nodes
        append_unique(traversed, temp_traversed)
    if results:
        return {"path": results, "traversed": traversed, "total_nodes": total_nodes}
    else:
        return {"error": "No path found"}
