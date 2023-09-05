The walking game levels both in the appendix of the report and in the directory that in the current folder.

the code directory contains all the code and the output results.



code structure and file explenation:

1. MCTS is the Monte Carlo Tree Search class. it has the search algorithm and the Nodes of the MCTS as a class also.

Each game has a GameState class which implements several methods and has several attributes that interface the MCTS algorithm and Node.

best_move_decision_functions.py is the file with all of the proposed methods in this project.

utils.py has an auxilary function to help randomize some selections.

the main files runs the experiments (per game) and saves the statistics in the results folders as csv files.
An experiment is performing an MCTS algorithm with each proposed method and several time budgets, levels, initializations ect. (depands on each game)

the graphs files outputs the relevant graphs for the report and saves them in the results folders.
