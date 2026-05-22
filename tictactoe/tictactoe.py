"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Count the number of X's and O's on the board to determine whose turn it is
    x_o_count = sum(row.count(X) + row.count(O) for row in board)
    # First player is X, so if the count of X's and O's is even, it's X's turn, otherwise it's O's turn
    if x_o_count % 2 == 0:
        return X
    else:
        return O
    

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Calculate which player's turn is it
    player_turn = player(board)
    # Check if the action is valid (i.e., the cell is empty)
    i, j = action
    if i < 0 or i >= len(board) or j < 0 or j >= len(board[i]):
        raise ValueError("Invalid action: Cell is out of bounds.")
    elif board[i][j] != EMPTY:
        raise ValueError("Invalid action: Cell is not empty.")
    
    # Create a deep copy of the board
    new_board = deepcopy(board)
    
    # Update the new board with the player's move
    new_board[i][j] = player_turn
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows for winner
    for row in board:
        if row[0] == row[1] == row[2] != EMPTY:
            return row[0]
    
    # Check columns for winner
    for j in range(len(board[0])):
        if board[0][j] == board[1][j] == board[2][j] != EMPTY:
            return board[0][j]
    
    # Check diagonals for winner
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]
    
    # If no winner, return None
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Check for winners
    if winner(board) is not None:
        return True
    # Check for draw (i.e., no empty cells left)
    if all(cell != EMPTY for row in board for cell in row):
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    # Store the winner in a local variable to avoid shadowing the
    # `winner` function name and to make the intent explicit.
    game_winner = winner(board)
    if game_winner == X:
        return 1
    elif game_winner == O:
        return -1
    else:
        return 0
    

def minvalue(board, alpha=-math.inf, beta=math.inf):
    """
    Returns the minimum utility value for a given board state.
    """
    if terminal(board):
        return utility(board)
    
    v = math.inf
    for action in actions(board):
        # Evaluate the child node (the maximizing player's response)
        # while threading current alpha/beta bounds for pruning.
        v = min(v, maxvalue(result(board, action), alpha, beta))

        # Update the beta bound after considering this child.
        beta = min(beta, v)

        # If alpha >= beta, the maximizing ancestor already has a
        # better option and will avoid this branch — prune remaining
        # children for efficiency.
        if alpha >= beta:
            break
    return v


def maxvalue(board, alpha=-math.inf, beta=math.inf):
    """
    Returns the maximum utility value for a given board state.
    """
    if terminal(board):
        return utility(board)
    
    v = -math.inf
    for action in actions(board):
        # Evaluate the child node (the minimizing player's response)
        # and let deeper calls use the current alpha/beta bounds.
        v = max(v, minvalue(result(board, action), alpha, beta))

        # Update alpha with the best value found so far at this node.
        alpha = max(alpha, v)

        # Prune remaining children when alpha >= beta.
        if alpha >= beta:
            break
    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    current_player = player(board)
    
    if current_player == X:
        best_value = -math.inf
        best_action = None
        alpha = -math.inf
        beta = math.inf
        for action in actions(board):
            # For `X` (maximizer): evaluate each possible action using
            # the `minvalue` helper and thread alpha/beta to enable pruning.
            value = minvalue(result(board, action), alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action

            # Update alpha and prune if possible.
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
        return best_action
    
    else:  # current_player == O
        best_value = math.inf
        best_action = None
        alpha = -math.inf
        beta = math.inf
        for action in actions(board):
            # For `O` (minimizer): evaluate each action via `maxvalue`,
            # thread bounds and prune when the branch cannot improve the
            # ancestor's decision.
            value = maxvalue(result(board, action), alpha, beta)
            if value < best_value:
                best_value = value
                best_action = action

            # Update beta and prune if alpha >= beta.
            beta = min(beta, best_value)
            if alpha >= beta:
                break
        return best_action