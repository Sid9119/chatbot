
import tkinter as tk
from tkinter import messagebox

# Initialize the board
board = [[' ' for _ in range(3)] for _ in range(3)]

def is_winner(board, player):
    win_conditions = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[2][0], board[1][1], board[0][2]]
    ]
    return [player, player, player] in win_conditions

def is_draw(board):
    return all(cell != ' ' for row in board for cell in row)

def minimax(board, depth, is_maximizing, alpha, beta):
    if is_winner(board, 'X'):
        return -10 + depth
    if is_winner(board, 'O'):
        return 10 - depth
    if is_draw(board):
        return 0
    
    if is_maximizing:
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    eval = minimax(board, depth + 1, False, alpha, beta)
                    board[i][j] = ' '
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        return max_eval
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    eval = minimax(board, depth + 1, True, alpha, beta)
                    board[i][j] = ' '
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        return min_eval
        return min_eval

def best_move(board):
    best_val = float('-inf')
    move = (-1, -1)
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                move_val = minimax(board, 0, False, float('-inf'), float('inf'))
                board[i][j] = ' '
                if move_val > best_val:
                    move = (i, j)
                    best_val = move_val
    return move

def on_click(row, col):
    if board[row][col] == ' ' and not is_game_over():
        board[row][col] = 'X'
        buttons[row][col].config(text='X', state=tk.DISABLED, disabledforeground='blue')
        
        if is_winner(board, 'X'):
            messagebox.showinfo("Game Over", "Player X wins!")
        elif is_draw(board):
            messagebox.showinfo("Game Over", "It's a draw match !")
        else:
            row, col = best_move(board)
            board[row][col] = 'O'
            buttons[row][col].config(text='O', state=tk.DISABLED, disabledforeground='red')
            
            if is_winner(board, 'O'):
                messagebox.showinfo("Game Over", "Player O wins!")
            elif is_draw(board):
                messagebox.showinfo("Game Over", "It's a draw match !")

def is_game_over():
    return is_winner(board, 'X') or is_winner(board, 'O') or is_draw(board)

def reset_game():
    global board, buttons
    board = [[' ' for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            buttons[i][j].config(text='', state=tk.NORMAL, bg='lightyellow')

# Initialize the GUI
root = tk.Tk()
root.title("Tic-Tac-Toe be Neeraj Rana ")

buttons = [[None for _ in range(3)] for _ in range(3)]
for i in range(3):
    for j in range(3):
        buttons[i][j] = tk.Button(root, text='', width=10, height=3,
                                  command=lambda row=i, col=j: on_click(row, col),
                                  bg='lightyellow', font=('Arial', 24))
        buttons[i][j].grid(row=i, column=j)

reset_button = tk.Button(root, text="Reset", command=reset_game, bg='pink', font=('Arial', 18))
reset_button.grid(row=3, column=0, columnspan=3)

root.mainloop()
