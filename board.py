ROWS = 10
COLS = 9


def in_bounds(x, y):
    return 0 <= x < COLS and 0 <= y < ROWS


class Board:

    def __init__(self):
        self.board = self.init_board()

    def init_board(self):
        return [
            ['b_c', 'b_m', 'b_x', 'b_s', 'b_j', 'b_s', 'b_x', 'b_m', 'b_c'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', 'b_p', '.', '.', '.', '.', '.', 'b_p', '.'],
            ['b_z', '.', 'b_z', '.', 'b_z', '.', 'b_z', '.', 'b_z'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['r_z', '.', 'r_z', '.', 'r_z', '.', 'r_z', '.', 'r_z'],
            ['.', 'r_p', '.', '.', '.', '.', '.', 'r_p', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['r_c', 'r_m', 'r_x', 'r_s', 'r_j', 'r_s', 'r_x', 'r_m', 'r_c'],
        ]

    def move(self, sx, sy, ex, ey):
        if not in_bounds(sx, sy) or not in_bounds(ex, ey):
            raise ValueError(f"move out of bounds: {(sx, sy)} -> {(ex, ey)}")

        piece = self.board[sy][sx]
        if piece == '.':
            raise ValueError(f"no piece at source square: {(sx, sy)}")

        captured = self.board[ey][ex]
        if captured != '.' and captured[0] == piece[0]:
            raise ValueError(f"cannot capture own piece: {piece} -> {captured}")

        self.board[sy][sx] = '.'
        self.board[ey][ex] = piece

        return captured

    def undo_move(self, sx, sy, ex, ey, captured):
        if not in_bounds(sx, sy) or not in_bounds(ex, ey):
            raise ValueError(f"undo out of bounds: {(sx, sy)} <- {(ex, ey)}")

        piece = self.board[ey][ex]
        if piece == '.':
            raise ValueError(f"no piece at destination square during undo: {(ex, ey)}")

        self.board[sy][sx] = piece
        self.board[ey][ex] = captured

    def get_piece(self, x, y):
        if not in_bounds(x, y):
            raise ValueError(f"square out of bounds: {(x, y)}")
        return self.board[y][x]

    def is_empty(self, x, y):
        if not in_bounds(x, y):
            raise ValueError(f"square out of bounds: {(x, y)}")
        return self.board[y][x] == '.'
