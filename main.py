from board import Board
from ui import UI
from sound import SoundManager

def main():
    board = Board()
    sound_manager = SoundManager()
    ui = UI(board,sound_manager)
    ui.run()

if __name__ == "__main__":
    main()