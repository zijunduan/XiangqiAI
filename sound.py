import pygame

class SoundManager:
    def __init__(self):
        pygame.mixer.init()

        # 防止多个音效抢占
        pygame.mixer.set_num_channels(16)

        # 加载音效
        self.move_sound = pygame.mixer.Sound("sounds/move.wav")
        self.capture_sound = pygame.mixer.Sound("sounds/capture.wav")
        self.check_sound = pygame.mixer.Sound("sounds/check.wav")

        # 音量
        self.move_sound.set_volume(0.7)
        self.capture_sound.set_volume(0.7)
        self.check_sound.set_volume(0.8)

        pygame.mixer.music.load("sounds/高山流水.mp3")
        pygame.mixer.music.set_volume(0.4)
        pygame.mixer.music.play(-1)
    def play_move(self):
        self.move_sound.play()

    def play_capture(self):
        self.capture_sound.play()

    def play_check(self):
        self.check_sound.play()

    def play_win(self):
        self.win_sound.play()