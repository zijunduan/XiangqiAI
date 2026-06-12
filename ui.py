import pygame
import sys
import math
from rules import get_legal_moves
from rules import get_game_state
from rules import is_in_check
from board import Board
from ai import greedy_move, MinimaxAI, MCTSAI, StrongAI, evaluate_fast

#大小
size= 60
offset = size*2//3
size_chess=size*5//6

width= offset*2 + size*8
height= offset*2 + size*9

#颜色
board_color= (205, 170, 125)   # 木色
line_color= (60, 40, 20)       # 深棕
river_color= (180, 150, 100)   # 河界稍浅


class UI:
    def __init__(self, board,sound_manager):
        pygame.init()
        self.screen = pygame.display.set_mode((width,height))
        pygame.display.set_caption("中国象棋")
        self.board = board
        self.load_images()
        self.selected = None
        self.valid_moves=[]
        self.turn='r'

        self.animating = False
        self.anim_piece = None
        self.anim_start = None
        self.anim_end = None
        self.anim_progress = 0  # 0 ~ 1
        self.speed=0.02

        self.history = []
        self.position_stack = []
        self.position_counts = {}

        self.game_over = False
        self.winner_text = ""

        self.sound_manager=sound_manager

        self.clock = pygame.time.Clock()

        self.check_chain = 0
        self.last_check_side = None
        self.menu_active = True
        self.menu_buttons = []
        self.menu_started_at = pygame.time.get_ticks()
        self.menu_action_tick = 0
        self.last_menu_action = None
        self.game_mode = 'human'
        self.ai_mode = 'minimax'
        self.red_ai_mode = 'minimax'
        self.black_ai_mode = 'minimax'
        self.minimax_depth = 8
        self.minimax_time = 8.0
        self.red_minimax_depth = 8
        self.red_minimax_time = 8.0
        self.black_minimax_depth = 8
        self.black_minimax_time = 8.0
        self.mcts_time = 2.0
        self.red_mcts_time = 2.0
        self.black_mcts_time = 2.0
        self.ai_engine = None
        self.ai_engines = {}
        self.last_move = None
        self.music_faded = False
        self.configure_ai()
        self.reset_repetition_tracker()

    def ai_mode_label(self, mode):
        return {
            'greedy': '浅谋',
            'minimax': '深算',
            'mcts': '试势',
            'strong': '强弈',
        }.get(mode, mode)

    def minimax_settings(self, side=None):
        if self.game_mode == 'arena':
            if side == 'r':
                return self.red_minimax_depth, self.red_minimax_time
            if side == 'b':
                return self.black_minimax_depth, self.black_minimax_time
        return self.minimax_depth, self.minimax_time

    def make_ai_engine(self, mode, side=None):
        if mode == 'greedy':
            return None
        if mode == 'mcts':
            return MCTSAI(time_limit=self.mcts_settings(side))
        depth, time_limit = self.minimax_settings(side)
        if mode == 'strong':
            return StrongAI(max_depth=max(8, depth + 2),
                            time_limit=time_limit, use_mcts=True)
        return MinimaxAI(max_depth=depth, time_limit=time_limit, use_neural=False)

    def mcts_settings(self, side=None):
        if self.game_mode == 'arena':
            if side == 'r':
                return self.red_mcts_time
            if side == 'b':
                return self.black_mcts_time
        return self.mcts_time

    def configure_ai(self):
        if self.game_mode == 'arena':
            self.ai_engines = {
                'r': self.make_ai_engine(self.red_ai_mode, 'r'),
                'b': self.make_ai_engine(self.black_ai_mode, 'b'),
            }
            self.ai_engine = self.ai_engines['b']
        else:
            self.ai_engines = {'r': None, 'b': self.make_ai_engine(self.ai_mode, 'b')}
            self.ai_engine = self.ai_engines['b']

    def load_images(self):
        self.images = {}
        pieces = ['b_c','b_j','b_x','b_m','b_z','b_s','b_p','r_c','r_j','r_x','r_s','r_p','r_m','r_z']
        for p in pieces:
            img = pygame.image.load(f"images/{p}.png")
            img = pygame.transform.scale(img, (size_chess, size_chess))
            self.images[p] = img

        bg = pygame.image.load(f"images/bg.jpg")
        self.bg = pygame.transform.scale(bg, (width, height))

    def to_screen(self,x,y):
        return offset+x*size, offset+y*size
    def get_grid_pos(self, mx, my):
        x = round((mx - offset) / size)
        y = round((my - offset) / size)
        return x, y
    def location(self,i,j):
        return j*size+offset-size_chess//2, i*size+offset-size_chess//2

    def draw_board(self):
        #画背景
        self.screen.blit(self.bg, (0, 0))
        #画网格
        for i in range(10):
            pygame.draw.line(self.screen, line_color, self.to_screen(0,i), self.to_screen(8,i),2)
        for j in range(9):
            pygame.draw.line(self.screen, line_color, self.to_screen(j,0), self.to_screen(j,4), 2)
            pygame.draw.line(self.screen, line_color, self.to_screen(j,5), self.to_screen(j,9), 2)
        pygame.draw.line(self.screen, line_color, self.to_screen(0, 4), self.to_screen(0, 5), 2)
        pygame.draw.line(self.screen, line_color, self.to_screen(8, 4), self.to_screen(8, 5), 2)
        #画九宫格
        # 上九宫
        pygame.draw.line(self.screen, line_color,
                         self.to_screen(3, 0), self.to_screen(5, 2), 2)
        pygame.draw.line(self.screen, line_color,
                         self.to_screen(5, 0), self.to_screen(3, 2), 2)
        # 下九宫
        pygame.draw.line(self.screen, line_color,
                         self.to_screen(3, 7), self.to_screen(5, 9), 2)
        pygame.draw.line(self.screen, line_color,
                         self.to_screen(5, 7), self.to_screen(3, 9), 2)
        #楚河汉界
        y = offset + size * 4 + size // 2 - 20
        self.draw_wood_text("楚  河", offset + size * 1.0, y)
        self.draw_wood_text("汉  界", offset + size * 5.0, y)

        #炮兵标记
        for pos in [(1, 2), (7, 2), (1, 7), (7, 7)]:
            self.draw_mark(*pos,'med')
        #兵标记
        for x in [2, 4, 6]:
            self.draw_mark(x, 3,'med')
            self.draw_mark(x, 6,'med')
        self.draw_mark(0,3,'left')
        self.draw_mark(0,6,'left')
        self.draw_mark(8,3,'right')
        self.draw_mark(8,6,'right')

    def draw_wood_text(self, text, x, y):
        font = pygame.font.Font("fonts/simkai.ttf", 42)
        main_color = (60, 30, 20)  # 木刻主色
        shadow_color = (30, 20, 10)  # 阴影
        # 阴影（偏移）
        shadow = font.render(text, True, shadow_color)
        self.screen.blit(shadow, (x + 2, y + 2))
        # 主文字
        main = font.render(text, True, main_color)
        self.screen.blit(main, (x, y))
        # 轻微“浮雕高光”（上偏白一点点）
        highlight = font.render(text, True, (140, 110, 80))
        self.screen.blit(highlight, (x - 1, y - 1))

    def draw_menu_text(self, text, center, font_size=28, color=(60, 30, 20)):
        font = pygame.font.Font("fonts/simkai.ttf", font_size)
        main = font.render(text, True, color)
        rect = main.get_rect(center=center)
        self.screen.blit(main, rect)

    def menu_ease(self):
        elapsed = pygame.time.get_ticks() - self.menu_started_at
        t = max(0.0, min(1.0, elapsed / 420))
        return 1 - (1 - t) * (1 - t)

    def play_menu_tap(self):
        try:
            self.sound_manager.play_move()
        except Exception:
            pass

    def add_menu_button(self, rect, action, label, active=False, font_size=26):
        self.menu_buttons.append((pygame.Rect(rect), action))
        rect = pygame.Rect(rect)
        hover = rect.collidepoint(pygame.mouse.get_pos())
        now = pygame.time.get_ticks()
        pulse = 0.5 + 0.5 * math.sin(now / 210)

        if rect.w <= 46 and rect.h <= 40:
            center = rect.center
            radius = min(rect.w, rect.h) // 2
            fill = (126, 49, 36) if active or hover else (92, 58, 36)
            pygame.draw.circle(self.screen, (58, 37, 22), center, radius)
            pygame.draw.circle(self.screen, fill, center, radius - 3)
            pygame.draw.circle(self.screen, (181, 151, 101), center, radius - 7, 1)
            self.draw_menu_text(label, center, font_size=font_size,
                                color=(228, 205, 168))
            return

        fill = (202, 174, 122) if active else (181, 151, 101)
        if hover and not active:
            fill = (194, 163, 111)
        left = rect.x
        right = rect.right
        top = rect.y
        bottom = rect.bottom
        notch = 14
        points = [
            (left + notch, top), (right - notch, top),
            (right, rect.centery), (right - notch, bottom),
            (left + notch, bottom), (left, rect.centery),
        ]
        pygame.draw.polygon(self.screen, (62, 42, 25), points)
        inner = [(x, y + 1) for x, y in points]
        pygame.draw.polygon(self.screen, fill, inner)
        pygame.draw.line(self.screen, (211, 188, 142),
                         (left + 28, top + 8), (right - 28, top + 8), 1)
        pygame.draw.line(self.screen, (119, 82, 47),
                         (left + 24, bottom - 8), (right - 24, bottom - 8), 1)

        if active:
            y = bottom - 9
            start = left + 28
            end = right - 28
            bright = int(118 + 24 * pulse)
            pygame.draw.line(self.screen, (bright, 39, 29), (start, y), (end, y), 3)
            pygame.draw.circle(self.screen, (124, 41, 31), (right - 26, top + 17), 11)
            self.draw_menu_text("定", (right - 26, top + 17), 15,
                                color=(226, 201, 165))
        elif hover:
            pygame.draw.circle(self.screen, (103, 54, 39), (right - 26, top + 17), 5)

        self.draw_menu_text(label, rect.center, font_size=font_size,
                            color=(54, 34, 22))

    def draw_scroll_panel(self, panel):
        ratio = self.menu_ease()
        scroll_h = max(28, int(panel.h * ratio))
        panel = pygame.Rect(panel.x, panel.centery - scroll_h // 2, panel.w, scroll_h)
        pygame.draw.rect(self.screen, (74, 50, 31),
                         (35, panel.y, 16, panel.h), border_radius=8)
        pygame.draw.rect(self.screen, (74, 50, 31),
                         (width - 51, panel.y, 16, panel.h), border_radius=8)
        for x in (43, width - 43):
            pygame.draw.circle(self.screen, (53, 34, 20), (x, panel.y), 10)
            pygame.draw.circle(self.screen, (53, 34, 20), (x, panel.bottom), 10)
            pygame.draw.circle(self.screen, (112, 84, 50), (x, panel.y), 4)
            pygame.draw.circle(self.screen, (112, 84, 50), (x, panel.bottom), 4)

        pygame.draw.rect(self.screen, (199, 174, 128), panel, border_radius=4)
        pygame.draw.rect(self.screen, (62, 42, 25), panel, 3, border_radius=4)
        inner = panel.inflate(-20, -20)
        pygame.draw.rect(self.screen, (137, 99, 58), inner, 1, border_radius=2)
        for y in range(panel.y + 38, panel.bottom - 24, 28):
            pygame.draw.line(self.screen, (179, 150, 105),
                             (panel.x + 26, y), (panel.right - 26, y), 1)
        return panel

    def draw_title_plaque(self):
        plaque = pygame.Rect(width // 2 - 118, 68, 236, 54)
        pygame.draw.rect(self.screen, (76, 50, 30), plaque, border_radius=4)
        pygame.draw.rect(self.screen, (137, 100, 61), plaque.inflate(-8, -8), 1)
        self.draw_menu_text("楚汉对弈", plaque.center, 42, color=(216, 190, 146))
        seal = pygame.Rect(width // 2 + 98, 74, 42, 42)
        pygame.draw.rect(self.screen, (123, 39, 30), seal, border_radius=3)
        pygame.draw.rect(self.screen, (83, 27, 22), seal, 2, border_radius=3)
        self.draw_menu_text("弈", seal.center, 28, color=(226, 201, 165))

    def draw_start_menu(self):
        self.screen.blit(self.bg, (0, 0))
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((68, 52, 35, 106))
        self.screen.blit(overlay, (0, 0))
        self.menu_buttons = []

        panel = pygame.Rect(45, 55, width - 90, height - 105)
        visible_panel = self.draw_scroll_panel(panel)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(visible_panel)
        self.draw_title_plaque()
        self.draw_menu_text("红先黑后  一局定势", (width // 2, 136), 21,
                            color=(82, 47, 25))

        modes = [
            ('greedy', '浅谋', '贪吃逐利，落子甚疾'),
            ('minimax', '深算', '层层推演，攻守兼顾'),
            ('mcts', '试势', '多番演局，以势取胜'),
            ('strong', '强弈', '深搜复核，防漏招送子'),
        ]
        self.add_menu_button(pygame.Rect(96, 158, 155, 44),
                             ('game_mode', 'human'), "人 机",
                             active=self.game_mode == 'human', font_size=24)
        self.add_menu_button(pygame.Rect(width - 251, 158, 155, 44),
                             ('game_mode', 'arena'), "观 战",
                             active=self.game_mode == 'arena', font_size=24)

        if self.game_mode == 'human':
            y = 210
            for key, title, desc in modes:
                active = self.ai_mode == key
                rect = pygame.Rect(82, y, width - 164, 43)
                self.add_menu_button(rect, ('mode', key), title,
                                     active=active, font_size=25)
                self.draw_menu_text(desc, (width // 2, y + 48), 16,
                                    color=(80, 56, 37))
                y += 58
        else:
            self.draw_menu_text("红黑自弈  坐看攻守", (width // 2, 230), 22,
                                color=(82, 47, 25))
            for row_y, side, label, current in [
                (268, 'r', '红方', self.red_ai_mode),
                (328, 'b', '黑方', self.black_ai_mode),
            ]:
                self.draw_menu_text(label, (105, row_y + 22), 23,
                                    color=(78, 47, 28))
                x = 142
                for key, title, _desc in modes:
                    self.add_menu_button(
                        pygame.Rect(x, row_y, 78, 42),
                        ('arena_mode', (side, key)),
                        title,
                        active=current == key,
                        font_size=21
                    )
                    x += 88

        param_y = 456 if self.game_mode == 'human' else 412
        uses_minimax = (
            self.ai_mode in ('minimax', 'strong') if self.game_mode == 'human'
            else (self.red_ai_mode in ('minimax', 'strong') or
                  self.black_ai_mode in ('minimax', 'strong'))
        )
        uses_mcts = (
            self.ai_mode == 'mcts' if self.game_mode == 'human'
            else self.red_ai_mode == 'mcts' or self.black_ai_mode == 'mcts'
        )
        if self.game_mode == 'human':
            if uses_minimax:
                self.draw_menu_text("算深", (104, param_y), 22)
                self.add_menu_button(pygame.Rect(141, param_y - 18, 34, 32),
                                     ('minimax_depth', -1), "-", font_size=22)
                self.draw_menu_text(str(self.minimax_depth), (192, param_y), 24)
                self.add_menu_button(pygame.Rect(211, param_y - 18, 34, 32),
                                     ('minimax_depth', 1), "+", font_size=22)

                self.draw_menu_text("候时", (312, param_y), 22)
                self.add_menu_button(pygame.Rect(346, param_y - 18, 34, 32),
                                     ('minimax_time', -0.5), "-", font_size=22)
                self.draw_menu_text(f"{self.minimax_time:.1f}s", (405, param_y), 22)
                self.add_menu_button(pygame.Rect(438, param_y - 18, 34, 32),
                                     ('minimax_time', 0.5), "+", font_size=22)
            elif uses_mcts:
                self.draw_menu_text("演局", (172, param_y), 22)
                self.add_menu_button(pygame.Rect(230, param_y - 18, 34, 32),
                                     ('mcts_time', -0.5), "-", font_size=22)
                self.draw_menu_text(f"{self.mcts_time:.1f}s", (296, param_y), 22)
                self.add_menu_button(pygame.Rect(346, param_y - 18, 34, 32),
                                     ('mcts_time', 0.5), "+", font_size=22)
            else:
                self.draw_menu_text("无须设局  即刻应手", (width // 2, param_y), 23)
        else:
            rows = []
            if self.red_ai_mode == 'minimax':
                rows.append(('minimax', 'r', '红深',
                             self.red_minimax_depth, self.red_minimax_time))
            elif self.red_ai_mode == 'strong':
                rows.append(('minimax', 'r', '红强',
                             self.red_minimax_depth, self.red_minimax_time))
            elif self.red_ai_mode == 'mcts':
                rows.append(('mcts', 'r', '红势', None, self.red_mcts_time))
            if self.black_ai_mode == 'minimax':
                rows.append(('minimax', 'b', '黑深',
                             self.black_minimax_depth, self.black_minimax_time))
            elif self.black_ai_mode == 'strong':
                rows.append(('minimax', 'b', '黑强',
                             self.black_minimax_depth, self.black_minimax_time))
            elif self.black_ai_mode == 'mcts':
                rows.append(('mcts', 'b', '黑势', None, self.black_mcts_time))

            row_y = 412 if len(rows) == 2 else 434
            for kind, side, label, depth, time_limit in rows:
                self.draw_menu_text(label, (92, row_y), 20)
                if kind == 'minimax':
                    self.add_menu_button(pygame.Rect(132, row_y - 17, 32, 30),
                                         ('arena_minimax_depth', (side, -1)), "-", font_size=20)
                    self.draw_menu_text(str(depth), (184, row_y), 22)
                    self.add_menu_button(pygame.Rect(204, row_y - 17, 32, 30),
                                         ('arena_minimax_depth', (side, 1)), "+", font_size=20)

                    self.draw_menu_text("候时", (292, row_y), 20)
                    self.add_menu_button(pygame.Rect(326, row_y - 17, 32, 30),
                                         ('arena_minimax_time', (side, -0.5)), "-", font_size=20)
                    self.draw_menu_text(f"{time_limit:.1f}s", (391, row_y), 20)
                    self.add_menu_button(pygame.Rect(428, row_y - 17, 32, 30),
                                         ('arena_minimax_time', (side, 0.5)), "+", font_size=20)
                else:
                    self.draw_menu_text("演局", (172, row_y), 20)
                    self.add_menu_button(pygame.Rect(226, row_y - 17, 32, 30),
                                         ('arena_mcts_time', (side, -0.5)), "-", font_size=20)
                    self.draw_menu_text(f"{time_limit:.1f}s", (288, row_y), 20)
                    self.add_menu_button(pygame.Rect(342, row_y - 17, 32, 30),
                                         ('arena_mcts_time', (side, 0.5)), "+", font_size=20)
                row_y += 44

        self.add_menu_button(pygame.Rect(125, height - 96, width - 250, 52),
                             ('start', None),
                             "落 子 开 局" if self.game_mode == 'human' else "开 局 观 战",
                             active=True, font_size=31)
        self.screen.set_clip(old_clip)

    def handle_menu_action(self, action):
        now = pygame.time.get_ticks()
        if (self.last_menu_action == action and self.menu_action_tick and
                now - self.menu_action_tick < 120):
            return
        kind, value = action
        self.menu_action_tick = now
        self.last_menu_action = action
        self.play_menu_tap()
        if kind == 'game_mode':
            self.game_mode = value
        elif kind == 'mode':
            self.ai_mode = value
        elif kind == 'arena_mode':
            side, mode = value
            if side == 'r':
                self.red_ai_mode = mode
            else:
                self.black_ai_mode = mode
        elif kind == 'minimax_depth':
            self.minimax_depth = max(1, min(100, self.minimax_depth + value))
        elif kind == 'minimax_time':
            self.minimax_time = max(0.5, min(100, self.minimax_time + value))
        elif kind == 'arena_minimax_depth':
            side, delta = value
            if side == 'r':
                self.red_minimax_depth = max(1, min(100, self.red_minimax_depth + delta))
            else:
                self.black_minimax_depth = max(1, min(100, self.black_minimax_depth + delta))
        elif kind == 'arena_minimax_time':
            side, delta = value
            if side == 'r':
                self.red_minimax_time = max(0.5, min(100, self.red_minimax_time + delta))
            else:
                self.black_minimax_time = max(0.5, min(100, self.black_minimax_time + delta))
        elif kind == 'arena_mcts_time':
            side, delta = value
            if side == 'r':
                self.red_mcts_time = max(0.5, min(100, self.red_mcts_time + delta))
            else:
                self.black_mcts_time = max(0.5, min(100, self.black_mcts_time + delta))
        elif kind == 'mcts_time':
            self.mcts_time = max(0.5, min(100, self.mcts_time + value))
        elif kind == 'start':
            self.configure_ai()
            self.restart_game()
            self.menu_active = False

    def handle_menu_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.handle_menu_action(('start', None))
            elif event.key == pygame.K_1:
                self.handle_menu_action(('mode', 'greedy'))
            elif event.key == pygame.K_2:
                self.handle_menu_action(('mode', 'minimax'))
            elif event.key == pygame.K_3:
                self.handle_menu_action(('mode', 'mcts'))
            elif event.key == pygame.K_4:
                self.handle_menu_action(('mode', 'strong'))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            for rect, action in self.menu_buttons:
                if rect.collidepoint(mx, my):
                    self.handle_menu_action(action)
                    break

    def draw_mark(self, x, y, state):
        cx = x * size + offset
        cy = y * size + offset
        l = 6
        color = (60, 40, 20)
        if state!='left':
            pygame.draw.line(self.screen, color, (cx - 3, cy - 3), (cx - 3 - l, cy - 3), 2)
            pygame.draw.line(self.screen, color, (cx - 3, cy - 3), (cx - 3, cy - 3 - l), 2)
            pygame.draw.line(self.screen, color, (cx - 3, cy + 3), (cx - 3 - l, cy + 3), 2)
            pygame.draw.line(self.screen, color, (cx - 3, cy + 3), (cx - 3, cy + 3 + l), 2)
        if state!='right':
            pygame.draw.line(self.screen, color, (cx + 3, cy - 3), (cx + 3 + l, cy - 3), 2)
            pygame.draw.line(self.screen, color, (cx + 3, cy - 3), (cx + 3, cy - 3 - l), 2)
            pygame.draw.line(self.screen, color, (cx + 3, cy + 3), (cx + 3+ l, cy + 3), 2)
            pygame.draw.line(self.screen, color, (cx + 3, cy + 3), (cx + 3, cy + 3 + l), 2)

    def position_key(self, turn=None):
        return (turn or self.turn, tuple(tuple(row) for row in self.board.board))

    def reset_repetition_tracker(self):
        self.position_stack = []
        self.position_counts = {}
        self.record_current_position()

    def record_current_position(self):
        key = self.position_key()
        self.position_stack.append(key)
        self.position_counts[key] = self.position_counts.get(key, 0) + 1

    def pop_current_position(self):
        if not self.position_stack:
            return
        key = self.position_stack.pop()
        count = self.position_counts.get(key, 0)
        if count <= 1:
            self.position_counts.pop(key, None)
        else:
            self.position_counts[key] = count - 1

    def move_would_repeat(self, sx, sy, ex, ey, side, seen_limit=2):
        captured = self.board.move(sx, sy, ex, ey)
        try:
            next_turn = 'b' if side == 'r' else 'r'
            key = self.position_key(next_turn)
            return self.position_counts.get(key, 0) >= seen_limit
        finally:
            self.board.undo_move(sx, sy, ex, ey, captured)

    def all_non_repeating_moves(self, side, seen_limit=2):
        moves = []
        for y in range(10):
            for x in range(9):
                piece = self.board.board[y][x]
                if piece == '.' or piece[0] != side:
                    continue
                for ex, ey in get_legal_moves(self.board, x, y):
                    if not self.move_would_repeat(x, y, ex, ey, side, seen_limit):
                        moves.append((x, y, ex, ey))
        return moves

    def fallback_non_repeating_move(self, side, seen_limit=2):
        moves = self.all_non_repeating_moves(side, seen_limit)
        if not moves:
            return None

        try:
            depth, time_limit = self.minimax_settings(side)
            fallback_ai = MinimaxAI(
                max_depth=max(2, min(4, depth)),
                time_limit=max(0.2, min(0.6, time_limit * 0.25)),
                use_neural=False,
            )
            fallback_ai.set_repetition_context(self.position_counts, seen_limit)
            move = fallback_ai.get_move(self.board, side)
            if move in moves:
                return move
        except Exception as exc:
            print(f"[AI] repetition fallback failed: {exc}")

        def score_move(move):
            sx, sy, ex, ey = move
            captured = self.board.move(sx, sy, ex, ey)
            try:
                score = evaluate_fast(self.board.board)
                return score if side == 'b' else -score
            finally:
                self.board.undo_move(sx, sy, ex, ey, captured)

        return max(moves, key=score_move)

    def legal_non_repeating_moves(self, sx, sy, side):
        return [
            (ex, ey)
            for ex, ey in get_legal_moves(self.board, sx, sy)
            if not self.move_would_repeat(sx, sy, ex, ey, side)
        ]


    def start_animation(self, sx, sy, ex, ey):
        self.animating = True
        self.anim_piece = self.board.board[sy][sx]
        self.anim_start = (sx, sy)
        self.anim_end = (ex, ey)
        self.anim_progress = 0
        # 先把目标位置的棋子记录（用于吃子）
        self.captured = self.board.board[ey][ex]
        if self.captured != '.':
            self.sound_manager.play_capture()
        else:
            self.sound_manager.play_move()

    def update_animation(self):
        self.anim_progress += self.speed

        if self.anim_progress >= 1:
            self.anim_progress = 1
            self.animating = False

            sx, sy = self.anim_start
            ex, ey = self.anim_end

            self.history.append((sx, sy, ex, ey, self.captured, self.turn))

            self.board.move(sx, sy, ex, ey)
            self.last_move = (sx, sy, ex, ey)

            self.turn = 'b' if self.turn == 'r' else 'r'
            self.record_current_position()
            state = get_game_state(self.board, self.turn)
            if state == "checkmate":
                self.winner_text = "红方得势" if self.turn == 'b' else "黑方破局"
                self.game_over = True
            elif state == "stalemate":
                self.winner_text = "和局收枰"
                self.game_over = True
            if is_in_check(self.board.board, self.turn):
                self.sound_manager.play_check()


    def draw_anim_piece(self):
        sx, sy = self.anim_start
        ex, ey = self.anim_end
        t = self.anim_progress
        t = t * t * (3 - 2 * t)  # smoothstep
        # 插值（线性）
        x = sx + (ex - sx) * t
        y = sy + (ey - sy) * t
        px = x * size + offset - 25
        lift = -8 * (1 - abs(2 * t - 1))  # 中间最高
        if t > 0.9:
            squash = 1 + 0.08 * (t - 0.9) / 0.1
        else:
            squash = 1
        py = y * size + offset - 25 + lift
        shadow = pygame.Surface((50, 50), pygame.SRCALPHA)
        pygame.draw.circle(shadow, (0, 0, 0, 80), (25, 25), 22)
        self.screen.blit(shadow, (px + 3, py + 3))
        img = pygame.transform.scale(self.images[self.anim_piece],(50, int(50 / squash)))
        self.screen.blit(img, (px, py))


    def draw_pieces(self):
        for i in range(10):
            for j in range(9):
                if self.animating and (j, i) == self.anim_start:
                    continue
                piece = self.board.board[i][j]
                if piece != '.':
                    x, y = self.location(i, j)
                    # 阴影
                    shadow = pygame.Surface((50, 50), pygame.SRCALPHA)
                    pygame.draw.circle(shadow, (0, 0, 0, 60), (25, 25), 22)
                    self.screen.blit(shadow, (x + 3, y + 3))
                    # 棋子
                    self.screen.blit(self.images[piece], (x, y))
        if self.animating:
            self.draw_anim_piece()

    def draw_last_move(self):
        if not self.last_move:
            return
        sx, sy, ex, ey = self.last_move
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        for x, y, alpha, arm in (
            (sx, sy, 45, 9),
            (ex, ey, 88, 13),
        ):
            cx = x * size + offset
            cy = y * size + offset
            color = (93, 49, 30, alpha)
            gap = 19
            for dx, dy, hx, hy in (
                (-gap, -gap, arm, 0), (-gap, -gap, 0, arm),
                (gap, -gap, -arm, 0), (gap, -gap, 0, arm),
                (-gap, gap, arm, 0), (-gap, gap, 0, -arm),
                (gap, gap, -arm, 0), (gap, gap, 0, -arm),
            ):
                start = (cx + dx, cy + dy)
                end = (cx + dx + hx, cy + dy + hy)
                pygame.draw.line(surface, color, start, end, 2)
            pygame.draw.circle(surface, (110, 57, 33, alpha // 2),
                               (cx, cy), 3)
        self.screen.blit(surface, (0, 0))

    #轻触发光
    def draw_hover(self):
        mx, my = pygame.mouse.get_pos()
        x = (mx - offset + size // 2) // size
        y = (my - offset + size // 2) // size

        if 0 <= x < 9 and 0 <= y < 10:
            piece = self.board.board[y][x]
            if piece != '.':
                cx = x * size + offset
                cy = y * size + offset
                surface = pygame.Surface((60, 60), pygame.SRCALPHA)
                pygame.draw.circle(surface, (200, 180, 140, 40), (30, 30), 25)
                self.screen.blit(surface, (cx - 30, cy - 30))

    def draw_selected(self):
        if self.selected:
            x, y = self.selected
            cx = x * size + offset
            cy = y * size + offset

            d = size_chess//2  # 距离中心
            l = size_chess//5  # 线长度

            surface = pygame.Surface((width, height), pygame.SRCALPHA)
            alpha = 220 + 30 * math.sin(pygame.time.get_ticks() * 0.005)
            color = (140, 60, 50, int(alpha))

            pygame.draw.line(surface, color, (cx - d, cy - d), (cx - d + l, cy - d), 3)
            pygame.draw.line(surface, color, (cx - d, cy - d), (cx - d, cy - d + l), 3)
            pygame.draw.line(surface, color, (cx + d, cy - d), (cx + d - l, cy - d), 3)
            pygame.draw.line(surface, color, (cx + d, cy - d), (cx + d, cy - d + l), 3)
            pygame.draw.line(surface, color, (cx - d, cy + d), (cx - d + l, cy + d), 3)
            pygame.draw.line(surface, color, (cx - d, cy + d), (cx - d, cy + d - l), 3)
            pygame.draw.line(surface, color, (cx + d, cy + d), (cx + d - l, cy + d), 3)
            pygame.draw.line(surface, color, (cx + d, cy + d), (cx + d, cy + d - l), 3)

            self.screen.blit(surface, (0, 0))


    def draw_valid_moves(self):
        for (x, y) in self.valid_moves:
            cx = x * size + offset
            cy = y * size + offset

            target = self.board.board[y][x]

            if target == '.':
                # 淡墨
                surface = pygame.Surface((40, 40), pygame.SRCALPHA)
                pygame.draw.circle(surface, (235, 225, 210, 120), (20, 20), 5)
                self.screen.blit(surface, (cx - 20, cy - 20))

            else:
                # 吃子 深色圆
                surface = pygame.Surface((40, 40), pygame.SRCALPHA)
                pygame.draw.circle(surface, (120, 40, 35, 220), (20, 20), 8)
                self.screen.blit(surface, (cx - 20, cy - 20))

    def draw_game_over(self):
        if not self.game_over:
            return
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((46, 30, 18, 145))
        for i in range(12):
            alpha = 7 + i
            pygame.draw.rect(overlay, (16, 8, 4, alpha),
                             (i * 7, i * 7, width - i * 14, height - i * 14), 12)
        self.screen.blit(overlay, (0, 0))

        panel = pygame.Rect(width // 2 - 165, height // 2 - 82, 330, 164)
        pygame.draw.rect(self.screen, (89, 50, 24),
                         (panel.x - 18, panel.y + 8, 12, panel.h - 16), border_radius=6)
        pygame.draw.rect(self.screen, (89, 50, 24),
                         (panel.right + 6, panel.y + 8, 12, panel.h - 16), border_radius=6)
        for x in (panel.x - 12, panel.right + 12):
            pygame.draw.circle(self.screen, (62, 33, 15), (x, panel.y + 8), 9)
            pygame.draw.circle(self.screen, (62, 33, 15), (x, panel.bottom - 8), 9)

        pygame.draw.rect(self.screen, (226, 196, 136), panel, border_radius=4)
        pygame.draw.rect(self.screen, (74, 42, 18), panel, 3, border_radius=4)
        pygame.draw.rect(self.screen, (170, 103, 48), panel.inflate(-18, -18), 1)

        self.draw_menu_text(self.winner_text, (width // 2, height // 2 - 18),
                            56, color=(110, 35, 25))
        self.draw_menu_text("R 复局    Esc 设局", (width // 2, height // 2 + 48),
                            24, color=(83, 48, 25))

        if not self.music_faded:
            pygame.mixer.music.fadeout(3000)
            self.music_faded = True

    #悔棋
    def undo(self):
        def undo_single_step():
            if not self.history:
                return False
            sx, sy, ex, ey, captured, turn = self.history.pop()
            self.pop_current_position()
            self.board.undo_move(sx, sy, ex, ey, captured)
            self.turn = turn
            return True
        if undo_single_step():
            while self.game_mode == 'human' and self.turn == 'b' and self.history:
                undo_single_step()
        self.last_move = None
        self.selected = None
        self.valid_moves = []

    def restart_game(self):
        if self.music_faded:
            try:
                pygame.mixer.music.play(-1)
            except Exception:
                pass
            self.music_faded = False
        self.board = Board()
        self.turn = 'r'
        self.selected = None
        self.valid_moves = []
        self.game_over = False
        self.winner_text = ""
        self.history = []
        self.last_move = None
        self.reset_repetition_tracker()
        self.animating = False

    def is_valid_ai_move(self, move, side):
        if not isinstance(move, (tuple, list)) or len(move) != 4:
            return False
        sx, sy, ex, ey = move
        if not all(isinstance(v, int) for v in (sx, sy, ex, ey)):
            return False
        if not (0 <= sx <= 8 and 0 <= ex <= 8 and 0 <= sy <= 9 and 0 <= ey <= 9):
            return False
        piece = self.board.board[sy][sx]
        if piece == '.' or piece[0] != side:
            return False
        return (ex, ey) in get_legal_moves(self.board, sx, sy)

    def ai_mode_for_side(self, side):
        if self.game_mode == 'arena':
            return self.red_ai_mode if side == 'r' else self.black_ai_mode
        return self.ai_mode if side == 'b' else None

    def should_ai_move(self, side):
        if self.game_mode == 'arena':
            return True
        return side == 'b'

    def get_ai_move(self, side):
        mode = self.ai_mode_for_side(side)
        engine = self.ai_engines.get(side)
        try:
            if mode == 'greedy':
                move = greedy_move(self.board, side)
            elif engine is None:
                return None
            else:
                if hasattr(engine, 'set_repetition_context'):
                    engine.set_repetition_context(self.position_counts, seen_limit=2)
                move = engine.get_move(self.board, side)
        except Exception as exc:
            print(f"[AI] move generation failed: {exc}")
            return None
        if move is not None and not self.is_valid_ai_move(move, side):
            print(f"[AI] ignored invalid move: {move}")
            return None
        if move is not None and self.move_would_repeat(*move, side):
            fallback = self.fallback_non_repeating_move(side)
            if fallback is None:
                self.game_over = True
                self.winner_text = "三复成和"
                return None
            return fallback
        return move

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if self.menu_active:
                    self.handle_menu_event(event)
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if not self.animating:
                            self.menu_active = True
                            self.menu_started_at = pygame.time.get_ticks()
                    if event.key == pygame.K_z:
                        if not self.animating:
                            self.undo()
                    if event.key == pygame.K_r:
                        self.restart_game()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.game_over:
                        continue
                    if self.animating:
                        continue
                    if self.should_ai_move(self.turn):
                        continue
                    mx, my = pygame.mouse.get_pos()
                    x, y = self.get_grid_pos(mx, my)

                    # 防越界
                    if not (0 <= x <= 8 and 0 <= y <= 9):
                        continue
                    piece = self.board.board[y][x]

                    if self.selected is None:
                        if piece != '.' and piece[0]==self.turn:
                            self.selected = (x, y)
                            self.valid_moves = self.legal_non_repeating_moves(
                                x, y, self.turn)
                    else:
                        sx, sy = self.selected

                        if (x,y) in self.valid_moves:
                            self.start_animation(sx, sy, x, y)
                        elif (x, y) in get_legal_moves(self.board, sx, sy):
                            if not self.all_non_repeating_moves(self.turn):
                                self.game_over = True
                                self.winner_text = "三复成和"

                        self.selected = None
                        self.valid_moves = []
            if self.menu_active:
                self.draw_start_menu()
                pygame.display.flip()
                self.clock.tick(60)
                continue
            if self.animating:
                self.update_animation()

            if self.should_ai_move(self.turn) and not self.game_over and not self.animating:
                move = self.get_ai_move(self.turn)
                if move:
                    sx, sy, ex, ey = move
                    self.start_animation(sx, sy, ex, ey)

            self.draw_board()
            self.draw_last_move()
            self.draw_pieces()
            self.draw_hover()
            self.draw_selected()
            self.draw_valid_moves()
            self.draw_game_over()
            pygame.display.flip()
            self.clock.tick(120)
