import cv2
import numpy as np
import pyautogui
import mss
import time
import os
import traceback

# Настроим pyautogui - ноль задержек для мгновенной реакции, но фейлсейф оставим - мало ли что
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

class Config:
    """Настройки бота - тут всё, что можно подкрутить под себя"""
    # Где примерно стоит динозавр по умолчанию, если не нашли
    DINO_X_FALLBACK = 55
    # Какой ширины/высоты бывает наш герой
    DINO_WIDTH_RANGE = (35, 60)
    DINO_HEIGHT_RANGE = (40, 70)

    # Дистанции, на которых срабатывает прыжок - для больших кактусов, маленьких и дефолт
    TRIGGER_BASE_BIG = 110
    TRIGGER_BASE_SMALL = 130
    TRIGGER_BASE_DEFAULT = 100

    # Сколько держать пробел - обычный прыжок и короткий
    JUMP_HOLD_NORMAL = 0.095
    JUMP_HOLD_SHORT = 0.060

    # Насколько сдвинуть линию горизонта относительно динозавра
    HORIZON_SHIFT_FROM_DINO = -30

    # Параметры для группировки препятствий - если объект широкий - считаем его кластером
    CLUSTER_THRESHOLD = 45
    # Радиус, в котором препятствия считаются рядом
    ISOLATED_RADIUS = 80

    # Логика для птиц - когда приседать и когда отпускать
    BIRD_TRIGGER = 200
    BIRD_RELEASE = -50
    MIN_DUCK_TIME = 0.4  # Минимальное время приседа, чтобы не дёргалось

    # Ускорение игры - каждые 25 секунд увеличиваем сложность
    SPEED_INTERVAL = 25.0
    SPEED_FACTOR_EARLY = 1.20  # Коэффициент ускорения на старте
    SPEED_FACTOR_LATE = 1.17   # И позже, когда уже становится быстро
    SPEED_SWITCH_TIME = 60.0   # Через минуту переключаемся на поздний режим

    # Минимальные дистанции для разных типов препятствий
    DIST_MIN = 320
    DIST_MIN_SMALL = 340
    DIST_MIN_BIG = 320

    # Детекция смерти - сколько кадров без изменений = игра окончена
    DEATH_STATIC_FRAMES = 8
    DEATH_FRAME_THRESHOLD = 1200
    MIN_GAME_TIME_BEFORE_DEATH_CHECK = 3.0  # Не проверяем смерть в первые 3 секунды

    # Фаст-фолл: ускоренное приземление после прыжка
    FAST_FALL_DELAY = 0.08  # Ждём чуть-чуть после прыжка, прежде чем давить вниз
    FAST_FALL_MAX_TIME = 0.5  # Максимальная длительность фаст-фолла
    GROUND_CHECK_OFFSET = 10  # Насколько ниже «ног» проверять землю

    # Размер области игры, которую захватываем
    GAME_WIDTH = 1200
    GAME_HEIGHT = 300
    ROI_AHEAD = 800  # Насколько далеко вперёд смотрим на препятствия

    # Отладка: вкл/выкл и пропуск кадров, чтобы не перегружать экран
    DEBUG_MODE = True
    DEBUG_SKIP_FRAMES = 3


class DinoBot:
    def __init__(self):
        print("T-Rex Bot started", flush=True)

        # Координаты динозавра - заполнятся после поиска
        self.DINO_X = None
        self.dino_base_y = None
        self.ground_y = None
        self.dino_feet_y = None  # Точка, где у динозавра ноги это удобно для детекции приземления

        self.cfg = Config()
        self.GAME_AREA = None  # Область экрана, где идёт игра
        self.last_action_time = 0  # Чтобы не спамить клавишами
        self.frame_history = []  # Кэш кадров для детекции смерти
        self.static_frames = 0  # Счётчик неподвижных кадров
        self.game_start_time = 0
        self.is_dead = False
        self.last_obstacles = []  # Последние найденные препятствия
        self.is_ducking = False  # Флаг - приседаем ли сейчас
        self.duck_start_time = 0

        # Флаги для фаст-фолла
        self.is_jumping = False
        self.jump_obstacle_cleared = False  # Прошли ли уже препятствие в прыжке
        self.fast_fall_active = False
        self.last_jump_time = 0
        self.last_known_y = None  # Последняя известная высота динозавра

        # Счётчик FPS для отладки
        self.fps_counter = 0
        self.fps_last_time = time.time()
        self.fps_history = []

        self.DEBUG_MODE = Config.DEBUG_MODE
        self.DEBUG_SKIP_FRAMES = Config.DEBUG_SKIP_FRAMES
        self._debug_frame_counter = 0

    def find_game_area_by_dino(self):
        """Ищем игру на экране: сначала по шаблону, потом по контурам"""
        print("Searching for game...", flush=True)

        with mss.MSS() as sct:
            monitor = sct.monitors[1]
            screenshot = np.array(sct.grab(monitor))

        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
        h, w = gray.shape

        # Пробуем найти по шаблону - если есть картинка динозавра
        if os.path.exists('dino_template.png'):
            print("Found dino_template.png, using template matching...", flush=True)
            template = cv2.imread('dino_template.png', 0)
            if template is not None:
                res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.7)
                pts = list(zip(*loc[::-1]))

                if len(pts) > 0:
                    dx, dy = pts[0]
                    th, tw = template.shape
                    ground_y = dy + th + 5
                    self._set_game_area_from_dino(dx, dy, tw, th, ground_y, w, h)
                    print("Game found by template!", flush=True)
                    return True

        # Шаблон не сработал - ищем по контурам и линии земли
        print("Template not found, searching by contours...", flush=True)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Ищем самую насыщенную горизонтальную линию - это, скорее всего, земля
        search_start_y = int(h * 0.15)
        search_end_y = int(h * 0.75)

        ground_y = None
        max_pixels = 0

        for y in range(search_start_y, search_end_y):
            row = binary[y, :]
            count = cv2.countNonZero(row)
            if count > w * 0.10 and count > max_pixels:
                max_pixels = count
                ground_y = y

        if ground_y is None:
            print("Ground line not found. Using manual setup.", flush=True)
            return self._setup_manual_area()

        print(f"Ground line found at y={ground_y}", flush=True)

        # Ищем динозавра рядом с землёй
        roi_top = max(0, ground_y - 150)
        roi_bottom = min(h, ground_y + 20)
        roi_left = 0
        roi_right = w // 2

        roi = binary[roi_top:roi_bottom, roi_left:roi_right].copy()
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 600 < area < 6000:  # Фильтруем по размеру - динозавр не слишком маленький и не огромный
                x, y, bw, bh = cv2.boundingRect(cnt)
                ar = bw / float(bh) if bh > 0 else 0

                # Проверяем пропорции и близость к земле
                if 0.4 <= ar <= 1.5 and abs((y + roi_top + bh) - ground_y) < 30:
                    candidates.append((x + roi_left, y + roi_top, bw, bh))

        if candidates:
            candidates.sort(key=lambda c: c[0])  # Берём самого левого - это и есть наш динозавр
            dx, dy, dw, dh = candidates[0]
            self._set_game_area_from_dino(dx, dy, dw, dh, ground_y, w, h)
            return True

        print("Dino not found. Using manual setup.", flush=True)
        return self._setup_manual_area()

    def _set_game_area_from_dino(self, dx, dy, dw, dh, ground_y, screen_w, screen_h):
        """Сохраняем координаты игровой зоны относительно найденного динозавра"""
        game_left = max(0, dx - 100)
        game_top = max(0, ground_y - 200)
        game_width = min(self.cfg.GAME_WIDTH, screen_w - game_left)
        game_height = min(self.cfg.GAME_HEIGHT, screen_h - game_top)

        self.GAME_AREA = {
            'top': game_top,
            'left': game_left,
            'width': game_width,
            'height': game_height
        }
        self.DINO_X = dx - game_left + dw // 2
        self.dino_base_y = (dy + dh) - game_top
        self.dino_feet_y = dy + dh  # Запоминаем, где у динозавра начало ноги

        print(f"Game area set: {self.GAME_AREA}", flush=True)
        print(f"Dino X: {self.DINO_X}, Feet Y: {self.dino_feet_y}", flush=True)

    def _setup_manual_area(self):
        """Ручная настройка - если автопоиск не сработал"""
        print("\nManual setup required.", flush=True)
        input("Position cursor over dino and press ENTER...")
        x, y = pyautogui.position()

        self.GAME_AREA = {
            'top': max(0, y - 180),
            'left': max(0, x - 100),
            'width': self.cfg.GAME_WIDTH,
            'height': self.cfg.GAME_HEIGHT
        }
        self.DINO_X = 100
        self.ground_y = 220
        self.dino_feet_y = y + 20
        return True

    def capture(self):
        """Делаем скриншот игровой области"""
        area = self.GAME_AREA or {'top': 280, 'left': 1050, 'width': 763, 'height': 180}
        with mss.MSS() as sct:
            return cv2.cvtColor(np.array(sct.grab(area)), cv2.COLOR_BGRA2BGR)

    def preprocess(self, img):
        """Превращаем кадр в бинарную маску: чёрное - препятствия, белое - фон"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def find_ground(self, binary):
        """Определяем линию земли: либо от динозавра, либо «примерно»"""
        if self.dino_base_y is not None:
            return self.dino_base_y + self.cfg.HORIZON_SHIFT_FROM_DINO
        h, _ = binary.shape
        return h - 30

    def get_dino_y_position(self, binary):
        """Находим, где сейчас ноги динозавра - сканируем его колонку сверху вниз"""
        h, w = binary.shape
        check_x = max(0, min(w - 1, self.DINO_X))

        for y in range(int(self.dino_base_y or h / 2), max(0, int((self.dino_base_y or h / 2) - 60)), -1):
            if y < h and binary[y, check_x] > 0:
                return y
        return self.dino_base_y if self.dino_base_y else h - 30

    def is_dino_on_ground(self, binary, ground_y):
        """Проверяем: динозавр на земле или ещё в прыжке?"""
        current_y = self.get_dino_y_position(binary)
        self.last_known_y = current_y

        distance_to_ground = abs(current_y - ground_y)
        return distance_to_ground <= 5  # Допуск в 5 пикселей - достаточно точно

    def find_obstacles(self, binary, ground_y):
        """Ищем препятствия впереди динозавра: кактусы, птицы, кластеры"""
        h, w = binary.shape
        clean = binary.copy()
        gy = max(0, min(h, ground_y))
        clean[gy:h, :] = 0  # Убираем землю, чтобы не мешала

        roi_left = max(0, self.DINO_X - 20)
        roi_right = min(w, self.DINO_X + self.cfg.ROI_AHEAD)
        roi = clean[:, roi_left:roi_right]

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstacles = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 60 or area > 2500:  # Слишком мелкие или огромные - игнорируем
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            x += roi_left

            if x < self.DINO_X - 20:  # То, что уже позади - не интересно
                continue

            real_dist = x - self.DINO_X
            object_bottom = y + bh

            # Если объект очень близко и касается земли - скорее всего, это сам динозавр, пропускаем
            if abs(real_dist) < 25 and object_bottom > ground_y - 30:
                continue

            # Определяем тип: птица летает выше земли, остальное - кактусы
            if object_bottom < ground_y - 5 and real_dist > 15:
                otype, color = 'BIRD', (0, 255, 255)
            else:
                is_cluster = bw > self.cfg.CLUSTER_THRESHOLD or area > 450
                otype, color = ('CLUSTER', (0, 0, 255)) if is_cluster else ('SINGLE', (0, 255, 0))

            obstacles.append({
                'x': x, 'y': y, 'w': bw, 'h': bh, 'right': x + bw,
                'type': otype, 'real_dist': real_dist, 'color': color
            })

        obstacles.sort(key=lambda o: o['real_dist'])  # Ближайшие - в начале списка
        return obstacles

    def check_death_robust(self, current_frame):
        """Детекция смерти: если кадр не меняется несколько раз подряд — игра окончена"""
        elapsed = time.time() - self.game_start_time
        if elapsed < self.cfg.MIN_GAME_TIME_BEFORE_DEATH_CHECK:
            return False  # Не проверяем в первые секунды - динозавр ещё разгоняется

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        self.frame_history.append(gray)

        if len(self.frame_history) > 40:
            self.frame_history.pop(0)
        if len(self.frame_history) < 25:
            return False  # Ждём, пока накопится достаточно кадров

        old = self.frame_history[0]
        diff = cv2.absdiff(old, gray)
        score = np.sum(diff)

        self.static_frames = self.static_frames + 1 if score < self.cfg.DEATH_FRAME_THRESHOLD else 0
        return self.static_frames >= self.cfg.DEATH_STATIC_FRAMES

    def handle_death(self):
        """Игра окончена: отпускаем клавиши, ждём и рестартуем"""
        if self.is_ducking:
            pyautogui.keyUp('down')
            self.is_ducking = False

        print("\nGAME OVER - Restarting...", flush=True)
        time.sleep(1.5)
        pyautogui.press('space')
        time.sleep(0.5)

        # Сбрасываем все флаги и таймеры
        self.is_dead = False
        self.frame_history = []
        self.static_frames = 0
        self.game_start_time = time.time()
        self.last_speed_update = self.game_start_time
        self.last_obstacles = []
        self.last_action_time = 0
        self.duck_start_time = 0
        self.is_ducking = False
        self.fps_counter = 0
        self.fps_last_time = time.time()

        # И фаст-фолл тоже сбрасываем
        self.is_jumping = False
        self.jump_obstacle_cleared = False
        self.fast_fall_active = False
        self.last_jump_time = 0

    def is_isolated(self, target, obstacles):
        """Проверяем: препятствие одно или рядом есть другие?"""
        target_center = target['x'] + target['w'] // 2
        for obs in obstacles:
            if obs is target:
                continue
            if abs((obs['x'] + obs['w'] // 2) - target_center) < self.cfg.ISOLATED_RADIUS:
                return False
        return True

    def _get_scaled_trigger(self, base_value, dist_min_override=None):
        """Рассчитываем дистанцию прыжка с учётом ускорения игры"""
        elapsed = time.time() - self.game_start_time
        level = int(elapsed // self.cfg.SPEED_INTERVAL)
        factor = self.cfg.SPEED_FACTOR_LATE if elapsed > self.cfg.SPEED_SWITCH_TIME else self.cfg.SPEED_FACTOR_EARLY

        limit = dist_min_override if dist_min_override is not None else self.cfg.DIST_MIN
        return int(min(limit, round(base_value * (factor ** level))))

    def decide_jump(self, obstacles):
        """Принимаем решение: прыгать или нет, и если да, то как"""
        if not obstacles:
            return None, self._get_scaled_trigger(self.cfg.TRIGGER_BASE_DEFAULT), self.cfg.JUMP_HOLD_NORMAL

        for obs in obstacles:
            if obs['type'] in ['SINGLE', 'CLUSTER']:
                # Одиночный маленький кактус - можно прыгнуть короче
                if obs['type'] == 'SINGLE' and self.is_isolated(obs, obstacles):
                    base_trigger = self.cfg.TRIGGER_BASE_SMALL
                    hold = self.cfg.JUMP_HOLD_SHORT
                    dist_limit = self.cfg.DIST_MIN_SMALL
                else:
                    base_trigger = self.cfg.TRIGGER_BASE_BIG
                    hold = self.cfg.JUMP_HOLD_NORMAL
                    dist_limit = self.cfg.DIST_MIN_BIG

                scaled = self._get_scaled_trigger(base_trigger, dist_min_override=dist_limit)

                if 10 < obs['real_dist'] < scaled:
                    return obs, scaled, hold

        return None, self._get_scaled_trigger(self.cfg.TRIGGER_BASE_DEFAULT), self.cfg.JUMP_HOLD_NORMAL

    def act_jump(self, hold_time=0.04):
        """Нажимаем пробел на нужное время"""
        now = time.time()
        if now - self.last_action_time < 0.08:  # Анти-спам
            return False
        self.last_action_time = now

        self.is_jumping = True
        self.jump_obstacle_cleared = False
        self.last_jump_time = now

        print(f"JUMP! (hold={hold_time:.3f}s)", flush=True)

        pyautogui.keyDown('space')
        time.sleep(hold_time)
        pyautogui.keyUp('space')
        return True

    def update_speed(self):
        """Отслеживаем ускорение игры - для статистики в отладке"""
        now = time.time()
        elapsed = now - self.game_start_time

        if now - self.last_speed_update >= self.cfg.SPEED_INTERVAL:
            factor = self.cfg.SPEED_FACTOR_LATE if elapsed > self.cfg.SPEED_SWITCH_TIME else self.cfg.SPEED_FACTOR_EARLY
            self.last_speed_update = now
            pct = int((factor - 1) * 100)
            phase = "HARD" if elapsed > self.cfg.SPEED_SWITCH_TIME else "NORMAL"
            print(f"Speed up! {phase} +{pct}%", flush=True)
            return True
        return False

    def _render_debug(self, screen, binary, obstacles, target, trigger_dist, action):
        """Рисуем отладочную картинку: что видит бот и что планирует"""
        try:
            debug = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            gy = int(max(0, min(debug.shape[0], self.ground_y)))
            cv2.rectangle(debug, (0, gy), (debug.shape[1], debug.shape[0]), (30, 30, 30), -1)
            view = np.hstack([screen, debug])

            trigger_x = int(self.DINO_X + trigger_dist)
            for off_x in [0, screen.shape[1]]:
                cv2.line(view, (int(off_x), gy), (int(off_x + screen.shape[1]), gy), (0, 80, 255), 2)
                cv2.line(view, (int(off_x + trigger_x), 0),
                         (int(off_x + trigger_x), int(view.shape[0])), (0, 0, 255), 2)
                for obj in obstacles:
                    cv2.rectangle(view, (int(off_x + obj['x']), obj['y']),
                                  (int(off_x + obj['x'] + obj['w']), obj['y'] + obj['h']), obj['color'], 1)

            elapsed = time.time() - self.game_start_time
            speed_level = int(elapsed // self.cfg.SPEED_INTERVAL) + 1
            obs_info = f"[{target['type']}]" if target else ""
            status_color = (0, 255, 255) if self.is_ducking else (0, 255, 0)

            avg_fps = np.mean(self.fps_history) if self.fps_history else 0

            cv2.putText(view, f'{action or "WAIT"}', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(view, f'{trigger_dist}{obs_info}|L{speed_level}', (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(view, f'DINO@{self.DINO_X}', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            cv2.putText(view, f'FPS: {avg_fps:.1f}', (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Статус фаст-фолла
            if self.fast_fall_active:
                cv2.putText(view, 'FAST FALL', (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            elif self.is_jumping:
                cv2.putText(view, 'JUMPING', (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow('Debug', view)
        except Exception as e:
            print(f"Debug render error: {e}", flush=True)
            traceback.print_exc()

    def run(self):
        """Основной цикл бота"""
        print("1. Make sure the game is open", flush=True)
        print("2. Bot will find the game automatically", flush=True)
        print("3. If not found, click on dino and press Enter\n", flush=True)

        pyautogui.keyUp('down')
        self.is_ducking = False
        time.sleep(1)

        self.find_game_area_by_dino()

        time.sleep(1)
        self.game_start_time = time.time()
        self.last_speed_update = self.game_start_time
        self.fps_last_time = time.time()

        if self.DEBUG_MODE:
            cv2.namedWindow('Debug', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Debug', 1100, 320)

        while True:
            t0 = time.time()
            screen = self.capture()
            binary = self.preprocess(screen)
            self.ground_y = self.find_ground(binary)
            obstacles = self.find_obstacles(binary, self.ground_y)
            self.last_obstacles = obstacles
            self.update_speed()

            # Проверяем, не умерли ли
            if not self.is_dead and self.check_death_robust(screen):
                self.is_dead = True
                self.handle_death()
                continue

            action = None
            target, trigger_dist, jump_hold = self.decide_jump(obstacles)

            #Логика для птиц
            bird_in_zone = False
            for o in obstacles:
                if o['type'] == 'BIRD':
                    bird_right_dist = o.get('right', o['x']) - self.DINO_X
                    bird_left_dist = o['x'] - self.DINO_X
                    if bird_left_dist < self.cfg.BIRD_TRIGGER and bird_right_dist > self.cfg.BIRD_RELEASE:
                        bird_in_zone = True
                        break

            if bird_in_zone:
                if not self.is_ducking:
                    pyautogui.keyDown('down')
                    self.is_ducking = True
                    self.duck_start_time = time.time()
                    print("DUCK", flush=True)
                action = 'DUCK_HOLD'
                if self.fast_fall_active:
                    pyautogui.keyUp('down')
                    self.fast_fall_active = False
            else:
                if self.is_ducking:
                    if time.time() - self.duck_start_time > self.cfg.MIN_DUCK_TIME:
                        pyautogui.keyUp('down')
                        self.is_ducking = False
                        self.last_action_time = time.time()

                #Фаст-фолл: ускоряем приземление
                if self.is_jumping and not self.fast_fall_active:
                    # Смотрим, не осталось ли препятствие позади
                    for obs in obstacles:
                        if obs['type'] in ['SINGLE', 'CLUSTER']:
                            if obs['right'] < self.DINO_X - 15:
                                self.jump_obstacle_cleared = True
                                break

                    elapsed_since_jump = time.time() - self.last_jump_time

                    # Условия для активации фаст-фолла
                    if (self.jump_obstacle_cleared and
                            elapsed_since_jump > self.cfg.FAST_FALL_DELAY and
                            elapsed_since_jump < self.cfg.FAST_FALL_MAX_TIME):

                        on_ground = self.is_dino_on_ground(binary, self.ground_y)

                        if not on_ground:
                            pyautogui.keyDown('down')
                            self.fast_fall_active = True
                            print(f"FAST FALL! (elapsed={elapsed_since_jump:.3f}s)", flush=True)

                # Проверяем, приземлились ли
                if self.fast_fall_active:
                    on_ground = self.is_dino_on_ground(binary, self.ground_y)
                    elapsed_since_jump = time.time() - self.last_jump_time

                    if on_ground:
                        pyautogui.keyUp('down')
                        self.fast_fall_active = False
                        self.is_jumping = False
                        self.jump_obstacle_cleared = False
                        print("LANDED!", flush=True)
                    elif elapsed_since_jump > self.cfg.FAST_FALL_MAX_TIME:
                        # Страховка: если что-то пошло не так
                        pyautogui.keyUp('down')
                        self.fast_fall_active = False
                        self.is_jumping = False
                        print("FAST FALL timeout", flush=True)

                # Обычный прыжок, если нет других действий
                if not self.is_ducking and not self.fast_fall_active and target:
                    print(f"JUMP | {target['type']} | {target['real_dist']}px", flush=True)
                    self.act_jump(hold_time=jump_hold)
                    action = 'JUMP'

            # Отладочная визуализация
            if self.DEBUG_MODE:
                self._debug_frame_counter += 1
                if self._debug_frame_counter % self.DEBUG_SKIP_FRAMES == 0:
                    self._render_debug(screen, binary, obstacles, target, trigger_dist, action)

            key = cv2.waitKey(1) & 0xFF if self.DEBUG_MODE else -1
            if key == ord('q'):
                break

            # Считаем FPS
            self.fps_counter += 1
            elapsed_since_fps = time.time() - self.fps_last_time
            if elapsed_since_fps >= 1.0:
                current_fps = self.fps_counter / elapsed_since_fps
                self.fps_history.append(current_fps)
                if len(self.fps_history) > 10:
                    self.fps_history.pop(0)
                print(f"FPS: {current_fps:.1f} (avg: {np.mean(self.fps_history):.1f})", flush=True)
                self.fps_counter = 0
                self.fps_last_time = time.time()

            # Держим стабильный темп кадров
            frame_time = time.time() - t0
            if frame_time < 0.016:
                time.sleep(0.016 - frame_time)

        if self.DEBUG_MODE:
            cv2.destroyAllWindows()
        print("Stopped.", flush=True)


if __name__ == "__main__":
    DinoBot().run()