from re import sub
from serpent.game_agent import GameAgent
import collections
import serpent.utilities
from serpent.sprite_locator import SpriteLocator
import numpy as np
from datetime import datetime
from serpent.frame_grabber import FrameGrabber
import skimage.color
import skimage.measure
import serpent.cv
import serpent.utilities
import serpent.ocr
from serpent.sprite import Sprite

from serpent.input_controller import KeyboardEvent, KeyboardEvents
from serpent.input_controller import MouseEvent, MouseEvents

from serpent.config import config

from serpent.analytics_client import AnalyticsClient


class Environment:

    def __init__(self, name, game_api=None, input_controller=None):
        self.name = name

        self.game_api = game_api
        self.input_controller = input_controller

        self.analytics_client = AnalyticsClient(project_key=config["analytics"]["topic"])

    def perform_input(self, actions):
        discrete_keyboard_keys = set()
        discrete_keyboard_labels = set()

        for label, game_input, value in actions:
            # Discrete Space
            if value is None:
                if len(game_input) == 0:
                    discrete_keyboard_labels.add(label)
                    continue

                for game_input_item in game_input:
                    if isinstance(game_input_item, KeyboardEvent):
                        if game_input_item.event == KeyboardEvents.DOWN:
                            discrete_keyboard_keys.add(game_input_item.keyboard_key)
                            discrete_keyboard_labels.add(label)

        discrete_keyboard_keys_sent = False

        for label, game_input, value in actions:
            # Discrete
            if value is None:
                # Discrete - Keyboard
                if (len(discrete_keyboard_keys) == 0 and len(discrete_keyboard_labels) > 0) or isinstance(game_input[0] if len(game_input) else None, KeyboardEvent):
                    if not discrete_keyboard_keys_sent:
                        self.input_controller.handle_keys(list(discrete_keyboard_keys))

                        self.analytics_client.track(
                            event_key="GAME_INPUT",
                            data={
                                "keyboard": {
                                    "type": "DISCRETE",
                                    "label": " - ".join(sorted(discrete_keyboard_labels)),
                                    "inputs": sorted([keyboard_key.value for keyboard_key in discrete_keyboard_keys])
                                },
                                "mouse": {}
                            }
                        )

                        discrete_keyboard_keys_sent = True
                # Discrete - Mouse
                elif isinstance(game_input[0], MouseEvent):
                    for event in game_input:
                        if event.event == MouseEvents.CLICK:
                            self.input_controller.click(button=event.button)
                        elif event.event == MouseEvents.CLICK_DOWN:
                            self.input_controller.click_down(button=event.button)
                        elif event.event == MouseEvents.CLICK_UP:
                            self.input_controller.click_up(button=event.button)

                        self.analytics_client.track(
                            event_key="GAME_INPUT",
                            data={
                                "keyboard": {},
                                "mouse": {
                                    "type": "DISCRETE",
                                    "label": label,
                                    "label_technical": event.as_label,
                                    "input": event.as_input,
                                    "value": value
                                }
                            }
                        )
            # Continuous
            else:
                if isinstance(game_input[0], KeyboardEvent):
                    self.input_controller.tap_keys(
                        [event.keyboard_key for event in game_input],
                        duration=value
                    )

                    self.analytics_client.track(
                        event_key="GAME_INPUT",
                        data={
                            "keyboard": {
                                "type": "CONTINUOUS",
                                "label": label,
                                "inputs": sorted([event.keyboard_key.value for event in game_input]),
                                "duration": value
                            },
                            "mouse": {}
                        }
                    )
                elif isinstance(game_input[0], MouseEvent):
                    for event in game_input:
                        if event.event == MouseEvents.MOVE:
                            self.input_controller.move(x=event.x, y=event.y)

    def clear_input(self):
        self.input_controller.handle_keys([])


from serpent.machine_learning.reinforcement_learning.agents.rainbow_dqn_agent import RainbowDQNAgent

sprite_locator = SpriteLocator()
retry = skimage.io.imread('D:/Python/datasets/Jigoku_retry.png')[..., np.newaxis]
ranking = skimage.io.imread('D:/Python/datasets/jk_ranking.png')[..., np.newaxis]
hint = skimage.io.imread('D:/Python/datasets/jk_hint.png')[..., np.newaxis]
menu = skimage.io.imread('D:/Python/datasets/jk_menu.png')[..., np.newaxis]
sprite_retry = Sprite("Retry", image_data=retry)
sprite_ranking = Sprite("Ranking", image_data=ranking)
sprite_hint = Sprite("Hint", image_data=hint)
sprite_menu = Sprite("Damn it, Hakisa! Just play!", image_data=menu)
import enum


class InputControlTypes(enum.Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class SerpentTenshi2GameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.game_state = None

        self._reset_game_state()

    def _reset_game_state(self):
        self.game_state = {
            "hp": collections.deque(np.full((8,), 4), maxlen=8),
            "power": 1.0,
            "aura": collections.deque(np.full((8,), 100), maxlen=8),
            "score_multiplier": 1.00,
            "score": 0,
            "run_reward": 0,
            "current_run": 1,
            "current_run_steps": 0,
            "current_run_hp": 0,
            "current_run_power": 0,
            "curent_run_aura": 0,
            "current_run_score_mult": 0,
            "current_run_score": 0,
            "run_predicted_actions": 0,
            "last_run_duration": 0,
            "record_time_alive": dict(),
            "random_time_alive": None,
            "random_time_alives": list(),
            "run_timestamp": datetime.utcnow(),
        }


    def setup_play(self):
    
        self.game_inputs = [
            {
                "name": "CONTROLS",
                "control_type": InputControlTypes.DISCRETE,
                "inputs": self.game.api.combine_game_inputs(["MOVEMENT", "SHOOTING"]),
                "value": None
            }
        ]

        self.agent = RainbowDQNAgent("Tenshi", game_inputs=self.game_inputs)
        
        
        
        #self.agent.add_human_observations_to_replay_memory()
        

    def handle_play(self, game_frame):
        import pyautogui 
        search_retry = sprite_locator.locate(sprite=sprite_retry, game_frame=game_frame)
        search_hint = sprite_locator.locate(sprite=sprite_hint, game_frame=game_frame)
        if search_retry != None or search_hint != None:
            pyautogui.press('z')
        
        search_ranking = sprite_locator.locate(sprite=sprite_ranking, game_frame=game_frame)
        if search_ranking != None:
            pyautogui.press('Enter')

        search_menu = sprite_locator.locate(sprite=sprite_menu, game_frame=game_frame)
        if search_menu != None:
            import time
            pyautogui.press('Enter', presses=3, interval=1.5)
            time.sleep(2.5)

        hp = self._measure_hp(game_frame)
        aura = self._measure_aura(game_frame)

        self.game_state['hp'].appendleft(hp)
        self.game_state['power'] = self._measure_power(game_frame)
        self.game_state['aura'].appendleft(aura)
        self.game_state['score_multiplier'] = self._measure_mscore(game_frame)
        self.game_state['score'] = self._measure_score(game_frame)

        reward = self._reward(self.game_state, game_frame)
        if reward is None:
            reward = 0
        else:
            pass

        self.game_state['run_reward'] = reward
        
        
        self.agent.observe(reward=reward)
        
        frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")
        agent_actions = self.agent.generate_actions(frame_buffer)

        Environment.perform_input(self, actions=agent_actions)

        # Saving model each N steps:
        if self.agent.current_step % 1000 == 0:
            self.agent.save_model()

        
        serpent.utilities.clear_terminal()
        print(f"Current HP: {self.game_state['hp'][0]}")
        print(f"Current Power: {self.game_state['power']}")
        print(f"Current Aura: {self.game_state['aura'][0]}")
        print(f"Current Score: {self.game_state['score']}")
        print(f"Current Score Multiplier: {self.game_state['score_multiplier']}")
        print(f"Current Reward: {self.game_state['run_reward']}")
        print(f"Current Run steps: {self.game_state['current_run_steps']}")
        
            
    def _measure_hp(self, game_frame):
        score_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["Lifes"])
            
        score_grayscale = np.array(skimage.color.rgb2gray(score_area_frame) * 255, dtype="uint8")
            
        score = serpent.ocr.perform_ocr(image=score_grayscale, scale=10, order=5, horizontal_closing=10, vertical_closing=5,config='--psm 6 -c tessedit_char_blacklist=-')

        score = score.replace('S', '5')
        score = score.replace('s', '8')
        score = score.replace('e', '2')
        score = score.replace('O', '0')
        score = score.replace('B', '8')
        score = score.replace('I', '1')
        score = score.replace('l', '1')
        score = score.replace("o", "4")
        score = score.replace("b", "4")

        score = sub(r'[^0-9]\.', '', score)

        self.game_state['current_run_hp'] = score

        try:
            return float(score)
        except ValueError:
            return 1.0

    def _measure_score(self, game_frame):
        score_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["Score"])
            
        score_grayscale = np.array(skimage.color.rgb2gray(score_area_frame) * 255, dtype="uint8")
            
        score = serpent.ocr.perform_ocr(image=score_grayscale, scale=10, order=5, horizontal_closing=10, vertical_closing=5, config='--psm 6 -c tessedit_char_blacklist=.-')
        
        # Fixing OCR flaws. Unfortunately, both 5 and 9 are recognized as S.

        score = score.replace('S', '5')
        score = score.replace('s', '8')
        score = score.replace('e', '2')
        score = score.replace('O', '0')
        score = score.replace('B', '8')
        score = score.replace('I', '1')
        score = score.replace('l', '1')
        score = score.replace("o", "4")
        score = score.replace("b", "4")

        score = sub(r'[^0-9]', '', score)

        self.game_state['current_run_score'] = score

        try:
            return int(score)
        except ValueError:
            return 1

    def _measure_power(self, game_frame):
        score_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["Power"])
            
        score_grayscale = np.array(skimage.color.rgb2gray(score_area_frame) * 255, dtype="uint8")
            
        score = serpent.ocr.perform_ocr(image=score_grayscale, scale=10, order=5, horizontal_closing=10, vertical_closing=5, config='--psm 6 -c tessedit_char_blacklist=-')

        score = score.replace('S', '5')
        score = score.replace('s', '8')
        score = score.replace('e', '2')
        score = score.replace('O', '0')
        score = score.replace('B', '8')
        score = score.replace('I', '1')
        score = score.replace('l', '1')
        score = score.replace("o", "4")
        score = score.replace("b", "4")

        score = sub(r'[^0-9]\.', '', score)

        self.game_state['current_run_power'] = score

        try:
            score = float(score)
            if score > 5.0:
                return 5.0
            else:
                return score
        except ValueError:
            return 5.0
        
    def _measure_aura(self, game_frame):
        score_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["Aura"])
            
        score_grayscale = np.array(skimage.color.rgb2gray(score_area_frame) * 255, dtype="uint8")
            
        score = serpent.ocr.perform_ocr(image=score_grayscale, scale=10, order=5, horizontal_closing=10, vertical_closing=5, config='--psm 8 -c tessedit_char_blacklist=-')

        score = score.replace('S', '5')
        score = score.replace('s', '8')
        score = score.replace('e', '2')
        score = score.replace('O', '0')
        score = score.replace('B', '8')
        score = score.replace('I', '1')
        score = score.replace('l', '1')
        score = score.replace("o", "4")
        score = score.replace("b", "4")

        score = sub(r'[^0-9]', '', score)
            
        self.game_state['current_run_aura'] = score

        try:
            score = int(score)
            if score > 200:
                return 200
            else:
                return score
        except ValueError:
            return 200

    def _measure_mscore(self, game_frame):
        score_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["Multiplier_score"])
            
        score_grayscale = np.array(skimage.color.rgb2gray(score_area_frame) * 255, dtype="uint8")

        score = serpent.ocr.perform_ocr(image=score_grayscale, scale=10, order=5, horizontal_closing=10, vertical_closing=5, config='--psm 6 -c tessedit_char_blacklist=-')

        score = score.replace('S', '5')
        score = score.replace('s', '8')
        score = score.replace('e', '2')
        score = score.replace('O', '0')
        score = score.replace('B', '8')
        score = score.replace('I', '1')
        score = score.replace('l', '1')
        score = score.replace("o", "4")
        score = score.replace("b", "4")

        score = sub(r'[^0-9]\.', '', score)

        self.game_state['current_run_score_mult'] = score

        try:
            return float(score)
        except ValueError:
            return 1.00


    def _reward(self, game_state, game_frame):
        if self.game_state['hp'][0] is None:
            pass
        elif self.game_state['hp'][0] >= 1.00:
            return (self.game_state['score'] * self.game_state['score_multiplier']) + (self.game_state['power'] * (self.game_state['aura'][0]/100))
        else:
            return -(1000000/(self.game_state['score'] * self.game_state['score_multiplier']))
 
