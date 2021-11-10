import re
from serpent.game_agent import GameAgent
import collections
from serpent.input_controller import KeyboardKey
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

import time

from serpent.input_controller import KeyboardEvent, KeyboardEvents
from serpent.input_controller import MouseEvent, MouseEvents

from serpent.config import config

from serpent.analytics_client import AnalyticsClient


class Environment:

    def __init__(self, name, game_api=None, input_controller=None):
        self.name = name

        self.game_api = game_api
        self.input_controller = input_controller

        self.game_state = dict()

        self.analytics_client = AnalyticsClient(project_key=config["analytics"]["topic"])

        self.reset()

    @property
    def episode_duration(self):
        return time.time() - self.episode_started_at

    @property
    def episode_over(self):
        if self.episode_maximum_steps is not None:
            return self.episode_steps >= self.episode_maximum_steps
        else:
            return False

    @property
    def new_episode_data(self):
        return dict()

    @property
    def end_episode_data(self):
        return dict()

    def new_episode(self, maximum_steps=None, reset=False):
        self.episode_steps = 0
        self.episode_maximum_steps = maximum_steps

        self.episode_started_at = time.time()

        if not reset:
            self.episode += 1

        self.analytics_client.track(
            event_key="NEW_EPISODE",
            data={
                "episode": self.episode,
                "episode_data": self.new_episode_data,
                "maximum_steps": self.episode_maximum_steps
            }
        )

    def end_episode(self):
        self.analytics_client.track(
            event_key="END_EPISODE",
            data={
                "episode": self.episode,
                "episode_data": self.end_episode_data,
                "episode_steps": self.episode_steps,
                "maximum_steps": self.episode_maximum_steps
            }
        )

    def episode_step(self):
        self.episode_steps += 1
        self.total_steps += 1

        self.analytics_client.track(
            event_key="EPISODE_STEP",
            data={
                "episode": self.episode,
                "episode_step": self.episode_steps,
                "total_steps": self.total_steps
            }
        )

    def reset(self):
        self.total_steps = 0

        self.episode = 0
        self.episode_steps = 0

        self.episode_maximum_steps = None

        self.episode_started_at = None

    def update_game_state(self, game_frame):
        raise NotImplementedError()

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
                        elif event.event == MouseEvents.CLICK_SCREEN_REGION:
                            screen_region = event.kwargs["screen_region"]
                            self.input_controller.click_screen_region(button=event.button, screen_region=screen_region)
                        elif event.event == MouseEvents.SCROLL:
                            self.input_controller.scroll(direction=event.direction)

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
                        if event.event == MouseEvents.CLICK_SCREEN_REGION:
                            screen_region = event.kwargs["screen_region"]
                            self.input_controller.click_screen_region(button=event.button, screen_region=screen_region)
                        elif event.event == MouseEvents.MOVE:
                            self.input_controller.move(x=event.x, y=event.y)
                        elif event.event == MouseEvents.MOVE_RELATIVE:
                            self.input_controller.move(x=event.x, y=event.y, absolute=False)
                        elif event.event == MouseEvents.DRAG_START:
                            screen_region = event.kwargs.get("screen_region")
                            coordinates = self.input_controller.ratios_to_coordinates(value, screen_region=screen_region)

                            self.input_controller.move(x=coordinates[0], y=coordinates[1], duration=0.1)
                            self.input_controller.click_down(button=event.button)
                        elif event.event == MouseEvents.DRAG_END:
                            screen_region = event.kwargs.get("screen_region")
                            coordinates = self.input_controller.ratios_to_coordinates(value, screen_region=screen_region)

                            self.input_controller.move(x=coordinates[0], y=coordinates[1], duration=0.1)
                            self.input_controller.click_up(button=event.button)

    def clear_input(self):
        self.input_controller.handle_keys([])


from serpent.machine_learning.reinforcement_learning.agents.rainbow_dqn_agent import RainbowDQNAgent


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
                "inputs": self.game.api.combine_game_inputs(["MOVEMENT", "SHOOTING"])
            }
        ]

        self.agent = RainbowDQNAgent("Tenshi", game_inputs=self.game_inputs)
        
        
        
        #self.agent.add_human_observations_to_replay_memory()
        

    def handle_play(self, game_frame):     

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
        if self.agent.current_step % 5000 == 0:
            self.agent.save_model()

        
        serpent.utilities.clear_terminal()
        print(f"Current HP: {self.game_state['hp'][0]}")
        print(f"Current Power: {self.game_state['power']}")
        print(f"Current Aura: {self.game_state['aura'][0]}")
        print(f"Current Score: {self.game_state['score']}")
        print(f"Current Score Multiplier: {self.game_state['score_multiplier']}")
        print(f"Current Reward: {self.game_state['run_reward']}")
        
            
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

        score = re.sub(r'[^0-9]\.', '', score)

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

        score = re.sub(r'[^0-9]', '', score)

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

        score = re.sub(r'[^0-9]\.', '', score)

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

        score = re.sub(r'[^0-9]', '', score)
            
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

        score = re.sub(r'[^0-9]', '', score)

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
  
