import os
import time
from collections import defaultdict

from colorama import Fore, Back, Style, init
from genrl.blockchain import SwarmCoordinator
from genrl.communication import Communication
from genrl.communication.hivemind.hivemind_backend import HivemindBackend
from genrl.data import DataManager
from genrl.game import BaseGameManager
from genrl.game.game_manager import DefaultGameManagerMixin
from genrl.logging_utils.global_defs import get_logger
from genrl.logging_utils.system_utils import get_system_info
from genrl.rewards import RewardManager
from genrl.roles import RoleManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from huggingface_hub import login, whoami

from rgym_exp.src.utils.name_utils import get_name_from_peer_id
from rgym_exp.src.prg_module import PRGModule

# Khởi tạo colorama
init(autoreset=True)


class ColoredLogger:
    """Wrapper cho logger với màu sắc"""
    
    @staticmethod
    def info(msg):
        print(f"{Fore.CYAN}ℹ️  {msg}{Style.RESET_ALL}")
        get_logger().info(msg)
    
    @staticmethod
    def success(msg):
        print(f"{Fore.GREEN}✅ {msg}{Style.RESET_ALL}")
        get_logger().info(msg)
    
    @staticmethod
    def warning(msg):
        print(f"{Fore.YELLOW}⚠️  {msg}{Style.RESET_ALL}")
        get_logger().warning(msg)
    
    @staticmethod
    def error(msg):
        print(f"{Fore.RED}❌ {msg}{Style.RESET_ALL}")
        get_logger().error(msg)
    
    @staticmethod
    def debug(msg):
        print(f"{Fore.MAGENTA}🔍 {msg}{Style.RESET_ALL}")
        get_logger().debug(msg)
    
    @staticmethod
    def highlight(msg):
        print(f"{Fore.WHITE}{Back.BLUE} {msg} {Style.RESET_ALL}")
        get_logger().info(msg)
    
    @staticmethod
    def game(msg):
        print(f"{Fore.YELLOW}🎮 {msg}{Style.RESET_ALL}")
        get_logger().info(msg)
    
    @staticmethod
    def swarm(msg):
        print(f"{Fore.GREEN}🐝 {msg}{Style.RESET_ALL}")
        get_logger().info(msg)
    
    @staticmethod
    def blockchain(msg):
        print(f"{Fore.BLUE}⛓️  {msg}{Style.RESET_ALL}")
        get_logger().info(msg)


class SwarmGameManager(BaseGameManager, DefaultGameManagerMixin):
    """GameManager that orchestrates a game using a SwarmCoordinator."""

    def __init__(
        self,
        coordinator: SwarmCoordinator,
        max_stage: int,
        max_round: int,
        game_state: GameState,
        reward_manager: RewardManager,
        trainer: TrainerModule,
        data_manager: DataManager,
        communication: Communication,
        role_manager: RoleManager | None = None,
        run_mode: str = "train",
        log_dir: str = "logs",
        hf_token: str | None = None,
        hf_push_frequency: int = 20,
        **kwargs,
    ):

        super().__init__(
            max_stage=max_stage,
            max_round=max_round,
            game_state=game_state,
            reward_manager=reward_manager,
            trainer=trainer,
            data_manager=data_manager,
            communication=communication,
            role_manager=role_manager,
            run_mode=run_mode,
        )

        assert isinstance(self.communication, HivemindBackend)
        self.train_timeout = 60 * 60 * 24 * 31  # 1 month
        self.logger = ColoredLogger()

        # Logging Setup
        self.peer_id = self.communication.get_id()
        self.state.peer_id = self.peer_id
        self.animal_name = get_name_from_peer_id(self.peer_id, True)

        # Register peer_id and get current round from the chain
        self.coordinator = coordinator
        self.coordinator.register_peer(self.peer_id)
        round, _ = self.coordinator.get_round_and_stage()
        self.state.round = round

        self.communication.step_ = (
            self.state.round
        )  # initialize communication module to contract's round

        # enable push to HF if token was provided
        self.hf_token = hf_token
        if self.hf_token not in [None, "None"]:
            self._configure_hf_hub(hf_push_frequency)

        # Welcome message với màu sắc
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        self.logger.highlight(f"🐱 SWARM AGENT INITIALIZED 🐈")
        print(f"{Fore.GREEN}├─ Agent Name: {Fore.YELLOW}{get_name_from_peer_id(self.peer_id)}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}├─ Peer ID: {Fore.WHITE}{self.peer_id}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}├─ Model: {Fore.CYAN}{self.trainer.model.config.name_or_path}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}└─ Bootnodes: {Fore.MAGENTA}{kwargs.get('bootnodes', [])}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

        with open(os.path.join(log_dir, f"system_info.txt"), "w") as f:
            f.write(get_system_info())

        self.batched_signals = 0.0
        self.time_since_submit = time.time()  # seconds
        self.submit_period = 3.0  # hours
        self.submitted_this_round = False

        # PRG Game
        self.prg_module = PRGModule(log_dir, **kwargs)
        self.prg_game = self.prg_module.prg_game

    def _get_total_rewards_by_agent(self):
        rewards_by_agent = defaultdict(int)
        for stage in range(self.state.stage):
            rewards = self.rewards[stage]
            for agent_id, agent_rewards in rewards.items():
                for batch_id, batch_rewards in agent_rewards.items():
                    tot = 0
                    for generation_rewards in batch_rewards:
                        tot += sum(generation_rewards)
                    rewards_by_agent[agent_id] += tot

        return rewards_by_agent

    def _get_my_rewards(self, signal_by_agent):
        if len(signal_by_agent) == 0:
            return 0
        if self.peer_id in signal_by_agent:
            my_signal = signal_by_agent[self.peer_id]
        else:
            my_signal = 0
        my_signal = (my_signal + 1) * (my_signal > 0) + my_signal * (my_signal <= 0)
        return my_signal

    def _try_submit_to_chain(self, signal_by_agent):
        elapsed_time_hours = (time.time() - self.time_since_submit) / 3600
        if elapsed_time_hours > self.submit_period:
            try:
                self.coordinator.submit_reward(
                    self.state.round, 0, int(self.batched_signals), self.peer_id
                )
                self.batched_signals = 0.0
                
                if len(signal_by_agent) > 0:
                    max_agent, max_signal = max(
                        signal_by_agent.items(), key=lambda x: x[1]
                    )
                    self.logger.blockchain(
                        f"Submitting winner: {get_name_from_peer_id(max_agent)} "
                        f"with signal: {Fore.YELLOW}{max_signal}{Style.RESET_ALL}"
                    )
                else:  # if we have no signal_by_agents, just submit ourselves.
                    max_agent = self.peer_id
                    self.logger.blockchain(f"No signals received, submitting self as winner")

                self.coordinator.submit_winners(
                    self.state.round, [max_agent], self.peer_id
                )
                self.time_since_submit = time.time()
                self.submitted_this_round = True
                
                self.logger.success(
                    f"Successfully submitted to chain for round {Fore.CYAN}{self.state.round}{Style.RESET_ALL}"
                )
            except Exception as e:
                self.logger.debug(f"Chain submission failed: {str(e)}")

    def _hook_after_rewards_updated(self):
        try:
            signal_by_agent = self._get_total_rewards_by_agent()
            my_rewards = self._get_my_rewards(signal_by_agent)
            self.batched_signals += my_rewards
            
            self.logger.info(
                f"Rewards updated | My signal: {Fore.GREEN}{my_rewards:.2f}{Style.RESET_ALL} | "
                f"Batched: {Fore.YELLOW}{self.batched_signals:.2f}{Style.RESET_ALL}"
            )
        except Exception as e:
            self.logger.debug(f"Error getting total rewards by agent: {e}")
            signal_by_agent = {}
        self._try_submit_to_chain(signal_by_agent)

    def _hook_after_round_advanced(self):
        print(f"\n{Fore.MAGENTA}{'─'*60}{Style.RESET_ALL}")
        self.logger.swarm(f"Round {Fore.CYAN}{self.state.round}{Style.RESET_ALL} completed!")
        print(f"{Fore.MAGENTA}{'─'*60}{Style.RESET_ALL}\n")
        
        try:
            if self.prg_game:
                self.logger.game("Starting PRG game...")
                prg_history_dict = self.prg_module.prg_history_dict
                results_dict = self.trainer.play_prg_game_logits(prg_history_dict)
                self.prg_module.play_prg_game(results_dict, self.peer_id)
                self.logger.success("PRG game completed")
        except Exception as e:
            self.logger.error(f"Error playing PRG game: {str(e)}")

        self._save_to_hf()

        # Try to submit to chain again if necessary, but don't update our signal twice
        if not self.submitted_this_round:
            try:
                signal_by_agent = self._get_total_rewards_by_agent()
            except Exception as e:
                self.logger.debug(f"Error getting total rewards by agent: {e}")
                signal_by_agent = {}
            self._try_submit_to_chain(signal_by_agent)

        # Reset flag for next round
        self.submitted_this_round = False

        # Block until swarm round advances
        self.agent_block()

    def _hook_after_game(self):
        print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        self.logger.highlight("🎮 GAME COMPLETED 🎮")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}\n")
        self._save_to_hf()

    def _configure_hf_hub(self, hf_push_frequency):
        username = whoami(token=self.hf_token)["name"]
        model_name = self.trainer.model.config.name_or_path.split("/")[-1]
        model_name += "-Gensyn-Swarm"
        model_name += f"-{self.animal_name}"
        self.trainer.args.hub_model_id = f"{username}/{model_name}"
        self.hf_push_frequency = hf_push_frequency
        
        self.logger.info("Logging into Hugging Face Hub...")
        login(self.hf_token)
        self.logger.success(f"HF Hub configured: {Fore.CYAN}{self.trainer.args.hub_model_id}{Style.RESET_ALL}")

    def _save_to_hf(self):
        if (
            self.hf_token not in [None, "None"]
            and self.state.round % self.hf_push_frequency == 0
        ):
            self.logger.info(f"Pushing model to Hugging Face...")
            try:
                repo_id = self.trainer.args.hub_model_id

                self.trainer.model.push_to_hub(
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message=f"rl-swarm: round {self.state.round}, agent {self.animal_name}",
                    tags=[
                        "rl-swarm",
                        "genrl-swarm",
                        "grpo",
                        "gensyn",
                        f"I am {self.animal_name}",
                    ],
                )
                self.logger.success(
                    f"Model pushed to HF Hub (Round {Fore.CYAN}{self.state.round}{Style.RESET_ALL})"
                )
            except Exception:
                self.logger.error("Failed to push model to the Hugging Face Hub")
                get_logger().exception(
                    "Failed to push model to the Hugging Face Hub. When you conclude training please try manually pushing it yourself using the instructions here: https://huggingface.co/docs/hub/en/models-uploading",
                    stack_info=True,
                )

    def agent_block(
        self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 15
    ):
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = check_interval
        
        self.logger.info(f"Waiting for next round...")
        
        while time.monotonic() - start_time < self.train_timeout:
            curr_time = time.monotonic()
            _ = self.communication.dht.get_visible_maddrs(latest=True)

            # Retrieve current round and stage.
            try:
                round_num, stage = self.coordinator.get_round_and_stage()
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    self.logger.debug(
                        f"Could not fetch round and stage: {e}. Next check in {check_interval}s."
                    )
                    fetch_log_time = curr_time

                time.sleep(check_interval)
                continue

            if round_num >= self.state.round:
                self.logger.swarm(f"Joining round: {Fore.CYAN}{round_num}{Style.RESET_ALL}")
                check_backoff = check_interval
                self.state.round = round_num
                return
            else:
                self.logger.info(
                    f"Already finished round: {Fore.YELLOW}{round_num}{Style.RESET_ALL}. "
                    f"Next check in {check_backoff}s."
                )
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            if round_num == self.max_round - 1:
                return

        self.logger.warning("Training timed out!")
