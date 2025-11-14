import os
import time
from collections import defaultdict

import ollama

from genrl.blockchain import SwarmCoordinator
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

from code_gen_exp.src.utils.name_utils import get_name_from_peer_id

# Enhanced colorful logging
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    class MockColor:
        CYAN = GREEN = RED = YELLOW = MAGENTA = BLUE = WHITE = LIGHTGREEN_EX = ""
    class MockStyle:
        RESET_ALL = BRIGHT = ""
    Fore = MockColor()
    Style = MockStyle()


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
        communication_kwargs: dict,
        role_manager: RoleManager | None = None,
        run_mode: str = "train",
        log_dir: str = "logs",
        hf_token: str | None = None,
        hf_push_frequency: int = 20,
        **kwargs,
    ):
        initial_peers = coordinator.get_bootnodes()
        communication_kwargs['initial_peers'] = initial_peers
        get_logger().info(f"{Fore.CYAN}🌐 [BOOTNODES] {initial_peers}{Style.RESET_ALL}")
        
        rewards_ollama_model = kwargs.get("rewards_ollama_model", 'qwen2.5-coder:1.5b-instruct')
        communication = HivemindBackend(**communication_kwargs)

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

        # Setup peer identity
        self.peer_id = self.communication.get_id()
        self.state.peer_id = self.peer_id
        self.animal_name = get_name_from_peer_id(self.peer_id, True)

        # Register with blockchain and sync
        self.coordinator = coordinator
        self.coordinator.register_peer(self.peer_id)
        round, _ = self.coordinator.get_round_and_stage()
        self.state.round = round
        self.communication.step_ = self.state.round
        self.data_manager.initialize(self.communication)

        # Model name for display
        self.model_display_name = self._get_clean_model_name()

        # Setup Hugging Face integration
        self.hf_token = hf_token
        if self.hf_token not in [None, "None"]:
            self._configure_hf_hub(hf_push_frequency)

        # Setup Ollama model
        self._setup_ollama_model(rewards_ollama_model)

        # Write system info
        with open(os.path.join(log_dir, f"system_info.txt"), "w") as f:
            f.write(get_system_info())

        # Submission tracking
        self.batched_signals = 0.0
        self.time_since_submit = time.time()
        self.submit_period = 3.0  # hours
        self.submitted_this_round = False
        self.round_counter = 0
        self._last_countdown_log = 0

        # Log initialization status
        self._log_initialization()

    def _get_clean_model_name(self) -> str:
        """Get clean model name for display"""
        try:
            model_name = self.trainer.model.config.name_or_path
            clean_name = model_name.split("/")[-1] if "/" in model_name else model_name
            
            # Remove common suffixes
            for suffix in ["-Instruct", "-Chat", "-Base", "-v1", "-v2", "-v3"]:
                if clean_name.endswith(suffix):
                    clean_name = clean_name[:-len(suffix)]
                    break
            return clean_name
        except:
            return "UnknownModel"

    def _setup_ollama_model(self, rewards_ollama_model):
        """Setup Ollama model with clean logging"""
        get_logger().info(f"{Fore.CYAN}🤖 [OLLAMA] Checking model: {rewards_ollama_model}{Style.RESET_ALL}")
        
        try:
            models = ollama.list()
            model_names = [model["model"] for model in models["models"]]
            
            if rewards_ollama_model not in model_names:
                get_logger().info(f"{Fore.YELLOW}📥 [OLLAMA] Downloading {rewards_ollama_model}...{Style.RESET_ALL}")
                ollama.pull(rewards_ollama_model)
                get_logger().info(f"{Fore.GREEN}✅ [OLLAMA] Model ready{Style.RESET_ALL}")
            else:
                get_logger().info(f"{Fore.GREEN}✅ [OLLAMA] Model already available{Style.RESET_ALL}")
        except Exception as e:
            get_logger().error(f"{Fore.RED}❌ [OLLAMA] Error: {e}{Style.RESET_ALL}")
            raise e

    def _log_initialization(self):
        """Log initialization status with beautiful formatting"""
        get_logger().info(
            f"\n{Fore.MAGENTA}{Style.BRIGHT}{'='*60}\n"
            f"🐝 CODEZERO SWARM INITIALIZED 🐝\n"
            f"{'='*60}{Style.RESET_ALL}\n"
            f"{Fore.CYAN}🤖 Model:{Style.RESET_ALL}      {Fore.WHITE}{self.model_display_name}{Style.RESET_ALL}\n"
            f"{Fore.CYAN}🐾 Agent:{Style.RESET_ALL}      {Fore.WHITE}{self.animal_name}{Style.RESET_ALL}\n"
            f"{Fore.CYAN}📍 Peer ID:{Style.RESET_ALL}    {Fore.WHITE}{self.peer_id[:16]}...{Style.RESET_ALL}\n"
            f"{Fore.CYAN}🔄 Round:{Style.RESET_ALL}      {Fore.WHITE}{self.state.round}{Style.RESET_ALL}\n"
            f"{Fore.CYAN}⏰ Submit:{Style.RESET_ALL}     {Fore.WHITE}Every {self.submit_period} hours{Style.RESET_ALL}\n"
            f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}"
        )

    def _configure_hf_hub(self, hf_push_frequency):
        """Configure Hugging Face Hub with clean logging"""
        try:
            username = whoami(token=self.hf_token)["name"]
            model_name = self.trainer.model.config.name_or_path.split("/")[-1]
            model_name += "-Gensyn-Swarm"
            model_name += f"-{self.animal_name}"
            self.trainer.args.hub_model_id = f"{username}/{model_name}"
            self.hf_push_frequency = hf_push_frequency
            
            get_logger().info(f"{Fore.GREEN}✅ [HUGGING FACE] Connected as {username}{Style.RESET_ALL}")
            login(self.hf_token)
        except Exception as e:
            get_logger().warning(f"{Fore.YELLOW}⚠️ [HUGGING FACE] Setup failed: {e}{Style.RESET_ALL}")

    def _get_total_rewards_by_agent(self):
        """Calculate total rewards by agent"""
        rewards_by_agent = defaultdict(int)
        for stage in range(self.state.stage):
            rewards = self.rewards[stage]
            for agent_id, agent_rewards in rewards.items():
                for batch_id, batch_rewards in agent_rewards.items():
                    total = sum(sum(generation_rewards) for generation_rewards in batch_rewards)
                    rewards_by_agent[agent_id] += total
        return rewards_by_agent

    def _get_my_rewards(self, signal_by_agent):
        """Get rewards for this agent"""
        if not signal_by_agent:
            return 0
        my_signal = signal_by_agent.get(self.peer_id, 0)
        return (my_signal + 1) * (my_signal > 0) + 0 * (my_signal <= 0)

    def _format_time_remaining(self, hours_remaining):
        """Format time remaining in a readable way"""
        if hours_remaining >= 1:
            return f"{hours_remaining:.1f}h"
        else:
            minutes = hours_remaining * 60
            if minutes >= 1:
                return f"{minutes:.0f}m"
            else:
                seconds = minutes * 60
                return f"{seconds:.0f}s"

    def _log_submit_countdown(self, hours_remaining):
        """Log countdown with visual progress bar"""
        current_time = time.time()
        
        # Log every 5 minutes to avoid spam
        if current_time - self._last_countdown_log < 300:
            return
        
        self._last_countdown_log = current_time
        
        # Calculate progress percentage
        elapsed_hours = self.submit_period - hours_remaining
        progress_pct = (elapsed_hours / self.submit_period) * 100
        
        # Create visual progress bar
        bar_length = 20
        filled_length = int(bar_length * progress_pct / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        time_str = self._format_time_remaining(hours_remaining)
        
        get_logger().info(
            f"{Fore.YELLOW}⏳ [SUBMIT COUNTDOWN] "
            f"{Fore.WHITE}[{bar}] {progress_pct:.0f}% "
            f"{Fore.CYAN}| Next in: {time_str} "
            f"{Fore.MAGENTA}| Points: {int(self.batched_signals)}{Style.RESET_ALL}"
        )

    def _try_submit_to_chain(self, signal_by_agent):
        """Submit results to blockchain with countdown"""
        elapsed_hours = (time.time() - self.time_since_submit) / 3600
        hours_remaining = self.submit_period - elapsed_hours
        
        if elapsed_hours > self.submit_period:
            try:
                points = int(self.batched_signals)
                
                get_logger().info(
                    f"{Fore.CYAN}⛓️ [BLOCKCHAIN] Submitting Round {self.state.round}...{Style.RESET_ALL}"
                )
                get_logger().info(
                    f"{Fore.BLUE}📊 [POINTS] Submitting {points} points{Style.RESET_ALL}"
                )
                
                # Submit reward
                self.coordinator.submit_reward(
                    self.state.round, 0, points, self.peer_id
                )
                
                # Determine winner
                if signal_by_agent:
                    max_agent, max_signal = max(signal_by_agent.items(), key=lambda x: x[1])
                    winner_name = get_name_from_peer_id(max_agent, True)
                    get_logger().info(
                        f"{Fore.YELLOW}👑 [WINNER] {winner_name} with {max_signal} points{Style.RESET_ALL}"
                    )
                else:
                    max_agent = self.peer_id
                
                # Submit winners
                self.coordinator.submit_winners(
                    self.state.round, [max_agent], self.peer_id
                )
                
                # Reset counters
                self.batched_signals = 0.0
                self.time_since_submit = time.time()
                self.submitted_this_round = True
                self._last_countdown_log = 0  # Reset countdown log
                
                get_logger().info(
                    f"{Fore.GREEN}✅ [BLOCKCHAIN] Submission successful!{Style.RESET_ALL}"
                )
                
            except Exception as e:
                get_logger().error(
                    f"{Fore.RED}❌ [BLOCKCHAIN] Submission failed: {e}{Style.RESET_ALL}"
                )
        else:
            # Show countdown
            self._log_submit_countdown(hours_remaining)

    def _hook_after_rewards_updated(self):
        """Handle reward updates"""
        try:
            signal_by_agent = self._get_total_rewards_by_agent()
            old_signals = self.batched_signals
            self.batched_signals += self._get_my_rewards(signal_by_agent)
            
            # Log reward gain
            reward_gained = self.batched_signals - old_signals
            if reward_gained > 0:
                get_logger().info(
                    f"{Fore.GREEN}💎 [REWARDS] +{reward_gained:.1f} points "
                    f"(Total: {int(self.batched_signals)}){Style.RESET_ALL}"
                )
        except Exception as e:
            get_logger().debug(f"Error getting rewards: {e}")
            signal_by_agent = {}

        self._try_submit_to_chain(signal_by_agent)

        # Send response data
        for stage in range(self.state.stage):
            root_state = self.state.get_stage_state(stage)
            self.data_manager.send_response(self.rewards[stage], root_state)

    def _hook_after_round_advanced(self):
        """Handle round advancement"""
        self.round_counter += 1
        
        # Beautiful round advancement log
        get_logger().info(
            f"\n{Fore.MAGENTA}{Style.BRIGHT}{'='*60}\n"
            f"🚀 NEW ROUND STARTED! 🚀\n"
            f"{'='*60}{Style.RESET_ALL}\n"
            f"{Fore.CYAN}📍 Round:{Style.RESET_ALL}      {Fore.WHITE}{self.state.round}{Style.RESET_ALL}\n"
            f"{Fore.CYAN}🏆 Total:{Style.RESET_ALL}      {Fore.WHITE}{self.round_counter} rounds{Style.RESET_ALL}\n"
            f"{Fore.CYAN}💎 Points:{Style.RESET_ALL}     {Fore.WHITE}{int(self.batched_signals)} pending{Style.RESET_ALL}\n"
            f"{Fore.CYAN}🐾 Agent:{Style.RESET_ALL}      {Fore.WHITE}{self.animal_name}{Style.RESET_ALL}\n"
            f"{Fore.MAGENTA}{'='*60}{Style.RESET_ALL}"
        )
        
        # Save to HF
        self._save_to_hf()

        # Final submit if needed
        if not self.submitted_this_round:
            try:
                signal_by_agent = self._get_total_rewards_by_agent()
            except Exception as e:
                get_logger().debug(f"Error getting rewards: {e}")
                signal_by_agent = {}
            
            self._try_submit_to_chain(signal_by_agent)

        # Reset flag
        self.submitted_this_round = False

        # Block until next round
        self.agent_block()

    def _hook_after_game(self):
        """Handle game completion"""
        get_logger().info(
            f"{Fore.MAGENTA}🎉 [GAME COMPLETE] Final save...{Style.RESET_ALL}"
        )
        self._save_to_hf()

    def _save_to_hf(self):
        """Save model to Hugging Face Hub"""
        if (self.hf_token not in [None, "None"] and 
            self.state.round % self.hf_push_frequency == 0):
            
            get_logger().info(
                f"{Fore.CYAN}📤 [HUGGING FACE] Uploading round {self.state.round}...{Style.RESET_ALL}"
            )
            
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
                
                get_logger().info(
                    f"{Fore.GREEN}✅ [HUGGING FACE] Upload complete!{Style.RESET_ALL}"
                )
                
            except Exception as e:
                get_logger().error(
                    f"{Fore.RED}❌ [HUGGING FACE] Upload failed: {e}{Style.RESET_ALL}"
                )

    def agent_block(
        self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 15
    ):
        """Block agent until swarm advances to next round"""
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = check_interval
        
        get_logger().info(
            f"{Fore.YELLOW}⏳ [BLOCKING] Waiting for swarm advancement...{Style.RESET_ALL}"
        )
        
        while time.monotonic() - start_time < self.train_timeout:
            curr_time = time.monotonic()
            
            # Refresh DHT
            try:
                _ = self.communication.dht.get_visible_maddrs(latest=True)
            except Exception as e:
                get_logger().debug(f"DHT refresh error: {e}")

            # Check round status
            try:
                round_num, stage = self.coordinator.get_round_and_stage()
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    get_logger().debug(
                        f"{Fore.YELLOW}⚠️ [ROUND CHECK] Failed: {e}{Style.RESET_ALL}"
                    )
                    fetch_log_time = curr_time
                time.sleep(check_interval)
                continue

            # Check if we can advance
            if round_num >= self.state.round:
                get_logger().info(
                    f"{Fore.GREEN}🚀 [ADVANCE] Joining round {round_num}!{Style.RESET_ALL}"
                )
                check_backoff = check_interval
                self.state.round = round_num
                return
            else:
                get_logger().info(
                    f"{Fore.WHITE}ℹ️ [STATUS] Already finished round {round_num}. "
                    f"Next check in {check_backoff:.0f}s{Style.RESET_ALL}"
                )
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            # Check for final round
            if round_num == self.max_round - 1:
                get_logger().info(
                    f"{Fore.MAGENTA}🎉 [COMPLETE] Reached final round!{Style.RESET_ALL}"
                )
                return

        get_logger().info(
            f"{Fore.YELLOW}🕐 [TIMEOUT] Training timeout reached{Style.RESET_ALL}"
        )
