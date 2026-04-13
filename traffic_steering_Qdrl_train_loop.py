import argparse
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dqn_quantum_discrete_state import DQNAgentQuantum, linear_schedule
from ts_env_raoulQuantum import TrafficSteeringEnv as TSQuantumEnv

os.environ["WANDB_MODE"] = "online"


def save_qrl_model(model: torch.nn.Module, output_path: str, scenario: str, n_ue: int, episode: int) -> None:
    model_dir = os.path.join(output_path, "models", "qdqn")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(
        model_dir,
        f"qdqn_{scenario.lower()}_{n_ue}ues_episode_{episode:03d}.pth",
    )
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Saved QDQN checkpoint to {save_path}")


def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    flat_obs = np.asarray(obs, dtype=np.float32).flatten()
    flat_obs[~np.isfinite(flat_obs)] = 0.0
    return flat_obs


def sample_batch(replay_buffer: deque, batch_size: int, device: torch.device):
    transitions = random.sample(replay_buffer, batch_size)
    states = torch.tensor(np.stack([t[0] for t in transitions]), dtype=torch.float32, device=device)
    actions = torch.tensor([t[1] for t in transitions], dtype=torch.long, device=device)
    rewards = torch.tensor([t[2] for t in transitions], dtype=torch.float32, device=device)
    next_states = torch.tensor(np.stack([t[3] for t in transitions]), dtype=torch.float32, device=device)
    dones = torch.tensor([t[4] for t in transitions], dtype=torch.float32, device=device)
    return states, actions, rewards, next_states, dones


def soft_update(target_network: torch.nn.Module, q_network: torch.nn.Module, tau: float) -> None:
    for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
        target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QDQN on the traffic steering environment")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/openlab/ns-o-ran-gym/src/environments/scenario_configurations/ts_use_case.json",
        help="Path to the configuration file",
    )
    parser.add_argument("--output_folder", type=str, default="output", help="Path to the output folder")
    parser.add_argument(
        "--ns3_path",
        type=str,
        default="/home/openlab/ns-3-mmwave-oran",
        help="Path to the ns-3 mmWave O-RAN environment",
    )
    parser.add_argument("--num_steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of episodes")
    parser.add_argument("--optimized", action="store_true", help="Enable optimization mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--buffer_size", type=int, default=1000, help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--start_e", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--end_e", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--exploration_fraction", type=float, default=1.0, help="Exploration fraction")
    parser.add_argument("--learning_starts", type=int, default=100, help="Training warmup steps")
    parser.add_argument("--train_frequency", type=int, default=10, help="Training frequency")
    parser.add_argument("--target_network_frequency", type=int, default=500, help="Target network update frequency")
    parser.add_argument("--tau", type=float, default=1.0, help="Target network interpolation factor")
    parser.add_argument("--num_qubits", type=int, default=8, help="Number of qubits")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of quantum layers")
    parser.add_argument("--lr_input_scaling", type=float, default=1e-3, help="Learning rate for input scaling")
    parser.add_argument("--lr_weights", type=float, default=1e-3, help="Learning rate for variational weights")
    parser.add_argument("--lr_output_scaling", type=float, default=1e-3, help="Learning rate for classical output head")
    parser.add_argument("--device", type=str, default="lightning.qubit", help="Pennylane device")
    parser.add_argument("--diff_method", type=str, default="adjoint", help="Pennylane differentiation method")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config) as params_file:
        scenario_configuration = json.load(params_file)

    scenario = "NO_UE_POS"
    n_ue = int(scenario_configuration["ues"][0]) if isinstance(scenario_configuration["ues"], list) else int(scenario_configuration["ues"])
    scenario_configuration["ues"] = [n_ue]

    env = TSQuantumEnv(
        ns3_path=args.ns3_path,
        scenario_configuration=scenario_configuration,
        output_folder=args.output_folder,
        optimized=args.optimized,
        verbose=args.verbose,
    )

    obs, _ = env.reset()
    state = preprocess_observation(obs)
    obs_dim = state.shape[0]
    num_actions = env.action_space.n

    q_config = {
        "num_qubits": args.num_qubits,
        "num_layers": args.num_layers,
        "device": args.device,
        "diff_method": args.diff_method,
        "n_ue": n_ue,
        "n_gnbs": env.n_gnbs,
        "n_features": env.n_features,
        "bond_dim": 8  # Vettore di riassunto per trasportare le informazioni nei rami
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = DQNAgentQuantum(obs_dim, num_actions, q_config).to(device)
    target_network = DQNAgentQuantum(obs_dim, num_actions, q_config).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(
        [
            {"params": q_network.input_scaling, "lr": args.lr_input_scaling},
            {"params": q_network.weights, "lr": args.lr_weights},
            {"params": q_network.output_layer.parameters(), "lr": args.lr_output_scaling},
        ]
    )
    replay_buffer = deque(maxlen=args.buffer_size)

    if args.wandb:
        wandb.init(
            name=f"QDQN_digitalTwin_{n_ue}ues",
            project="traffic-steering",
            tags=[f"QDQN_digitalTwin_{n_ue}ues"],
            entity="danyrichwell",
            reinit=True,
        )

    global_step = 0
    start_time = time.time()

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset()
        state = preprocess_observation(obs)
        cumulative_reward = 0.0
        epsilon = linear_schedule(
                    args.start_e,
                    args.end_e,
                    max(1, int(args.exploration_fraction * args.episodes)),
                    episode
                    ) 

        for step in range(2, args.num_steps):
            print(f'Step {step} ', end='', flush=True)
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
                    action = int(torch.argmax(q_network(state_tensor), dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_observation(next_obs)
            done = terminated or truncated
            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state
            cumulative_reward += reward
            global_step += 1

            if len(replay_buffer) >= args.batch_size and global_step >= args.learning_starts:
                if global_step % args.train_frequency == 0:
                    states, actions, rewards, next_states, dones = sample_batch(
                        replay_buffer, args.batch_size, device
                    )
                    with torch.no_grad():
                        next_q = target_network(next_states).max(dim=1).values
                        td_target = rewards + args.gamma * next_q * (1.0 - dones)

                    current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    loss = F.mse_loss(current_q, td_target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if global_step % args.target_network_frequency == 0:
                        soft_update(target_network, q_network, args.tau)

                    if args.wandb:
                        wandb.log(
                            {
                                "td_loss": loss.item(),
                                "q_values": current_q.mean().item(),
                                "epsilon": epsilon,
                                "global_step": global_step,
                            },
                            commit=False,
                        )
            
            if terminated or truncated or step == args.num_steps - 1:
                avg_thr = float(np.mean(env.average_throughputs)) if env.average_throughputs else 0.0
                sps = int(global_step / max(time.time() - start_time, 1e-6))
                print(
                    f"episode {episode} reward={cumulative_reward:.3f} "
                    f"avg_thr={avg_thr:.3f} handovers={env.n_handovers} epsilon={epsilon:.3f}"
                )
                if args.wandb:
                    wandb.log(
                        {
                            "episode": episode,
                            "cumulative_reward": cumulative_reward,
                            "average_throughput": avg_thr,
                            "n_handovers_tot": env.n_handovers,
                            "epsilon": epsilon,
                            "SPS": sps,
                        },
                        commit=True,
                    )
                save_qrl_model(q_network, args.output_folder, scenario, n_ue, episode)
                env._reset_stats()
                break

    env.close()
    if args.wandb:
        wandb.finish()
