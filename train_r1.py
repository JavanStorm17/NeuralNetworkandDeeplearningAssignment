from pathlib import Path
from sumo_rl import SumoEnvironment
from stable_baselines3 import DQN
from compat import OldGymToGymnasium, DictObsToArray, ScalarActionToDict, DictRewardToFloat

ROOT = Path(__file__).resolve().parent
NET = ROOT/'sumo-rl/sumo_rl/nets/2way-single-intersection/single-intersection.net.xml'
ROU = ROOT/'sumo-rl/sumo_rl/nets/2way-single-intersection/single-intersection-gen.rou.xml'

raw = SumoEnvironment(net_file=str(NET),
                      route_file=str(ROU),
                      use_gui=False, delta_time=5,
                      yellow_time=3, min_green=8)

env = OldGymToGymnasium(raw)   # fix reset/step API
env = DictObsToArray(env)      # dict → vector
env = ScalarActionToDict(env)  # scalar → dict
env = DictRewardToFloat(env)
model = DQN("MlpPolicy", env,
            policy_kwargs=dict(net_arch=[128,128]),
            learning_rate=2.5e-4, buffer_size=50_000,
            exploration_fraction=0.1, verbose=1)

model.learn(total_timesteps=500_000)
model.save("dqn_sumo_intersection")
env.close()
print("✅  RL training finished → dqn_sumo_intersection.zip")

