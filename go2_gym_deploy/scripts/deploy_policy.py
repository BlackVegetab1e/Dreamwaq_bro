import glob
import pickle as pkl
import lcm
import sys

#
sys.path.append("/home/unitree/DreamWaq_mjc/legged_gym")

from go2_gym_deploy.utils.deployment_runner import DeploymentRunner
from go2_gym_deploy.envs.lcm_agent import LCMAgent
from go2_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go2_gym_deploy.utils.command_profile import *

import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def load_and_run_policy(label, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # load agent

    logdir = "../../logs/Go2/20260128"

    from envs.GO2.go2_config import Go2_Cfg

    cfg = Go2_Cfg()
    # 如果 cfg 里没有 control、sim 等 env 字段，但包含 'env'（即 train_cfg + env_cfg 的结构），
    # 则使用 cfg['env'] 作为传给 LCMAgent 的环境配置。
    if isinstance(cfg, dict) and "control" not in cfg and "env" in cfg:
        print("Detected train_cfg with nested 'env' -> using cfg['env'] for LCMAgent")
        cfg_for_agent = cfg["env"]
    else:
        cfg_for_agent = cfg

    print(cfg_for_agent)


    se = StateEstimator(lc)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

    # hardware_agent = LCMAgent(cfg, se, command_profile)
    hardware_agent = LCMAgent(cfg_for_agent, se, command_profile)
    se.spin()

    from go2_gym_deploy.envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)

    policy = load_policy(logdir)

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None,
                                         log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000000
    print(f'max steps {max_steps}')

    deployment_runner.run(max_steps=max_steps, logging=True)

def load_policy(logdir):
    actor=torch.jit.load(logdir + '/actor.pt')
    encoder=torch.jit.load(logdir + "/encoder.pt")
    encode_mean_latent=torch.jit.load(logdir + '/encode_mean_latent.pt')
    encode_logvar_latent=torch.jit.load(logdir + '/encode_logvar_latent.pt')
    encode_mean_vel=torch.jit.load(logdir + '/encode_mean_vel.pt')
    encode_logvar_vel=torch.jit.load(logdir + '/encode_logvar_vel.pt')


    def policy(obs, info):
        def act_inference(observations,obs_history):
            _,_,_,latent= cenet_forward(obs_history)
            mean_vel,logvar_vel,mean_latent,logvar_latent=latent
            observations = torch.cat((mean_vel,mean_latent,observations),dim=-1)
            actions_mean = actor(observations)
            return actions_mean

        def reparameterise(mean,logvar):
            var = torch.exp(logvar*0.5)
            code_temp = torch.randn_like(var)
            return mean + var*code_temp
        
        def cenet_forward(obs_history):
            encoded = encoder(obs_history)
            mean_latent = encode_mean_latent(encoded)
            logvar_latent = encode_logvar_latent(encoded)
            mean_vel = encode_mean_vel(encoded)
            logvar_vel = encode_logvar_vel(encoded)
            code_latent = reparameterise(mean_latent,logvar_latent)
            code_vel = reparameterise(mean_vel,logvar_vel)
            
            code = torch.cat((code_vel,code_latent),dim=-1)
            decode = None
            return (code),(code_vel,code_latent),(decode),(mean_vel,logvar_vel,mean_latent,logvar_latent)



        # h = encoder(obs["obs_history"].to('cpu').float())
        # mu = fc_mu(h)
        # log_var = fc_var(h)
        # std = torch.exp(0.5 * log_var)
        # eps = torch.randn_like(std)
        # latent = mu + eps * std
        # print(latent.shape,'----------------------')
        # # action = actor(torch.cat((obs["obs"].to('cpu'), latent), dim=-1))
        # action = actor(torch.cat((latent, obs["obs"].to('cpu')), dim=-1))
        # info['latent'] = latent
        return action

    return policy


if __name__ == '__main__':
    label = "gait-conditioned-agility/pretrain-v0/train"

    experiment_name = "example_experiment"

    load_and_run_policy(label, experiment_name=experiment_name, max_vel=0.75, max_yaw_vel=1.0)
