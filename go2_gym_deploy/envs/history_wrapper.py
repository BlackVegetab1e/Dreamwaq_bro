# import isaacgym

# assert isaacgym, "import isaacgym before pytorch"
import torch


class HistoryWrapper:
    # def __init__(self, env):
    #     self.env = env

    #     if isinstance(self.env.cfg, dict):
    #         self.obs_history_length = self.env.cfg["env"]["num_observation_history"]
    #     else:
    #         self.obs_history_length = self.env.cfg.env.num_observation_history
    #     self.num_obs_history = self.obs_history_length * self.env.num_obs
    #     self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
    #                                    device=self.env.device, requires_grad=False)
    #     self.num_privileged_obs = self.env.num_privileged_obs
    def __init__(self, env):
        self.env = env

        # 支持多种可能的配置键名并提供默认值
        # 可能存在的键名：num_observation_history, num_obs_hist, num_obs_history
        default_history_len = 5

        def _get_history_len_from_cfg(cfg):
            # cfg 可以是 dict 或对象（有属性）
            # 返回第一个存在的值或 None
            candidates = ["num_observation_history", "num_obs_hist", "num_obs_history"]
            if isinstance(cfg, dict):
                env_section = cfg.get("env", cfg)
                for key in candidates:
                    if key in env_section:
                        return env_section[key]
                return None
            else:
                # cfg is object with attributes
                env_obj = getattr(cfg, "env", cfg)
                for key in candidates:
                    if hasattr(env_obj, key):
                        return getattr(env_obj, key)
                return None

        history_len = _get_history_len_from_cfg(self.env.cfg)
        if history_len is None:
            history_len = default_history_len

        self.obs_history_length = int(history_len)

        # env.num_obs (or env.num_observations) used elsewhere; try both
        try:
            num_obs = self.env.num_obs
        except AttributeError:
            # fallback: try cfg value names
            try:
                num_obs = self.env.cfg["env"].get("num_observations", None)
            except Exception:
                num_obs = None
            if num_obs is None:
                # last fallback: try num_obs_hist related naming
                num_obs = getattr(self.env.cfg.get("env", {}), "num_observations", None) if isinstance(self.env.cfg, dict) else getattr(self.env.cfg, "num_observations", None)
            if num_obs is None:
                raise RuntimeError("Cannot determine number of observations (env.num_obs or cfg.env.num_observations)")

        self.num_obs = int(num_obs)
        self.num_obs_history = self.obs_history_length * self.num_obs

        # allocate history buffer
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)

        # privileged obs count: try env attribute then cfg keys, fallback to 0
        try:
            self.num_privileged_obs = self.env.num_privileged_obs
        except AttributeError:
            try:
                self.num_privileged_obs = int(self.env.cfg["env"].get("num_privileged_obs", 0))
            except Exception:
                self.num_privileged_obs = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        privileged_obs = info["privileged_obs"]

        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}, rew, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}

    def get_obs(self):
        obs = self.env.get_obs()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = self.env.reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        ret = self.env.reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}

    def __getattr__(self, name):
        return getattr(self.env, name)
