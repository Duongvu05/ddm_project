module.exports = {
  apps: [
    {
      name: "hm-train",
      cwd: "/home/vungocduong/ddm_project",
      script: "uv",
      args: "run python scripts/train.py --config configs/default.yaml --output outputs/lgbm_model.pkl",
      autorestart: false,
      watch: false,
      max_restarts: 0,
      time: true,
    },
    {
      name: "hm-ensemble",
      cwd: "/home/vungocduong/ddm_project",
      script: "uv",
      args: "run python -m ddm_project.ensemble",
      autorestart: false,
      watch: false,
      max_restarts: 0,
      time: true,
    },
    {
      name: "hm-evaluate",
      cwd: "/home/vungocduong/ddm_project",
      script: "uv",
      args: "run python scripts/evaluate.py --config configs/default.yaml",
      autorestart: false,
      watch: false,
      max_restarts: 0,
      time: true,
    },
  ],
};
