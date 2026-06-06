# PPO LunarLander-v3 Reinforcement Learning Agent

This project trains, evaluates, records, and optionally uploads a reinforcement learning agent for the `LunarLander-v3` Gymnasium environment. The agent uses Proximal Policy Optimization (PPO) from Stable-Baselines3 and is designed to run smoothly in Google Colab.

## Project Files

| File | Description |
| --- | --- |
| `LunarLander_v3_RL.ipynb` | Main notebook containing installation, training, evaluation, Hugging Face upload, model loading, and video recording code. |
| `download.mp4` | Demo video of the trained LunarLander agent. |

## Demo Video

Watch the trained agent demo:

[Open demo video](./download.mp4)

If GitHub does not preview the video directly inside the README, click the link above. GitHub can preview `.mp4` files when opened from the repository file browser.

## What This Project Does

The notebook performs the complete reinforcement learning workflow:

1. Installs system and Python dependencies.
2. Starts a virtual display for rendering videos in Colab.
3. Creates the `LunarLander-v3` environment.
4. Trains a PPO agent using 16 parallel environments.
5. Saves the trained model locally.
6. Evaluates the agent over multiple episodes.
7. Uploads the trained model to Hugging Face Hub.
8. Loads the model back from Hugging Face Hub.
9. Records a video of the agent playing.

## Environment

The project uses:

- Python
- Gymnasium with Box2D
- Stable-Baselines3
- PPO reinforcement learning algorithm
- Hugging Face Hub
- `huggingface_sb3`
- PyVirtualDisplay
- FFmpeg
- Google Colab

## Installation

The notebook installs the required dependencies automatically in Colab:

```bash
apt-get update -y
apt-get install -y python3-opengl ffmpeg xvfb swig cmake
pip install gymnasium[box2d]
pip install stable-baselines3[extra]
pip install huggingface_sb3
pip install huggingface_hub
pip install pyvirtualdisplay
```

For local execution, install the same Python packages in a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install "gymnasium[box2d]" "stable-baselines3[extra]" huggingface_sb3 huggingface_hub pyvirtualdisplay
```

On Windows, Box2D dependencies may require extra build tools. Colab is recommended for the easiest setup.

## Training Details

The model is trained with PPO:

```python
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1
)

model.learn(total_timesteps=1_000_000)
model.save("ppo-LunarLander-v3")
```

Important configuration:

- Environment: `LunarLander-v3`
- Algorithm: PPO
- Policy: `MlpPolicy`
- Parallel environments: `16`
- Training steps: `1,000,000`
- Saved model name: `ppo-LunarLander-v3`

## Evaluation

The notebook evaluates the trained model using:

```python
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=10,
    deterministic=True
)
```

For LunarLander, a mean reward above `200` is commonly considered solved.

## Hugging Face Hub Upload

The notebook includes code to upload the trained model:

```python
repo_id = "praveenkumar23/ppo-LunarLander-v3"

package_to_hub(
    model=model,
    model_name="ppo-LunarLander-v3",
    model_architecture="PPO",
    env_id="LunarLander-v3",
    eval_env=DummyVecEnv([lambda: gym.make("LunarLander-v3", render_mode="rgb_array")]),
    repo_id=repo_id,
    commit_message="Upload PPO LunarLander-v3 trained agent"
)
```

Before uploading, log in inside Colab:

```python
from huggingface_hub import notebook_login
notebook_login()
```

Change `repo_id` to your own Hugging Face username and model repository name before running the upload cell.

## Recording a Video

The notebook records gameplay using Gymnasium's `RecordVideo` wrapper:

```python
eval_env = gym.make("LunarLander-v3", render_mode="rgb_array")
eval_env = RecordVideo(
    eval_env,
    video_folder="./videos",
    name_prefix="lunarlander",
    episode_trigger=lambda ep: True
)
```

The recorded video can be downloaded from Colab and added to the repository as an `.mp4` file.

## How to Upload This Project to GitHub

From this project folder:

```bash
cd "D:\Compiler-Design"
git status
git add README.md LunarLander_v3_RL.ipynb download.mp4
git commit -m "Add PPO LunarLander project README and demo video"
git push origin main
```

If this is a new repository and no remote is connected yet:

```bash
git init
git add README.md LunarLander_v3_RL.ipynb download.mp4
git commit -m "Initial PPO LunarLander project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` with your actual GitHub account and repository name.

## Adding Videos to GitHub

This demo video is small enough to commit directly to Git:

```bash
git add download.mp4
git commit -m "Add LunarLander demo video"
git push
```

GitHub has a normal file size limit of `100 MB`. If your video is larger than that, use Git LFS:

```bash
git lfs install
git lfs track "*.mp4"
git add .gitattributes
git add download.mp4
git commit -m "Add demo video with Git LFS"
git push
```

After pushing, keep the README video link as:
## Demo Video

https://github.com/Praveen23-kk/ppo-LunarLander-v3/blob/main/download.mp4

## Recommended Repository Structure

```text
.
├── README.md
├── LunarLander_v3_RL.ipynb
└── download.mp4
```

## Notes

- Colab is recommended because it handles Linux system packages and rendering setup more easily.
- Use `LunarLander-v3`; older versions such as `LunarLander-v2` are deprecated.
- If you upload to Hugging Face Hub, update the notebook's `repo_id` before running upload cells.
- For better GitHub presentation, rename `download.mp4` to something descriptive like `lunarlander-demo.mp4`.
