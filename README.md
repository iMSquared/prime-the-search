# Search Tree augmented by Language Model (STaLM)

---

This is an official code for STaLM (submitted to IJRR)

![Figure][media/pipeline.png]

## Set up

```bash
# Setup conda environment
conda env create -f environment.yaml

```

STaLM makes use of GPT-4 api, so please prepare openai api key. Upon initial learning, you will be prompted to register the openai key

```bash
    OpenAI API token is not found at ~/STaLM/LLM/api_token.json. Please enter your OpenAI API token.
    Please enter your OpenAI API token:

```

## Run

To run the STaLM, use

```bash
python Simulation/pybullet/experiment_shop.py --override_scenario_num $X --num_episodes $Y --override_num_sims $Z --baseline STaLM

```

Here, X is the scenario number of the experiment, Y is the number experiments to repeat, and Z is the number of simulation for STaLM to run during WarmStartUCT procedure. To reproduce the result, you can test scenario 1~6 (i.e. X=1~6) with Y=50 and Z=30.

Besides STaLM, you can also test other baselines in the paper by simply giving the name instead of STaLM in the command above. The supported baselines are LLM_MCTS, NO_UCT, MCTS_UCT, Iterative_Replanning, SAYCAN.