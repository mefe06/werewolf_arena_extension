# Werewolf The Social Deduction Game
This repository provides code for [Werewolf Arena](https://arxiv.org/abs/2407.13943) - a framework for evaluating the social reasoning skills of large language models (LLMs) through the game of Werewolf.

## Set up the environment

### Create a Python Virtual Environment
conda create -n $env-name python=3.12

### 
conda activate $env-name

###
pip install -r requirements.txt

### Set OpenAI API Key for using GPTs. create .env file, add following to it.

OPENAI_API_KEY=<your api key>

### The program will read from this environment variable.

### Set up GCP for using Gemini
 - [Install the gcloud cli](https://cloud.google.com/sdk/docs/install)
 - Authenticate and set your GCP project
 - Create the application default credentials by running 
 ```
 gcloud auth application-default login
 ```

## Run a single game

`python3 main.py --run --v_models=gpt-4.1-nano --w_models=gpt-4.1-nano`


## Run games between all model combinations

`python3 main.py --eval --num_games=5 --v_models=pro1.5,flash --w_models=gpt4,gpt4o`
