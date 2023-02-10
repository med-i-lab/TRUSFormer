# TRUSnet
Research code for ai-based automatic prostate cancer detection

## Summary
This repository provides access to dataset and algorithm implementations for PCA detection, as well as useful library code for preprocessing, data augmentations, visualization, analysis, etc. It is designed to be as flexible as possible and serve as a one-stop-shop for these implementations. 

Features: 
- dataset implementations, preprocessing library code (`src/data`)
- various training methods for running experiments (`src/driver`)
- lots of utility library code (`src/data`)
- model implementations (`src/modeling`), model registry (`src.modeling.create_model(...)`)
- various algorithm implementations (we mostly prefer using PyTorch Lightning)
- experiments can be run through a single shared entry point `main.py`. eg. `python main.py experiment=01_TRUSFormer_reprod_PWilson_2023-01-19`

Libraries Used: 
- PyTorch (required)
- PyTorch Lightning (optional) - this library simplifies writing training loops and removes some boiler plate
- Hydra (optional) - this is a configuration library which hugely helps with managing configurations (`configs/`) don't be intimidated at first by hydra. It is simpler than it looks

## Quick start
How to use: 

1. Clone the repository: 
```bash 
git clone https://github.com/med-i-lab/TRUSnet
cd TRUSnet
```

2. Install the requirements: 
```bash
pip install -r requirements.txt
```

3. Create a copy of the .env.example file, call it `.env`, 
```bash
cat .env.example > .env
```
and fill in the required field (at a minimum, you will likely want need to fill in `SERVER_USERNAME` (your username on the image server) and `SERVER_PASSWORD` (password on image server) to avoid being constantly prompted to log into the image server where the data is stored

4. Now you can explore the repository. I'd recomment running the TRUSFormer experiment
 ```bash
 python main.py experiment=01_TRUSFormer_reprod_PWilson_2023-01-19
 ``` 
 (you'll be prompted to log in to WandB) or simply walk through the notebook `notebooks/1_onboarding_Mohamed_Harmanani_2023-01-30.ipynb` to get a sense of what to do with the repo. 
 
## Implementing your own training loop
Don't be intimidated by hydra. Is just a very sophisticated `yaml` parser! Hydra lets you tell `main.py` what configurations to run. Try running it with no experiment specified: 
```bash
python main.py
``` 
You will see that it prints the configuration (it got this by just reading `configs/config.yaml` and getting the bare minimum of configs). Then it will crash with the error "no driver specified in config". This is because `main.py` proceeds by instantiating a `driver` object (this can be anything that implements a `driver.run(self)` method) from the config.driver field. However, without adding an experiment, this field is not defined. 

Now, let's try the basic example experiment: 
```bash 
python main.py experiment=examples/basic.yaml
```
Now, it actually runs a not very interesting experiment where it just prints the experiment settings and leaves. What have we done? We told hydra to go look in the experiments config group (`configs/experiments`) and find the file examples/basic.yaml, and add the configurations defined here to the basic configurations. If you go look at the file, you will see the following: 
```yaml
driver: 
  _target_: src.driver.example.BasicExample
  setting1: 'Hello'
  setting2: 'World'
```
This means the config object will have a field `driver` with three fields and values. By defining both the `_target_` field, which specifies a class in the source code, as well as the constructor arguments `setting1` and `setting2`, the program will use this to instantiate the target object (the experiment driver) and call its `run` method. If you go look at the source code `src.driver.example.BasicExample` you will see the `BasicExample` class that is being instantiated and run

Therefore, in order to run your own experiments, there are 3 simple steps: 
1. Write a driver class for the experiment (or use one of our premades) in the source code (under `src/driver/...`)
2. Make an yaml file "1_my_test_experiment_MyName_DATE.yaml" in the experiments section following the examples/basic.yaml template, which defines the driver by giving a target and constructor args
3. Run the driver `python main.py experiment=1_my_test_experiment_MyName_DATE.yaml
 
To see another example that actually does something, run: 
```bash
python main.py experiment=examples/mnist.yaml
```

## TRUSFormer

For our paper "TRUSformer: Improving Prostate Cancer Detection from Micro-Ultrasound Using Attention and Self-Supervision", the main config file to run the experiment is `configs/experiment/experiment=01_TRUSFormer_reprod_PWilson_2023-01-19`. To see the full configuration, run
```bash 
python main.py experiment=01_TRUSFormer_reprod_PWilson_2023-01-19 --cfg job
```

