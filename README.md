# Carsharing Optimisation Experiments

This repository contains tools and scripts to run experiments on carsharing fleet management. It focuses on optimising the work of service teams (relocation and refuelling) and tuning the free-floating zone. Optimisation uses a configurable, distributed Island Genetic Algorithm (IGA) combined with a fast Cython based simulator.

## Features

- Distributed genetic optimisation of service actions.
- Zone optimisation for free-floating carsharing.
- Generation of demand, directions and revenue matrices for simulations.
- Zone preparation utilities for converting GIS data into model inputs.
- Example scripts for creating default configuration and service teams.

## Repository structure

- `core/` – demand processing, direction modelling and data access helpers.
- `fleet_manager/` – genetic algorithm code and optimisation tasks.
- `service_designer/` – zone optimisation experiment runner and simulation utilities.
- `prepare_zone_spatial_data/` – scripts for building zone geofiles.

## Setup

Code should run on UNIX-like systems (Linux, macOS). It is tested on Ubuntu 22 and 24.
 To run the experiments, you need to set up the environment and install the required dependencies. The following steps outline the process:

### Python and virtual environment 

- install `uv`: https://docs.astral.sh/uv/#installation
- install python 3.12 or later:
- ```bash
  uv python install 3.12
  ```
- install dependencies for Cython https://cython.readthedocs.io/en/latest/src/quickstart/install.html. For Ubuntu and Debian this should suffice:
  ```bash
  sudo apt install build-essential python3-dev
  ```
- create a virtual environment:
  ```bash
  uv venv .venv --python 3.12
  ```
- activate the virtual environment:
  ```bash
  source .venv/bin/activate
  ```
- install the required dependencies:
  ```bash
  uv sync
  ```
- copy the example configuration file:
  ```bash
  cp .env.example .env
  ```
- edit the `.env` file to set up the required environment variables. MAIN_PATH should point to the directory where the data folder is located. 

NOTE: `HERE_API_KEY` and `HERE_API_ID` inv `.env` are not mandatory, but are required for zone data preparation used in the project. You can get them by signing up at https://developer.here.com/.
However, you can test the project without them by using provided sample data. (see below).

### Sample data

Synthetic datasets containing rents, vehicle statuses and zone borders along with reference resources (hexagonal grid, travel time and distance matrices and petrol station locations) 

can be downloaded from AWS S3 at https://carsharing-sample-data.s3.eu-west-1.amazonaws.com/data_structure_with_synthetic_data.zip.
Unzip the downloaded file. PLEASE NOTE! The `MAIN_PATH` variable in the `.env` file should point to this unzipped directory.

After files are in place and `.env` file is configured, you can create example configuration files by running:
```bash
python -m helpers.create_sample_configs
```
This step is not necessary as all the configs are already provided in the package, but if you want to experiment with different configurations, you can use this command to generate new ones. 

See the `helpers/create_sample_configs.py` script for more details.


### Redis
Redis is used as a storage for asynchronous tasks and results.
See installation guide: https://redis.io/docs/latest/operate/oss_and_stack/install/install-stack/

Remember to start the Redis server before running the experiments!

## Running experiments
We have provided 3 example scripts to run experiments. Each script is configured in the `main` function.
By default, the `main` function runs the experiment without data preparation. You can change that by running `with_data_preparation` version of the function.

PLEASE NOTE! All the experiments run in parallel. `run_fleetmanager_optimisation.py` and `run_service_designer_experiment.py` require Huey task consumer to be run (read about huey here https://github.com/coleifer/huey). 

Before running the experiments, make sure that the task consumer is working:
```bash
./run_task_consumer.sh
```
In those scripts, python only enqueues the tasks, and the actual work is done by the task consumer.
You can watch progress in the terminal where the task consumer is running.

### Fleetmanager optimisation
This showcases the distributed genetic algorithm for optimising service actions (relocation and refuelling) in a carsharing fleet. Algorithm uses parallel dynamic island model, where islands are created using spectral clustering method every n epochs.   It uses the `fleet_manager` module to run the optimisation process.
To run the fleet manager optimisation experiment, use the following command:
```bash
python run_fleetmanager_optimisation.py
```
The result of the optimisation will be saved in the `<MAIN_PATH>/results/<zone name>_<datetime of the optimisation>` directory.

### Service designer experiment
This script runs a service designer experiment, which optimises the free-floating zone and simulates the carsharing fleet performance. It uses the `service_designer` module to run the experiment.
To run the service designer experiment, use the following command:
```bash
python run_service_designer_experiment.py
```
The result of the experiment will be saved in the `<MAIN_PATH>/sdesigner/experiments_results/<experiment name>` directory.


### Genetic algorithm with static topographies experiment
This script runs a genetic algorithm experiment with static island topographies, which optimises the service actions in a carsharing fleet (result is the same as in the fleetmanager experiment). It uses the `fleet_manager` module to run the experiment.
You can see abstract examples of the topographies in the [fleet_manager/genetic_algorithm/connection_topography.py](fleet_manager/genetic_algorithm/connection_topography.py) file.
To run the experiment, use the following command:
```bash
python run_genetic_algorithm_cpu_topography.py
```

## Using your own data and creating your own experiments
It is entirely possible to use your own data and create your own experiments. The project is designed to be modular and extensible. You can create your own zones, service teams, vehicles as well as your own rents and app data.
The provided example scripts can be used as a starting point for your own experiments.

#### Zone Data Preparation
Zone preparation scripts are located in the `prepare_zone_spatial_data` directory. You can use the provided scripts to convert your own GIS data into the required format. The scripts use HERE MATRIX API to get the baseline travel times and distances.
See [prepare_zone_spatial_data/README.md](prepare_zone_spatial_data/README.md) for more details and instructions.

#### Raw data preparation
Convert your rents data into the required format (see the example, synthetic data). 
Please note that rents data should have a `cell_id` column. After zone files are prepared (see the point above), run the script to assign the rents to the cells:
```bash
python -m helpers.assign_cell_ids_to_rents
```
Remember to provide `zone_id` in the main function of the script. The script will read the rents data from the `rents` directory and assign the cell ids to the rents based on the referential grid.

## License
The project is released under the Mozilla Public License 2.0.
