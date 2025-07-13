# Carsharing Optimisation Experiments

This repository contains tools and scripts to run experiments on carsharing fleet management. It focuses on optimising the work of service teams (relocation and refuelling) and tuning the free-floating zone. Optimisation uses a configurable, distributed Island Genetic Algorithm (IGA) combined with a fast Cython based simulator.

## Features

- Distributed genetic optimisation of service actions.
- Generation of demand, directions and revenue matrices for simulations.
- Zone preparation utilities for converting GIS data into model inputs.
- Example scripts for creating default configuration and service teams.

## Repository structure

- `core/` – demand processing, direction modelling and data access helpers.
- `fleet_manager/` – genetic algorithm code and optimisation tasks.
- `service_designer/` – zone optimisation experiment runner and simulation utilities.
- `prepare_zone_spatial_data/` – scripts for building zone geofiles.

## Sample data

Synthetic datasets containing rents, user app openings, vehicle statuses and zone borders along with reference resources (hexagonal grid, travel time and distance matrices and petrol station locations) can be downloaded from AWS S3 at `[https://carsharing-sample-data.s3.eu-west-1.amazonaws.com/]`.

## License
The project is released under the Mozilla Public License 2.0.
