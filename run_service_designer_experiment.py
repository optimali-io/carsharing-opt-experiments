from service_designer.tasks.run_experiments import run_gridsearch

def run_sdesigner_experiment_with_data_preparation(experiment_id: str):
    run_gridsearch(experiment_id=experiment_id, prepare_data=True)

def run_sdesigner_experiment_without_data_preparation(experiment_id: str):
    run_gridsearch(experiment_id=experiment_id, prepare_data=False)


if __name__ == "__main__":
    # Example usage
    experiment_name = "lodz_synth_experyment"
    run_sdesigner_experiment_without_data_preparation(experiment_name)
    print(f"Service Designer experiment {experiment_name} started. Switch to the Huey consumer to monitor progress.")

