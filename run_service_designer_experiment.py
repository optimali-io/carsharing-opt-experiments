from service_designer.tasks.run_experiments import run_gridsearch

def run_sdesigner_experiment(experiment_id: str, prepare_data: bool = True):
    run_gridsearch(experiment_id=experiment_id, prepare_data=prepare_data)


if __name__ == "__main__":
    # Example usage
    experiment_name = "lodz_synth_experyment"
    run_sdesigner_experiment(experiment_name, prepare_data=False)

