from config import huey
from core.db.data_access_facade_basic import DataAccessFacadeBasic
from service_designer.experiments.kpis import Kpis
from service_designer.simulator.parallel_simulator import ParallelSimulateIn, SimulationResult
from service_designer.simulator.simulator_daybyday import Simulator


@huey.task(context=True)
def simulate(simulate_in: ParallelSimulateIn, task=None):
    simulator = Simulator(simulate_in.simulate_in)
    simulator.simulate_weeks()

    kpis = Kpis(
        simulator.simulate_out.vehicles,
        simulate_in.simulate_in.base_data,
        simulate_in.price_list,
        simulate_in.simulate_in.config,
    )
    kpis.calculate_and_clear_working_data()

    result = SimulationResult.create(
        kpis=kpis,
        experiment_id=simulate_in.simulate_in.experiment_id,
        simulate_in=simulate_in,
    )
    daf = DataAccessFacadeBasic()
    daf.save_simulation_result(result)