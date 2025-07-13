from cython import boundscheck, wraparound
from cython.parallel import prange
from libc.stdlib cimport rand, srand


@boundscheck(False)
@wraparound(False)
def fast_simulate_day(unsigned short [:, :] location_row_vehicle,
                      int simulations_nbr,
                      unsigned short [:, :, :] demand_sample_hour_cell,
                      int demand_sample_nbr,
                      unsigned short [:, :, :] distance_hour_cell_cell,
                      unsigned short [:, :, :] time_hour_cell_cell,
                      unsigned short [:, :, :] destination_hour_cell_sample,
                      int [:, :] destination_sidx_hour_cell,
                      int destination_sample_nbr,
                      unsigned short [:] region_cell,
                      int regions_nbr,
                      int rows_nbr,
                      int cells_nbr,
                      int vehicles_nbr,
                      int hours_nbr,
                      float [:, :] out_revenue_row_simulation,
                      unsigned short [:, :] out_rents_row_simulation,
                      float [:, :, :] out_satisfied_demand_row_simulation_region,
                      int trace,
                      int seed,
                      float [:, :, :, :] out_revenue_row_simulation_vehicle_hour,
                      unsigned short [:, :, :, :] out_location_row_simulation_vehicle_hour,
                      unsigned short [:, :, :, :] out_distance_row_simulation_vehicle_hour,
                      unsigned short [:, :, :, :] out_time_row_simulation_vehicle_hour,
                      unsigned char [:, :, :, :] out_rents_row_simulation_vehicle_hour):
    cdef int ridx
    cdef int sidx
    srand(seed)
    for ridx in prange(rows_nbr, nogil=True):
        c_simulate_samples(location_row_vehicle[ridx],
                           simulations_nbr,
                           demand_sample_hour_cell,
                           demand_sample_nbr,
                           distance_hour_cell_cell,
                           time_hour_cell_cell,
                           destination_hour_cell_sample,
                           destination_sidx_hour_cell,
                           destination_sample_nbr,
                           region_cell,
                           regions_nbr,
                           cells_nbr,
                           vehicles_nbr,
                           hours_nbr,
                           out_revenue_row_simulation[ridx],
                           out_rents_row_simulation[ridx],
                           out_satisfied_demand_row_simulation_region[ridx],
                           trace,
                           out_revenue_row_simulation_vehicle_hour,
                           out_location_row_simulation_vehicle_hour,
                           out_distance_row_simulation_vehicle_hour,
                           out_time_row_simulation_vehicle_hour,
                           out_rents_row_simulation_vehicle_hour,
                           ridx)


@boundscheck(False)
@wraparound(False)
cdef void c_simulate_samples(unsigned short[:] location_vehicle,
                             int simulations_nbr,
                             unsigned short [:, :, :] demand_sample_hour_cell,
                             int demand_sample_nbr,
                             unsigned short [:, :, :] distance_hour_cell_cell,
                             unsigned short [:, :, :] time_hour_cell_cell,
                             unsigned short [:, :, :] destination_hour_cell_sample,
                             int [:, :] destination_sidx_hour_cell,
                             int destination_sample_nbr,
                             unsigned short [:] region_cell,
                             int regions_nbr,
                             int cells_nbr,
                             int vehicles_nbr,
                             int hours_nbr,
                             float [:] out_revenue_simulation,
                             unsigned short [:] out_rents_simulation,
                             float [:, :] out_satisfied_demand_simulation_region,
                             int trace,
                             float [:, :, :, :] out_revenue_row_simulation_vehicle_hour,
                             unsigned short [:, :, :, :] out_location_row_simulation_vehicle_hour,
                             unsigned short [:, :, :, :] out_distance_row_simulation_vehicle_hour,
                             unsigned short [:, :, :, :] out_time_row_simulation_vehicle_hour,
                             unsigned char [:, :, :, :] out_rents_row_simulation_vehicle_hour,
                             int ridx) nogil:
    cdef:
        unsigned short tmp_location_vehicle[2048]
        unsigned short tmp_moves_cell[4096]
        unsigned short tmp_region_moves[256]
        unsigned short tmp_region_demand[256]
        float out_day_revenue[1]
        unsigned short out_day_rents[1]
        int sidx
        int demand_sample_idx


    out_day_revenue[0] = 0.0
    out_day_rents[0] = 0


    for sidx in range(simulations_nbr):
        demand_sample_idx = rand() % demand_sample_nbr

        c_fill_zeros(tmp_region_moves, regions_nbr)
        c_fill_zeros(tmp_region_demand, regions_nbr)

        c_simulate_day(location_vehicle,
                       tmp_location_vehicle,
                       tmp_moves_cell,
                       tmp_region_moves,
                       tmp_region_demand,
                       demand_sample_hour_cell[demand_sample_idx],
                       distance_hour_cell_cell,
                       time_hour_cell_cell,
                       destination_hour_cell_sample,
                       destination_sidx_hour_cell,
                       destination_sample_nbr,
                       region_cell,
                       regions_nbr,
                       cells_nbr,
                       vehicles_nbr,
                       hours_nbr,
                       out_day_revenue,
                       out_day_rents,
                       trace,
                       out_revenue_row_simulation_vehicle_hour,
                       out_location_row_simulation_vehicle_hour,
                       out_distance_row_simulation_vehicle_hour,
                       out_time_row_simulation_vehicle_hour,
                       out_rents_row_simulation_vehicle_hour,
                       ridx,
                       sidx)

        out_revenue_simulation[sidx] = out_day_revenue[0]
        out_rents_simulation[sidx] = out_day_rents[0]

        for rg_idx in range(regions_nbr):
            if tmp_region_demand[rg_idx] > 0:
                out_satisfied_demand_simulation_region[sidx, rg_idx] = float(tmp_region_moves[rg_idx]) / float(tmp_region_demand[rg_idx])
            else:
                out_satisfied_demand_simulation_region[sidx, rg_idx] = 0


@boundscheck(False)
@wraparound(False)
cdef void c_simulate_day(unsigned short[:] location_vehicle,
                         unsigned short [] tmp_location_vehicle,
                         unsigned short [] tmp_moves_cell,
                         unsigned short [] tmp_region_moves,
                         unsigned short [] tmp_region_demand,
                         unsigned short [:, :] demand_hour_cell,
                         unsigned short [:, :, :] distance_hour_cell_cell,
                         unsigned short [:, :, :] time_hour_cell_cell,
                         unsigned short [:, :, :] destination_hour_cell_sample,
                         int [:, :] destination_sidx_hour_cell,
                         int destination_sample_nbr,
                         unsigned short [:] region_cell,
                         int regions_nbr,
                         int cells_nbr,
                         int vehicles_nbr,
                         int hours_nbr,
                         float[] out_day_revenue,
                         unsigned short[] out_day_rents,
                         int trace,
                         float [:, :, :, :] out_revenue_row_simulation_vehicle_hour,
                         unsigned short [:, :, :, :] out_location_row_simulation_vehicle_hour,
                         unsigned short [:, :, :, :] out_distance_row_simulation_vehicle_hour,
                         unsigned short [:, :, :, :] out_time_row_simulation_vehicle_hour,
                         unsigned char [:, :, :, :] out_rents_row_simulation_vehicle_hour,
                         int ridx,
                         int sidx) nogil:
    cdef:
        float out_hour_revenue[1]
        unsigned short out_hour_rents[1]
        float revenue_sum = 0.0
        unsigned short rents_sum = 0
        int i
        int hidx

    for i in range(vehicles_nbr):
        tmp_location_vehicle[i] = location_vehicle[i]

    out_hour_revenue[0] = 0.0
    out_hour_rents[0] = 0

    for hidx in range(hours_nbr):
        c_simulate_hour(tmp_location_vehicle,
                        tmp_moves_cell,
                        tmp_region_moves,
                        tmp_region_demand,
                        demand_hour_cell[hidx],
                        distance_hour_cell_cell,
                        time_hour_cell_cell,
                        hidx,
                        vehicles_nbr,
                        cells_nbr,
                        destination_hour_cell_sample,
                        destination_sidx_hour_cell,
                        destination_sample_nbr,
                        region_cell,
                        regions_nbr,
                        out_hour_revenue,
                        out_hour_rents,
                        trace,
                        out_revenue_row_simulation_vehicle_hour,
                        out_location_row_simulation_vehicle_hour,
                        out_distance_row_simulation_vehicle_hour,
                        out_time_row_simulation_vehicle_hour,
                        out_rents_row_simulation_vehicle_hour,
                        ridx,
                        sidx,
                        hidx)
        revenue_sum += out_hour_revenue[0]
        rents_sum += out_hour_rents[0]

    out_day_revenue[0] = revenue_sum
    out_day_rents[0] = rents_sum


@boundscheck(False)
@wraparound(False)
cdef void c_simulate_hour(unsigned short [] location_vehicle,
                          unsigned short [] moves_cell,
                          unsigned short [] tmp_region_moves,
                          unsigned short [] tmp_region_demand,
                          unsigned short [:] demand_cell,
                          unsigned short [:, :, :] distance_hour_cell_cell,
                          unsigned short [:, :, :] time_hour_cell_cell,
                          int h,
                          int vehicles_nbr,
                          int cells_nbr,
                          unsigned short [:, :, :] destination_hour_cell_sample,
                          int [:, :] destination_sidx_hour_cell,
                          int destination_sample_nbr,
                          unsigned short [:] region_cell,
                          int regions_nbr,
                          float[] out_hour_revenue,
                          unsigned short[] out_hour_rents,
                          int trace,
                          float [:, :, :, :] out_revenue_row_simulation_vehicle_hour,
                          unsigned short [:, :, :, :] out_location_row_simulation_vehicle_hour,
                          unsigned short [:, :, :, :] out_distance_row_simulation_vehicle_hour,
                          unsigned short [:, :, :, :] out_time_row_simulation_vehicle_hour,
                          unsigned char [:, :, :, :] out_rents_row_simulation_vehicle_hour,
                          int ridx,
                          int sidx,
                          int hidx) nogil:

    cdef:
        double revenue = 0
        double revenue_sum = 0
        unsigned short rents_sum = 0
        double satisfied_demand_sum = 0
        int demand_cells_nbr = 0
        unsigned short src_cell
        unsigned short dst_cell
        int i
        int sum
        int region_idx

    c_fill_zeros(moves_cell, cells_nbr)

    for vidx in range(vehicles_nbr):
        src_cell = location_vehicle[vidx]

        if moves_cell[src_cell] < demand_cell[src_cell]:
            dst_cell = c_get_dst_cell(destination_sidx_hour_cell, destination_hour_cell_sample, destination_sample_nbr, h, src_cell)
            revenue = calculate_ride_price(distance_hour_cell_cell[hidx, src_cell, dst_cell],
                                           time_hour_cell_cell[hidx, src_cell, dst_cell])
            revenue_sum += revenue
            rents_sum += 1
            location_vehicle[vidx] = dst_cell
            moves_cell[src_cell] += 1
            if trace == 1:
                out_revenue_row_simulation_vehicle_hour[ridx, sidx, vidx, hidx] = revenue
                out_distance_row_simulation_vehicle_hour[ridx, sidx, vidx, hidx] = distance_hour_cell_cell[hidx, src_cell, dst_cell]
                out_time_row_simulation_vehicle_hour[ridx, sidx, vidx, hidx] = time_hour_cell_cell[hidx, src_cell, dst_cell]
                out_rents_row_simulation_vehicle_hour[ridx, sidx, vidx, hidx] += 1

        if trace == 1:
            out_location_row_simulation_vehicle_hour[ridx, sidx, vidx, hidx] = location_vehicle[vidx]

    for cell in range(cells_nbr):
        region_idx = region_cell[cell]
        tmp_region_moves[region_idx] += moves_cell[cell]
        tmp_region_demand[region_idx] += demand_cell[cell]

    out_hour_revenue[0] = revenue_sum
    out_hour_rents[0] = rents_sum


@boundscheck(False)
@wraparound(False)
cdef void c_fill_zeros(unsigned short [] array, int len) nogil:
    cdef int i
    for i in range(len):
        array[i] = 0


@boundscheck(False)
@wraparound(False)
cdef unsigned short c_get_dst_cell(int [:, :] destination_sidx_hour_cell,
                                   unsigned short [:, :, :] destination_hour_cell_sample,
                                   int destination_sample_nbr,
                                   int hour,
                                   int src_cell) nogil:
    cdef int sample_idx = destination_sidx_hour_cell[hour, src_cell]
    destination_sidx_hour_cell[hour, src_cell] = (sample_idx + 1) % destination_sample_nbr
    cdef unsigned short dst_cell = destination_hour_cell_sample[hour, src_cell, sample_idx]
    return dst_cell


@boundscheck(False)
@wraparound(False)
cpdef double calculate_ride_price(unsigned short meters, unsigned short seconds) nogil:
    cdef double price
    price = 0
    price += (meters + 999) // 1000 * 0.8
    price += (seconds + 59) // 60 * 0.5
    if price < 2:
        price = 2
    return price
