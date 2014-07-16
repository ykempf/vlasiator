#include "spatial_cell_funcs.hpp"
#include <vector>
#include "../vlasovsolver/cpu_acc_semilag.hpp"
using namespace spatial_cell;

// Use same values for all dimensions for this test
const int spatial_cell_side_length = 25; // 30 is the realistic case, 10 or 100 good for testing
const Real v_min = -1e6;
const Real v_max = 1e6;
const Real min_value = 1e-17;

// Required parameters hard coded here
int  Parameters::sparseBlockAddWidthV = 0;
Real Parameters::q = 1.60217653e-19;
Real Parameters::maxSlAccelerationRotation=10.0;
Real P::lorentzHallMinimumRho=1.0;

// Initializes the SpatialCell static variables to values given above
void init_spatial_cell_static(void) {
    SpatialCell::vx_length = spatial_cell_side_length;
    SpatialCell::vy_length = spatial_cell_side_length;
    SpatialCell::vz_length = spatial_cell_side_length;
    SpatialCell::vx_min = v_min;
    SpatialCell::vx_max = v_max;
    SpatialCell::vy_min = v_min;
    SpatialCell::vy_max = v_max;
    SpatialCell::vz_min = v_min;
    SpatialCell::vz_max = v_max;
    SpatialCell::velocity_block_min_value = min_value;
    
    
    // Rest are calculated from the previous
    SpatialCell::max_velocity_blocks =
    SpatialCell::vx_length * SpatialCell::vy_length * \
                                                SpatialCell::vz_length;
    SpatialCell::grid_dvx = SpatialCell::vx_max - \
                                                SpatialCell::vx_min;
    SpatialCell::grid_dvy = SpatialCell::vy_max - \
                                                SpatialCell::vy_min;
    SpatialCell::grid_dvz = SpatialCell::vz_max - \
                                                SpatialCell::vz_min;
    SpatialCell::block_dvx = SpatialCell::grid_dvx /               \
                                                SpatialCell::vx_length;
    SpatialCell::block_dvy = SpatialCell::grid_dvy /               \
                                                SpatialCell::vy_length;
    SpatialCell::block_dvz = SpatialCell::grid_dvz /               \
                                                SpatialCell::vz_length;
    SpatialCell::cell_dvx = SpatialCell::block_dvx /               \
                                                block_vx_length;
    SpatialCell::cell_dvy = SpatialCell::block_dvy / block_vy_length;
    SpatialCell::cell_dvz = SpatialCell::block_dvz / block_vz_length;
}

// Sets the data on all existing cells to zero.
void clear_data(SpatialCell *spacell) {
    unsigned int ind;
    Velocity_Block* block_ptr;
    for(int i = 0; i < spacell->number_of_blocks; i++) {
        ind = spacell->velocity_block_list[i];
        block_ptr = spacell->at(ind);
        for (int j = 0; j < WID3; j++) {
            block_ptr->data[j] = (Real)0.0;
        }
    }
}

// Prints information about the velocity blocks on the CPU. Used to check correct transfer to GPU.
void print_blocks(SpatialCell *cell) {
    velocity_block_indices_t indices;
    unsigned int ind;
    for(unsigned int i=0; i<cell->number_of_blocks; i++) {
        ind = cell->velocity_block_list[i];
        printf("%5.0u: ", ind);
        Velocity_Block* block_ptr = cell->at(ind);
        indices = SpatialCell::get_velocity_block_indices(ind);
        printf("(%4i, %4i, %4i) %7.1f\n", indices[0], indices[1], indices[2], block_ptr->data[0]);
    }
}

SpatialCell *create_index_test_cell(void) {
    const int ids_len = 8;
    int ids[] = {1, 10, 100, 101, 999, 1001, 9999, 10005};
    
    SpatialCell *cell;
    cell = new SpatialCell();
    
    // Add blocks to the given ids
    for (int i=0; i<ids_len; i++) {
        int ind = ids[i];
        cell->add_velocity_block(ind);
        Velocity_Block* block_ptr = cell->at(ind);
        block_ptr->data[0]=ind; // Put some data into each velocity cell
    }
    return cell;
}

// Projects the velocity grid to xy-plane by summing each block to a single value
float *xy_projection(SpatialCell *spacell) {
    float *projection = new float[SpatialCell::vx_length * SpatialCell::vy_length](); // Initialize to 0

    velocity_block_indices_t indices;
    unsigned int ind;
    float sum;
    for(unsigned int i=0; i < spacell->number_of_blocks; i++) {
        ind = spacell->velocity_block_list[i];
        Velocity_Block* block_ptr = spacell->at(ind);
        indices = SpatialCell::get_velocity_block_indices(ind);
        
        sum = 0.;
        for (unsigned int j = 0; j < VELOCITY_BLOCK_LENGTH; j++) {
            //if (j==i) printf("%i %f\n", i, block_ptr->data[j]);
            sum += block_ptr->data[j];
        }
        projection[indices[0] + SpatialCell::vx_length * indices[1]] += sum;
    }
    
    return projection;
}

void fprint_projection(float *projection, std::string filename) {
    using namespace std;
    ofstream fout(filename);
    if (!fout.is_open()) {
        cout<< "Could not open " << filename << " for output.";
        return;
    }
    for(unsigned int i = 0; i < SpatialCell::vx_length; i++) {
        for (unsigned int j = 0; j < SpatialCell::vy_length; j++) {
            fout << projection[i] << " ";
        }
        fout << endl;
    }
    fout.close();
}

// Returns values for the given index
Real Maxwell(Real vx, Real vy, Real vz, Real T, Real rho) {
    Real temp;
    Real val = rho;
    val *= pow(physicalconstants::MASS_PROTON / (2.0 * M_PI * \
    physicalconstants::K_B * T), 1.5);
    //if (val == 0.0) printf("Error at %f %f %f\n", vx, vy, vz);
    temp = (vx * vx + vy * vy + vz * vz) / (2.0 * physicalconstants::K_B * T);
//printf("%e %e\n", val, -physicalconstants::MASS_PROTON * temp);
    val *= exp(-physicalconstants::MASS_PROTON * temp);
    //if (val == 0.0) printf("Error at %f %f %f\n", vx, vy, vz);
    return val;
}

// Returns a spatial cell with a Maxwellian velocity space of size len_side^3
SpatialCell *create_maxwellian(Real T, Real rho) {
    SpatialCell *spacell;
    spacell = new SpatialCell();
    // Add some parameters by hand
    spacell->parameters[CellParams::BGBX] = 1.0e-9;
    spacell->parameters[CellParams::BGBY] = 1.0e-9;
    spacell->parameters[CellParams::BGBZ] = 1.0e-9;
    spacell->parameters[CellParams::BGBXVOL] = 1.0e-9;
    spacell->parameters[CellParams::BGBYVOL] = 1.0e-9;
    spacell->parameters[CellParams::BGBZVOL] = 1.0e-9;
    spacell->parameters[CellParams::PERBXVOL] = 0.0;
    spacell->parameters[CellParams::PERBYVOL] = 0.0;
    spacell->parameters[CellParams::PERBZVOL] = 0.0;
    spacell->derivativesBVOL[bvolderivatives::dPERBXVOLdy] = 0.0;
    spacell->derivativesBVOL[bvolderivatives::dPERBXVOLdz] = 0.0;
    spacell->derivativesBVOL[bvolderivatives::dPERBYVOLdx] = 0.0;
    spacell->derivativesBVOL[bvolderivatives::dPERBYVOLdz] = 0.0;
    spacell->derivativesBVOL[bvolderivatives::dPERBZVOLdx] = 0.0;
    spacell->derivativesBVOL[bvolderivatives::dPERBZVOLdy] = 0.0;
    spacell->parameters[CellParams::DX] = 100;
    spacell->parameters[CellParams::DY] = 100;
    spacell->parameters[CellParams::DZ] = 100;
    // Loop over the whole velocity space
    Real val;
    Real block_vx;
    Real block_vy;
    Real block_vz;
    Real cell_vx;
    Real cell_vy;
    Real cell_vz;
    velocity_cell_indices_t cell_indices;
    // Loop over blocks
    for (unsigned int i = 0; i < SpatialCell::vx_length; i++) {
        for (unsigned int j = 0; j < SpatialCell::vy_length; j++) {
            for (unsigned int k = 0; k < SpatialCell::vz_length; k++) {
                // Calculate the velocities corresponding to the block indices
                block_vx = SpatialCell::vx_min + SpatialCell::block_dvx * i;
                block_vy = SpatialCell::vy_min + SpatialCell::block_dvy * j;
                block_vz = SpatialCell::vz_min + SpatialCell::block_dvz * k;
                for (unsigned int ci = 0; ci < VELOCITY_BLOCK_LENGTH; ci++) {
                    // Add the offset for each cell
                    cell_indices = SpatialCell::get_velocity_cell_indices(ci);
                    cell_vx = block_vx + cell_indices[0]*SpatialCell::cell_dvx;
                    cell_vy = block_vy + cell_indices[1]*SpatialCell::cell_dvy;
                    cell_vz = block_vz + cell_indices[2]*SpatialCell::cell_dvz;
                    
                    val = Maxwell(cell_vx, cell_vy, cell_vz, T, rho);
                    /*
                    if (ci == 0 && i == j && j == k) {
                        printf("%6.2e %6.2e %6.2e: ", cell_vx, cell_vy, cell_vz);
                        printf("%i %e\n", i, val);
                    }
                    */
                    spacell->set_value(cell_vx, cell_vy, cell_vz, val);
                }
            }
        }
    }
    // Remove unnecessary blocks from spatial cell
    std::vector<SpatialCell*> neighbor_ptrs;
    spacell->update_velocity_block_content_lists();
    spacell->adjust_velocity_blocks(neighbor_ptrs,true);
    
    return spacell;
}

// Sorts the blockid list and block_data according to blockids
std::vector<int>* sorted_velocity_block_list(SpatialCell * spacell) {
    // Container for sorting
    class sort_indices {
        private:
        unsigned int* mparr;
        public:
        sort_indices(unsigned int* parr) : mparr(parr) {}
        bool operator()(int i, int j) { return mparr[i]<mparr[j]; }
    };
    std::vector<int> *indices = new std::vector<int>(spacell->number_of_blocks);
    // Initialize the index array
    for (int i = 0; i < spacell->number_of_blocks; i++) (*indices)[i] = i;
    // Sort indices to find the correct order
    std::sort(indices->begin(), indices->end(), sort_indices(spacell->velocity_block_list.data()));

    // Put the values to correct positions
    return indices;
}

// Wrapper for the Vlasiator function
void cpu_acc_cell(SpatialCell *spacell, const Real dt) {
    cpu_accelerate_cell(spacell, dt);
}