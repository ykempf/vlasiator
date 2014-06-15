#include "spatial_cell_funcs.hpp"

using namespace spatial_cell;

int main(void) {
    init_spatial_cell_static();
    
    /*
    SpatialCell *cell = create_index_test_cell();
    // Print data as it is on CPU
    print_blocks(cell);
    */
    
    SpatialCell *cell = create_maxwellian(1.0e6, 1.0e5);
    printf("%u\n", cell->number_of_blocks);
    float *proj = xy_projection(cell);
    fprint_projection(proj, "maxwell_proj.out");
    printf("%e %e\n", physicalconstants::MASS_PROTON, physicalconstants::K_B);
    return 0;
}
