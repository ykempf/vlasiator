#include <stdio.h>
#include "cpu_1d_column_interpolation.hpp"
#include "common.h"

#define index(i,j,k)   ( k + WID + j * (blocks_per_dim_z + 2) * WID + i * (blocks_per_dim_z + 2) * blocks_per_dim_y * WID2 )
#define colindex(i,j)   ( j * (blocks_per_dim_z + 2) * WID + i * (blocks_per_dim_z + 2) * blocks_per_dim_y * WID2 )

/*print all values in the vector valued values array. In this array
  there are blocks_per_dim blocks with a width of WID*/
void print_values(int step, Real *values, uint blocks_per_dim, Real v_min, Real dv){
  char name[256];
  sprintf(name,"dist_%03d.dat",step);

  FILE* fp=fopen(name,"w");
  for(int i=0; i < blocks_per_dim * WID; i++){
    Real v = v_min + i*dv;
    fprintf(fp,"%20.12g %20.12g\n", v, values[i + WID]);
  }
  fclose(fp);
}


// Target needs to be allocated
void propagate(Real *values, Real *target, uint  blocks_per_dim, Real v_min, Real dv,
       uint i_block, uint i_cell, uint j_block, uint j_cell,
       Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk){

  Real a[RECONSTRUCTION_ORDER + 1];
  //Real *target = new Real[((int)spatial_cell::SpatialCell::vx_length+2)*WID];
  /*clear target*/
  for (uint k=0; k<WID* (blocks_per_dim + 2); ++k){
       target[k] = 0.0;
  }
   
   /* intersection_min is the intersection z coordinate (z after
      swaps that is) of the lowest possible z plane for each i,j
      index 
   */
  const Real intersection_min = intersection +
     (i_block * WID + i_cell) * intersection_di + 
     (j_block * WID + j_cell) * intersection_dj;
  

  /*compute some initial values, that are used to set up the
   * shifting of values as we go through all blocks in
   * order. See comments where they are shifted for
   * explanations of their meening*/

  /*loop through all blocks in column and compute the mapping as integrals*/
  for (unsigned int k_block = 0; k_block<blocks_per_dim;k_block++) {
    for (uint k_cell=0; k_cell<WID; ++k_cell){ 
      /*v_l, v_r are the left and right velocity coordinates of source cell*/
      Real v_l = v_min + (k_block * WID + k_cell) * dv;
      Real v_r = v_l + dv;
      /*left(l) and right(r) k values (global index) in the target
	lagrangian grid, the intersecting cells. Again old right is new left*/               
      const int target_gk_l = (int)((v_l - intersection_min)/intersection_dk);
      const int target_gk_r = (int)((v_r - intersection_min)/intersection_dk);

      for(int gk = target_gk_l; gk <= target_gk_r; gk++){
         //the velocity limits  for the integration  to put mass
         //in the targe cell. If both v_r and v_l are in same target cell
         //then v_int_l,v_int_r should be between v_l and v_r.
         //v_int_norm_l and v_int_norm_r normalized to be between 0 and 1 in the cell.
  const Real v_int_l = min( max((Real)(gk) * intersection_dk + intersection_min, v_l), v_r);
  const Real v_int_norm_l = (v_int_l - v_l)/dv;
  const Real v_int_r = min((Real)(gk + 1) * intersection_dk + intersection_min, v_r);
  const Real v_int_norm_r = (v_int_r - v_l)/dv;

  uint k = k_block * WID + k_cell;
  #ifdef ACC_SEMILAG_PLM
    compute_plm_coeff_explicit_columns(values, a, k);
  #endif
  #ifdef ACC_SEMILAG_PPM
    compute_ppm_coeff_explicit_columns(values, a, k);
  #endif
	 /*compute left and right integrand*/
#ifdef ACC_SEMILAG_PLM
	 Real target_density_l =
	   v_int_norm_l * a[0] +
	   v_int_norm_l * v_int_norm_l * a[1];
	 Real target_density_r =
	   v_int_norm_r * a[0] +
	   v_int_norm_r * v_int_norm_r * a[1];
#endif
#ifdef ACC_SEMILAG_PPM
	 Real target_density_l =
	   v_int_norm_l * a[0] +
	   v_int_norm_l * v_int_norm_l * a[1] +
	   v_int_norm_l * v_int_norm_l * v_int_norm_l * a[2];
	 Real target_density_r =
	   v_int_norm_r * a[0] +
	   v_int_norm_r * v_int_norm_r * a[1] +
	   v_int_norm_r * v_int_norm_r * v_int_norm_r * a[2];
#endif
	 /*total value of integrand, if it is wihtin bounds*/
         if ( gk >= 0 && gk <= blocks_per_dim * WID )
	   target[gk + WID] +=  target_density_r - target_density_l;
      }
    }
  }
  /*copy target to values*/
  /*
  for (unsigned int k_block = 0; k_block<blocks_per_dim;k_block++){
     for (uint k=0; k<WID; ++k){
        values[k_block * WID + k + WID] = target[k_block * WID + k + WID];
     }
  }
  */
}


// void propagate_old(Real *values, uint  blocks_per_dim, Real v_min, Real dv,
//          uint i_block, uint i_cell, uint j_block, uint j_cell,
//          Real intersection, Real intersection_di, Real intersection_dj, Real intersection_dk){
//   Real a[MAX_BLOCKS_PER_DIM*WID][RECONSTRUCTION_ORDER + 1];  
//   Real target[(MAX_BLOCKS_PER_DIM+2)*WID]; 
  
  
// #ifdef ACC_SEMILAG_PLM
//   compute_plm_coeff_explicit_columns(values, blocks_per_dim, a);
// #endif
// #ifdef ACC_SEMILAG_PPM
//   compute_ppm_coeff_explicit_columns(values, blocks_per_dim, a);
// #endif

//   /*clear temporary taret*/
//   for (uint k=0; k<WID* (blocks_per_dim + 2); ++k){ 
//        target[k] = 0.0;
//   }
   
//    /* intersection_min is the intersection z coordinate (z after
//       swaps that is) of the lowest possible z plane for each i,j
//       index 
//    */  
//   const Real intersection_min = intersection +
//      (i_block * WID + i_cell) * intersection_di + 
//      (j_block * WID + j_cell) * intersection_dj;
  

//   /*compute some initial values, that are used to set up the
//    * shifting of values as we go through all blocks in
//    * order. See comments where they are shifted for
//    * explanations of their meening*/

//   /*loop through all blocks in column and compute the mapping as integrals*/
//   for (unsigned int k_block = 0; k_block<blocks_per_dim;k_block++){
//     for (uint k_cell=0; k_cell<WID; ++k_cell){ 
//       /*v_l, v_r are the left and right velocity coordinates of source cell*/
//       Real v_l = v_min + (k_block * WID + k_cell) * dv;
//       Real v_r = v_l + dv;
//       /*left(l) and right(r) k values (global index) in the target
//   lagrangian grid, the intersecting cells. Again old right is new left*/               
//       const int target_gk_l = (int)((v_l - intersection_min)/intersection_dk);
//       const int target_gk_r = (int)((v_r - intersection_min)/intersection_dk);

//       for(int gk = target_gk_l; gk <= target_gk_r; gk++){
//          //the velocity limits  for the integration  to put mass
//          //in the targe cell. If both v_r and v_l are in same target cell
//          //then v_int_l,v_int_r should be between v_l and v_r.
//          //v_int_norm_l and v_int_norm_r normalized to be between 0 and 1 in the cell.
//   const Real v_int_l = min( max((Real)(gk) * intersection_dk + intersection_min, v_l), v_r);
//   const Real v_int_norm_l = (v_int_l - v_l)/dv;
//   const Real v_int_r = min((Real)(gk + 1) * intersection_dk + intersection_min, v_r);
//   const Real v_int_norm_r = (v_int_r - v_l)/dv;
        
//    /*compute left and right integrand*/
// #ifdef ACC_SEMILAG_PLM
//    Real target_density_l =
//      v_int_norm_l * a[k_block * WID + k_cell][0] +
//      v_int_norm_l * v_int_norm_l * a[k_block * WID + k_cell][1];
//    Real target_density_r =
//      v_int_norm_r * a[k_block * WID + k_cell][0] +
//      v_int_norm_r * v_int_norm_r * a[k_block * WID + k_cell][1];
// #endif
// #ifdef ACC_SEMILAG_PPM
//    Real target_density_l =
//      v_int_norm_l * a[k_block * WID + k_cell][0] +
//      v_int_norm_l * v_int_norm_l * a[k_block * WID + k_cell][1] +
//      v_int_norm_l * v_int_norm_l * v_int_norm_l * a[k_block * WID + k_cell][2];
//    Real target_density_r =
//      v_int_norm_r * a[k_block * WID + k_cell][0] +
//      v_int_norm_r * v_int_norm_r * a[k_block * WID + k_cell][1] +
//      v_int_norm_r * v_int_norm_r * v_int_norm_r * a[k_block * WID + k_cell][2];
// #endif
//    /*total value of integrand, if it is wihtin bounds*/
//          if ( gk >= 0 && gk <= blocks_per_dim * WID )
//      target[gk + WID] +=  target_density_r - target_density_l;
//       }
//     }
//   }
//   /*copy target to values*/
//   for (unsigned int k_block = 0; k_block<blocks_per_dim;k_block++){
//      for (uint k=0; k<WID; ++k){ 
//         values[k_block * WID + k + WID] = target[k_block * WID + k + WID];
//      }
//   }   
// }