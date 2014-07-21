/*
This file is part of Vlasiator.
Copyright 2013, 2014 Finnish Meteorological Institute

*/

#ifndef CPU_1D_COLUMN_INTERPS_H
#define CPU_1D_COLUMN_INTERPS_H

#include "algorithm"
#include "math.h"
#include "cpu_slope_limiter.hpp"

using namespace std;


/*Compute face values of cell k. Based on explicit h6 estimate*/
inline void compute_h6_face_values(Real *values, Real &fv_l, Real &fv_r, int k) {

   /*compute left value*/
   fv_l = 1.0/60.0 * (values[k - 3 + WID]  - 8.0 * values[k - 2 + WID]  + 37.0 * values[k - 1 + WID] +
          37.0 * values[k  + WID] - 8.0 * values[k + 1 + WID] + values[k + 2 + WID]); // Same as right face of previous cell right face
   /*set right value*/
   ++k;
   fv_r = 1.0/60.0 * (values[k - 3 + WID]  - 8.0 * values[k - 2 + WID]  + 37.0 * values[k - 1 + WID] +
          37.0 * values[k  + WID] - 8.0 * values[k + 1 + WID] + values[k + 2 + WID]);
}


inline void filter_extrema(Real *values, Real &fv_l, Real &fv_r, int k) {
   //Coella1984 eq. 1.10, detect extrema and make algorithm constant if it is
   Real extrema_check = ((fv_r - values[k + WID]) * (values[k + WID] - fv_l));
   fv_l = extrema_check < 0 ? values[k + WID]: fv_l;
   fv_r = extrema_check < 0 ? values[k + WID]: fv_r;
}

/*Filter according to Eq. 19 in White et al.*/
inline void filter_boundedness(Real *values, Real &fv_l, Real &fv_r, int k) {
   /*First Eq. 19 & 20*/
   bool do_fix_bounds =
      (values[k - 1 + WID] - fv_l) * (fv_l - values[k + WID]) < 0 ||
      (values[k + 1 + WID] - fv_r) * (fv_r - values[k + WID]) < 0;
   if(do_fix_bounds) {
      Real slope_abs,slope_sign;
      slope_limiter_(values[k -1 + WID], values[k + WID], values[k + 1 + WID], slope_abs, slope_sign);
      //detect and  fix boundedness, as in WHITE 2008
      fv_l = (values[k -1 + WID] - fv_l) * (fv_l - values[k + WID]) < 0 ?
         values[k + WID] - slope_sign * std::min((Real)0.5 * slope_abs, (Real)abs(fv_l - values[k + WID])) :
         fv_l;
      fv_r = (values[k + 1 + WID] - fv_r) * (fv_r - values[k + WID]) < 0 ?
         values[k + WID] + slope_sign * std::min( (Real)0.5 * slope_abs, (Real)abs(fv_r- values[k + WID])) :
         fv_r;
   }
}


/*!
 Compute PLM coefficients
 f(v) = a[0] + a[1]/2.0*t 
 t=(v-v_{i-0.5})/dv where v_{i-0.5} is the left face of a cell
 The factor 2.0 is in the polynom to ease integration, then integral is a[0]*t + a[1]*t**2
*/
inline void compute_plm_coeff_explicit_columns(Real *values, Real a[RECONSTRUCTION_ORDER + 1], uint k){ 
   const Real d_cv=slope_limiter_(values[k - 1 + WID], values[k + WID], values[k + 1 + WID]);
   a[0] = values[k + WID] - d_cv * 0.5;
   a[1] = d_cv * 0.5;
}

/*
  Compute parabolic reconstruction with an explicit scheme
  
  Note that value array starts with an empty block, thus values[k + WID]
  corresponds to the current (centered) cell.
*/

inline void compute_ppm_coeff_explicit_columns(Real *values, Real a[RECONSTRUCTION_ORDER + 1], uint k){
   Real p_face;
   Real m_face;
   Real fv_l; /*left face value, extra space for ease of implementation*/
   Real fv_r; /*right face value*/

   // compute_h6_face_values(values,n_cblocks,fv_l, fv_r); 
   // filter_boundedness(values,n_cblocks,fv_l, fv_r); 
   // filter_extrema(values,n_cblocks,fv_l, fv_r);

   compute_h6_face_values(values,fv_l, fv_r, k);
   filter_boundedness(values,fv_l, fv_r, k);
   filter_extrema(values,fv_l, fv_r, k);
   m_face = fv_l;
   p_face = fv_r;
   
   //Coella et al, check for monotonicity   
   m_face = (p_face - m_face) * (values[k + WID] - 0.5 * (m_face + p_face)) > (p_face - m_face)*(p_face - m_face) / 6.0 ?
      3 * values[k + WID] - 2 * p_face : m_face;
   p_face = -(p_face - m_face) * (p_face - m_face) / 6.0 > (p_face - m_face) * (values[k + WID] - 0.5 * (m_face + p_face)) ?
      3 * values[k + WID] - 2 * m_face : p_face;

   //Fit a second order polynomial for reconstruction see, e.g., White
   //2008 (PQM article) (note additional integration factors built in,
   //contrary to White (2008) eq. 4
   a[0] = m_face;
   a[1] = 3.0 * values[k + WID] - 2.0 * m_face - p_face;
   a[2] = (m_face + p_face - 2.0 * values[k + WID]);
}



#endif
