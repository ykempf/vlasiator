#include <iostream>
#include <array>
#include <functional>
#include <cmath>

#define VELSPACE_SIZE 40
typedef double Real;


int main(int argc, char** argv) {

   if(argc <3) {
      std::cerr << "Syntax: coneCoverage <angle> <Bx> <By> <Bz>" << std::endl;
      return 1;
   }

   //std::array< std::array<Real, VELSPACE_SIZE>, VELSPACE_SIZE> velspace;
   Real cosAngle = atof(argv[1]) / 360. * 2. * M_PI;
   std::array<Real, 3> B({atof(argv[2]), atof(argv[3]), atof(argv[4])});
   Real normB = sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
   for (uint i=0; i<3; i++){
      B[i] /= normB;
   }

   // Signed distance function of the loss cone.
   // Adapted from https://mercury.sexy/hg_sdf/
   // Returns distance from the point p to the cone surface.
   // Positive values: outside of the cone
   // Negative values: inside of the cone
   std::function<Real(std::array<Real,3>)> coneSDF = [&B,cosAngle](std::array<Real,3> p) -> Real {

      Real pDotB = B[0]*p[0] + B[1]*p[1] + B[2]*p[2];
      Real pCrossB = sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2] - pDotB*pDotB);

      // Go to 2D coordinate system, where y is along cone direction and tip is at 0,0
      std::array<Real,2> q = {pCrossB, pDotB};
      std::array<Real,2> mantleDir = {sqrt(1 - cosAngle*cosAngle), cosAngle};

      // Are we in front of, or behind the tip?
      Real projected = q[0]*mantleDir[1] + q[1]*-mantleDir[0];

      // Distance to mantle
      Real distance = q[0]*mantleDir[0] + q[1] * mantleDir[1];
      // Distance to tip
      if(q[1] < 0 && projected < 0) {
         distance = std::max(distance, sqrt(q[0]*q[0]+q[1]*q[1]));
      }

      return distance;
   };

   // Determine how much the (sub) cell at centre point v with extents dv
   // is overlapped by the loss cone, by recursive (k-d tree) subdivision.
   std::function<Real(std::array<Real,3>, std::array<Real,3>, int, int)> coneCoverage = [&coneSDF,&coneCoverage](std::array<Real,3> v, std::array<Real,3> dv, int maxIteration, int dim) -> Real {
      Real lossconeDistance = coneSDF(v);
      Real diagonalSqr = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2];

      // If we are more than one cell diagonal distance away from the loss cone boundary,
      // we are either fully outside or fully inside
      if(lossconeDistance*lossconeDistance > diagonalSqr) {
         if(lossconeDistance > 0) {
            return 0.;
         } else {
            return 1.;
         }
      }

      // If our iterations are exhausted, return coverage approximation from the SDF distance.
      if(maxIteration==0) {
         Real diagonalLength = sqrt(diagonalSqr);
         // Percentage of the equivalent sphere that is covered by the cone
         // (if distance = 0, => 50% cover.
         //  if distance = diagonal, 0% cover.
         //  if distance = -diagonal, 100% cover)
         Real cover = 0.5 * (diagonalLength - lossconeDistance) / diagonalLength;
         cover = std::max(0.,cover);
         cover = std::min(1.,cover);
         return cover;
      }

      // Otherwise, split the cell and continue recursively.
      Real result = 0.;
      std::array<Real, 3> newV = v;
      std::array<Real, 3> newDv = dv;
      newDv[dim] *= 0.5;
      newV[dim]+= 0.5*dv[dim];
      result += 0.5*coneCoverage(newV, newDv, maxIteration-1, (dim+1)%3);
      newV[dim]-= dv[dim];
      result += 0.5*coneCoverage(newV, newDv, maxIteration-1, (dim+1)%3);

      return result;
   };

   for(int i=0; i<VELSPACE_SIZE; i++) {
      for(int j=0; j<VELSPACE_SIZE; j++) {
         Real x = (Real)i - VELSPACE_SIZE/2.;
         Real y = (Real)j - VELSPACE_SIZE/2.;

         std::cout << coneCoverage({x,y,0},{1,1,1},6,0) << " ";
      }
      std::cout << std::endl;
   }

   return 0;
}
