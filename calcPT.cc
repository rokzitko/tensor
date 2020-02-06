#include <itensor/all.h>
#include <itensor/util/args.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>

#include <omp.h>

#include "FindGS.h"

double FindPT(InputGroup &input, store &s, params &p);

int main(int argc, char* argv[]){
  params p;
  store s;
  InputGroup input = parse_cmd_line(argc, argv, p);

	double PTgamma = FindPT(input, s, p);

  std::cout << "Phase transition at Gamma = " << PTgamma << "\n";  

}


//Calculates Delta = E_odd - E_even = (-1)^(n_GS%2) * min( E_nGS+1 - E_nGS, E_nGS-1 - E_nGS)
double DeltaE(InputGroup &input, store &s, params &p) {

	FindGS(input, s, p);

	//Find the sector with the global GS:
 	int N_GS;
  double EGS = std::numeric_limits<double>::infinity();
  for(auto ntot: p.numPart){
    if (s.GSEstore[ntot] < EGS) {
      EGS = s.GSEstore[ntot];
      N_GS = ntot;
    }
  }
  
  auto prefactor = pow(-1, N_GS%2);

  double Delta;
	if (s.GSEstore.find(N_GS+1) == s.GSEstore.end()) {
	// N_GS+1 key not found
		Delta = s.GSEstore[N_GS-1] - EGS;
  }
  else if (s.GSEstore.find(N_GS-1) == s.GSEstore.end()) {
	// N_GS-1 key not found
		Delta = s.GSEstore[N_GS+1] - EGS;
  }
  else {
  	Delta = std::min(s.GSEstore[N_GS+1] - EGS, s.GSEstore[N_GS-1] - EGS); 
  }
  return prefactor * Delta;

}


// find the phase transition point Gamma for a given set of parameters, using the secant method; gamma0, gamma1 are initial points
double FindPT(InputGroup &input, store &s, params &p)  {
 	
 	p.gamma = p.gamma0;	//overwrite gamma and calculate DeltaE with it
 	double gamma0 = p.gamma0;	//save the first initial guess
 	double Delta0 = DeltaE(input, s, p);

 	p.gamma = p.gamma1;	//overwrite gamma and calculate DeltaE with it
 	double gamma1 = p.gamma1;	//save the second initial guess 
	double Delta1 = DeltaE(input, s, p);

	double gamma2 = gamma1 - Delta1 * ((gamma1 - gamma0)/(Delta1 - Delta0));
	p.gamma = gamma2;	//overwrite gamma and calculate DeltaE with it
	double Delta2 = DeltaE(input, s, p);

	int iteration=0;
	while ( abs(Delta2) > p.precision && iteration < p.maxIter ) {
		iteration++;

		gamma0 = gamma1;
		Delta0 = Delta1;

		gamma1 = gamma2;
		Delta1 = Delta2;

		gamma2 = gamma1 - Delta1 * ((gamma1 - gamma0)/(Delta1 - Delta0));
		p.gamma = gamma2;
		Delta2 = DeltaE(input, s, p);


		std::cout << "\nIteration " << iteration << ": gamma = " << gamma2 << ", Delta = " << Delta2 << "\n";  

	} // end while

	return gamma2;
}























