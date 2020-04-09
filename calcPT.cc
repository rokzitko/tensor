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

	return 0;
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
// https://en.wikipedia.org/wiki/Secant_method
double FindPT(InputGroup &input, store &s, params &p)  {
 	
	std::cout << "\nStarting iteration\n";  


 	p.gamma = p.gamma0;	//overwrite gamma and calculate DeltaE with it
 	double gamma0 = p.gamma0;	//save the first initial guess
 	double Delta0 = DeltaE(input, s, p);

 	std::cout << "Delta0 = " << Delta0 << ", gamma0 = " << gamma0 << "\n";  

 	p.gamma = p.gamma1;	//overwrite gamma and calculate DeltaE with it
 	double gamma1 = p.gamma1;	//save the second initial guess 
	double Delta1 = DeltaE(input, s, p);

 	std::cout << "Delta1 = " << Delta1 << ", gamma1 = " << gamma1 << "\n";  

	double gamma2 = std::max(0.0, gamma1 - Delta1 * ((gamma1 - gamma0)/(Delta1 - Delta0)) ); //if gamma is negative, set it to zero. gamma < 0 results in a zero norm error.

	std::cout << "Next gamma is:" << gamma2 << "\n"; //debugging purposes 

	p.gamma = gamma2;	//overwrite gamma and calculate DeltaE with it
	double Delta2 = DeltaE(input, s, p);

	int iteration=0;

	std::cout << "\nIteration " << iteration << ": gamma = " << gamma2 << ", Delta = " << Delta2 << "\n";  

	while ( abs(Delta2) > p.precision && iteration < p.maxIter ) { //iteration ends if Delta is smalled than precision, if maxIter is hit or if the previous gamma is the same as the new one.
		iteration++;

		gamma0 = gamma1;
		Delta0 = Delta1;

		gamma1 = gamma2;
		Delta1 = Delta2;

		gamma2 = std::max(0.0, gamma1 - Delta1 * ((gamma1 - gamma0)/(Delta1 - Delta0)) );
		p.gamma = gamma2;
		Delta2 = DeltaE(input, s, p);


		std::cout << "\nIteration " << iteration << ": gamma = " << gamma2 << ", Delta = " << Delta2 << "\n";  

		if (abs(gamma2 - gamma1) < p.precision) break;

	} // end while

	return gamma2;
}























