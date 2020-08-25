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
  double Sz_GS;
  double EGS = std::numeric_limits<double>::infinity();
  for(auto ntot: p.numPart){
    for(auto Sz: p.Szs[ntot]){
      if (s.GSEstore[std::make_pair(ntot, Sz)] < EGS) {
        EGS = s.GSEstore[std::make_pair(ntot, Sz)];
        N_GS = ntot;
        Sz_GS = Sz;
      }
    }
  }
  
  auto prefactor = pow(-1, N_GS%2);

  //set Ep at infinity. If state Ngs+1, SzGS+0.5 is computed, take that as Ep. If the state with SzGS-0.5 is also computed, take the minimum of these two values as Ep.
  //In the case where the nGS+1 (or -1) state is not computed, Ep (Em) will remain at infinity. The last line takes Delta as the smallest between Ep and Em.  
  double Ep = std::numeric_limits<double>::infinity();	
  double Em = std::numeric_limits<double>::infinity();	

  if (s.GSEstore.find(std::make_pair(N_GS+1, Sz_GS+0.5)) != s.GSEstore.end()) {
  	Ep = s.GSEstore[std::make_pair(N_GS+1, Sz_GS+0.5)];
  }
  else if (s.GSEstore.find(std::make_pair(N_GS+1, Sz_GS-0.5)) != s.GSEstore.end()) {
	Ep = std::min( s.GSEstore[std::make_pair(N_GS+1, Sz_GS-0.5)], Ep);
  }

  //do the same for the nGS-1 case, save to Em.
  //THIS REPEATS BASICALLY THE SAME CODE, EASY WAY TO AVOID THIS? (IT IS MORE READABLE LIKE THIS TOUGH)
  if (s.GSEstore.find(std::make_pair(N_GS-1, Sz_GS+0.5)) != s.GSEstore.end()) {
  	Em = s.GSEstore[std::make_pair(N_GS-1, Sz_GS+0.5)];
  }
  else if (s.GSEstore.find(std::make_pair(N_GS-1, Sz_GS-0.5)) != s.GSEstore.end()) {
	Em = std::min( s.GSEstore[std::make_pair(N_GS-1, Sz_GS-0.5)], Em);
  }

  double Delta = std::min(Ep, Em);

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























