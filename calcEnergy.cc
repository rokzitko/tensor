#include <itensor/all.h>
#include <itensor/util/args.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <chrono>

#include "FindGS.h"

// Given a saved MPS (MPS and sites), its quantum numbers n, Sz, i and an input file, calculates <psi|H|psi> for the hamiltonian defined in the params from the inputFile.

int main(int argc, char* argv[]) {
  
	if (argc != 7) throw std::runtime_error("Please provide the names of the state and the inputFile. Usage: executable inputFile SITES MPS n Sz i");

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	
 	auto inputFile = argv[1];
 	auto SITESname = argv[2];
 	auto MPSname = argv[3];
 	
 	charge n = atoi(argv[4]);
 	spin Sz = atof(argv[5]);
 	int i = atoi(argv[6]);
 	
 	subspace_t sub = std::make_pair(n, Sz);
 	state_t st = es(sub, i);

	//load the sites and the MPS objects
	Hubbard sites;
	readFromFile(SITESname, sites);
	MPS psi(sites);
	readFromFile(MPSname, psi);

	
	// create the fake argv to pass in to the parse_cmd_line function 	
	char *fakeargv[] = {
	    (char*)argv[0],
	    (char*)inputFile,
	    NULL
	};

	params p;
	parse_cmd_line(2, fakeargv, p);
	
	p.sites = sites; // it is important to overwrite the sites object, so that the indeces in psi and H match!

	std::cout << "initializing H \n";
	// initialize H
	MPO H = p.problem->initH(st, p);

	std::cout << "calculating E \n";
	// get energy
 	auto res = inner(psi, H, psi);

	std::cout << "energy: " << res << std::endl;

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  	std::cout << std::endl << "Wall time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " s" << std::endl;
}
