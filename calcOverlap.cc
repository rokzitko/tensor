#include <itensor/all.h>
#include <itensor/util/args.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <chrono>

#include "FindGS.h"

MPS loadMPS(std::string sitesName, std::string mpsName){
	
	Hubbard sites;
	readFromFile(sitesName, sites);
	MPS psi(sites);
	readFromFile(mpsName, psi);

	return psi;
}

int main(int argc, char* argv[]) {
  
	if (argc != 5)
    throw std::runtime_error("Please provide the names of the two states. Usage: executable <SITES name1> <MPS name 1> <SITES name 2> <MPS name 2>");

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	
 	auto SITESname1 = argv[1];
 	auto MPSname1 = argv[2];
 	auto SITESname2 = argv[3];
  	auto MPSname2 = argv[4];

	//load the two MPSs and their sites
	MPS psi1 = loadMPS(SITESname1, MPSname1);
	MPS psi2 = loadMPS(SITESname2, MPSname2);

	// compute the overlap
	auto res = inner(psi1, psi2);

	std::cout << "overlap: " << res << std::endl;

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  	std::cout << std::endl << "Wall time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " s" << std::endl;
}
