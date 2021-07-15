#include <itensor/all.h>
#include <itensor/util/args.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <chrono>

#include "FindGS.h"

MPS loadMPS(std::string name){
	
	Hubbard sites;
	readFromFile("SITES_" + name, sites);
	MPS psi(sites);
	readFromFile("MPS_" + name, psi);

	return psi;
}

int main(int argc, char* argv[]) {
  
	if (argc != 3)
    throw std::runtime_error("Please provide the names of the two states. Usage: executable <MPS name 1> <MPS name 2>");

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

 
  auto name1 = argv[1];
  auto name2 = argv[2];

	//load the two MPSs and their sites
	MPS psi1 = loadMPS(name1);
	MPS psi2 = loadMPS(name2);

	// compute the overlap
	auto res = inner(psi1, psi2);

	std::cout << "overlap: " << res << std::endl;

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << std::endl << "Wall time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " s" << std::endl;
}
