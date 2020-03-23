#include <itensor/all.h>
#include <itensor/util/args.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>

#include <omp.h>

#include "FindGS.h"

int main(int argc, char* argv[]){
  params p;
  store s;
  InputGroup input = parse_cmd_line(argc, argv, p);

  // calculates the ground state in different particle number sectors
  FindGS(input, s, p);
  // calculates observables and prints them
  calculateAndPrint(input, s, p);

  return 0;
}