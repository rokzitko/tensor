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
  InputGroup input = parse_cmd_line(argc, argv, p);

  //THESE ARE ALL TO BE PASSED TO calculateAndPrint()
  std::map<int, MPS> psiStore;      //stores ground states
  std::map<int, double> GSEstore;   //stores ground state energies
  std::map<int, MPS> ESpsiStore;    //stores excited states
  std::map<int, double> ESEstore;   //stores excited state energies

  //These quantities require the knowledge of H, so they are calculated in FindGS and saved here.
  std::map<int, double> GS0bisStore; //stores <GS|H|GS>
  std::map<int, double> deltaEStore; //stores sqrt(<GS|H^2|GS> - <GS|H|GS>^2)
  std::map<int, double> residuumStore; //stores <GS|H|GS> - GSE*<GS|GS>

  //calculates the ground state in different particle number sectors according to n0, nrange, refisn0 and stores ground states and energies
  FindGS(input, psiStore, GSEstore, ESpsiStore, ESEstore, GS0bisStore, deltaEStore, residuumStore, p);
  calculateAndPrint(input, psiStore, GSEstore, ESpsiStore, ESEstore, GS0bisStore, deltaEStore, residuumStore, p);
}
