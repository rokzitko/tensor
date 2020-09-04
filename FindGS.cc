#include <itensor/all.h>
#include <itensor/util/args.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <stdexcept>
#include <limits> // quiet_NaN

#include <omp.h>

#include "FindGS.h"

#include "SC_BathMPO.h"
#include "SC_BathMPO_MiddleImp.h"
#include "SC_BathMPO_Ec.h"
#include "SC_BathMPO_Ec_V.h"
#include "SC_BathMPO_MiddleImp_TwoChannel.h"

// Make an array centered at nref
auto n_list(int nref, int nrange) {
  std::vector<int> n;
    n.push_back(nref);
  for (auto i : range1(nrange)) {
        n.push_back(nref+i);
        n.push_back(nref-i);
  }
    return n;
}

InputGroup parse_cmd_line(int argc, char *argv[], params &p) {
  if(argc!=2){
    std::cout<<"Please provide input file. Usage ./executable inputfile.txt" << std::endl;
    exit(1);
  }

  //read parameters from the input file
  p.inputfn = {argv[1]};
  auto input = InputGroup{p.inputfn, "params"}; //get input parameters using InputGroup from itensor

  p.MPO = input.getString("MPO", "std");

  p.NImp = 1; // TODO: class problem, which contains all relevant objects (imp,bath,hyb) ?
  {
    double U = input.getReal("U", 0);
    p.qd = std::make_unique<imp>(U, input.getReal("epsimp", -U/2.), input.getReal("EZ_imp", 0.));
  }
  
  p.N = input.getInt("N", 0);
  if (p.N != 0) {
    p.NBath = p.N-p.NImp;
  }
  else { // N not specified, try NBath
    p.NBath = input.getInt("NBath", 0);
    if (p.NBath == 0) {
      std::cout << "specify either N or NBath!" << std::endl;
      exit(1);
    }
    p.N = p.NBath+p.NImp;
  } 
  if (p.MPO == "middle") {
    assert(p.NBath%2 == 0);   // NBath must be even
    p.impindex = 1+p.NBath/2;
  } else if (p.MPO == "std" || p.MPO == "Ec" || p.MPO == "Ec_V") {
    p.impindex = 1;
  } else
    throw std::runtime_error("Unknown MPO type");
  std::cout << "N=" << p.N << " NBath=" << p.NBath << " impindex=" << p.impindex << std::endl;

  p.sc = std::make_unique<SCbath>(p.NBath, input.getReal("alpha", 0), input.getReal("Ec", 0), input.getReal("n0", p.N-1));
  
  // sites is an ITensor thing. it defines the local hilbert space and
  // operators living on each site of the lattice
  // for example sites.op("N",1) gives the pariticle number operator
  // on the first site
  p.sites = Hubbard(p.N);

  p.excited_state = input.getYesNo("excited_state", false);
  p.randomMPSb = input.getYesNo("randomMPS", false);
  p.printDimensions = input.getYesNo("printDimensions", false);
  p.calcweights = input.getYesNo("calcweights", false);
  p.refisn0 = input.getYesNo("refisn0", false);
  p.parallel = input.getYesNo("parallel", false);
  p.verbose = input.getYesNo("verbose", false);
  p.band_level_shift = input.getYesNo("band_level_shift", false);
  p.computeEntropy = input.getYesNo("computeEntropy", false);
  p.printTotSpinZ = input.getYesNo("printTotSpinZ", false);

  p.impNupNdn = input.getYesNo("impNupNdn", false);
  p.chargeCorrelation = input.getYesNo("chargeCorrelation", false);
  p.pairCorrelation = input.getYesNo("pairCorrelation", false);
  p.spinCorrelation = input.getYesNo("spinCorrelation", false);
  p.hoppingExpectation = input.getYesNo("hoppingExpectation", false);

  p.EnergyErrgoal = input.getReal("EnergyErrgoal", 1e-16);
  p.nrH = input.getInt("nrH", 5);
  p.nrange = input.getInt("nrange", 1);

  p.Gamma = std::make_unique<hyb>(input.getReal("gamma", 0));
  p.V12 = input.getReal("V", 0);

  p.EZ_bulk = input.getReal("EZ_bulk", 0.);

  //TWO CHANNEL PARAMETERS
  p.alpha1 = input.getReal("alpha1", 0);
  p.alpha2 = input.getReal("alpha2", 0);
  p.n01 = input.getReal("n01", (p.N-1)/2);
  p.n02 = input.getReal("n02", (p.N-1)/2);
  p.gamma1 = input.getReal("gamma1", 0);
  p.gamma2 = input.getReal("gamma2", 0);
  p.Ec1 = input.getReal("Ec1", 0);
  p.Ec2 = input.getReal("Ec2", 0);
  p.EZ_bulk1 = input.getReal("EZ_bulk1", 0);
  p.EZ_bulk2 = input.getReal("EZ_bulk2", 0);

  p.SCSCinteraction = input.getReal("SCSCinteraction", 0);

  // for Ec_trick mapping
  p.Ueff = p.qd->U() + 2.*p.sc->Ec();                            // effective impurity e-e repulsion
  // p.epseff cannot be set here, because it depends on ntot (number of electrons in a given sector)

  const int nhalf = p.N; // total nr of electrons at half-filling
  const int nref = (p.refisn0 ? ( round(p.sc->n0() + 0.5 - (p.qd->eps()/p.qd->U())) ) : nhalf); //calculation of the energies is centered around this n
  p.numPart = n_list(nref, p.nrange);
  
  bool magnetic_field = ((p.qd->EZ()!=0 || p.EZ_bulk!=0) ? true : false); // true if there is magnetic field, implying that Sz=0.5 states are NOT degenerate
  // Sz values for n are in Szs[n]
  for (size_t i=0; i<p.numPart.size(); i++){
    int ntot = p.numPart[i];
    if (ntot%2==0) {
      p.Szs[ntot].push_back(0);
    }
    else {
      p.Szs[ntot].push_back(0.5);
      if (magnetic_field) p.Szs[ntot].push_back(-0.5);
    }
  }

  p.iterateOver={};
  for (size_t i=0; i<p.numPart.size(); i++){
    for (size_t j=0; j<p.Szs[p.numPart[i]].size(); j++){
      p.iterateOver.push_back(std::make_pair(p.numPart[i], p.Szs[p.numPart[i]][j]));
    }
  }


  // parameters used in the phase transition point iteration
  p.PTgamma0 = input.getReal("PTgamma0", 0.5);
  p.PTgamma1 = input.getReal("PTgamma1", 1.5);
  p.PTprecision = input.getReal("PTprecision", 1e-5);
  p.PTmaxIter = input.getInt("PTmaxIter", 30);

  return input;
}

//calculates the groundstates and the energies of the relevant particle number sectors
void FindGS(InputGroup &input, store &s, params &p){

  //The sweeps object defines the accuracy used for each update cycle in DMRG.
  //The used parameters are read from the input file with the following meaning:
  // maxdim:  maximal bond dimension -> start low approach ground state roughly, then increase
  // mindim:  minimal bond dimension used, can be helpfull to improve DMRG convergence
  // cutoff:  truncated weight i.e.: sum of all discared squared Schmidt values 
  // niter:   uring DMRG, one has to solve an eigenvalue problem usually using a krylov 
  //          method. niter is the number of krylov vectors used. Since DMRG is variational
  //          it does not matter if we use only very few krylov vectors. We only want to 
  //          move in the direction of the ground state - it seems that in late sweeps it is 
  //          a good idea to have niter at least 5 or 7.
  // noise:   if non-zero, a so-called noise term is added to DMRG which improves the convergence.
  //          This term must be zero towards the end, as the ground state is otherwise incorrect.
  auto inputsw = InputGroup(p.inputfn,"sweeps");
  auto sw_table = InputGroup(inputsw,"sweeps");
  int nrsweeps = input.getInt("nrsweeps", 15);
  auto sweeps = Sweeps(nrsweeps,sw_table);

  #pragma omp parallel for if(p.parallel) 
  for (size_t i=0; i<p.iterateOver.size(); i++){

      int ntot = std::get<0>(p.iterateOver[i]);
      double Sz = std::get<1>(p.iterateOver[i]);

      std::cout << "\nSweeping in the sector with " << ntot << " particles, Sz = " << Sz << ".\n";

      //initialize H and psi
      auto [H, Eshift] = initH(ntot, p);
      auto psi = initPsi(ntot, Sz, p);

      Args args; //args is used to store and transport parameters between various functions

      //Apply the MPO a couple of times to get DMRG started, otherwise it might not converge.
      for(auto i : range1(p.nrH)){
        psi = applyMPO(H,psi,args);
        psi.noPrime().normalize();
        }

      auto [GS0, GS] = dmrg(H,psi,sweeps,{"Silent",p.parallel, "Quiet",!p.printDimensions, "EnergyErrgoal",p.EnergyErrgoal}); // call itensor dmrg 

      double GSenergy = GS0+Eshift;

      s.psiStore[std::make_pair(ntot, Sz)] = GS;
      s.GSEstore[std::make_pair(ntot, Sz)] = GSenergy; 

      //values that need H are calculated here and stored, in order to avoid storing the entire hamiltonian
      double GS0bis = inner(GS, H, GS);
      double deltaE = sqrt(inner(H, GS, H, GS) - pow(inner(GS, H, GS),2));
      double residuum = inner(GS,H,GS) - GS0*inner(GS,GS);

      s.GS0bisStore[std::make_pair(ntot, Sz)] = GS0bis; 
      s.deltaEStore[std::make_pair(ntot, Sz)] = deltaE;
      s.residuumStore[std::make_pair(ntot, Sz)] = residuum;

      if (p.excited_state) {
        auto wfs = std::vector<MPS>(1);
        wfs.at(0) = GS;
        auto [ESenergy, ES] = dmrg(H,wfs,psi,sweeps,{"Silent",p.parallel,"Quiet",true,"Weight",11.0});
        ESenergy += Eshift;
        s.ESEstore[std::make_pair(ntot, Sz)]=ESenergy;
        s.ESpsiStore[std::make_pair(ntot, Sz)]=ES;
      }

  }
}//end FindGS


//initialize the Hamiltonian
std::tuple<MPO, double> initH(int ntot, params &p){
  auto eps = p.sc->eps(p.band_level_shift);
  auto V = p.Gamma->V(p.sc->Nbath());
  if (p.verbose) {
    std::cout << "eps=" << eps << std::endl;
    std::cout << "V=" << V << std::endl;
  }

  double Eshift;  // constant term in the Hamiltonian
  MPO H(p.sites); // MPO is the hamiltonian in "MPS-form" after this line it is still a trivial operator
  if (p.MPO == "std") {
    assert(p.V12 != 0.0);
    Eshift = p.sc->Ec()*pow(ntot-p.sc->n0(),2); // occupancy dependent effective energy shift
    double epseff = p.qd->eps() - 2.*p.sc->Ec()*(ntot-p.sc->n0()) + p.sc->Ec();
    Fill_SCBath_MPO(H, eps, V, epseff, p); // defined in SC_BathMPO.h, fills the MPO with the necessary entries
  } else if (p.MPO == "middle") {
    assert(p.V12 != 0.0);
    Eshift = p.sc->Ec()*pow(ntot-p.sc->n0(),2); // occupancy dependent effective energy shift
    double epseff = p.qd->eps() - 2.*p.sc->Ec()*(ntot-p.sc->n0()) + p.sc->Ec();
    Fill_SCBath_MPO_MiddleImp(H, eps, V, epseff, p);
  } else if (p.MPO == "Ec") {
    assert(p.V12 != 0.0);
    Eshift = p.sc->Ec()*pow(p.sc->n0(), 2);
    Fill_SCBath_MPO_Ec(H, eps, V, p);
  } else if (p.MPO == "Ec_V") {
    Eshift = p.sc->Ec()*pow(p.sc->n0(), 2) + p.V12 * p.sc->n0() * p.qd->nu();
    double epseff = p.qd->eps() - p.V12 * p.sc->n0();
    double epsishift = -p.V12 * p.qd->nu();
    Fill_SCBath_MPO_Ec_V(H, eps, V, epseff, epsishift, p);
  } else if (p.MPO == "middle_2C") {
    Eshift = p.Ec1*pow(p.n01, 2) + p.Ec2*pow(p.n02, 2);
    Fill_SCBath_MPO_MiddleImp_TwoChannel(H, eps, V, p);
  } else
    throw std::runtime_error("Unknown MPO type");
  
  Eshift += p.qd->U()/2.; // RZ, for convenience
  return std::make_tuple(H, Eshift);
}

//initialize the MPS in a product state with the correct particle number
MPS initPsi(int ntot, float Sz, params &p){
  auto state = InitState(p.sites);

  const int nsc = ntot-1;  // number of electrons in the SC in Gamma->0 limit
  const int npair = nsc/2; // number of pairs in the SC
  int tot = 0; // for assertion test
  int Sztot = 0.;

  //Up electron at the impurity site and npair UpDn pairs. 
  if (Sz < 0.) {
    state.set(p.impindex, "Dn");
    Sztot += -0.5; 
  }
  else {
    state.set(p.impindex, "Up"); 
    Sztot += 0.5;
  }
  tot++;

  int j=0;
  int i=1;
  for(; j < npair; i++){   
    if (i!=p.impindex){             //In order to avoid adding a pair to the impurity site 
      j++;                          //i counts sites, j counts added pairs.
      state.set(i, "UpDn");
      tot += 2;
    }
  }

  if (nsc%2 == 1) { //If ncs is odd, add another electron according to EZ_bulk preference, but not to the impurity site.
    
    if (i!=p.impindex){
      state.set(i,"Dn"); 
      tot++;
    }
    else{
      state.set(i+1,"Dn"); 
      tot++;
    }
  }

  assert(tot == n);
  MPS psi(state);
  if (p.randomMPSb) psi = randomMPS(state);
  return psi;
}

//Loops over all particle sectors and prints relevant quantities.
void calculateAndPrint(InputGroup &input, store &s, params &p){
  //print data for each sector
  for(auto ntot: p.numPart) {
    for (double Sz:p.Szs[ntot]){
      auto nSz = std::make_pair(ntot, Sz);
      
      printfln("\n");
      printfln("RESULTS FOR THE SECTOR WITH %i PARTICLES, Sz %i:", ntot, Sz);

      printfln("Ground state energy = %.17g",s.GSEstore[nSz]);

      MPS & GS = s.psiStore[nSz];
      
      //norm
      double normGS = inner(GS, GS);
      printfln("norm = %.17g", normGS);
      //occupancy; site pairing; v and u amplitudes 
      MeasureOcc(GS, p);
      MeasurePairing(GS, p);
      MeasureAmplitudes(GS, p);

      if (p.computeEntropy) PrintEntropy(GS, p);

      if (p.impNupNdn) ImpurityUpDn(GS, p);

      if (p.chargeCorrelation) ChargeCorrelation(GS, p);
      if (p.spinCorrelation) SpinCorrelation(GS, p);
      if (p.pairCorrelation) PairCorrelation(GS, p);
      if (p.hoppingExpectation) expectedHopping(GS, p);
      if (p.printTotSpinZ) TotalSpinz(GS, p);

      double & GS0 = s.GSEstore[nSz];
      double & GS0bis = s.GS0bisStore[nSz];
      double & deltaE = s.deltaEStore[nSz];
      double & residuum = s.residuumStore[nSz];
      //various measures of convergence (energy deviation, residual value)
      printfln("Eigenvalue(bis): <GS|H|GS> = %.17g",GS0bis);
      printfln("diff: E_GS - <GS|H|GS> = %.17g", GS0-GS0bis);
      printfln("deltaE: sqrt(<GS|H^2|GS> - <GS|H|GS>^2) = %.17g", deltaE);
      printfln("residuum: <GS|H|GS> - E_GS*<GS|GS> = %.17g", residuum);

      if (p.excited_state){
        MPS & ES = s.ESpsiStore[nSz];
        double & ESenergy = s.ESEstore[nSz];
        MeasureOcc(ES, p);
        printfln("Excited state energy = %.17g",ESenergy);
       }
    } //end of Sz for loop 
  } //end of ntot for loop

  printfln("");
  //Print out energies again:
  for(auto ntot: p.numPart){
    for(auto Sz: p.Szs[ntot]){
      printfln("n = %.17g  Sz = %.17g  E = %.17g", ntot, Sz, s.GSEstore[std::make_pair(ntot, Sz)]);
  
    }
  }

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
  printfln("N_GS = %i",N_GS);
  printfln("Sz_GS = %i",Sz_GS);

  //Calculate the spectral weights:
  if (p.calcweights) {
    MPS & psiGS = s.psiStore[std::make_pair(N_GS, Sz_GS)];

    printfln(""); 
    printfln("Spectral weights:");
    printfln("(Spectral weight is the square of the absolute value of the number.)");


    if ( s.GSEstore.find(std::make_pair(N_GS+1, Sz_GS+0.5)) != s.GSEstore.end() ){ //if the N_GS+1 state was computed, print the <N+1|c^dag|N> terms
      MPS & psiNp = s.psiStore[std::make_pair(N_GS+1, Sz_GS+0.5)];
      ExpectationValueAddEl(psiNp, psiGS, "up", p);
    }  
    else printfln("ERROR: we don't have info about the N_GS+1, Sz_GS+0.5 occupancy sector.");

    if ( s.GSEstore.find(std::make_pair(N_GS+1, Sz_GS-0.5)) != s.GSEstore.end() ){ //if the N_GS+1 state was computed, print the <N+1|c^dag|N> terms
      MPS & psiNp = s.psiStore[std::make_pair(N_GS+1, Sz_GS-0.5)];
      ExpectationValueAddEl(psiNp, psiGS, "dn", p);
    }  
    else printfln("ERROR: we don't have info about the N_GS+1, Sz_GS-0.5 occupancy sector.");


    if ( s.GSEstore.find(std::make_pair(N_GS-1, Sz_GS-0.5)) != s.GSEstore.end() ){ //if the N_GS-1 state was computed, print the <N-1|c|N> terms
      MPS & psiNm = s.psiStore[std::make_pair(N_GS-1, Sz_GS-0.5)];   
      ExpectationValueTakeEl(psiNm, psiGS, "up", p);
    }
    else printfln("ERROR: we don't have info about the N_GS-1, Sz_GS+0.5 occupancy sector.");

    if ( s.GSEstore.find(std::make_pair(N_GS-1, Sz_GS+0.5)) != s.GSEstore.end() ){ //if the N_GS-1 state was computed, print the <N-1|c|N> terms
      MPS & psiNm = s.psiStore[std::make_pair(N_GS-1, Sz_GS+0.5)];   
      ExpectationValueTakeEl(psiNm, psiGS, "dn", p);
    }
    else printfln("ERROR: we don't have info about the N_GS-1, Sz_GS-0.5 occupancy sector.");

  } //end of if (calcweights)

} //end of calculateAndPrint()

//calculates <psi1|c_dag|psi2>, according to http://itensor.org/docs.cgi?vers=cppv3&page=formulas/mps_onesite_op
void ExpectationValueAddEl(MPS psi1, MPS psi2, std::string spin, const params &p){

  psi2.position(p.impindex); //set orthogonality center
  auto newTensor = noPrime(op(p.sites,"Cdag"+spin, p.impindex)*psi2(p.impindex)); //apply the local operator
  psi2.set(p.impindex,newTensor); //plug in the new tensor, with the operator applied

  auto res = inner(psi1, psi2);

  std::cout << "weight w+ " << spin << ": " << res << "\n";
}

//calculates <psi1|c|psi2>
void ExpectationValueTakeEl(MPS psi1, MPS psi2, std::string spin, const params &p){

  psi2.position(p.impindex);
  auto newTensor = noPrime(op(p.sites,"C"+spin, p.impindex)*psi2(p.impindex));
  psi2.set(p.impindex,newTensor);

  auto res = inner(psi1, psi2);

  std::cout << "weight w- " << spin << ": " << res << "\n";
}

//CORRELATION FUNCTIONS BETWEEN THE IMPURITY AND ALL SC LEVELS:
//according to: http://www.itensor.org/docs.cgi?vers=cppv3&page=formulas/correlator_mps

// <n_imp n_i>
void ChargeCorrelation(MPS& psi, const params &p){
  std::cout << "charge correlation = ";

  auto impOp = op(p.sites, "Ntot", p.impindex);

  double tot=0;
  for(auto j: range1(2, length(psi))) {

    auto scOp = op(p.sites, "Ntot", j);

    double result = ImpurityCorrelator(psi, impOp, j, scOp, p);

    std::cout << std::setprecision(17) << result << " ";
    tot+=result;
  }

  std::cout << std::endl;
  std::cout << "charge correlation tot = " << tot << "\n"; 
}

// <S_imp S_i> = <Sz_imp Sz_i> + 1/2 ( <S+_imp S-_i> + <S-_imp S+_i> )
void SpinCorrelation(MPS& psi, const params &p){
  std::cout << "spin correlations:\n";

  //impurity spin operators
  auto impSz = 0.5*( op(p.sites, "Nup", p.impindex) - op(p.sites, "Ndn", p.impindex) );
  auto impSp = op(p.sites, "Cdagup*Cdn", p.impindex);
  auto impSm = op(p.sites, "Cdagdn*Cup", p.impindex);

  auto SzSz = 0.25 * ( op(p.sites, "Nup*Nup", p.impindex) - op(p.sites, "Nup*Ndn", p.impindex) - op(p.sites, "Ndn*Nup", p.impindex) + op(p.sites, "Ndn*Ndn", p.impindex));

  //squares of the impurity operators; for on-site terms
  double tot = 0;


  //SzSz term
  std::cout << "SzSz correlations: ";
  
  psi.position(p.impindex);

  //on site term
  auto onSiteSzSz = elt(psi(p.impindex) * SzSz *  dag(prime(psi(p.impindex),"Site")));
  
  std::cout << std::setprecision(17) << onSiteSzSz << " ";
  tot+=onSiteSzSz;

  for(auto j: range1(2, length(psi))) {

    auto scSz = 0.5*( op(p.sites, "Nup", j) - op(p.sites, "Ndn", j) );
    
    double result = ImpurityCorrelator(psi, impSz, j, scSz, p);

    std::cout << std::setprecision(17) << result << " ";
    tot+=result;
  }
  std::cout << std::endl;

  //S+S- term
  std::cout << "S+S- correlations: ";

  psi.position(p.impindex);

  //on site term
  auto onSiteSpSm = elt(psi(p.impindex) * op(p.sites, "Cdagup*Cdn*Cdagdn*Cup", p.impindex) *  dag(prime(psi(p.impindex),"Site")));

  std::cout << std::setprecision(17) << onSiteSpSm << " ";
  tot+=0.5*onSiteSpSm; 

  for(auto j: range1(2, length(psi))) {

    auto scSm = op(p.sites, "Cdagdn*Cup", j);
    
    double result = ImpurityCorrelator(psi, impSp, j, scSm, p);

    std::cout << std::setprecision(17) << result << " ";
    tot+=0.5*result;
  }
  std::cout << std::endl;


  //S- S+ term
  std::cout << "S-S+ correlations: ";

  psi.position(p.impindex);

  //on site term
  auto onSiteSmSp = elt(psi(p.impindex) * op(p.sites, "Cdagdn*Cup*Cdagup*Cdn", p.impindex) *  dag(prime(psi(p.impindex),"Site")));

  std::cout << std::setprecision(17) << onSiteSmSp << " ";
  tot+=0.5*onSiteSmSp; 

  for(auto j: range1(2, length(psi))) {

    auto scSp = op(p.sites, "Cdagup*Cdn", j);
    
    double result = ImpurityCorrelator(psi, impSm, j, scSp, p);

    std::cout << std::setprecision(17) << result << " ";
    tot+=0.5*result;
  }
  std::cout << std::endl;

  std::cout << "spin correlation tot = " << tot << "\n";
}

void PairCorrelation(MPS& psi, const params &p){
  std::cout << "pair correlation = ";

  auto impOp = op(p.sites, "Cup*Cdn", p.impindex);

  double tot=0;
  for(auto j: range1(2, length(psi))) {

    auto scOp = op(p.sites, "Cdagdn*Cdagup", j);

    double result = ImpurityCorrelator(psi, impOp, j, scOp, p);

    std::cout << std::setprecision(17) << result << " ";
    tot+=result;
  }

  std::cout << std::endl;
  std::cout << "pair correlation tot = " << tot << "\n";
}

//Prints <d^dag c_i + c_i^dag d> for each i. the sum of this expected value, weighted by 1/sqrt(N)
//gives <d^dag f_0 + f_0^dag d>, where f_0 = 1/sqrt(N) sum_i c_i. This is the expected value of hopping.
void expectedHopping(MPS& psi, const params &p){

  auto impOpUp = op(p.sites, "Cup", p.impindex);
  auto impOpDagUp = op(p.sites, "Cdagup", p.impindex);

  auto impOpDn = op(p.sites, "Cdn", p.impindex);
  auto impOpDagDn = op(p.sites, "Cdagdn", p.impindex);

  double totup=0;
  double totdn=0;

  // hopping expectation values for spin up
  std::cout << "hopping spin up = ";
  for (auto j : range1(2, length(psi))){

    auto scDagOp = op(p.sites, "Cdagup", j);
    auto scOp = op(p.sites, "Cup", j);

    double result = ImpurityCorrelator(psi, impOpUp, j, scDagOp, p);    // <d c_i^dag>
    double resultdag = ImpurityCorrelator(psi, impOpDagUp, j, scOp, p); // <d^dag c_i>

    std::cout << std::setprecision(17) << result << " " << resultdag << " ";
    std::cout << std::setprecision(17) << result+resultdag << " ";
    totup+=result+resultdag;
  }

  std::cout << std::endl;
  std::cout << "hopping correlation up tot = " << totup << "\n";

  // hopping expectation values for spin up
  std::cout << "hopping spin down = ";
  for (auto j : range1(2, length(psi))){

    auto scDagOp = op(p.sites, "Cdagdn", j);
    auto scOp = op(p.sites, "Cdn", j);

    double result = ImpurityCorrelator(psi, impOpDn, j, scDagOp, p);    // <d c_i^dag>
    double resultdag =  ImpurityCorrelator(psi, impOpDagDn, j, scOp, p); // <d^dag c_i>

    std::cout << std::setprecision(17) << result << " " << resultdag << " ";
    //std::cout << std::setprecision(17) << result+resultdag << " ";
    totdn+=result+resultdag;
  }
  
  std::cout << std::endl;
  std::cout << "hopping correlation down tot = " << totdn << "\n";
  
  std::cout << "total hopping correlation = " << totup + totdn << "\n";
}

//computes the correlation between operator impOp on impurity and operator opj on j
double ImpurityCorrelator(MPS& psi, auto impOp, int j, auto opj, const params &p){

  psi.position(p.impindex);
  
  MPS psidag = dag(psi);
  psidag.prime("Link");

  auto li_1 = leftLinkIndex(psi,p.impindex);

  auto C = prime(psi(p.impindex),li_1)*impOp;
  C *= prime(psidag(p.impindex),"Site");
  for (int k=p.impindex+1; k<j; ++k){
    C*=psi(k);
    C*=psidag(k);
  }

  auto lj=rightLinkIndex(psi,j);
  
  C *= prime(psi(j),lj)*opj;
  C *= prime(psidag(j),"Site");
  
  return elt(C);
}

//prints the occupation number Nup and Ndn at the impurity
void ImpurityUpDn(MPS& psi, const params &p){
  
  std::cout << "impurity nup ndn = ";
  psi.position(p.impindex);
  auto valnup = psi.A(p.impindex) * p.sites.op("Nup",p.impindex)* dag(prime(psi.A(p.impindex),"Site"));
  auto valndn = psi.A(p.impindex) * p.sites.op("Ndn",p.impindex)* dag(prime(psi.A(p.impindex),"Site"));

  std::cout << std::setprecision(17) << std::real(valnup.cplx()) << " " << std::real(valndn.cplx()) << "\n";
}

//prints total Sz of the state
void TotalSpinz(MPS& psi, const params &p){

  double totNup=0.;
  double totNdn=0.;

  for(auto j: range1(length(psi))) {
    psi.position(j);

    auto Nupi = psi.A(j) * p.sites.op("Nup",j)* dag(prime(psi.A(j),"Site"));
    auto Ndni = psi.A(j) * p.sites.op("Ndn",j)* dag(prime(psi.A(j),"Site"));

    totNup += std::real(Nupi.cplx());
    totNdn += std::real(Ndni.cplx());

  }

  std::cout << std::setprecision(17) << "Total spin z: " << " Nup = " << totNup << " Ndn = " << totNdn << " Sztot = " << 0.5*(totNup-totNdn) <<  "\n";

}

//prints the occupation number of an MPS psi
//instructive to learn how to calculate local observables
void MeasureOcc(MPS& psi, const params &p){
  
  
  std::cout << "site occupancies = ";
  double tot = 0;
  for(auto i : range1(length(psi)) ){
    //position call very important! otherwise one would need to 
    //contract the whole tensor network of <psi|O|psi>
    //this way, only the local operator at site i is needed
    psi.position(i);
    auto val = psi.A(i) * p.sites.op("Ntot",i)* dag(prime(psi.A(i),"Site"));
    std::cout << std::setprecision(17) << std::real(val.cplx()) << " ";
    tot += std::real(val.cplx());
  }
  std::cout << std::endl;
  Print(tot);

}

// The sum (tot) corresponds to \bar{\Delta}', Eq. (4) in Braun, von Delft, PRB 50, 9527 (1999), first proposed by Dan Ralph.
// It reduces to Delta_BCS in the thermodynamic limit (if the impurity is decoupled, Gamma=0).
// For Gamma>0, there is no guarantee that all values 'sq' are real, because the expr <C+CC+C>-<C+C><C+C> may be negative.
void MeasurePairing(MPS& psi, const params &p){
  std::cout << "site pairing = ";
  std::complex<double> tot = 0;
  for(auto i : range1(length(psi))){
    psi.position(i);
    auto val2  = psi.A(i) * p.sites.op("Cdagup*Cup*Cdagdn*Cdn", i) * dag(prime(psi.A(i),"Site"));
    auto val1u = psi.A(i) * p.sites.op("Cdagup*Cup", i) * dag(prime(psi.A(i),"Site"));
    auto val1d = psi.A(i) * p.sites.op("Cdagdn*Cdn", i) * dag(prime(psi.A(i),"Site"));
    auto sq = p.sc->g() * sqrt( val2.cplx() - val1u.cplx() * val1d.cplx() );
    std::cout << std::setprecision(17) << sq << " ";
    if (i != p.impindex) tot += sq; // exclude the impurity site in the sum
  }
  std::cout << std::endl;
  Print(tot);
}

// See von Delft, Zaikin, Golubev, Tichy, PRL 77, 3189 (1996)
void MeasureAmplitudes(MPS& psi, const params &p){
  std::cout << "amplitudes vu = ";
  std::complex<double> tot = 0;
  for(auto i : range1(length(psi)) ){
    psi.position(i);
    auto valv = psi.A(i) * p.sites.op("Cdagup*Cdagdn*Cdn*Cup", i) * dag(prime(psi.A(i),"Site"));
    auto valu = psi.A(i) * p.sites.op("Cdn*Cup*Cdagup*Cdagdn", i) * dag(prime(psi.A(i),"Site"));
    auto v = sqrt( std::real(valv.cplx()) );
    auto u = sqrt( std::real(valu.cplx()) );
    auto pdt = v*u;
    auto element = p.sc->g() * pdt;
    std::cout << "[v=" << v << " u=" << u << " pdt=" << pdt << "] ";
    if (i != p.impindex) tot += element; // exclude the impurity site in the sum
  }
  std::cout << std::endl;
  Print(tot);
}


// Computed entanglement/von Neumann entropy between the impurity and the system.
// Copied from https://www.itensor.org/docs.cgi?vers=cppv3&page=formulas/entanglement_mps
void PrintEntropy(MPS& psi, const params &p){

  psi.position(p.impindex);

  //SVD this wavefunction to get the spectrum of density-matrix eigenvalues
  auto l = leftLinkIndex(psi, p.impindex);
  auto s = siteIndex(psi, p.impindex);
  auto [U,S,V] = svd(psi(p.impindex), {l,s});
  auto u = commonIndex(U,S);

  //Apply von Neumann formula to the squares of the singular values
  double SvN = 0.;
  for(auto n : range1(dim(u)))
      {
      auto Sn = elt(S,n,n);
      auto pp = sqr(Sn);
      if(pp > 1E-12) SvN += -pp*log(pp);
      }

  printfln("Entanglement entropy across impurity bond b=%d, SvN = %.10f",p.impindex,SvN);

}
