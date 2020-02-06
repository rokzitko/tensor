#include <itensor/all.h>
#include <itensor/util/args.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>

#include <omp.h>

#include "FindGS.h"

#ifndef MIDDLE_IMP
 #include "SC_BathMPO.h"
#else
 #include "SC_BathMPO_MiddleImp.h"
#endif

InputGroup parse_cmd_line(int argc, char *argv[], params &p) {
  if(argc!=2){
    std::cout<<"Please provide input file. Usage ./executable inputfile.txt" << std::endl;
    exit(1);
  }

  //read parameters from the input file
  p.inputfn = {argv[1]};
  auto input = InputGroup{p.inputfn, "params"}; //get input parameters using InputGroup from itensor

  p.NImp = 1;
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
#ifdef MIDDLE_IMP
  assert(p.NBath%2 == 0);   // NBath must be even
  p.impindex = 1+p.NBath/2;
#else
  p.impindex = 1;
#endif
  std::cout << "N=" << p.N << " NBath=" << p.NBath << " impindex=" << p.impindex << std::endl;

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

  p.EnergyErrgoal = input.getReal("EnergyErrgoal", 1e-16);
  p.nrH = input.getInt("nrH", 5);
  p.nrange = input.getInt("nrange", 1);

  p.n0 = input.getReal("n0", p.N-1);
  p.alpha = input.getReal("alpha");
  p.d = 2./p.NBath;
  p.g = p.alpha * p.d;
  p.U = input.getReal("U");
  p.gamma = input.getReal("gamma");
  p.Ec = input.getReal("Ec", 0);
  p.epsimp = -p.U/2.;    
  p.Ueff = p.U + 2.*p.Ec;

  p.numPart={};
  const int nhalf = p.N; // total nr of electrons at half-filling
  const int nref = (p.refisn0 ? round(p.n0)+1 : nhalf); //calculation of the energies is centered around this n
  p.numPart.push_back(nref);
  for (int i = 1; i <= p.nrange; i++) {
    p.numPart.push_back(nref+i);
    p.numPart.push_back(nref-i);
  }
  
  return input;
}

//calculates the groundstates and the energies of the relevant particle number sectors
void FindGS(InputGroup &input,
  std::map<int, MPS>& psiStore, std::map<int, double>& GSEstore, std::map<int, MPS> ESpsiStore, std::map<int, double>& ESEstore, std::map<int, double>& GS0bisStore, std::map<int, double>& deltaEStore, std::map<int, double>& residuumStore,
           const params &p){
  
  std::vector<double> GSenergies(0); // result: lowest energy in each occupancy sector
  std::vector<double> ESenergies(0); // optionally: first excited state in each occupancy sector

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

  std::vector<double> eps;        // vector containing on-site energies of the bath !one-indexed!
  std::vector<double> V;          //vector containing hopping amplitudes to the bath !one-indexed!
  
  #pragma omp parallel for if(p.parallel) private(eps, V)
  for (int i=0; i<p.numPart.size(); i++){
    auto ntot = p.numPart[i];
    std::cout << "\nSweeping in the sector with " << ntot << " particles.\n";  

    //Read the bath parameters (fills in the vectors eps and V).
    const double epseff = p.epsimp - 2.*p.Ec*(ntot-p.n0) + p.Ec;  //effective impurity on-site potential
    GetBathParams(epseff, eps, V, p);

    //initialize H and psi
    MPO H = initH(eps, V, p);  //This does not necessarily compile with older(?) compilers, as
    MPS psi = initPsi(ntot, p);               //it dislikes 'auto sites' as a function parameter. Works on spinon.

    Args args; //args is used to store and transport parameters between various functions
    
    //Apply the MPO a couple of times to get DMRG started, otherwise it might not converge.
    for(auto i : range1(p.nrH)){
      psi = applyMPO(H,psi,args);
      psi.noPrime().normalize();
      }
    
    auto [GS0, GS] = dmrg(H,psi,sweeps,{"Silent",p.parallel, "Quiet",!p.printDimensions, "EnergyErrgoal",p.EnergyErrgoal}); // call itensor dmrg 
    
    double shift = p.Ec*pow(ntot-p.n0,2); // occupancy dependent effective energy shift
    shift += p.U/2.; // RZ, for convenience    
    
    double GSenergy = GS0+shift;

    psiStore[ntot] = GS;
    GSEstore[ntot] = GSenergy; 

    //values that need H are calculated here and stored, in order to avoid storing the entire hamiltonian
    double GS0bis = inner(GS, H, GS);
    double deltaE = sqrt(inner(H, GS, H, GS) - pow(inner(GS, H, GS),2));
    double residuum = inner(GS,H,GS) - GS0*inner(GS,GS);

    GS0bisStore[ntot] = GS0bis; 
    deltaEStore[ntot] = deltaE;
    residuumStore[ntot] = residuum;

    if (p.excited_state) {
      auto wfs = std::vector<MPS>(1);
      wfs.at(0) = GS;
      auto [ESenergy, ES] = dmrg(H,wfs,psi,sweeps,{"Quiet",true,"Weight",11.0});
      ESenergy += shift;
      ESEstore[ntot]=ESenergy;
      ESpsiStore[ntot]=ES;
    }
  }//end for loop
}//end FindGS

//initialize the Hamiltonian
MPO initH(std::vector<double> eps, std::vector<double> V, const params &p){
  MPO H(p.sites); // MPO is the hamiltonian in "MPS-form" after this line it is still a trivial operator
  Fill_SCBath_MPO(H, eps, V, p); // defined in SC_BathMPO.h, fills the MPO with the necessary entries
  return H;
}

//initialize the MPS in a product state with the correct particle number
MPS initPsi(int ntot, const params &p){
  auto state = InitState(p.sites);

  const int nsc = ntot-1;  // number of electrons in the SC in Gamma->0 limit
  const int npair = nsc/2; // number of pairs in the SC
  int tot = 0; // for assertion test
  
  //Up electron at the impurity site and npair UpDn pairs. 
  state.set(p.impindex, "Up"); 
  tot++;

  int j=0;
  int i=1;
  for(i; j < npair; i++){ //In order to avoid adding a pair to the impurity site   
    if (i!=p.impindex){             //i counts sites, j counts added pairs.
      j++;
      state.set(i, "UpDn");
      tot += 2;
    }
  }     

  //If ncs is odd, add another Dn electron, but not to the impurity site.
  if (nsc%2 == 1) {
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
void calculateAndPrint(InputGroup &input, std::map<int, MPS> psiStore, std::map<int, double> GSEstore,std::map<int, MPS> ESpsiStore, std::map<int, double> ESEstore, std::map<int, double> GS0bisStore, std::map<int, double> deltaEStore, std::map<int, double> residuumStore,
                       const params &p){
  //print data for each sector
  for(auto ntot: p.numPart) {
    printfln("\n");
    printfln("RESULTS FOR THE SECTOR WITH %i PARTICLES:", ntot);

    printfln("Ground state energy = %.20f",GSEstore[ntot]);
    
    MPS & GS = psiStore[ntot];
    //norm
    double normGS = inner(GS, GS);
    printfln("norm = %.20f", normGS);
    //occupancy; site pairing; v and u amplitudes 
    MeasureOcc(GS, p);
    MeasurePairing(GS, p);
    MeasureAmplitudes(GS, p);
    
    double & GS0 = GSEstore[ntot];
    double & GS0bis = GS0bisStore[ntot];
    double & deltaE = deltaEStore[ntot];
    double & residuum = residuumStore[ntot];
    //various measures of convergence (energy deviation, residual value)
    printfln("Eigenvalue(bis): <GS|H|GS> = %.20f",GS0bis);
    printfln("diff: E_GS - <GS|H|GS> = %.20f", GS0-GS0bis);
    printfln("deltaE: sqrt(<GS|H^2|GS> - <GS|H|GS>^2) = %.20f", deltaE);
    printfln("residuum: <GS|H|GS> - E_GS*<GS|GS> = %.20f", residuum);

    if (p.excited_state){
      MPS & ES = ESpsiStore[ntot];
      double & ESenergy = ESEstore[ntot]; 
      MeasureOcc(ES, p);
      printfln("Excited state energy = %.20f",ESenergy);
     } 
  } //end of n-for loop

  printfln("");
  //Print out energies again:
  for(auto ntot: p.numPart){
    printfln("ntot = %.20f  E = %.20f", ntot, GSEstore[ntot]);  
  }

  //Find the sector with the global GS:
  int N_GS;
  double EGS = std::numeric_limits<double>::infinity();
  for(auto ntot: p.numPart){
    if (GSEstore[ntot] < EGS) {
      EGS = GSEstore[ntot];
      N_GS = ntot;
    }
  }
  printfln("N_GS = %i",N_GS);

  //Calculate the spectral weights:
  if (p.calcweights) {
    MPS & psiGS = psiStore[N_GS];

    printfln(""); 
    printfln("Spectral weights:");
    printfln("(Spectral weight is the square of the absolute value of the number.)");

    if ( GSEstore.find(N_GS+1) != GSEstore.end() ){ //if the N_GS+1 state was computed, print the <N+1|c^dag|N> terms

      MPS & psiNp = psiStore[N_GS+1];

      ExpectationValueAddEl(psiNp, psiGS, "up", p);
      ExpectationValueAddEl(psiNp, psiGS, "dn", p);
    }

    else {      
      printfln("ERROR: we don't have info about the N_GS+1 occupancy sector.");
    }

    if ( GSEstore.find(N_GS-1) != GSEstore.end() ){ //if the N_GS-1 state was computed, print the <N-1|c|N> terms

      MPS & psiNm = psiStore[N_GS-1];   

      ExpectationValueTakeEl(psiNm, psiGS, "up", p);
      ExpectationValueTakeEl(psiNm, psiGS, "dn", p);
    }
  
    else {      
      printfln("ERROR: we don't have info about the N_GS-1 occupancy sector.");
    }

  } //end of if (calcweights)  

} //end of calculateAndPrint()

//calculates <psi1|c_dag|psi2>, according to http://itensor.org/docs.cgi?vers=cppv3&page=formulas/mps_onesite_op
double ExpectationValueAddEl(MPS psi1, MPS psi2, std::string spin, const params &p){

  psi2.position(p.impindex); //set orthogonality center
  auto newTensor = noPrime(op(p.sites,"Cdag"+spin, p.impindex)*psi2(p.impindex)); //apply the local operator
  psi2.set(p.impindex,newTensor); //plug in the new tensor, with the operator applied

  auto res = inner(psi1, psi2);
    
  std::cout << "weight w+ " << spin << ": " << res << "\n";  
}

//calculates <psi1|c|psi2>
double ExpectationValueTakeEl(MPS psi1, MPS psi2, std::string spin, const params &p){
  
  psi2.position(p.impindex);
  auto newTensor = noPrime(op(p.sites,"C"+spin, p.impindex)*psi2(p.impindex)); 
  psi2.set(p.impindex,newTensor);

  auto res = inner(psi1, psi2);
    
  std::cout << "weight w- " << spin << ": " << res << "\n";  
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
    std::cout << std::real(val.cplx()) << " ";
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
  for(auto i : range1(length(psi)) ){
    psi.position(i);
    auto val2  = psi.A(i) * p.sites.op("Cdagup*Cup*Cdagdn*Cdn", i) * dag(prime(psi.A(i),"Site"));
    auto val1u = psi.A(i) * p.sites.op("Cdagup*Cup", i) * dag(prime(psi.A(i),"Site"));
    auto val1d = psi.A(i) * p.sites.op("Cdagdn*Cdn", i) * dag(prime(psi.A(i),"Site"));
    auto sq = p.g * sqrt( val2.cplx() - val1u.cplx() * val1d.cplx() );
    std::cout << sq << " ";
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
    auto element = p.g * pdt;
    std::cout << "[v=" << v << " u=" << u << " pdt=" << pdt << "] ";
    if (i != p.impindex) tot += element; // exclude the impurity site in the sum
  }
  std::cout << std::endl;
  Print(tot);
}

//fills the vectors eps and V with the correct values for given gamma and number of bath sites
void GetBathParams(double epseff, std::vector<double>& eps, std::vector<double>& V, const params &p) {
  double dEnergy = 2./p.NBath;
  double Vval = std::sqrt( 2*p.gamma/(M_PI*p.NBath) ); // pi!
  std::cout << "Vval=" << Vval << std::endl;
  eps.resize(0);
  V.resize(0);
  eps.push_back(epseff);
  V.push_back(0.);
  for(auto k: range1(p.NBath)){
    eps.push_back( -1 + (k-0.5)*dEnergy );
    V.push_back( Vval );
  }
}
