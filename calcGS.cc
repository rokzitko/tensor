#include <itensor/all.h>
#include <itensor/util/args.h>

#ifndef MIDDLE_IMP
 #include "SC_BathMPO.h"
#else
 #include "SC_BathMPO_MiddleImp.h"
#endif

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>

#include <omp.h>

using namespace itensor;

void FindGS(std::string inputfn, InputGroup &input, auto sites, int N, int NBath, 
  std::map<int, MPS>& psiStore, std::map<int, double>& GSEstore, std::map<int, MPS> ESpsiStore, std::map<int, double>& ESEstore, 
  std::map<int, double>& GS0bisStore, std::map<int, double>& deltaEStore, std::map<int, double>& residuumStore);
void calculateAndPrint(InputGroup &input, int N, auto sites, 
  std::map<int, MPS> psiStore, std::map<int, double> GSEstore,std::map<int, MPS> ESpsiStore, std::map<int, double> ESEstore, 
  std::map<int, double> GS0bisStore, std::map<int, double> deltaEStore, std::map<int, double> residuumStore);
void GetBathParams(double U, double epsimp, double gamma, std::vector<double>& eps, std::vector<double>& V, int NBath);
MPO initH(auto sites, int n, std::vector<double> eps, std::vector<double> V, double Ueff, double g);
MPS initPsi(auto sites, int n);
double ExpectationValueAddEl(auto sites, MPS psi1, MPS psi2, std::string spin);
double ExpectationValueTakeEl(auto sites, MPS psi1, MPS psi2, std::string spin);
void MyDMRG(MPS& psi, MPO& H, double& energy, Args args);
void MeasureOcc(MPS& psi, const SiteSet& sites);
void MeasurePairing(MPS& psi, const SiteSet& sites, double);
void MeasureAmplitudes(MPS& psi, const SiteSet& sites, double);

//all bools have default value false
bool writetofiles;    //if true writes MPS and MPO to file
bool excited_state;   //if true computes the first excited state
bool randomMPSb;      //it true sets the initial state to random
bool printDimensions; //if true prints dmrg() prints info during the sweep
bool calcweights;     //if true calculates the spectral weights of the two closes spectroscopically availabe excitations
bool refisn0;         //if true the energies will be computed in the sectors centered around the one with n = round(n0) + 1
bool parallel;        //if true enables openMP parallel calculation of the for loop in findGS()

double EnergyErrgoal; //the convergence value at which dmrg() will stop the sweeps; default = machine precision
int nrH;              //number of times to apply H to psi before comencing the sweep - akin to a power method; default = 5
int nrange;           //the number of energies computed is 2*nrange + 1

int impindex;         //impurity position in the chain (1 based)


int main(int argc, char* argv[]){ 
  
  if(argc!=2){
    std::cout<<"Please provide input file. Usage ./calcGS inputfile.txt" << std::endl;
    return 0;
  }


  //read parameters from the input file
  string inputfn{argv[1]};
  auto input = InputGroup(inputfn, "params"); //get input parameters using InputGroup from itensor
  
  int NImp = 1; // number of impurity orbitals
  int N, NBath;   // N=number of sites, Nbath=number of bath sites
  N = input.getInt("N", 0);
  if (N != 0) {
    NBath = N-NImp;
  } 
  else { // N not specified, try NBath
    NBath = input.getInt("NBath", 0);
    if (NBath == 0) {
      std::cout << "specify either N or NBath!" << std::endl;
      exit(1);
    }
    N = NBath+NImp;
  }
#ifdef MIDDLE_IMP
  assert(NBath%2 == 0);   // NBath must be even
  impindex = 1+NBath/2;
#else
  impindex = 1;
#endif
  std::cout << "N=" << N << " NBath=" << NBath << " impindex=" << impindex << std::endl;
  
  // Global variables:
  writetofiles = input.getYesNo("writetofiles", false);
  excited_state = input.getYesNo("excited_state", false);
  randomMPSb = input.getYesNo("randomMPS", false);
  printDimensions = input.getYesNo("printDimensions", false);
  calcweights = input.getYesNo("calcweights", false);
  refisn0 = input.getYesNo("refisn0", false);
  parallel = input.getYesNo("parallel", false);

  nrH = input.getInt("nrH", 5);
  EnergyErrgoal = input.getReal("EnergyErrgoal", 1e-16);
  nrange = input.getInt("nrange", 1);

  //THESE ARE ALL TO BE PASSED TO calculateAndPrint()
  std::map<int, MPS> psiStore;      //stores ground states
  std::map<int, double> GSEstore;   //stores ground state energies
  std::map<int, MPS> ESpsiStore;    //stores excited states
  std::map<int, double> ESEstore;   //stores excited state energies

  //These quantities require the knowledge of H, so they are calculated in FindGS and saved here.
  std::map<int, double> GS0bisStore; //stores <GS|H|GS>
  std::map<int, double> deltaEStore; //stores sqrt(<GS|H^2|GS> - <GS|H|GS>^2)
  std::map<int, double> residuumStore; //stores <GS|H|GS> - GSE*<GS|GS>

  //sites is an ITensor thing. it defines the local hilbert space and
  //operators living on each site of the lattice
  //for example sites.op("N",1) gives the pariticle number operator 
  //on the first site
  auto sites = Hubbard(N); 

  //calculates the ground state in different particle number sectors according to n0, nrange, refisn0 and stores ground states and energies 
  FindGS(inputfn, input, sites, N, NBath, psiStore, GSEstore, ESpsiStore, ESEstore, GS0bisStore, deltaEStore, residuumStore); 
  calculateAndPrint(input, N, sites, psiStore, GSEstore, ESpsiStore, ESEstore, GS0bisStore, deltaEStore, residuumStore); 

}

//calculates the groundstates and the energies of the relevant particle number sectors
void FindGS(std::string inputfn, InputGroup &input, auto sites, int N, int NBath, 
  std::map<int, MPS>& psiStore, std::map<int, double>& GSEstore, std::map<int, MPS> ESpsiStore, std::map<int, double>& ESEstore, std::map<int, double>& GS0bisStore, std::map<int, double>& deltaEStore, std::map<int, double>& residuumStore){

  double U = input.getReal("U"), alpha = input.getReal("alpha"), gamma = input.getReal("gamma");
  double Ec = input.getReal("Ec", 0), n0 = input.getReal("n0", NBath);
  
  std::vector<int> numPart(0); // input: occupancies of interest
  std::vector<double> GSenergies(0); // result: lowest energy in each occupancy sector
  std::vector<double> ESenergies(0); // optionally: first excited state in each occupancy sector
  int nhalf = N; // total nr of electrons at half-filling
  int nref = (refisn0 ? round(n0)+1 : nhalf); //calculation of the energies is centered around this n
  numPart.push_back(nref);

  for (int i = 1; i <= nrange; i++) {
    numPart.push_back(nref+i);
    numPart.push_back(nref-i);
  }

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
  auto inputsw = InputGroup(inputfn,"sweeps");
  auto sw_table = InputGroup(inputsw,"sweeps");
  int nrsweeps = input.getInt("nrsweeps", 15);
  auto sweeps = Sweeps(nrsweeps,sw_table);

  std::vector<double> eps;        // vector containing on-site energies of the bath !one-indexed! 
  std::vector<double> V;          //vector containing hopping amplitudes to the bath !one-indexed!     
  const double epsimp = -U/2.;    
  const double Ueff = U + 2.*Ec;  //effective impurity couloumb potential term
  const double d = 2./NBath;      //d=2D/NBath, level spacing
  const double g = alpha * d;     //strenght of the SC coupling
  
  #pragma omp parallel for if(parallel) private(eps, V)
  for (int i=0; i<numPart.size(); i++){
    auto ntot = numPart[i];
  
  //for(auto ntot : numPart) {  -- omp dislikes this for loop
    
    std::cout << "\nSweeping in the sector with " << ntot << " particles.\n";  

    const double epseff = epsimp - 2.*Ec*(ntot-n0) + Ec;  //effective impurity on-site potential
    
    //Read the bath parameters (fills in the vectors eps and V).
    GetBathParams(U, epseff, gamma, eps, V, NBath);

    //initialize H and psi
    MPO H = initH(sites, ntot, eps, V, Ueff, g);  //This does not necessarily compile with older(?) compilers, as
    MPS psi = initPsi(sites, ntot);               //it dislikes 'auto sites' as a function parameter. Works on spinon.

    Args args; //args is used to store and transport parameters between various functions
    
    //Apply the MPO a couple of times to get DMRG started, otherwise it might not converge.
    for(auto i : range1(nrH)){
      psi = applyMPO(H,psi,args);
      psi.noPrime().normalize();
      }
    
    auto [GS0, GS] = dmrg(H,psi,sweeps,{"Silent",parallel, "Quiet",!printDimensions, "EnergyErrgoal",EnergyErrgoal}); // call itensor dmrg 
    
    double shift = Ec*pow(ntot-n0,2); // occupancy dependent effective energy shift
    shift += U/2.; // RZ, for convenience    
    
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


    if (excited_state) {
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
MPO initH(auto sites, int n, std::vector<double> eps, std::vector<double> V, double Ueff, double g){
  
  MPO H(sites); // MPO is the hamiltonian in "MPS-form" after this line it is still a trivial operator

  Fill_SCBath_MPO(H, sites, eps, V, Ueff, g); // defined in SC_BathMPO.h, fills the MPO with the necessary entries
  if (writetofiles) {
    std::ofstream out("output" + std::to_string(n) + ".txt");
    std::cout.rdbuf(out.rdbuf());
  }
  return H;
}

//initialize the MPS in a product state with the correct particle number
MPS initPsi(auto sites, int n){
  
  auto state = InitState(sites);

  int tot = 0;  
  int nsc = n-1; // number of electrons in the SC in Gamma->0 limit
  int npair = nsc/2; // number of pairs in the SC
  
  //Up electron at the impurity site and npair UpDn pairs. 
  state.set(impindex, "Up"); 
  tot++;

  int j=0;
  int i=1;
  for(i; j < npair; i++){ //In order to avoid adding a pair to the impurity site   
    if (i!=impindex){             //i counts sites, j counts added pairs.
      j++;
      state.set(i, "UpDn");
      tot += 2;
    }
  }     

  //If ncs is odd, add another Dn electron, but not to the impurity site.
  if (nsc%2 == 1) {
    if (i!=impindex){
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
  if (randomMPSb) {
    psi = randomMPS(state);
  }
 
  return psi;
}


//Loops over all particle sectors and prints relevant quantities.
void calculateAndPrint(InputGroup &input, int N, auto sites, std::map<int, MPS> psiStore, std::map<int, double> GSEstore,std::map<int, MPS> ESpsiStore, std::map<int, double> ESEstore, std::map<int, double> GS0bisStore, std::map<int, double> deltaEStore, std::map<int, double> residuumStore){
  
  //set up numPart, a vector of all calculated ns, and Nbath and g
  double n0 = input.getReal("n0", N-1);   
  double alpha = input.getReal("alpha");

  const int NBath = N-1;
  const double d = 2./NBath;      //d=2D/NBath, level spacing
  const double g = alpha * d;     //strenght of the SC coupling

  std::vector<int> numPart(0); // calculated occupancies
  int nref = (refisn0 ? round(n0)+1 : N); //calculation of the energies is centered around this n
  numPart.push_back(nref);
  
  for (int i = 1; i <= nrange; i++) {
    numPart.push_back(nref+i);
    numPart.push_back(nref-i);
  } 

  //print data for each sector
  for(auto n: numPart) {

    printfln("\n");
    printfln("RESULTS FOR THE SECTOR WITH %i PARTICLES:", n);

    printfln("Ground state energy = %.20f",GSEstore[n]);
    
    MPS & GS = psiStore[n];
    //norm
    double normGS = inner(GS, GS);
    printfln("norm = %.20f", normGS);
    //occupancy; site pairing; v and u amplitudes 
    MeasureOcc(GS, sites);
    MeasurePairing(GS, sites, g);
    MeasureAmplitudes(GS, sites, g);
    
    double & GS0 = GSEstore[n];
    double & GS0bis = GS0bisStore[n];
    double & deltaE = deltaEStore[n];
    double & residuum = residuumStore[n];
    //various measures of convergence (energy deviation, residual value)
    printfln("Eigenvalue(bis): <GS|H|GS> = %.20f",GS0bis);
    printfln("diff: E_GS - <GS|H|GS> = %.20f", GS0-GS0bis);
    printfln("deltaE: sqrt(<GS|H^2|GS> - <GS|H|GS>^2) = %.20f", deltaE);
    printfln("residuum: <GS|H|GS> - E_GS*<GS|GS> = %.20f", residuum);

    if (excited_state){
      MPS & ES = ESpsiStore[n];
      double & ESenergy = ESEstore[n]; 
      MeasureOcc(ES, sites);
      printfln("Excited state energy = %.20f",ESenergy);
     } 
  } //end of n-for loop

  printfln("");
  //Print out energies again:
  for(auto n: numPart){
    printfln("n = %.20f  E = %.20f", n, GSEstore[n]);  
  }

  //Find the sector with the global GS:
  int N_GS;
  double EGS = std::numeric_limits<double>::infinity();
  for(auto n: numPart){
    if (GSEstore[n] < EGS) {
      EGS = GSEstore[n];
      N_GS = n;
    }
  }
  printfln("N_GS = %i",N_GS);

  //Calculate the spectral weights:
  if (calcweights) {
    
    MPS & psiGS = psiStore[N_GS];

    printfln(""); 
    printfln("Spectral weights:");
    printfln("(Spectral weight is the square of the absolute value of the number.)");

    if ( GSEstore.find(N_GS+1) != GSEstore.end() ){ //if the N_GS+1 state was computed, print the <N+1|c^dag|N> terms

      MPS & psiNp = psiStore[N_GS+1];

      ExpectationValueAddEl(sites, psiNp, psiGS, "up");
      ExpectationValueAddEl(sites, psiNp, psiGS, "dn");
    }

    else {      
      printfln("ERROR: we don't have info about the N_GS + 1 occupancy sector.");
    }

    if ( GSEstore.find(N_GS+1) != GSEstore.end() ){ //if the N_GS-1 state was computed, print the <N-1|c|N> terms

      MPS & psiNm = psiStore[N_GS-1];   

      ExpectationValueTakeEl(sites, psiNm, psiGS, "up");
      ExpectationValueTakeEl(sites, psiNm, psiGS, "dn");
    }
  
    else {      
      printfln("ERROR: we don't have info about the N_GS - 1 occupancy sector.");
    }

  } //end of if (calcweights)  

} //end of calculateAndPrint()

//calculates <psi1|c_dag|psi2>, according to http://itensor.org/docs.cgi?vers=cppv3&page=formulas/mps_onesite_op
double ExpectationValueAddEl(auto sites, MPS psi1, MPS psi2, std::string spin){

  psi2.position(impindex); //set orthogonality center
  auto newTensor = noPrime(op(sites,"Cdag"+spin, impindex)*psi2(impindex)); //apply the local operator
  psi2.set(impindex,newTensor); //plug in the new tensor, with the operator applied

  auto res = inner(psi1, psi2);
    
  std::cout << "Electron in with spin " << spin << ": " << res << "\n";  
}

//calculates <psi1|c|psi2>
double ExpectationValueTakeEl(auto sites, MPS psi1, MPS psi2, std::string spin){
  
  psi2.position(impindex);
  auto newTensor = noPrime(op(sites,"C"+spin, impindex)*psi2(impindex)); 
  psi2.set(impindex,newTensor);

  auto res = inner(psi1, psi2);
    
  std::cout << "Electron out with spin " << spin << ": " << res << "\n";  
}
	

//prints the occupation number of an MPS psi
//instructive to learn how to calculate local observables
void MeasureOcc(MPS& psi, const SiteSet& sites){
  std::cout << "site occupancies = ";
  double tot = 0;
  for(auto i : range1(length(psi)) ){
    //position call very important! otherwise one would need to 
    //contract the whole tensor network of <psi|O|psi>
    //this way, only the local operator at site i is needed
    psi.position(i);
    auto val = psi.A(i) * sites.op("Ntot",i)* dag(prime(psi.A(i),"Site"));
    std::cout << std::real(val.cplx()) << " ";
    tot += std::real(val.cplx());
  }
  std::cout << std::endl;
  Print(tot);
}

// The sum (tot) corresponds to \bar{\Delta}', Eq. (4) in Braun, von Delft, PRB 50, 9527 (1999), first proposed by Dan Ralph.
// It reduces to Delta_BCS in the thermodynamic limit (if the impurity is decoupled, Gamma=0).
// For Gamma>0, there is no guarantee that all values 'sq' are real, because the expr <C+CC+C>-<C+C><C+C> may be negative.
void MeasurePairing(MPS& psi, const SiteSet& sites, double g){
  std::cout << "site pairing = ";
  std::complex<double> tot = 0;
  for(auto i : range1(length(psi)) ){
    psi.position(i);
    auto val2  = psi.A(i) * sites.op("Cdagup*Cup*Cdagdn*Cdn", i) * dag(prime(psi.A(i),"Site"));
    auto val1u = psi.A(i) * sites.op("Cdagup*Cup", i) * dag(prime(psi.A(i),"Site"));
    auto val1d = psi.A(i) * sites.op("Cdagdn*Cdn", i) * dag(prime(psi.A(i),"Site"));
    auto sq = g * sqrt( val2.cplx() - val1u.cplx() * val1d.cplx() );
    std::cout << sq << " ";
    if (i != impindex) tot += sq; // exclude the impurity site in the sum
  }
  std::cout << std::endl;
  Print(tot);
}

// See von Delft, Zaikin, Golubev, Tichy, PRL 77, 3189 (1996)
void MeasureAmplitudes(MPS& psi, const SiteSet& sites, double g){
  std::cout << "amplitudes vu = ";
  std::complex<double> tot = 0;
  for(auto i : range1(length(psi)) ){
    psi.position(i);
    auto valv = psi.A(i) * sites.op("Cdagup*Cdagdn*Cdn*Cup", i) * dag(prime(psi.A(i),"Site"));
    auto valu = psi.A(i) * sites.op("Cdn*Cup*Cdagup*Cdagdn", i) * dag(prime(psi.A(i),"Site"));
    auto v = sqrt( std::real(valv.cplx()) );
    auto u = sqrt( std::real(valu.cplx()) );
    auto pdt = v*u;
    auto element = g * pdt;
    std::cout << "[v=" << v << " u=" << u << " pdt=" << pdt << "] ";
    if (i != impindex) tot += element; // exclude the impurity site in the sum
  }
  std::cout << std::endl;
  Print(tot);
}

//fills the vectors eps and V with the correct values for given gamma and number of bath sites
void GetBathParams(double U, double epsimp, double gamma, std::vector<double>& eps, std::vector<double>& V, int NBath) {
  double dEnergy = 2./NBath;
  double Vval = std::sqrt( 2*gamma/(M_PI*NBath) ); // pi!
  std::cout << "Vval=" << Vval << std::endl;
  eps.resize(0);
  V.resize(0);
  eps.push_back(epsimp);
  V.push_back(0.);
  for(auto k: range1(NBath)){
    eps.push_back( -1 + (k-0.5)*dEnergy );
    V.push_back( Vval );
  }
}

