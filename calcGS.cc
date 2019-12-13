#include <itensor/all.h>
#include <itensor/util/args.h>
#include "SC_BathMPO.h"
// #include "lanczos.h" // Daniel's DMRG

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace itensor;

void GetBathParams(double U, double epsimp, double gamma, std::vector<double>& eps, std::vector<double>& V, int NBath);
void MyDMRG(MPS& psi, MPO& H, double& energy, Args args);
void FindGS(std::string inputfn, InputGroup &input, int N, int NBath);
void MeasureOcc(MPS& psi, const SiteSet& sites);

bool writetofiles;
bool excited_state;
bool randomMPSb;
int nrH;

int main(int argc, char* argv[]){ 
  if(argc!=2){
    std::cout<<"Please provide input file. Usage ./calcGS inputfile.txt" << std::endl;
    return 0;
  }
  string inputfn{argv[1]};
  auto input = InputGroup(inputfn, "params"); //get input parameters using InputGroup from itensor
  int NImp = 1; // number of impurity orbitals
  int N, NBath;   // N=number of sites, Nbath=number of bath sites
  N = input.getInt("N", 0);
  if (N != 0) {
    NBath = N-NImp;
  } else { // N not specified, try NBath
    NBath = input.getInt("NBath", 0);
    if (NBath == 0) {
      std::cout << "specify either N or NBath!" << std::endl;
      exit(1);
    }
    N = NBath+NImp;
  }
  std::cout << "N=" << N << " NBath=" << NBath << std::endl;
  // Global variables:
  writetofiles = input.getYesNo("writetofile", false);
  excited_state = input.getYesNo("excited_state", false);
  randomMPSb = input.getYesNo("randomMPS", false);
  nrH = input.getInt("nrH", 5);
  FindGS(inputfn, input, N, NBath); //calculates the ground state in different sectors
}

//calculates the groundstates and the energie of the three relevant particle number sectors
void FindGS(std::string inputfn, InputGroup &input, int N, int NBath){
  double U = input.getReal("U"), alpha = input.getReal("alpha"), gamma = input.getReal("gamma");
  double Ec = input.getReal("Ec", 0), n0 = input.getReal("n0", NBath);
  std::vector<int> numPart(0); // input: occupancies of interest
  std::vector<double> GSenergies(0); // result: lowest energy in each occupancy sector
  std::vector<double> ESenergies(0); // optionally: first excited state in each occupancy sector
  int nhalf = N; // total nr of electrons at half-filling
  numPart.push_back(nhalf);
  const int nrange = input.getInt("nrange", 1);
  for (int i = 1; i <= nrange; i++) {
    numPart.push_back(nhalf+i);
    numPart.push_back(nhalf-i);
  }
  //the sweeps object defines the accuracy used for each update cycle in DMRG
  //the used parameters are read from the input file with the following meaning:
  // maxdim:  maximal bond dimension -> start low approach ground state roughly, then increase
  // mindim:  minimal bond dimension used, can be helpfull to improve DMRG convergence
  // cutoff:  truncated weight i.e.: sum of all discared squared Schmidt values 
  // niter:   during DMRG, one has to solve an eigenvalue problem usually using a krylov 
  //          method. niter is the number of krylov vectors used. Since DMRG is variational
  //          it does not matter if we use only very few krylov vectors. We only want to 
  //          move in the direction of the ground state.
  // noise:   if non-zero a so-called noise term is added to DMRG which improves the convergence.
  //          this term must be zero towards the end, as the ground state is otherwise incorrect.
  auto inputsw = InputGroup(inputfn,"sweeps");
  auto sw_table = InputGroup(inputsw,"sweeps");
  int nrsweeps = input.getInt("nrsweeps", 15);
  auto sweeps = Sweeps(nrsweeps,sw_table);
  std::streambuf *coutbuf = std::cout.rdbuf();
  for(auto ntot: numPart) {
    std::vector<double> eps; // vector containing on-site energies of the bath !one-indexed! 
    std::vector<double> V; //vector containing hoppinga amplitudes to the bath !one-indexed! 
    const double epsimp = -U/2.;
    const double epseff = epsimp - 2.*Ec*(ntot-n0) + Ec;
    GetBathParams(U, epseff, gamma, eps, V, NBath);
    //sites is an ITensor thing. it defines the local hilbert space and
    //operators living on each site of the lattice
    //for example sites.op("N",1) gives the pariticle number operator 
    //on the first site
    auto sites = Hubbard(N);
    MPO H(sites); // MPO is the hamiltonian in "MPS-form" after this line it is still a trivial operator
    const double Ueff = U + 2.*Ec;
    const double d = 2./NBath;  // d=2D/NBath
    const double g = alpha * d;
    Fill_SCBath_MPO(H, sites, eps, V, Ueff, g); // defined in SC_BathMPO.h, fills the MPO with the necessary entries
    if (writetofiles) {
      std::ofstream out("output" + std::to_string(ntot) + ".txt");
      std::cout.rdbuf(out.rdbuf());
    }
    //initialize the MPS in a product state with the correct particle number
    //since the itensor conserves the number of particles, in this step we 
    //choose in which sector of particle number we perform the calculation
    auto state = InitState(sites);
    int tot = 0;
    int nsc = ntot-1; // number of electrons in the SC in Gamma->0 limit
    int npair = nsc/2; // number of pairs in the SC
    state.set(1, "Up"); // Particle on the impurity site
    tot++;
    for(auto i : range1(npair)) {
      state.set(1+i,"UpDn");
      tot += 2;
    }
    if (nsc%2 == 1) {
      state.set(1+npair+1,"Dn"); // additional spin-down electron (Sz=0)
      tot++;
    }
    assert(tot == ntot);
    std::cout <<"Using sector with " << ntot << " number of Particles"<<std::endl;
    MPS psi(state);
    Args args; //args is used to store and transport parameters between various functions
    if (randomMPSb) {
      psi = randomMPS(state);
    }
    //apply the MPO a couple of times to get DMRG started. otherwise it might not converge
    for(auto i : range1(nrH)){
      psi = applyMPO(H,psi,args);
      psi.noPrime().normalize();
    }
    auto [GS0, GS] = dmrg(H,psi,sweeps,{"Quiet",true}); // call itensor dmrg
    printfln("Eigenvalue = %.20f",GS0);
    double shift = Ec*pow(ntot-n0,2); // occupancy dependent effective energy shift
    shift += U/2.; // RZ, for convenience
    double GSenergy = GS0+shift;
    printfln("Ground state energy = %.20f",GSenergy);
    MeasureOcc(GS, sites);
    GSenergies.push_back(GSenergy);
    // Norm
    double normGS = inner(GS, GS);
    printfln("norm = %.20f", normGS);
    double GS0bis = inner(GS, H, GS);
    printfln("Eigenvalue(bis) = %.20f",GS0bis);
    printfln("diff = %.20f", GS0-GS0bis);
    // Energy deviation, residual value
    double deltaE = sqrt(inner(H, GS, H, GS) - pow(inner(GS, H, GS),2));
    double residuum = inner(GS,H,GS) - GS0*inner(GS,GS);
    printfln("deltaE = %.20f", deltaE);
    printfln("residuum = %.20f", residuum);
    if (excited_state) {
      auto wfs = std::vector<MPS>(1);
      wfs.at(0) = GS;
      auto [ESenergy, ES] = dmrg(H,wfs,psi,sweeps,{"Quiet",true,"Weight",11.0});
      ESenergy += shift;
      printfln("Excited state energy = %.20f",ESenergy);
      MeasureOcc(ES, sites);
      ESenergies.push_back(ESenergy);
    }
  } 
  std::cout.rdbuf(coutbuf);
  for(auto i : range(GSenergies.size())) {
    std::cout<< "n = "<< numPart[i] << "  E= "<< std::setprecision(16) << GSenergies.at(i);
    if (excited_state)
      std::cout << " " << ESenergies.at(i);
    std::cout<< std::endl;
  }
}

//prints the occupation number of an MPS psi
//instructive to learn how to calculate local observables
void MeasureOcc(MPS& psi, const SiteSet& sites){
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
