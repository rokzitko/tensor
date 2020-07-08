#ifndef _calcGS_h_
#define _calcGS_h_

using namespace itensor;

// parameters from the input file 
struct params {
  string inputfn;       // filename of the input file
  InputGroup input;     // itensor input parser

  string MPO = "std";   // which MPO representation to use

  int N;                // number of sites
  int NImp;             // number of impurity orbitals
  int NBath;            // number of bath sites
  int impindex;         // impurity position in the chain (1 based)

  Hubbard sites;        // itensor object

  // all bools have default value false
  bool excited_state;    // computes the first excited state
  bool randomMPSb;       // sets the initial state to random
  bool printDimensions;  // prints dmrg() prints info during the sweep
  bool calcweights;      // calculates the spectral weights of the two closes spectroscopically availabe excitations
  bool refisn0;          // the energies will be computed in the sectors centered around the one with n = round(n0) + 1
  bool parallel;         // enables openMP parallel calculation of the for loop in findGS()
  bool verbose;          // verbosity level
  bool band_level_shift; // shifts the band levels for a constant in order to make H particle-hole symmetric
  bool computeEntropy;   // prints von Neumann entropy at the bond between impurity and next site. Works as intended is p.impindex=1.
  bool printTotSpinZ;    // prints total Nup, Ndn and Sz.

  bool impNupNdn;        // print the number of up and dn electrons on the impurity

  bool chargeCorrelation;// compute the impurity-superconductor correlation <n_imp n_i>
  bool pairCorrelation;  // compute the impurity-superconductor correlation <d d c_i^dag c_i^dag>
  bool spinCorrelation;  // compute the impurity-superconductor correlation <S_imp S_i>
  bool hoppingExpectation;//compute the hopping expectation value 1/sqrt(N) \sum_sigma \sum_i <d^dag c_i> + <c^dag_i d>

  double EnergyErrgoal; // the convergence value at which dmrg() will stop the sweeps; default is machine precision
  int nrH;              // number of times to apply H to psi before comencing the sweep - akin to a power method; default = 5
  int nrange;           // the number of energies computed is 2*nrange + 1

  bool calcspin1;       

  double alpha;         // e-e coupling
  double n0;            // charge offset
  double d;             // d=2D/NBath, level spacing
  double g;             // strength of the SC coupling

  double U;             // e-e on impurity site
  double gamma;         // hybridisation
  double Ec;            // charging energy
  double V12;           // QD-SC capacitive coupling
  double epsimp;        // impurity level
  double nu;            // nu=1/2-epsimp/U, computed after epsimp parsed
  double Ueff;          // effective e-e on impurity site (after Ec_trick mapping)

  double EZ_imp;        // impurity Zeeman energy
  double EZ_bulk;        // bulk Zeeman energy

  std::vector<int> numPart; // range of total occupancies of interest

  //parameters for the phase transition point iteration
  double gamma0;        // initial guess
  double gamma1;        // initial guess
  double precision;     // required precision
  int maxIter;          // maximal number of iterations of the secant method

};

// lists of quantities calculated in FindGS 
struct store
{
  std::map<int, MPS> psiStore;      // ground states
  std::map<int, double> GSEstore;   // ground state energies
  std::map<int, MPS> ESpsiStore;    // excited states
  std::map<int, double> ESEstore;   // excited state energies

  //These quantities require the knowledge of H, so they are calculated in FindGS and saved here.
  std::map<int, double> GS0bisStore; // <GS|H|GS>
  std::map<int, double> deltaEStore; // sqrt(<GS|H^2|GS> - <GS|H|GS>^2)
  std::map<int, double> residuumStore; // <GS|H|GS> - GSE*<GS|GS>
};


void FindGS(InputGroup &input, store &s, params &p);
void calculateAndPrint(InputGroup &input, store &s, params &p);
void GetBathParams(std::vector<double>& eps, std::vector<double>& V, params &p);
std::tuple<MPO, double> initH(std::vector<double> eps, std::vector<double> V, int ntot, params &p);
MPS initPsi(int ntot, params &p);
void ExpectationValueAddEl(MPS psi1, MPS psi2, std::string spin, const params &p);
void ExpectationValueTakeEl(MPS psi1, MPS psi2, std::string spin, const params &p);
void ChargeCorrelation(MPS& psi, const params &p);
void SpinCorrelation(MPS& psi, const params &p);
void PairCorrelation(MPS& psi, const params &p);
void expectedHopping(MPS& psi, const params &p);
double ImpurityCorrelator(MPS& psi, auto impOp, int j, auto opj, const params &p);
void MyDMRG(MPS& psi, MPO& H, double& energy, Args args);
void ImpurityUpDn(MPS& psi, const params &p);
void TotalSpinz(MPS& psi, const params &p);
void MeasureOcc(MPS& psi, const params &);
void MeasurePairing(MPS& psi, const params &);
void MeasureAmplitudes(MPS& psi, const params &);
InputGroup parse_cmd_line(int, char * [], params &p);
void PrintEntropy(MPS& psi, const params &p);


#endif
