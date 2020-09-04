#ifndef _calcGS_h_
#define _calcGS_h_

using namespace itensor;

// Class containing impurity parameters
class imp {
 private:
   double _U;
   double _eps;
   double _EZ;
 public:
   imp(double U, double eps, double EZ) : _U(U), _eps(eps), _EZ(EZ) {};
   auto U() const { return _U; }
   auto eps() const { return _eps; }
   auto EZ() const { return _EZ; }
   auto nu() const { return _U != 0.0 ? 0.5-_eps/_U : std::numeric_limits<double>::quiet_NaN(); }
};

// Class containing bath parameters
class bath { // normal-state bath
 private:
   int _Nbath; // number of levels
   double _D; // half-bandwidth
 public:
   bath(int Nbath, double D = 1.0) : _Nbath(Nbath), _D(D) {};
   auto Nbath() const { return _Nbath; }
   auto d() const { // inter-level spacing
     return 2.0*_D/_Nbath;
   }
   auto eps() const {
     std::vector<double> eps; // note: zero-based indexing here!
     for (auto k: range1(_Nbath))
       eps.push_back( -_D + (k-0.5)*d() );
     return eps;
   }
};

class SCbath : public bath { // superconducting island bath
 private:
   double _alpha; // pairing strength
   double _Ec; // charging energy
   double _n0; // offset
 public:
   SCbath(int Nbath, double alpha, double Ec, double n0) :
     bath(Nbath), _alpha(alpha), _Ec(Ec), _n0(n0) {};
   auto alpha() const { return _alpha; }
   auto Ec() const { return _Ec; }
   auto n0() const { return _n0; }
   auto g() const { return _alpha*d(); }
   auto eps(bool band_level_shift = true) const {
     auto eps = bath::eps();
     if (band_level_shift)
       for (auto x: eps)
         x += -g()/2.0;
     return eps;
   };
};

// parameters from the input file 
struct params {
  string inputfn;       // filename of the input file
  InputGroup input;     // itensor input parser

  string MPO = "std";   // which MPO representation to use

  int N;                // number of sites
  int NImp;             // number of impurity orbitals
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


//  double U;             // e-e on impurity site
//  double epsimp;        // impurity level
//  double nu;            // nu=1/2-epsimp/U, computed after epsimp parsed
  std::unique_ptr<imp> qd; // replaces {U, epsimp, nu}
  double Ueff;          // effective e-e on impurity site (after Ec_trick mapping)
  
  std::unique_ptr<SCbath> sc;
  int NBath;            // number of bath sites
  double d;             // d=2D/NBath, level spacing
  double g;             // strength of the SC coupling
  double n0;            // charge offset
//  double alpha;         // e-e coupling
//  double Ec;            // charging energy

  double gamma;         // hybridisation
  double V12;           // QD-SC capacitive coupling

  double EZ_imp;        // impurity Zeeman energy
  double EZ_bulk;        // bulk Zeeman energy


  // TWO CHANNEL PARAMETERS
  double alpha1, alpha2;
  double n01, n02;
  double g1, g2;
  double gamma1, gamma2;
  double Ec1, Ec2;
  double EZ_bulk1, EZ_bulk2;

  int SCSCinteraction; //test parameter for the 2 channel MPO

  std::vector<int> numPart; // range of total occupancies of interest
  std::map<int, std::vector<double>> Szs; // Szs for each n in numPart
  std::vector<std::pair<int, double>> iterateOver; // a zipped vector of off (n, Sz) combinations

  //parameters for the phase transition point iteration
  double PTgamma0;        // initial guess
  double PTgamma1;        // initial guess
  double PTprecision;     // required precision
  int PTmaxIter;          // maximal number of iterations of the secant method

};

// lists of quantities calculated in FindGS 
struct store
{
  std::map<std::pair<int, double>, MPS> psiStore;      // ground states
  std::map<std::pair<int, double>, double> GSEstore;   // ground state energies
  std::map<std::pair<int, double>, MPS> ESpsiStore;    // excited states
  std::map<std::pair<int, double>, double> ESEstore;   // excited state energies

  //These quantities require the knowledge of H, so they are calculated in FindGS and saved here.
  std::map<std::pair<int, double>, double> GS0bisStore; // <GS|H|GS>
  std::map<std::pair<int, double>, double> deltaEStore; // sqrt(<GS|H^2|GS> - <GS|H|GS>^2)
  std::map<std::pair<int, double>, double> residuumStore; // <GS|H|GS> - GSE*<GS|GS>
};


void FindGS(InputGroup &input, store &s, params &p);
void calculateAndPrint(InputGroup &input, store &s, params &p);
std::tuple<MPO, double> initH(int ntot, params &p);
MPS initPsi(int ntot, float Sz, params &p);
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
