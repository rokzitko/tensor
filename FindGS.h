#ifndef _calcGS_h_
#define _calcGS_h_

//#include "hash_for_tuples.h"

using namespace itensor;

inline bool even(int i) { return i%2 == 0; }
inline bool odd(int i) { return i%2 != 0; }

template <typename T>
  std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, " "));
    return out;
  }

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
     std::vector<double> eps;
     eps.push_back(std::numeric_limits<double>::quiet_NaN()); // we use 1-based indexing
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
   double _EZ;
 public:
   SCbath(int Nbath, double alpha, double Ec, double n0, double EZ) :
     bath(Nbath), _alpha(alpha), _Ec(Ec), _n0(n0), _EZ(EZ) {};
   auto alpha() const { return _alpha; }
   auto Ec() const { return _Ec; }
   auto n0() const { return _n0; }
   auto EZ() const { return _EZ; }
   auto g() const { return _alpha*d(); }
   auto eps(bool band_level_shift = true) const {
     auto eps = bath::eps();
     if (band_level_shift)
       for (auto &x: eps)
         x += -g()/2.0;
     return eps;
   };
};

class hyb {
 private:
   double _Gamma;
 public:
   hyb(double Gamma) : _Gamma(Gamma) {};
   auto Gamma() const { return _Gamma; }
   auto V(int Nbath) const {
     std::vector<double> V;
     V.push_back(std::numeric_limits<double>::quiet_NaN()); // we use 1-based indexing
     for (auto k: range1(Nbath))
       V.push_back( std::sqrt( 2.0*_Gamma/(M_PI*Nbath) ) );
     return V;
   }
};

using spin = double;

constexpr auto spin0 = spin(0);
constexpr auto spinp = spin(0.5);
constexpr auto spinm = spin(-0.5);

#ifdef OLD
// Quantum numbers for an invariant subspace
class subspace {
 private:
   int _n;
   spin _sz;
 public:
   subspace(int n, spin sz) : _n(n), _sz(sz) {}
   auto n() const { return _n; }
   auto sz() const { return _sz; }
   auto get() const { return std::make_tuple(n(), sz()); }
};

// Quantum numbers for a state (i=0 is GS, i=1 is 1st excited state, etc.)
class state : public subspace {
 private:
   int _i;
 public:
   state(int n, spin sz, int i) : subspace(n,sz), _i(i) {}
   auto i() const { return _i; }
};
#endif

using subspace = std::pair<int, spin>;
using state = std::pair<subspace, int>;

class eigenpair {
 private:
   Real _E = 0;
   MPS _psi;
 public:
   eigenpair() {}
   eigenpair(Real E, MPS psi) : _E(E), _psi(psi) {}
   auto E() const { return _E; }
   MPS & psi() { return _psi; } // not const !
};

class psi_stats {
 private:
   double _norm = 0;
   double _Ebis = 0;
   double _deltaE = 0;
   double _residuum = 0;
 public:
   psi_stats() {}
   psi_stats(double E, MPS &psi, MPO &H) {
     _norm = inner(psi, psi);
     _Ebis = inner(psi, H, psi);
     _deltaE = sqrt( inner(H, psi, H, psi) - pow(_Ebis,2) );
     _residuum = _Ebis-E*_norm;
   }
   auto norm() const { return _norm; }
   auto Ebis() const { return _Ebis; }
   auto deltaE() const { return _deltaE; }
   auto residuum() const { return _residuum; }
};

// parameters from the input file
struct params {
  string inputfn;       // filename of the input file
  InputGroup input;     // itensor input parser

  string MPO = "std";   // which MPO representation to use

  int N;                // number of sites
  int NBath;            // number of bath sites
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

  std::unique_ptr<imp> qd; // replaces {U, epsimp, nu}
  std::unique_ptr<SCbath> sc;
  std::unique_ptr<hyb> Gamma;
  double V12;           // QD-SC capacitive coupling

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
  std::vector<subspace> iterateOver; // a zipped vector of off (n, Sz) combinations
};

// lists of quantities calculated in FindGS 
struct store
{
  std::map<subspace, eigenpair> eigen0, eigen1; // 0=GS, 1=1st ES, etc.
  std::map<subspace, psi_stats> stats0;
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
