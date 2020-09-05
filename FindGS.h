#ifndef _calcGS_h_
#define _calcGS_h_

#include <itensor/all.h>
#include <itensor/util/args.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <limits> // quiet_NaN
#include <tuple>

#include <omp.h>

#include <highfive/H5Easy.hpp>
using namespace H5Easy;

template <class T>
inline DataSet dumpreal(File& file,
                        const std::string& path,
                        const std::complex<T>& data,
                        DumpMode mode = DumpMode::Create)
{
  const T realdata = std::real(data);
  return dump(file, path, realdata, mode);
}

template <class T>
inline DataSet dumpreal(File& file,
                        const std::string& path,
                        const std::vector<std::complex<T>>& data,
                        DumpMode mode = DumpMode::Create)
{
  std::vector<T> realdata;
  for (const auto &z : data)
    realdata.push_back(std::real(z));
  return dump(file, path, realdata, mode);
}

using namespace itensor;

using complex_t = std::complex<double>;

constexpr auto full = std::numeric_limits<double>::max_digits10;

inline bool even(int i) { return i%2 == 0; }
inline bool odd(int i) { return i%2 != 0; }

template <typename T>
  std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, " "));
    return out;
  }

template <typename U, typename V>
  std::ostream& operator<< (std::ostream& out, const std::pair<U,V>& v) {
    out << v.first << "," << v.second;
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
     for (auto k: range1(Nbath))
       V.push_back( std::sqrt( 2.0*_Gamma/(M_PI*Nbath) ) );
     return V;
   }
};

using spin = double;

constexpr auto spin0 = spin(0);
constexpr auto spinp = spin(0.5);
constexpr auto spinm = spin(-0.5);

using subspace = std::pair<int, spin>;
using state = std::pair<subspace, int>;

inline std::string str(subspace &sub)
{
  std::ostringstream ss;
  auto [n, sz] = sub;
  ss << "/" << n << "/" << sz;
  return ss.str();
}

inline std::string str(subspace &sub, std::string s)
{
  return str(sub) + "/" + s;
}

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
   void dump(auto &file, std::string path) const {
     H5Easy::dump(file, path + "/norm", _norm);
     H5Easy::dump(file, path + "/Ebis", _Ebis);
     H5Easy::dump(file, path + "/deltaE", _deltaE);
     H5Easy::dump(file, path + "/residuum", _residuum);
   }
};

class problem_type {
 public:
   virtual int imp_index(int) = 0;
};

class imp_first : public problem_type
{
 public:
   int imp_index(int) override { return 1; }
};

class imp_middle : public problem_type
{
 public:
   int imp_index(int NBath) override {
     assert(even(NBath));
     return 1+NBath/2;
   }
};

namespace prob {
   class std : public imp_first {};
   class Ec : public imp_first {};
   class Ec_V : public imp_first {};
   class middle : public imp_middle {};
   class middle_2channel : public imp_middle {};
}

inline std::unique_ptr<problem_type> set_problem(std::string str)
{
  if (str == "std") return std::make_unique<prob::std>();
  if (str == "Ec") return std::make_unique<prob::Ec>();
  if (str == "Ec_V") return std::make_unique<prob::Ec_V>();
  if (str == "middle") return std::make_unique<prob::middle>();
  if (str == "middle_2channel") return std::make_unique<prob::middle_2channel>();
  throw std::runtime_error("Unknown MPO type");
}

// parameters from the input file
struct params {
  string inputfn;       // filename of the input file
  InputGroup input;     // itensor input parser

  string MPO = "std";   // which MPO representation to use
  std::unique_ptr<problem_type> problem = set_problem(MPO);

  int N;                // number of sites
  int NBath;            // number of bath sites
  int NImp;             // number of impurity orbitals
  int impindex;         // impurity position in the chain (1 based)

  Hubbard sites;        // itensor object

  // all bools have default value false
  bool computeEntropy;   // von Neumann entropy at the bond between impurity and next site. Works as intended if p.impindex=1.
  bool impNupNdn;        // print the number of up and dn electrons on the impurity

  bool excited_state;    // computes the first excited state
  bool printDimensions;  // prints dmrg() prints info during the sweep
  bool calcweights;      // calculates the spectral weights of the two closes spectroscopically availabe excitations
  bool refisn0;          // the energies will be computed in the sectors centered around the one with n = round(n0) + 1
  bool parallel;         // enables openMP parallel calculation of the for loop in findGS()
  bool verbose;          // verbosity level
  bool band_level_shift; // shifts the band levels for a constant in order to make H particle-hole symmetric
  bool printTotSpinZ;    // prints total Nup, Ndn and Sz.


  bool chargeCorrelation;// compute the impurity-superconductor correlation <n_imp n_i>
  bool pairCorrelation;  // compute the impurity-superconductor correlation <d d c_i^dag c_i^dag>
  bool spinCorrelation;  // compute the impurity-superconductor correlation <S_imp S_i>
  bool hoppingExpectation;//compute the hopping expectation value 1/sqrt(N) \sum_sigma \sum_i <d^dag c_i> + <c^dag_i d>

  double EnergyErrgoal; // the convergence value at which dmrg() will stop the sweeps; default is machine precision
  int nrH;              // number of times to apply H to psi before comencing the sweep - akin to a power method; default = 5
  int nrange;           // the number of energies computed is 2*nrange + 1

  std::unique_ptr<imp>    qd;
  std::unique_ptr<SCbath> sc;
  std::unique_ptr<hyb>    Gamma;
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

InputGroup parse_cmd_line(int, char * [], params &p);
void FindGS(InputGroup &input, store &s, params &p);
void calculateAndPrint(InputGroup &input, store &s, params &p);

#endif
