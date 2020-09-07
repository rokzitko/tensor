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
#include <cassert>

#include <omp.h>

#include <highfive/H5Easy.hpp>
using namespace H5Easy;

#define my_assert(x) do { \
  if (!(x)) { \
    std::cout << "Failed assertion, file " << __FILE__ << ", line " << __LINE__ << std::endl; \
    throw std::runtime_error("Failed assertion"); \
  }} while(0)

template <typename T>
inline DataSet dumpreal(File& file,
                        const std::string& path,
                        const std::complex<T>& data,
                        DumpMode mode = DumpMode::Create)
{
  const T realdata = std::real(data);
  return dump(file, path, realdata, mode);
}

template <typename T>
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

// Convert 0-based vector to 1-based vector
inline auto shift1(const std::vector<double> &a) {
  std::vector<double> b;
  b.push_back(std::numeric_limits<double>::quiet_NaN());
  for(const auto & x: a)
        b.push_back(x);
    return b;
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
   int _NBath; // number of levels
   double _D;  // half-bandwidth
 public:
   bath(int NBath, double D = 1.0) : _NBath(NBath), _D(D) {};
   auto NBath() const { return _NBath; }
   auto d() const { // inter-level spacing
     return 2.0*_D/_NBath;
   }
   auto eps() const {
     std::vector<double> eps;
     for (auto k: range1(_NBath))
       eps.push_back( -_D + (k-0.5)*d() );
     return eps;
   }
   void set_NBath(int NBath) { _NBath = NBath; }
};

class SCbath : public bath { // superconducting island bath
 private:
   double _alpha; // pairing strength
   double _Ec;    // charging energy
   double _n0;    // offset
   double _EZ;    // Zeeman energy
 public:
   SCbath(int NBath, double alpha, double Ec, double n0, double EZ) :
     bath(NBath), _alpha(alpha), _Ec(Ec), _n0(n0), _EZ(EZ) {};
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
   double _Gamma; // hybridisation strength
 public:
   hyb(double Gamma) : _Gamma(Gamma) {};
   auto Gamma() const { return _Gamma; }
   auto V(int NBath) const {
     std::vector<double> V;
     for (auto k: range1(NBath))
       V.push_back( std::sqrt( 2.0*_Gamma/(M_PI*NBath) ) );
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
   Real _E = 0; // eigenenergy
   MPS _psi;    // eigenvector
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
     H5Easy::dump(file, path + "/residuum", _residuum);
   }
};

class problem_type;
std::unique_ptr<problem_type> set_problem(std::string);

// parameters from the input file
struct params {
  string inputfn;       // filename of the input file
  InputGroup input;     // itensor input parser

//  string MPO = "std";   // which MPO representation to use
  std::unique_ptr<problem_type> problem = set_problem("std");

  int N;                // number of sites
  int NBath;            // number of bath sites
  int NImp;             // number of impurity orbitals
  int impindex;         // impurity position in the chain (1 based)

  Hubbard sites;        // itensor object

  // all bools have default value false
  bool computeEntropy;   // von Neumann entropy at the bond between impurity and next site. Works as intended if p.impindex=1.
  bool impNupNdn;        // print the number of up and dn electrons on the impurity
  bool chargeCorrelation;// compute the impurity-superconductor correlation <n_imp n_i>
  bool spinCorrelation;  // compute the impurity-superconductor correlation <S_imp S_i>
  bool pairCorrelation;  // compute the impurity-superconductor correlation <d d c_i^dag c_i^dag>
  bool hoppingExpectation;//compute the hopping expectation value 1/sqrt(N) \sum_sigma \sum_i <d^dag c_i> + <c^dag_i d>
  bool printTotSpinZ;    // prints total Nup, Ndn and Sz.

  bool calcweights;      // calculates the spectral weights of the two closes spectroscopically availabe excitations
  bool excited_state;    // computes the first excited state

  bool printDimensions;  // prints dmrg() prints info during the sweep
  bool refisn0;          // the energies will be computed in the sectors centered around the one with n = round(n0) + 1
  bool parallel;         // enables openMP parallel calculation of the for loop in findGS()
  bool verbose;          // verbosity level
  bool band_level_shift; // shifts the band levels for a constant in order to make H particle-hole symmetric


  double EnergyErrgoal; // the convergence value at which dmrg() will stop the sweeps; default is machine precision
  int nrH;              // number of times to apply H to psi before comencing the sweep - akin to a power method; default = 5
  int nrange;           // the number of energies computed is 2*nrange + 1

  std::unique_ptr<imp>    qd;
  std::unique_ptr<SCbath> sc;
  std::unique_ptr<hyb>    Gamma;
  double V12;           // QD-SC capacitive coupling

  std::unique_ptr<SCbath> sc1, sc2;
  std::unique_ptr<hyb> Gamma1, Gamma2;
  double SCSCinteraction = 0.0;  // test parameter for the 2 channel MPO

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

using H_t = std::tuple<MPO, double>;

class problem_type {
 public:
   virtual int imp_index(int) = 0;
   virtual H_t initH(int, params &) = 0;
};

class imp_first : virtual public problem_type
{
 public:
   int imp_index(int) override { return 1; }
};

class imp_middle : virtual public problem_type
{
 public:
   int imp_index(int NBath) override {
     my_assert(even(NBath));
     return 1+NBath/2;
   }
};

#include "SC_BathMPO.h"
#include "SC_BathMPO_Ec.h"
#include "SC_BathMPO_Ec_V.h"
#include "SC_BathMPO_MiddleImp.h"
#include "SC_BathMPO_MiddleImp_Ec.h"
#include "SC_BathMPO_MiddleImp_TwoChannel.h"

class single_channel : virtual public problem_type
{
 public:
   auto get_eps_V(auto & sc, auto & Gamma, params &p) {
     auto eps0 = sc->eps(p.band_level_shift);
     auto V0 = Gamma->V(sc->NBath());
     if (p.verbose) {
       std::cout << "eps=" << eps0 << std::endl;
       std::cout << "V=" << V0 << std::endl;
     }
     auto eps = shift1(eps0);
     auto V = shift1(V0);
     return std::make_pair(eps, V);
   }
};

template <class T>
  auto concat(const std::vector<T> &t1, const std::vector<T> &t2)
{
   std::vector<T> t;
   t.reserve(t1.size() + t2.size());
   t.insert(end(t), cbegin(t1), cend(t2));
   t.insert(end(t), cbegin(t2), cend(t2));
   return t;
}

class two_channel : virtual public problem_type
{
 public:
   auto get_eps_V(auto & sc1, auto & Gamma1, auto & sc2, auto & Gamma2, params &p) {
     auto eps1 = sc1->eps(p.band_level_shift);
     auto eps2 = sc2->eps(p.band_level_shift);
     auto V1 = Gamma1->V(sc1->NBath());
     auto V2 = Gamma2->V(sc2->NBath());
     if (p.verbose) {
       std::cout << "eps1=" << eps1 << std::endl << "eps2=" << eps2 << std::endl;
       std::cout << "V1=" << V1 << std::endl << "V2=" << V2 << std::endl;
     }
     auto eps = shift1(concat(eps1, eps2));
     auto V = shift1(concat(V1, V2));
     return std::make_pair(eps, V);
   }
};

namespace prob {
   class Std : public imp_first, public single_channel { // Note: avoid lowercase 'std'!!
    public:
      H_t initH(int ntot, params &p) override {
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites); // MPO is the hamiltonian in the "MPS-form"
        double Eshift = p.sc->Ec()*pow(ntot-p.sc->n0(),2) + p.qd->U()/2; // occupancy dependent effective energy shift
        double epseff = p.qd->eps() - 2.*p.sc->Ec()*(ntot-p.sc->n0()) + p.sc->Ec();
        double Ueff = p.qd->U() + 2.*p.sc->Ec();
        Fill_SCBath_MPO(H, eps, V, epseff, Ueff, p); // defined in SC_BathMPO.h, fills the MPO with the necessary entries
        return std::make_tuple(H, Eshift);
      }
   };

   class Ec : public imp_first, public single_channel {
    public:
      H_t initH(int ntot, params &p) override {
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc->Ec()*pow(p.sc->n0(), 2) + p.qd->U()/2;
        Fill_SCBath_MPO_Ec(H, eps, V, p);
        return std::make_tuple(H, Eshift);
      }
   };

   class Ec_V : public imp_first, public single_channel {
    public:
      H_t initH(int ntot, params &p) override {
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc->Ec()*pow(p.sc->n0(), 2) + p.V12 * p.sc->n0() * p.qd->nu() + p.qd->U()/2;
        double epseff = p.qd->eps() - p.V12 * p.sc->n0();
        double epsishift = -p.V12 * p.qd->nu();
        Fill_SCBath_MPO_Ec_V(H, eps, V, epseff, epsishift, p);
        return std::make_tuple(H, Eshift);
      }
   };

   class middle : public imp_middle, public single_channel {
    public:
      H_t initH(int ntot, params &p) override {
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc->Ec()*pow(ntot-p.sc->n0(),2) + p.qd->U()/2;
        double epseff = p.qd->eps() - 2.*p.sc->Ec()*(ntot-p.sc->n0()) + p.sc->Ec();
        double Ueff = p.qd->U() + 2.*p.sc->Ec();
        Fill_SCBath_MPO_MiddleImp(H, eps, V, epseff, Ueff, p);
        return std::make_tuple(H, Eshift);
      }
   };

   class middle_Ec : public imp_middle, public single_channel {
    public:
      H_t initH(int ntot, params &p) override {
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc->Ec()*pow(p.sc->n0(), 2) + p.qd->U()/2;
        Fill_SCBath_MPO_MiddleImp_Ec(H, eps, V, p);
        return std::make_tuple(H, Eshift);
      }
   };

   // For testing only!! This is the same as 'middle_Ec', but using the MPO for the
   // 2-ch problem. It uses Gamma for hybridisation, but alpha1,alpha2, etc. for
   // channel parameters. Use with care!
   class middle_2channel : public imp_middle, public single_channel {
    public:
      H_t initH(int ntot, params &p) override {
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc1->Ec()*pow(p.sc1->n0(), 2) + p.qd->U()/2;
        p.SCSCinteraction = 1.0; // IMPORTANT: single bath
        p.sc1->set_NBath(p.NBath); // override!
        p.sc2->set_NBath(p.NBath);
        Fill_SCBath_MPO_MiddleImp_TwoChannel(H, eps, V, p);
        return std::make_tuple(H, Eshift);
      }
   };

   class twoch : public imp_middle, public two_channel {
    public:
      H_t initH(int ntot, params &p) override {
        my_assert(even(p.NBath)); // in 2-ch problems, NBath is the total number of bath sites in both SCs !!
        my_assert(p.sc1->NBath() + p.sc2->NBath() == p.NBath);
        auto [eps, V] = get_eps_V(p.sc1, p.Gamma1, p.sc2, p.Gamma2, p);
        MPO H(p.sites);
        double Eshift = p.sc1->Ec()*pow(p.sc1->n0(), 2) + p.sc2->Ec()*pow(p.sc2->n0(), 2) + p.qd->U()/2;
        p.SCSCinteraction = 0.0; // IMPORTANT: separate baths
        Fill_SCBath_MPO_MiddleImp_TwoChannel(H, eps, V, p);
        return std::make_tuple(H, Eshift);
      }
   };
}

inline std::unique_ptr<problem_type> set_problem(std::string str)
{
  if (str == "std") return std::make_unique<prob::Std>();
  if (str == "Ec") return std::make_unique<prob::Ec>();
  if (str == "Ec_V") return std::make_unique<prob::Ec_V>();
  if (str == "middle") return std::make_unique<prob::middle>();
  if (str == "middle_Ec") return std::make_unique<prob::middle_Ec>();
  if (str == "middle_2channel") return std::make_unique<prob::middle_2channel>();
  if (str == "2ch") return std::make_unique<prob::twoch>();
  throw std::runtime_error("Unknown MPO type");
}

InputGroup parse_cmd_line(int, char * [], params &p);
void FindGS(InputGroup &input, store &s, params &p);
void calculateAndPrint(InputGroup &input, store &s, params &p);

#endif
