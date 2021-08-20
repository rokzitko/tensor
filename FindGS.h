#ifndef _calcGS_h_
#define _calcGS_h_

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <map>
#include <stdexcept>
#include <limits>
#include <tuple>
#include <cassert>
#include <utility>
#include <algorithm>
#include <execution>
#include <optional>

#include <gsl/gsl_assert>

#include <itertools/itertools.hpp>

#include <itensor/all.h>
#include <itensor/util/args.h>
#include <itensor/util/print_macro.h>
using namespace itensor;

#include "tdvp.h"
#include "basisextension.h"

#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#define H5_USE_BOOST
#include <highfive/H5Easy.hpp>

using complex_t = std::complex<double>;
using matrix_t = boost::numeric::ublas::matrix<double, boost::numeric::ublas::row_major>;

inline void h5_dump_matrix(H5Easy::File &file, const std::string &path, const matrix_t &m) {
  H5Easy::detail::createGroupsToDataSet(file, path);
  HighFive::DataSet dataset = file.createDataSet<double>(path, HighFive::DataSpace::From(m));
  dataset.write(m);
}

constexpr auto full = std::numeric_limits<double>::max_digits10;

using charge = int;
using spin = double;

constexpr auto spin0 = spin(0);
constexpr auto spinp = spin(0.5);
constexpr auto spinm = spin(-0.5);

using subspace_t = std::pair<charge, spin>;
using state_t = std::tuple<charge, spin, int>;

inline state_t gs(const subspace_t &sub)
{
  return {std::get<charge>(sub), std::get<spin>(sub), 0}; // ground state in the subspace
}

inline state_t es(const subspace_t &sub, const int n = 1) // es(sub,0) is the ground state!
{
  Expects(0 <= n);
  return {std::get<charge>(sub), std::get<spin>(sub), n}; // n-th excited state in the subspace
}

inline subspace_t subspace(const state_t &st) {
  return {std::get<0>(st), std::get<1>(st)};
}

inline void skip_line(std::ostream &o = std::cout)
{
  o << std::endl;
}

inline std::string Sz_string(const spin Sz) // custom formatting
{
    Expects(Sz == -1.0 || Sz == -0.5 || Sz == 0.0 || Sz == +0.5 || Sz == +1.0);
    if (Sz == -1.0) return "-1";
    if (Sz == -0.5) return "-0.5";
    if (Sz == 0.0) return "0";
    if (Sz == 0.5) return "0.5";
    if (Sz == 1.0) return "1";
    return "xxx";
}

inline auto subspace_path(const charge ntot, const spin Sz)
{
  return  fmt::format("{}/{}", ntot, Sz_string(Sz));
}

inline auto state_path(const charge ntot, const spin Sz, const int i)
{
  return fmt::format("{}/{}/{}", ntot, Sz_string(Sz), i);
}

inline auto state_path(const state_t st)
{
    const auto [ntot, Sz, i] = st;
    return state_path(ntot, Sz, i);
}

inline auto ij_path(const charge ntot, const spin Sz, const int i, const int j)
{
  return fmt::format("{}/{}/{}/{}", ntot, Sz_string(Sz), i, j);
}

class problem_type;
using type_ptr = std::unique_ptr<problem_type>;

using ndx_t = std::vector<int>;

template <typename T>
  inline H5Easy::DataSet dumpreal(H5Easy::File& file,
                        const std::string& path,
                        const std::complex<T>& data,
                        H5Easy::DumpMode mode = H5Easy::DumpMode::Create)
{
  const T realdata = std::real(data);
  return H5Easy::dump(file, path, realdata, mode);
}

template <typename T>
  inline H5Easy::DataSet dumpreal(H5Easy::File& file,
                        const std::string& path,
                        const std::vector<std::complex<T>>& data,
                        H5Easy::DumpMode mode = H5Easy::DumpMode::Create)
{
  std::vector<T> realdata;
  for (const auto &z : data)
    realdata.push_back(std::real(z));
  return H5Easy::dump(file, path, realdata, mode);
}

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

template< typename F, typename ...types >
  F for_all(F f, types &&... values)
{
  (f(std::forward<types>(values)), ...);
  return std::move(f);
}

template< typename F, typename ...types, std::size_t ...indices >
  F for_all_indices(F f, std::tuple<types...> const & t, std::index_sequence<indices...>)
{
  return for_all(std::move(f), std::get<indices>(t)...);
}

template< typename first, typename ...rest >
  std::ostream& operator<< (std::ostream& out, std::tuple<first, rest...> const & t)
{
  out << '[';
  for_all_indices([&out] (auto const & value) { out << value << ","; }, t, std::index_sequence_for<rest...>{});
  return out << std::get< sizeof...(rest) >(t) << ']';
}

// Convert 0-based vector to 1-based vector
inline auto shift1(const std::vector<double> &a) {
  std::vector<double> b;
  b.push_back(std::numeric_limits<double>::quiet_NaN());
  b.insert(b.end(), a.cbegin(), a.cend());
  return b;
}

// Vector of integers centered at nref
inline auto n_list(const int nref, const int nrange) {
  std::vector<int> n;
  n.push_back(nref);
  for (auto i : range1(nrange)) {
    n.push_back(nref+i);
    n.push_back(nref-i);
  }
  return n;
}

// Range of integers [a:b], end points included.
inline ndx_t range(int a, int b) // non-const because of swap!
{
  if (a > b) std::swap(a, b);
  ndx_t l(b - a + 1);
  std::iota(l.begin(), l.end(), a);
  return l;
}

// Concatenate two vectors
template<typename T>
auto concat(const std::vector<T> &t1, const std::vector<T> &t2)
{
   std::vector<T> t;
   t.reserve(t1.size() + t2.size());
   t.insert(t.end(), t1.cbegin(), t1.cend());
   t.insert(t.end(), t2.cbegin(), t2.cend());
   return t;
}

// Class containing impurity parameters
class imp {
 private:
   double _U;
   double _eps;
   double _EZ;
   double _EZx;
 public:
   imp(double U, double eps, double EZ, double EZx) : _U(U), _eps(eps), _EZ(EZ), _EZx(EZx) {};
   auto U() const { return _U; }
   auto eps() const { return _eps; }
   auto EZ() const { return _EZ; }
   auto EZx() const { return _EZx; }
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
   auto eps(bool flat_band = false, double flat_band_factor = 0) const {
     std::vector<double> eps;
     for (auto k: range1(_NBath))
       if (flat_band) {
        if (k < NBath()/2) eps.push_back(-1. * flat_band_factor);
        else if (k == NBath()/2) eps.push_back(0);
        else if (k > NBath()/2) eps.push_back(flat_band_factor); 
       }
       else eps.push_back( -_D + (k-0.5)*d() );
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
   double _EZx;    // Zeeman energy
   double _t;     // nearest neighbour hopping in SC
   double _lambda;// spin orbit coupling
 public:
   SCbath(int NBath, double alpha, double Ec, double n0, double EZ, double EZx, double t, double lambda) :
     bath(NBath), _alpha(alpha), _Ec(Ec), _n0(n0), _EZ(EZ), _EZx(EZx), _t(t), _lambda(lambda) {};
   auto alpha() const { return _alpha; }
   auto Ec() const { return _Ec; }
   auto n0() const { return _n0; }
   auto EZ() const { return _EZ; }
   auto EZx() const { return _EZx; }
   auto g() const { return _alpha*d(); }
   auto t() const { return _t; }
   auto lambda() const { return _lambda; }
   auto l() const { return _lambda*d(); }
   auto eps(bool band_level_shift = true, bool flat_band = false, double flat_band_factor = 0) const {
     auto eps = bath::eps(flat_band, flat_band_factor);
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
     return std::vector<double>(NBath, std::sqrt( 2.0*_Gamma/(M_PI*NBath) ));
   }
};

class eigenpair {
 private:
   Real _E = 0; // eigenenergy
   MPS _psi;    // eigenvector
 public:
   eigenpair() {}
   eigenpair(Real E, MPS psi) : _E(E), _psi(psi) {}
   auto E() const { return _E; }
   MPS & psi() { return _psi; } // not const !
   const MPS & psi() const { return _psi; }
};

class psi_stats {
 private:
   double _norm = 0;
   double _E = 0;
   double _deltaE2 = 0;
 public:
   psi_stats() {}
   psi_stats(const MPS &psi, MPO &H) {
     _norm = inner(psi, psi);
     _E = inner(psi, H, psi);
     _deltaE2 = inner(H, psi, H, psi) - pow(_E,2);
   }
   void dump() const {
     std::cout << fmt::format("norm: <psi|psi> = {}", _norm) << std::endl
       << fmt::format("E: <psi|H|psi> = {}", _E) << std::endl
       << fmt::format("deltaE^2: <psi|H^2|psi> - <psi|H|psi>^2 = {}", _deltaE2) << std::endl
       << fmt::format("rel error: (<psi|H^2|psi> - <psi|H|psi>^2)/<psi|H|psi>^2 = {}", _deltaE2/pow(_E,2)) << std::endl;
   }
};

// lists of quantities calculated in FindGS
struct store
{
  std::map<state_t, eigenpair> eigen;
  std::map<state_t, psi_stats> stats;
};

// parameters from the input file
struct params {
  string inputfn;       // filename of the input file
  InputGroup input;     // itensor input parser

  type_ptr problem;

  int N;                // number of sites
  int NBath;            // number of bath sites
  int NImp;             // number of impurity orbitals
  int impindex;         // impurity position in the chain (1 based)

  Hubbard sites;        // itensor object

  // all bools have default value false
  bool totalSpin;        // measure total spin of the state - uses autoMPO
  bool computeEntropy;   // von Neumann entropy at the bond between impurity and next site. Works as intended if p.impindex=1.
  bool impNupNdn;        // print the number of up and dn electrons on the impurity
  bool chargeCorrelation;// compute the impurity-superconductor correlation <n_imp n_i>
  bool spinCorrelation;  // compute the impurity-superconductor correlation <S_imp S_i>
  bool spinCorrelationMatrix;  // compute the full correlation matrix <S_i S_j>
  bool pairCorrelation;  // compute the impurity-superconductor correlation <d d c_i^dag c_i^dag>
  bool hoppingExpectation;//compute the hopping expectation value 1/sqrt(N) \sum_sigma \sum_i <d^dag c_i> + <c^dag_i d>
  bool printTotSpinZ;    // prints total Nup, Ndn and Sz
  bool transition_dipole_moment; // compute < i | nsc1 - ncs2 | j > 
  bool transition_quadrupole_moment; // compute < i | nsc1 + ncs2 | j > 
  bool overlaps;         // compute <i|j> overlap table in each subspace
  bool charge_susceptibility; // compute <i|nimp|j> overlap table in each subspace
  bool channel_parity;   // prints the channel parity

  bool calcweights;      // calculates the spectral weights of the two closes spectroscopically availabe excitations
  bool excited_state;    // computes excited state
  int excited_states;    // compute n excited states

  bool save;             // store computed energy/psi pair(s)
  int solve_ndx;         // which subproblem to solve [parsed directly from command line]
  int stop_n;            // stop calculation after level 'stop_n' has been computed
  int nrsweeps;          // number of DMRG sweeps to perform
  bool parallel;         // execution policy (also affects some defaults)
  bool Quiet, Silent;    // control output in dmrg()
  bool refisn0;          // the energies will be computed in the sectors centered around the one with n = round(n0) + 1
  bool verbose;          // verbosity level
  bool debug;            // debugging messages
  bool band_level_shift; // shifts the band levels for a constant in order to make H particle-hole symmetric
  bool sc_only;          // do not put any electrons on the QD in the initial state
  double Weight;         // parameter 'Weight' for the calculaiton of excited states

  double EnergyErrgoal;  // convergence value at which dmrg() will stop the sweeps; default is machine precision
  int nrH;               // number of times to apply H to psi before comencing the sweep - akin to a power method; default = 5
  int nref;              // central value of the occupancy.
  int nrange;            // number of occupancies considered is 2*nrange + 1, i.e. [nref-nrange:nref+nrange]
  bool spin1;            // include sz=1 for even charge sectors.
  bool flat_band;        // set energies of half of the levels to -flat_band_factor, and half of the levels to +flat_band_factor
  float flat_band_factor;

  bool chi;              // calculate Re[chi(omega_r+i eta_r)] by tabulating over [0:tau_max] in steps of tau_step
  double omega_r;
  double eta_r;
  double tau_max;
  double tau_step;
  int evol_krylovord;
  int evol_nr_expansion;
  int evol_nrsweeps;
  double evol_sweeps_cutoff;
  int evol_sweeps_maxdim;
  int evol_sweeps_niter;
  double evol_epsilonK1, evol_epsilonK2;
  int evol_numcenter;

  std::unique_ptr<imp>    qd;
  std::unique_ptr<SCbath> sc;
  std::unique_ptr<hyb>    Gamma;
  double V12;            // QD-SC capacitive coupling, for MPO_Ec_V
  double V1imp;          // QD-SC capacitive coupling, for MPO_2h_.*_V
  double V2imp;          // QD-SC capacitive coupling, for MPO_2h_.*_V
  double eta;            // coupling reduction factor, 0<=eta<=1, for MPO_Ec_eta

  bool magnetic_field() { return qd->EZ() != 0 || qd->EZx() != 0 || sc->EZ() != 0 || sc->EZx() != 0 || sc1->EZ() != 0 || sc1->EZx() != 0 || sc2->EZ() != 0 || sc2->EZx() != 0; } // true if there is magnetic field

  std::map<state_t, double> MFnSCs; // Maps an approximate nSC to a given state (n, Sz, i). Used for the iterative mean field calculation.

  double MF_precision;


  std::unique_ptr<SCbath> sc1, sc2;
  std::unique_ptr<hyb> Gamma1, Gamma2;
  double SCSCinteraction = 0.0;  // test parameter for the 2 channel MPO
};

class problem_type {
 public:
   virtual int imp_index() = 0;
   virtual ndx_t all_indexes() = 0;            // all indexes
   virtual ndx_t bath_indexes() = 0;           // all bath indexes
   virtual ndx_t bath_indexes(const int) = 0;  // per channel bath indexes
   virtual MPO initH(state_t, params &) = 0;
   virtual InitState initState(state_t, params &) = 0;
   virtual int numChannels() = 0;
   virtual bool spin_conservation() = 0;
};

class Sz_conserved : virtual public problem_type
{
 public:
   bool spin_conservation() override { return true; }
};

class Sz_non_conserved : virtual public problem_type
{
 public:
   bool spin_conservation() override { return false; }
};

class imp_first : virtual public problem_type
{
 private:
   int NBath;
 public:
   imp_first() = delete;
   imp_first(int _NBath) : NBath(_NBath) {}
   int imp_index() override { return 1; }
   ndx_t all_indexes() override { return range(1, NBath+1); }
   ndx_t bath_indexes() override { return range(2, NBath+1); }
   ndx_t bath_indexes(const int ch) override {
     Expects(ch == 1 || ch == 2);
     const auto ndx = bath_indexes();
     return ch == 1 ? ndx_t(ndx.begin(), ndx.begin() + NBath/2) : ndx_t(ndx.begin() + NBath/2, ndx.begin() + NBath);
   }
};

class imp_middle : virtual public problem_type
{
 private:
   int NBath;
 public:
   imp_middle() = delete;
   imp_middle(int _NBath) : NBath(_NBath) { Expects(even(NBath)); }
   int imp_index() override { return 1+NBath/2; }
   ndx_t all_indexes() override { return range(1, NBath+1); }
   ndx_t bath_indexes() override { return concat(range(1,NBath/2), range(2+NBath/2, NBath+1)); }
   ndx_t bath_indexes(const int ch) override {
     Expects(ch == 1 || ch == 2);
     const auto ndx = bath_indexes();
     return ch == 1 ? ndx_t(ndx.begin(), ndx.begin() + NBath/2) : ndx_t(ndx.begin() + NBath/2, ndx.begin() + NBath);
   }
};

inline void add_imp_electron(const double Sz, const int impindex, auto & state, charge & tot, spin & Sztot)
{
  if (Sz == -1 || Sz == -0.5) {
    state.set(impindex, "Dn");
    Sztot -= 0.5;
  }
  if (Sz == 0 || Sz == +0.5 || Sz == +1) {
    state.set(impindex, "Up");
    Sztot += 0.5;
  }
  tot++;
}

// nsc = number of electrons to add, Szadd = spin of the unpaired electron in the case of odd nsc
inline void add_bath_electrons(const int nsc, const spin & Szadd, const ndx_t &bath, auto & state, charge & tot, spin & Sztot)
{
  const size_t npair = nsc/2;            // number of pairs to add
  Expects(bath.size() >= npair);
  for (size_t j = 0; j < npair; j++)
    state.set(bath[j], "UpDn");
  tot += npair*2;                        // Sztot does not change!
  if (odd(nsc)) {                        // if ncs is odd, add one electron
    Expects(bath.size() >= npair+1);
    const auto i = bath[npair];          // note: vector bath is 0-based
    if (Szadd == 0.5)
      state.set(i, "Up");
    else if (Szadd == -0.5)
      state.set(i, "Dn");
    else throw std::runtime_error("oops! should not happen!");
    tot++;
    Sztot += Szadd;
  }
}

// The functions in these headers take a class params argument
#include "SC_BathMPO.h"
#include "SC_BathMPO_Ec.h"
#include "SC_BathMPO_Ec_MF.h"
#include "SC_BathMPO_Ec_V.h"
#include "SC_BathMPO_Ec_SO.h"
#include "SC_BathMPO_MiddleImp.h"
#include "SC_BathMPO_MiddleImp_Ec.h"
#include "SC_BathMPO_t_SConly.h"
#include "SC_BathMPO_Ec_t.h"
#include "SC_BathMPO_Ec_eta.h"
#include "SC_BathMPO_MiddleImp_TwoChannel.h"
#include "SC_BathMPO_ImpFirst_TwoChannel.h"
#include "SC_BathMPO_ImpFirst_TwoChannel_V.h"
#include "SC_BathMPO_ImpFirst_TwoChannel_hopping.h"
#include "autoMPO_1ch.h"
#include "autoMPO_1ch_so.h"
#include "autoMPO_2ch.h"

class single_channel : virtual public problem_type
{
 public:
   auto get_eps_V(auto & sc, auto & Gamma, params &p) const {
     auto eps0 = sc->eps(p.band_level_shift, p.flat_band, p.flat_band_factor);
     auto V0 = Gamma->V(sc->NBath());
     if (p.verbose) {
       std::cout << "eps=" << eps0 << std::endl;
       std::cout << "V=" << V0 << std::endl;
     }
     auto eps = shift1(eps0);
     auto V = shift1(V0);
     return std::make_pair(eps, V);
   }
   InitState initState(state_t st, params &p) override {
     const auto [ntot, Sz, ii] = st; // Sz is the z-component of the total spin.
     Expects(0 <= ntot && ntot <= 2*p.N);
     Expects(Sz == -1 || Sz == -0.5 || Sz == 0 || Sz == +0.5 || Sz == +1);
     int tot = 0;      // electron counter, for assertion test
     double Sztot = 0; // SZ counter, for assertion test
     auto state = InitState(p.sites);
     const auto nimp = p.sc_only ? 0 : 1;        // number of electrons in the impurity level
     const auto nsc = p.sc_only ? ntot : ntot-1; // number of electrons in the bath
     Ensures(nimp + nsc == ntot);
     // ** Add electron to the impurity site
     if (nimp)
       add_imp_electron(Sz, p.impindex, state, tot, Sztot);
     // ** Add electrons to the bath
     if (nsc) {
       ndx_t bath_sites = p.problem->bath_indexes();
       add_bath_electrons(nsc, Sz-Sztot, bath_sites, state, tot, Sztot);
     }
     Ensures(tot == ntot);
     Ensures(Sztot == Sz);
     return state;
   }
   int numChannels() {
    return 1;
   }
};

class single_channel_eta : public single_channel
{
 public:
   double y(const int i, const params &p) const {
     const auto L = p.NBath;
     assert(0.0 <= p.eta && p.eta <= 1.0);
     assert(i >= 1 && i <= L); // 1 is special site, 2,3,...,L are regular sites
     return i == 1 ? p.eta : sqrt( (L-p.eta*p.eta)/(L-1.) );
   }

   auto get_eps_V(auto & sc, auto & Gamma, params &p) const {
     auto eps0 = sc->eps(false, p.flat_band, p.flat_band_factor); // no band_level_shift here
     if (p.band_level_shift) { // we do it here, in a site-dependent way
       const auto L = p.NBath;
       for (unsigned int i = 0; i < eps0.size(); i++) 
         eps0[i] += -sc->g()/2.0 * pow(y(i+1, p), 2);
     }
     auto V0 = Gamma->V(sc->NBath());
     if (p.verbose) {
       std::cout << "eta=" << p.eta << std::endl;
       std::cout << "eps=" << eps0 << std::endl;
       std::cout << "V=" << V0 << std::endl;
     }
     auto eps = shift1(eps0);
     auto V = shift1(V0);
     return std::make_pair(eps, V);
   }
};

class two_channel : virtual public problem_type
{
 public:
   auto get_eps_V(auto & sc1, auto & Gamma1, auto & sc2, auto & Gamma2, params &p) const {
     auto eps1 = sc1->eps(p.band_level_shift, p.flat_band, p.flat_band_factor);
     auto eps2 = sc2->eps(p.band_level_shift, p.flat_band, p.flat_band_factor);
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
   InitState initState(state_t st, params &p) override {
     const auto [ntot, Sz, ii] = st; // Sz is the z-component of the total spin.
     Expects(0 <= ntot && ntot <= 2*p.N);
     Expects(Sz == -1 || Sz == -0.5 || Sz == 0 || Sz == +0.5 || Sz == +1);
     int tot = 0;      // electron counter, for assertion test
     double Sztot = 0; // SZ counter, for assertion test
     auto state = InitState(p.sites);

     const auto nimp = p.sc_only ? 0 : 1;        // number of electrons in the impurity level
     const auto nsc = p.sc_only ? ntot : ntot-1; // number of electrons in the bath
     auto nsc1 = nsc/2;                          // number of electrons in bath 1
     if (odd(nsc1) && nsc1) nsc1--;
     const auto nsc2 = nsc-nsc1;                 // number of electrons in bath 2
     Ensures(nimp + nsc1 + nsc2 == ntot);

     // ** Add electron to the impurity site
     if (nimp)
       add_imp_electron(Sz, p.impindex, state, tot, Sztot);
     // ** Add electrons to bath 1
     if (nsc1) {
       Expects(even(nsc1));
       ndx_t bath_sites = p.problem->bath_indexes(1);
       add_bath_electrons(nsc1, spin0, bath_sites, state, tot, Sztot);
     }
     if (nsc2) {
       ndx_t bath_sites = p.problem->bath_indexes(2);
       add_bath_electrons(nsc2, Sz-Sztot, bath_sites, state, tot, Sztot);
     }

     Ensures(tot == ntot);
     Ensures(Sztot == Sz);
     return state;
   }
   int numChannels() {
    return 2;
   }
};

namespace prob {
   class Std : public imp_first, public single_channel, public Sz_conserved { // Note: avoid lowercase 'std'!!
    public:
      Std(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=Std" << std::endl;
        auto [ntot, Sz, i] = st;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites); // MPO is the hamiltonian in the "MPS-form"
        double Eshift = p.sc->Ec()*pow(ntot-p.sc->n0(),2) + p.qd->U()/2; // occupancy dependent effective energy shift
        double epseff = p.qd->eps() - 2.*p.sc->Ec()*(ntot-p.sc->n0()) + p.sc->Ec();
        double Ueff = p.qd->U() + 2.*p.sc->Ec();
        Fill_SCBath_MPO(H, Eshift, eps, V, epseff, Ueff, p); // defined in SC_BathMPO.h, fills the MPO with the necessary entries
        return H;
      }
   };

   class Ec : public imp_first, public single_channel, public Sz_conserved {
    public:
      Ec(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=Ec" << std::endl;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc->Ec()*pow(p.sc->n0(), 2) + p.qd->U()/2;
        Fill_SCBath_MPO_Ec(H, Eshift, eps, V, p);
        return H;
      }
   };

   class Ec_MF : public imp_first, public single_channel, public Sz_conserved {
    public:
      Ec_MF(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=Ec_MF" << std::endl;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);

        double Eshift = p.sc->Ec() * ( pow(p.sc->n0(), 2) - pow(p.MFnSCs[st], 2) ) + p.qd->U()/2;
        Fill_SCBath_MPO_Ec_MF(H, p.MFnSCs[st], Eshift, eps, V, p);
        return H;
      }
   };

   class Ec_V : public imp_first, public single_channel, public Sz_conserved  {
    public:
      Ec_V(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=Ec_V" << std::endl;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc->Ec()*pow(p.sc->n0(), 2) + p.V12 * p.sc->n0() * p.qd->nu() + p.qd->U()/2;
        double epseff = p.qd->eps() - p.V12 * p.sc->n0();
        double epsishift = -p.V12 * p.qd->nu();
        Fill_SCBath_MPO_Ec_V(H, Eshift, eps, V, epseff, epsishift, p);
        return H;
      }
   };

   class Ec_SO : public imp_first, public single_channel, public Sz_non_conserved  {
    public:
      Ec_SO(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=Ec_SO" << std::endl;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc->Ec()*pow(p.sc->n0(), 2) + p.qd->U()/2;
        Fill_SCBath_MPO_Ec_SO(H, Eshift, eps, V, p);
        return H;
      }
   };

   class middle : public imp_middle, public single_channel, public Sz_conserved  {
    public:
      middle(const params &p) : imp_middle(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=middle" << std::endl;
        auto [ntot, Sz, i] = st;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc->Ec()*pow(ntot-p.sc->n0(),2) + p.qd->U()/2;
        double epseff = p.qd->eps() - 2.*p.sc->Ec()*(ntot-p.sc->n0()) + p.sc->Ec();
        double Ueff = p.qd->U() + 2.*p.sc->Ec();
        Fill_SCBath_MPO_MiddleImp(H, Eshift, eps, V, epseff, Ueff, p);
        return H;
      }
   };

   class middle_Ec : public imp_middle, public single_channel, public Sz_conserved  {
    public:
      middle_Ec(const params &p) : imp_middle(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=middle_Ec" << std::endl;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc->Ec()*pow(p.sc->n0(), 2) + p.qd->U()/2;
        Fill_SCBath_MPO_MiddleImp_Ec(H, Eshift, eps, V, p);
        return H;
      }
   };

   class t_SConly : public imp_first, public single_channel, public Sz_conserved  {
    public:
      t_SConly(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        //Expects(p.sc_only);
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=t_SConly" << std::endl;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc->Ec()*pow(p.sc->n0(), 2);
        Fill_SCBath_MPO_t_SConly(H, Eshift, eps, p);
        return H;
      }
   };

   class Ec_t : public imp_first, public single_channel, public Sz_conserved  {
    public:
      Ec_t(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=Ec_t" << std::endl;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc->Ec()*pow(p.sc->n0(), 2) + p.qd->U()/2;
        Fill_SCBath_MPO_Ec_t(H, Eshift, eps, V, p);
        return H;
      }
   };
   class Ec_eta : public imp_first, public single_channel_eta, public Sz_conserved  {
    public:
      Ec_eta(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=Ec_eta" << std::endl;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc->Ec()*pow(p.sc->n0(), 2) + p.qd->U()/2;
        Fill_SCBath_MPO_Ec_eta(H, Eshift, eps, V, p);
        return H;
      }
   };
   // For testing only!! This is the same as 'middle_Ec', but using the MPO for the
   // 2-ch problem. It uses Gamma for hybridisation, but alpha1,alpha2, etc. for
   // channel parameters. Use with care!
   class middle_2channel : public imp_middle, public single_channel, public Sz_conserved  {
    public:
      middle_2channel(const params &p) : imp_middle(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=middle_2channel" << std::endl;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.sc1->Ec()*pow(p.sc1->n0(), 2) + p.qd->U()/2;
        p.SCSCinteraction = 1.0; // IMPORTANT: single bath
        p.sc1->set_NBath(p.NBath); // override!
        p.sc2->set_NBath(p.NBath);
        Fill_SCBath_MPO_MiddleImp_TwoChannel(H, Eshift, eps, V, p);
        return H;
      }
   };

    class autoH_1ch : public imp_first, public single_channel, public Sz_conserved  {
    public:
      autoH_1ch(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=autoH_1ch" << std::endl;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.qd->U()/2;
        get_autoMPO_1ch(H, Eshift, eps, V, p);
        return H;
      }
   };

    class autoH_1ch_so : public imp_first, public single_channel, public Sz_conserved  {
    public:
      autoH_1ch_so(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=autoH_1ch_so" << std::endl;
        auto [eps, V] = get_eps_V(p.sc, p.Gamma, p);
        MPO H(p.sites);
        double Eshift = p.qd->U()/2;
        get_autoMPO_1ch_so(H, Eshift, eps, V, p);
        return H;
      }
   };

    class autoH_2ch : public imp_first, public two_channel, public Sz_conserved  {
    public:
      autoH_2ch(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=autoH_2ch" << std::endl;
        auto [eps, V] = get_eps_V(p.sc1, p.Gamma1, p.sc2, p.Gamma2, p);
        MPO H(p.sites);
        double Eshift = p.qd->U()/2;
        get_autoMPO_2ch(H, Eshift, eps, V, p);
        return H;
      }
   };

   class twoch : public imp_middle, public two_channel, public Sz_conserved  {
    public:
      twoch(const params &p) : imp_middle(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=twoch" << std::endl;
        Expects(even(p.NBath)); // in 2-ch problems, NBath is the total number of bath sites in both SCs !!
        Expects(p.sc1->NBath() + p.sc2->NBath() == p.NBath);
        auto [eps, V] = get_eps_V(p.sc1, p.Gamma1, p.sc2, p.Gamma2, p);
        MPO H(p.sites);
        double Eshift = p.sc1->Ec()*pow(p.sc1->n0(), 2) + p.sc2->Ec()*pow(p.sc2->n0(), 2) + p.qd->U()/2;
        p.SCSCinteraction = 0.0; // IMPORTANT: separate baths
        Fill_SCBath_MPO_MiddleImp_TwoChannel(H, Eshift, eps, V, p);
        return H;
      }
   };

   class twoch_impfirst : public imp_first, public two_channel, public Sz_conserved  {
    public:
      twoch_impfirst(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=twoch_impfirst" << std::endl;
        Expects(even(p.NBath)); // in 2-ch problems, NBath is the total number of bath sites in both SCs !!
        Expects(p.sc1->NBath() + p.sc2->NBath() == p.NBath);
        Expects(p.sc1->NBath() == p.sc2->NBath());
        auto [eps, V] = get_eps_V(p.sc1, p.Gamma1, p.sc2, p.Gamma2, p);
        MPO H(p.sites);
        double Eshift = p.sc1->Ec()*pow(p.sc1->n0(), 2) + p.sc2->Ec()*pow(p.sc2->n0(), 2) + p.qd->U()/2;
        Fill_SCBath_MPO_ImpFirst_TwoChannel(H, Eshift, eps, V, p);
        return H;
      }
   };


  class twoch_impfirst_V : public imp_first, public two_channel, public Sz_conserved  {
    public:
      twoch_impfirst_V(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=twoch_impfirst_V" << std::endl;
        Expects(even(p.NBath)); // in 2-ch problems, NBath is the total number of bath sites in both SCs !!
        Expects(p.sc1->NBath() + p.sc2->NBath() == p.NBath);
        Expects(p.sc1->NBath() == p.sc2->NBath());
        auto [eps, V] = get_eps_V(p.sc1, p.Gamma1, p.sc2, p.Gamma2, p);
        MPO H(p.sites);
        double Eshift = p.sc1->Ec()*pow(p.sc1->n0(), 2) + p.sc2->Ec()*pow(p.sc2->n0(), 2) + p.qd->U()/2 + p.V1imp * p.sc1->n0() * p.qd->nu() + p.V2imp * p.sc2->n0() * p.qd->nu();
        double epsishift1 = - p.V1imp * p.qd->nu();
        double epsishift2 = - p.V2imp * p.qd->nu();
        double epseff = p.qd->eps() - p.V1imp * p.sc1->n0() - p.V2imp * p.sc2->n0();
        Fill_SCBath_MPO_ImpFirst_TwoChannel_V(H, Eshift, epsishift1, epsishift2, epseff, eps, V, p);
        return H;
      }
   };

   class twoch_hopping : public imp_first, public two_channel, public Sz_conserved  {
    public:
      twoch_hopping(const params &p) : imp_first(p.NBath) {}
      MPO initH(state_t st, params &p) override {
        if (p.verbose) std::cout << "Building Hamiltonian, MPO=twoch_hopping" << std::endl;
        Expects(even(p.NBath)); // in 2-ch problems, NBath is the total number of bath sites in both SCs !!
        Expects(p.sc1->NBath() + p.sc2->NBath() == p.NBath);
        Expects(p.sc1->NBath() == p.sc2->NBath());
        auto [eps, V] = get_eps_V(p.sc1, p.Gamma1, p.sc2, p.Gamma2, p);
        MPO H(p.sites);
        double Eshift = p.sc1->Ec()*pow(p.sc1->n0(), 2) + p.sc2->Ec()*pow(p.sc2->n0(), 2) + p.qd->U()/2;
        Fill_SCBath_MPO_ImpFirst_TwoChannel_hopping(H, Eshift, eps, V, p);
        return H;
      }
   };
}

inline type_ptr set_problem(const std::string str, const params &p)
{
  if (str == "std") return std::make_unique<prob::Std>(p);
  if (str == "Ec") return std::make_unique<prob::Ec>(p);
  if (str == "Ec_MF") return std::make_unique<prob::Ec_MF>(p);
  if (str == "Ec_V") return std::make_unique<prob::Ec_V>(p);
  if (str == "Ec_SO") return std::make_unique<prob::Ec_SO>(p);
  if (str == "middle") return std::make_unique<prob::middle>(p);
  if (str == "middle_Ec") return std::make_unique<prob::middle_Ec>(p);
  if (str == "t_SConly") return std::make_unique<prob::t_SConly>(p);
  if (str == "Ec_t") return std::make_unique<prob::Ec_t>(p);
  if (str == "Ec_eta") return std::make_unique<prob::Ec_eta>(p);
  if (str == "autoH") return std::make_unique<prob::autoH_1ch>(p);
  if (str == "autoH_so") return std::make_unique<prob::autoH_1ch_so>(p);
  if (str == "autoH_2ch") return std::make_unique<prob::autoH_2ch>(p);
  if (str == "middle_2channel") return std::make_unique<prob::middle_2channel>(p);
  if (str == "2ch") return std::make_unique<prob::twoch>(p);
  if (str == "2ch_impFirst") return std::make_unique<prob::twoch_impfirst>(p);
  if (str == "2ch_impFirst_V") return std::make_unique<prob::twoch_impfirst_V>(p);
  if (str == "2ch_t") return std::make_unique<prob::twoch_hopping>(p);
  throw std::runtime_error("Unknown MPO type");
}

void parse_cmd_line(int, char * [], params &);
std::vector<subspace_t> init_subspace_lists(params &p);
void solve(const std::vector<subspace_t> &l, store &s, params &);
void process_and_save_results(store &, params &, std::string = "solution.h5");

std::vector<double> calcOccupancy(MPS &psi, const ndx_t &all_sites, const params &p); // this has to be defined here as it is used in MF_selfConsistent_calcGS.cc

#endif
