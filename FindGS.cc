#include "FindGS.h"
#include <cxxabi.h>
#include <type_traits>

std::vector<subspace_t> init_subspace_lists(params &p)
{
  const int nhalf = p.N;  // total nr of electrons at half-filling
  int nref = nhalf;       // default
  if (p.nref >= 0)        // override
    nref = p.nref;
  else if (p.refisn0){   // adaptable
    if ( p.problem->type() == "single channel impurity problem" )   nref = round(p.sc->n0() + 0.5 - (p.qd->eps()/p.qd->U())); // calculation of the energies is centered around this n
    else if ( p.problem->type() == "two channel impurity problem" ) nref = round(p.sc1->n0() + p.sc2->n0() + 0.5 - (p.qd->eps()/p.qd->U())); // calculation of the energies is centered around this n
    else if ( p.problem->type() == "chain problem" ) {
      auto temp_n = 0;
      for (auto &  x : p.chain_scis) temp_n += x->n0();
      for (auto &  x : p.chain_imps) temp_n += 0.5 - (x->eps()/x->U());
      nref = round(temp_n);
    }
    else if ( p.problem->type() == "qd-sc-qd problem" ) nref = round(p.sc->n0() + (0.5 - (p.qd_L->eps()/p.qd_L->U())) + (0.5 - (p.qd_R->eps()/p.qd_R->U())) );
  }
  std::vector<subspace_t> l;
  std::cout << "Constructing the list of subspace: " << std::endl;
  for (const auto &ntot : n_list(nref, p.nrange)) {
    spin szmin, szmax;
    // set up the defaults...
    if (p.problem->spin_conservation()) {
      szmax = even(ntot) ? ( p.spin1 ? 1 : 0) : 0.5;
      szmin = p.magnetic_field() ? -szmax : (even(ntot) ? 0 : 0.5);
    } else { // if there is no spin conservation, we ignore the parameter spin1 and the magnetic_field() setting
      szmax = even(ntot) ? 0 : 0.5;
      szmin = szmax;
    }
    // ...and override, if so requested
    if (p.sz_override) {
      szmax = even(ntot) ? p.sz_even_max : p.sz_odd_max;
      szmin = even(ntot) ? p.sz_even_min : p.sz_odd_min;
    }
    // half-integers are exactly representable as floating points, thus the following loop will work correctly
    for (spin sz = szmin; sz <= szmax; sz += 1.0) {
      assert(abs(abs(2.0*sz)-int(abs(2.0*sz))) < 1e-10);
      std::cout << " " << ntot << " " << sz << std::endl;
      l.push_back({ntot, sz});
    }
  }
  std::cout << std::endl;
  return l;
}

template<typename T> void my_assert(const bool condition, T message) {
  if (!condition) {
    std::cout << "Failed assertion: " << message << std::endl;
    exit(1);
  }
}

// Example of conditional compilation
// TO DO: to be fixed!
template<typename T>
void report_Sz_conserved(T *prob) {
  if constexpr (std::is_base_of_v<Sz_conserved, T>) {
    std::cout << "spin is conserved\n";
  } else {
    std::cout << "spin is not conserved\n";
  }
}

// parse pairing strength into a vector of ys. The pairing alpha_ij = alpha * y[i] * y[j]
auto parse_ys(auto& input, const int Nlevels, const std::string channel = "") {
  std::vector<double> yvec;
  const std::string y_ch = "y" + channel; // for the two channel problem this is "y1" and "y2", and "y" for single channel.
  std::cout.setstate(std::ios_base::failbit); // stops the output to stdout, from https://stackoverflow.com/questions/30184998/how-to-disable-cout-output-in-the-runtime
  const auto default_value = input.getReal(y_ch, 1.0); // NOTE: default is 1
  for (int i = 1; i <= Nlevels; i++)
    yvec.push_back(input.getReal(y_ch + "_" + std::to_string(i), default_value)); // parse y_NN from the input, if not found fall back to y.
  std::cout.clear();
  // print out all y in the same line:
  std::cout << "Got " << y_ch << ": ";
  for (auto &x : shift1(yvec)) std::cout << x << " ";
  std::cout << "\n";
  return shift1(yvec); // convert to 1-based vector
}

// v_NN is the transition rate from the impurity to the NNth level in the bath. Parsed into a map here and merged with V in class hyb.
std::map<int, double> parse_special_levels(auto& input, const int Nlevels, const std::string which, const std::string channel = ""){
  std::map<int, double> special_map;
  const std::string v_ch = which + channel; // which is the name of parameter, as of Jan 2021 "v" or "eps". For the two channel problem this is "v1" and "v2", and "v" for single channel.
  std::cout.setstate(std::ios_base::failbit);
  for (int i = 1; i <= Nlevels; i++){
    double v = input.getReal(v_ch + "_" + std::to_string(i), std::numeric_limits<double>::quiet_NaN()); // have NaN as the default value
    if (!std::isnan(v)) special_map.insert(std::pair<int, double>(i, v)); // check if this v_i value was given (ie. is not Nan by default), and add it to the map
  }
  std::cout.clear();
  std::cout << "Got " << v_ch << ": \n";
  for (auto const & x : special_map){
    std::cout << v_ch << "_" << x.first << " = " << x.second << "\n";
  }
  return special_map;
}

auto parse_chain_sci(const int i, auto & input, params & p){
  const std::string stri = std::to_string(i);
  // default values
  auto alpha = input.getReal("alpha", 0.);
  auto ys = parse_ys(input, p.SClevels, stri);
  auto eps = parse_special_levels(input, p.SClevels, "eps");
  auto Ec = input.getReal("Ec", 0.);
  auto n0 = input.getReal("n0", p.SClevels);
  auto EZ = input.getReal("EZ_bulk", 0.);
  auto EZx = input.getReal("EZx_bulk", 0.);
  auto t = input.getReal("t", 0.);
  auto lambda = input.getReal("lambda", 0.);
  return std::make_unique<SCbath>(p.SClevels, p.D, input.getReal("alpha"+stri, alpha), parse_ys(input, p.SClevels, stri), parse_special_levels(input, p.SClevels, "eps", stri), input.getReal("Ec"+stri, Ec), 
    input.getReal("n0"+stri, n0), input.getReal("EZ_bulk"+stri, EZ), input.getReal("EZx_bulk"+stri, EZx), input.getReal("t"+stri, t), input.getReal("lambda"+stri, lambda));
}

auto parse_chain_imp(const int i, auto & input, params & p){
  const std::string stri = std::to_string(i);
  auto U = input.getReal("U", 0);
  auto epsimp = input.getReal("epsimp", -U/2.);
  auto EZ = input.getReal("EZ_imp", 0.);
  auto EZx = input.getReal("EZx_imp", 0.);
  return std::make_unique<imp>(input.getReal("U"+stri, U), input.getReal("epsimp"+stri, epsimp), input.getReal("EZ_imp"+stri, EZ), input.getReal("EZx_imp"+stri, EZx));
}

auto parse_chain_hyb(const int i, auto &input, params & p){
  const std::string stri = std::to_string(i);
  auto gamma = input.getReal("gamma", 0.);
  return std::make_unique<hyb>(input.getReal("gamma"+stri, gamma), parse_special_levels(input, p.SClevels, "v", stri));
}

void parse_cmd_line(int argc, char *argv[], params &p) {
  if (!(argc == 2 || argc == 3 || argc == 4))
    throw std::runtime_error("Please provide input file. Usage: executable <input file> [solve_ndx] [stop_n]");
  p.solve_ndx = argc >= 3 ? atoi(argv[2]) : -1;
  p.stop_n = argc >= 4 ? atoi(argv[3]) : INT_MAX;
  p.inputfn = { argv[1] };                                 // read parameters from the input file
  auto input = InputGroup{p.inputfn, "params"};            // get input parameters using InputGroup from itensor
  p.N = input.getInt("N", 0);

  // TO DO: FIX THIS MESS WITH NBath INITIALIZATION (Luka, JAN 2023)
  // To call set_problem() the NBath value has to be set. But it depends on the number of imp levels in the system, which depends on the problem type.
  // Here define a dummy_problem, read out the number of imp levels - 

  // for now: if the mpo is qd_sc_qd, override chainLen, NImp and NSC manually.
  std::string mpoType = input.getString("MPO", "std");
  p.chainLen = input.getInt("chainLen", 0); // number of elements in a SC-QD-... chain

  std::cout << "HERE\n";

  auto dummy_problem = set_problem(input.getString("MPO", "std"), p); // set a dummy problem only to get NImp and NSC, and with it NBath or N
  p.NImp = dummy_problem->NImp();
  p.NSC = dummy_problem->NSC();  
  
  if (p.N != 0){
    p.NBath = p.N - p.NImp; 
  }
  else { // N not specified, try NBath
    p.NBath = input.getInt("NBath", 0);
    if (p.NBath == 0) throw std::runtime_error("specify either N or NBath!");
    p.N = p.NBath + p.NImp;
  }
  p.SClevels = p.NBath / p.NSC;

  // NOW DEFINE THE REAL PROBLEM, p.NBath is set
  p.problem = set_problem(input.getString("MPO", "std"), p);  // problem type

  p.impindex = p.problem->imp_index(); // XXX: redundant? // Jan 2023 - this is messy. Chain and qd-sc-qd have multiple imps. p.impindex defaults to 1 in those cases.

  p.D = input.getReal("D", 1.0);
  std::cout << "N=" << p.N << " NBath=" << p.NBath << " D=" << p.D << " impindex=" << p.impindex << std::endl;

  // parameters entering the problem definition
  const double U = input.getReal("U", 0); // need to parse it first because it enters the default value for epsimp just below
  p.qd = std::make_unique<imp>(U, input.getReal("epsimp", -U/2.), input.getReal("EZ_imp", 0.), input.getReal("EZx_imp", 0.));
  p.Gamma = std::make_unique<hyb>(input.getReal("gamma", 0), parse_special_levels(input, p.NBath, "v", ""));

  p.sc = std::make_unique<SCbath>(p.NBath, p.D, input.getReal("alpha", 0.), parse_ys(input, p.NBath, ""), parse_special_levels(input, p.NBath, "eps"), input.getReal("Ec", 0), input.getReal("n0", p.NBath), input.getReal("EZ_bulk", 0.), input.getReal("EZx_bulk", 0.), input.getReal("t", 0.), input.getReal("lambda", 0.));

  p.eta = input.getReal("eta", 1.0);
  p.etasite = input.getInt("etasite", p.NBath/2); // Fermi-level !
  p.etarescale = input.getYesNo("etarescale", true);
  p.band_level_shift = input.getYesNo("band_level_shift", false);

  p.V12 = input.getReal("V", 0); // handled in a special way
  p.V1imp = input.getReal("V1imp", 0); // capacitive coupling between sc1 and imp
  p.V2imp = input.getReal("V2imp", 0); // capacitive coupling between sc2 and imp
  p.tQD = input.getReal("tQD", 0); // hopping between the two qds in the qd-sc-qd problem

  // parameters for the 2-channel problem
  p.sc1 = std::make_unique<SCbath>(p.NBath/2, p.D, input.getReal("alpha1", 0.), parse_ys(input, p.NBath/2, "1"), parse_special_levels(input, p.NBath/2, "eps", "1"), input.getReal("Ec1", 0), input.getReal("n01", (p.N-1)/2), input.getReal("EZ_bulk1", 0), input.getReal("EZx_bulk1", 0.), input.getReal("t1", 0), input.getReal("lambda1", 0.));
  p.sc2 = std::make_unique<SCbath>(p.NBath/2, p.D, input.getReal("alpha2", 0.), parse_ys(input, p.NBath/2, "2"), parse_special_levels(input, p.NBath/2, "eps", "2"), input.getReal("Ec2", 0), input.getReal("n02", (p.N-1)/2), input.getReal("EZ_bulk2", 0), input.getReal("EZx_bulk2", 0.), input.getReal("t2", 0), input.getReal("lambda2", 0.));
  p.Gamma1 = std::make_unique<hyb>(input.getReal("gamma1", 0), parse_special_levels(input, p.NBath/2, "v", "1"));
  p.Gamma2 = std::make_unique<hyb>(input.getReal("gamma2", 0), parse_special_levels(input, p.NBath/2, "v", "2"));

  // parameters for the SC-QD-... chain problem
  for (int i = 1; i <= p.NSC; i++ ){
    p.chain_scis.push_back( parse_chain_sci(i, input, p) );
  }
  for (int i = 1; i <= p.NImp; i++){
    p.chain_imps.push_back( parse_chain_imp(i, input, p) );
  }
  for (int i = 1; i <= p.chainLen - 1; i++){
    p.chain_hybs.push_back( parse_chain_hyb(i, input, p) );
  }

  // parameters for the qd-sc-qd problem
  // p.sc is already parsed above
  // The left and right qd parameters are parsed here. The default value is always the parameter without _L or _R. Thus specifying just those (like for the normal sc-qd problem) will give two equal qds.
  const double U_L = input.getReal("U_L", p.qd->U());
  const double U_R = input.getReal("U_R", p.qd->U());
  p.qd_L = std::make_unique<imp>(U_L, input.getReal("epsimp_L", -U_L/2.), input.getReal("EZ_imp_L", p.qd->EZ()), input.getReal("EZx_imp_L", p.qd->EZx()));
  p.qd_R = std::make_unique<imp>(U_R, input.getReal("epsimp_R", -U_R/2.), input.getReal("EZ_imp_R", p.qd->EZ()), input.getReal("EZx_imp_R", p.qd->EZx()));

  double gamma = input.getReal("gamma", 0.); // parse the default gamma again
  p.Gamma_L = std::make_unique<hyb>(input.getReal("gamma_L", gamma), parse_special_levels(input, p.SClevels, "v", "L"), parse_special_levels(input, p.SClevels, "iv", "L"));
  p.Gamma_R = std::make_unique<hyb>(input.getReal("gamma_R", gamma), parse_special_levels(input, p.SClevels, "v", "R"), parse_special_levels(input, p.SClevels, "iv", "R"));

  // parameters controlling the calculation targets
  p.nref = input.getInt("nref", -1);
  p.nrange = input.getInt("nrange", 1);
  p.refisn0 = input.getYesNo("refisn0", false);
  p.spin1 = input.getYesNo("spin1", false);
  p.sz_override = input.getYesNo("sz_override", false);
  p.sz_even_max = input.getReal("sz_even_max", 0);
  p.sz_even_min = input.getReal("sz_even_min", 0);
  p.sz_odd_max = input.getReal("sz_odd_max", 0.5);
  p.sz_odd_min = input.getReal("sz_odd_min", 0.5);
  p.excited_state = input.getYesNo("excited_state", false);
  p.excited_states = input.getInt("excited_states", 0);
  if (p.excited_states >= 1)
    p.excited_state = true;
  if (p.excited_state && p.excited_states == 0)
    p.excited_states = 1; // override

  // parameters controlling the postprocessing and output
  p.result_verbosity = input.getInt("result_verbosity", 0);
  p.stdout_verbosity = input.getInt("stdout_verbosity", 0);
  p.computeEntropy = input.getYesNo("computeEntropy", false);
  p.computeEntropy_beforeAfter = input.getYesNo("computeEntropy_beforeAfter", false);
  p.measureAllDensityMatrices = input.getYesNo("measureAllDensityMatrices", false);
  p.chargeCorrelation = input.getYesNo("chargeCorrelation", false);
  p.spinCorrelation = input.getYesNo("spinCorrelation", false);
  p.spinCorrelationMatrix = input.getYesNo("spinCorrelationMatrix", false);
  p.channelDensityMatrix = input.getYesNo("channelDensityMatrix", false);
  p.pairCorrelation = input.getYesNo("pairCorrelation", false);
  p.hoppingExpectation = input.getYesNo("hoppingExpectation", false);
  p.calcweights = input.getYesNo("calcweights", false);
  p.charge_susceptibility = input.getYesNo("charge_susceptibility", false);
  p.measureChannelsEnergy = input.getYesNo("measureChannelsEnergy", false);
  p.measureParity = input.getYesNo("measureParity", false);
  // parameters controlling the calculation
  p.save = input.getYesNo("save", false) || (p.solve_ndx >= 0);
  p.nrsweeps = input.getInt("nrsweeps", 15);
  p.parallel = input.getYesNo("parallel", true); // parallel by default!
  p.Quiet = input.getYesNo("Quiet", true);
  p.Silent = input.getYesNo("Silent", p.parallel);
  p.verbose = input.getYesNo("verbose", !p.parallel); // add veryverbose ?
  p.debug = input.getYesNo("debug", false);
  p.EnergyErrgoal = input.getReal("EnergyErrgoal", 0);
  p.nrH = input.getInt("nrH", 5);
  p.sc_only = input.getYesNo("sc_only", false);
  p.Weight = input.getReal("Weight", p.N);  // weight is implemented as the energy cost of the overlap between the GS and the ES; as energy of the GS is on the order of -N/2, the default weight should probs be on this scale.  
  p.transition_dipole_moment = input.getYesNo("transition_dipole_moment", false);
  p.transition_quadrupole_moment = input.getYesNo("transition_quadrupole_moment", false);
  p.overlaps = input.getYesNo("overlaps", false);
  p.cdag_overlaps = input.getYesNo("cdag_overlaps", false);
  p.flat_band = input.getYesNo("flat_band", false);
  p.flat_band_factor = input.getReal("flat_band_factor", 0);
  p.band_rescale = input.getReal("band_rescale", 1.0);
  p.reverse_second_channel_eps = input.getYesNo("reverse_second_channel_eps", p.measureParity); //has to be true for measuring parity
  p.enforce_total_spin = input.getYesNo("enforce_total_spin", false);
  p.spin_enforce_weight = input.getReal("spin_enforce_weight", p.N);

  // dynamical charge susceptibility calculations
  p.chi = input.getYesNo("chi", false);
  p.omega_r = input.getReal("omega_r", 0);
  p.eta_r = input.getReal("eta_r", 0.01);
  p.tau_max = input.getReal("tau_max", 1.0);
  p.tau_step = input.getReal("tau_step", 0.1);
  p.evol_nr_expansion = input.getInt("evol_nr_expansion", 3);
  p.evol_krylovord = input.getInt("evol_krylovord", 3);
  p.evol_nrsweeps = input.getInt("evol_nrsweeps", 1);
  p.evol_sweeps_cutoff = input.getReal("evol_sweeps_cutoff", 1e-8);
  p.evol_sweeps_maxdim = input.getInt("evol_sweeps_maxdim", 2000);
  p.evol_sweeps_niter = input.getInt("evol_sweeps_niter", 10);
  p.evol_epsilonK1 = input.getReal("evol_epsilonK1", 1e-12);
  p.evol_epsilonK2 = input.getReal("evol_epsilonK2", 1e-12);
  p.evol_numcenter = input.getInt("evol_numcenter", 1);
  my_assert(1 <= p.evol_numcenter && p.evol_numcenter <= 2, "incorrect evol_numcenter");

  p.MF_precision = input.getReal("MF_precision", 1e-5);
  p.max_iter = input.getReal("max_iter", 5.);

  // If spin orbit coupling is turned on in any sc, turn of the spin conservation.
  std::cout << "\nspin conservation: " << (p.problem->spin_conservation() ? "yes" : "no") << "\n";
  p.sites = Electron(p.N, {"ConserveSz", p.problem->spin_conservation()});
  // report_Sz_conserved(p.problem.get()); // TO DO: to be fixed
}

void MeasureChannelsEnergy(MPS& psi, H5Easy::File & file, std::string path, params &p) {
  MPO Hch1(p.sites);
  MPO Hch2(p.sites);
  prob::twoch_impfirst_V new2chProblem(p);
  auto impOp = new2chProblem.get_one_channel_MPOs_and_impOp(Hch1, Hch2, p);
  double ch1EnergyGain = std::real(innerC(psi, Hch1, psi));
  double ch2EnergyGain = std::real(innerC(psi, Hch2, psi));
  psi.position(p.impindex);
  auto res = psi(p.impindex) * impOp * dag(prime(psi(p.impindex),"Site"));
  double impEnergy = std::real(res.cplx());
  std::cout << std::setprecision(full) << "Energy gain: " << std::endl;
  std::cout << std::setprecision(full) << "channel1 : " <<  ch1EnergyGain << std::endl;
  std::cout << std::setprecision(full) << "channel2 : " <<  ch2EnergyGain << std::endl;
  std::cout << std::setprecision(full) << "impurity : " <<  impEnergy << std::endl;
  std::cout << std::setprecision(full) << "sum = " << ch1EnergyGain + ch2EnergyGain + impEnergy << std::endl;
  H5Easy::dump(file, path + "/channel_energy_gain/1", ch1EnergyGain);
  H5Easy::dump(file, path + "/channel_energy_gain/2", ch2EnergyGain);
  H5Easy::dump(file, path + "/channel_energy_gain/imp", impEnergy);
}

#include "MPO_totalSpin.h"
//#include "autoMPO_S2.h"

//Measures the total spin of a site using a MPO
void MeasureTotalSpin(MPS& psi, auto & file, std::string path, params &p) {
  MPO S2(p.sites);
  makeS2_MPO(S2, p, 0.0, 1.0);
  // res2 is the result of \hat{S}^2 = S(S+1)
  // Actual S is obtained by solving the quadratic equation, always taking the largest solution.
  auto res2 = std::real(innerC(psi, S2, psi));
  auto res = 0.5 * std::max( -1 + std::sqrt(1 + 4*res2), -1 - std::sqrt(1 + 4*res2) );
  std::cout << std::setprecision(full) << "Total S = " <<  res << ", S^2 = " << res2 << std::endl;
  H5Easy::dump(file, path + "/S2", res);
}

// If i<j, computes <op_i op_j>. If i>j, computes <op_j op_i>=<op_i^dag op_j^dag>^*.
auto Correlator(MPS& psi, const int i, const auto op_i, const int j, const auto op_j, const params &p) {
  Expects(i != j);
  psi.position(i);
  MPS psidag = dag(psi);
  psidag.prime("Link");
  const auto [first, second] = std::minmax(i, j);
  // apply the operator to the first site
  const auto li_1 = leftLinkIndex(psi, first);
  auto C = prime(psi(first), li_1) * (first == i ? op_i : op_j) ;
  C *= prime(psidag(first), "Site");
  for (int k = first+1; k < second; ++k) {
    C *= psi(k);
    C *= psidag(k);
  }
  // apply the operator to the second site
  const auto lj = rightLinkIndex(psi, second);
  C *= prime(psi(second), lj) * (first == i ? op_j : op_i);
  C *= prime(psidag(second), "Site");
  return std::real(eltC(C));
}

auto ImpurityCorrelator(MPS& psi, const auto impOp, const int j, const auto opj, const params &p) {
  return Correlator(psi, p.impindex, impOp, j, opj, p);
}

//CORRELATION FUNCTIONS BETWEEN THE IMPURITY AND ALL SC LEVELS:
//according to: http://www.itensor.org/docs.cgi?vers=cppv3&page=formulas/correlator_mps

// <n_imp n_i>
auto calcChargeCorrelation(MPS& psi, const ndx_t bath_sites, const params &p) {
  std::vector<double> r;
  double tot = 0;
  auto impOp = op(p.sites, "Ntot", p.impindex);
  for (const auto j: bath_sites) {
    auto scOp = op(p.sites, "Ntot", j);
    double result = ImpurityCorrelator(psi, impOp, j, scOp, p);
    r.push_back(result);
    tot += result;
  }
  return std::make_pair(r, tot);
}

void MeasureChargeCorrelation(MPS& psi, auto & file, std::string path, const params &p) {
  const auto [r, tot] = calcChargeCorrelation(psi, p.problem->bath_indexes(), p);
  if (p.stdout_verbosity){
    std::cout << "charge correlation = " << std::setprecision(full) << r << std::endl;
    std::cout << "charge correlation tot = " << tot << std::endl;}
  else std::cout << "charge correlations computed" << std::endl;
  H5Easy::dump(file, path + "/charge_correlation", r);
  H5Easy::dump(file, path + "/charge_correlation_total", tot);
}

// Sz, Sp, Sm
auto Sz(const int i, const params &p) {
  return 0.5*( op(p.sites, "Nup", i) - op(p.sites, "Ndn", i) );
}

auto Sp(const int i, const params &p) {
  return op(p.sites, "Cdagup*Cdn", i);
}

auto Sm(const int i, const params &p) {
  return op(p.sites, "Cdagdn*Cup", i);
}

auto Sz_Sp_Sm(const int i, const params &p) {
  return std::make_tuple(Sz(i, p), Sp(i, p), Sm(i, p));
}

// Sz^2, SpSm, SmSp
auto SzSz_SpSm_SmSp(const int i, const params &p) {
  return std::make_tuple(0.25 * ( op(p.sites, "Nup*Nup", i) - op(p.sites, "Nup*Ndn", i) - op(p.sites, "Ndn*Nup", i) + op(p.sites, "Ndn*Ndn", i)),
                         op(p.sites, "Cdagup*Cdn*Cdagdn*Cup", i),
                         op(p.sites, "Cdagdn*Cup*Cdagup*Cdn", i));
}

auto vev(MPS &psi, const int i, auto &op) {
  psi.position(i);
  return std::real(eltC(psi(i) * op *  dag(prime(psi(i),"Site"))));
}

// <S_imp S_i> = <Sz_imp Sz_i> + 1/2 ( <S+_imp S-_i> + <S-_imp S+_i> )
auto calcSpinCorrelation(MPS& psi, const ndx_t &bath_sites, const params &p) {
  const auto [impSz, impSp, impSm] = Sz_Sp_Sm(p.impindex, p);   //impurity spin operators
  const auto [impSzSz, impSpSm, impSmSp] = SzSz_SpSm_SmSp(p.impindex, p);
  auto sum = [&bath_sites, &p, &psi](const auto &opimp, const auto &opbath, auto &results) {
    double total = 0;
    for(const auto j: bath_sites) {
      const auto result = ImpurityCorrelator(psi, opimp, j, opbath(j), p);
      results.push_back(result);
      total += result;
    }
    return total;
  };
  std::vector<double> rzz, rpm, rmp;  // vectors collecting individual terms <Sz_imp Sz_i>, <S+_imp S-_i>, <S-_imp S+_i>
  double tot = 0;                     // sum over all three contributions and over i
  // Sz Sz
  const auto onSiteSzSz = vev(psi, p.impindex, impSzSz);
  tot += onSiteSzSz; // VERY IMPORTANT WARNING: tot also contains <Simp.Simp> contribution!!!
  tot += sum(impSz, [&p](const int j){ return Sz(j, p); }, rzz);
  // S+ S-
  const auto onSiteSpSm = vev(psi, p.impindex, impSpSm);
  tot += 0.5*onSiteSpSm;
  tot += 0.5*sum(impSp, [&p](const int j){ return Sm(j, p); }, rpm);
  // S- S+
  const auto onSiteSmSp = vev(psi, p.impindex, impSmSp);
  tot += 0.5*onSiteSmSp;
  tot += 0.5*sum(impSm, [&p](const int j){ return Sp(j, p); }, rmp);
  return std::make_tuple(onSiteSzSz, onSiteSpSm, onSiteSmSp, rzz, rpm, rmp, tot);
}

void MeasureSpinCorrelation(MPS& psi, H5Easy::File & file, std::string path, const params &p) {
  const auto [onSiteSzSz, onSiteSpSm, onSiteSmSp, rzz, rpm, rmp, tot] = calcSpinCorrelation(psi, p.problem->bath_indexes(), p);
  if (p.stdout_verbosity >= 0){
    std::cout << "spin correlations:\n";
    std::cout << "SzSz correlations: ";
    std::cout << std::setprecision(full) << onSiteSzSz << " ";
    std::cout << std::setprecision(full) << rzz << std::endl;
    std::cout << "S+S- correlations: ";
    std::cout << std::setprecision(full) << onSiteSpSm << " ";
    std::cout << std::setprecision(full) << rpm << std::endl;
    std::cout << "S-S+ correlations: ";
    std::cout << std::setprecision(full) << onSiteSmSp << " ";
    std::cout << std::setprecision(full) << rmp << std::endl;
    std::cout << "spin correlation tot = " << tot << "\n";
  }
  else std::cout << "spin correlations computed" << std::endl;
  H5Easy::dump(file, path + "/spin_correlation_imp/zz", onSiteSzSz);
  H5Easy::dump(file, path + "/spin_correlation_imp/pm", onSiteSpSm);
  H5Easy::dump(file, path + "/spin_correlation_imp/mp", onSiteSmSp);
  H5Easy::dump(file, path + "/spin_correlation/zz", rzz);
  H5Easy::dump(file, path + "/spin_correlation/pm", rpm);
  H5Easy::dump(file, path + "/spin_correlation/mp", rmp);
  H5Easy::dump(file, path + "/spin_correlation_total", tot);
}

auto calcSS(MPS& psi, const int i, const int j, const params &p) {
  if (i != j) {
    const auto [Szi, Spi, Smi] = Sz_Sp_Sm(i, p);
    const auto [Szj, Spj, Smj] = Sz_Sp_Sm(j, p);
    const auto zz = Correlator(psi, i, Szi, j, Szj, p);
    const auto pm = Correlator(psi, i, Spi, j, Smj, p);
    const auto mp = Correlator(psi, i, Smi, j, Spj, p);
    return zz + 0.5*(pm + mp);
  } else {
    const auto [SzSz, SpSm, SmSp] = SzSz_SpSm_SmSp(i, p);
    const auto zz = vev(psi, i, SzSz);
    const auto pm = vev(psi, i, SpSm);
    const auto mp = vev(psi, i, SmSp);
    return zz + 0.5*(pm + mp);
  }
}

//for a qd-sc-qd problem, measures the qd-qd spin correlation
void MeasureImpImpSpinCorrelation(MPS& psi, H5Easy::File & file, std::string path, const params &p) {
  auto res = calcSS(psi, 1, p.N, p);
  std::cout << "imp-imp spin correlation = " << res << "\n";
  H5Easy::dump(file, path + "/imp_imp_spin_correlation", res);
}

auto calcCdagC(MPS& psi, const int i, const int j, const params &p) {
  if (i == j) {
//    auto res = psi(i) * p.sites.op("Ntot",i) * dag(prime(psi(i),"Site"));
    psi.position(i);
    const auto res = psi.A(i) * p.sites.op("Ntot",i) * dag(prime(psi.A(i),"Site"));
    return std::real(res.cplx());
  } else {
    auto cdagupi = op(p.sites, "Cdagup", i);
    auto cupj    = op(p.sites, "Cup", j);
    auto cdagdni = op(p.sites, "Cdagdn", i);
    auto cdnj    = op(p.sites, "Cdn", j);
    auto corup = Correlator(psi, i, cdagupi, j, cupj, p);
    auto cordn = Correlator(psi, i, cdagdni, j, cdnj, p);
    return corup + cordn;
  }
}

auto calcMatrix(const std::string which, MPS& psi, const ndx_t &all_sites, const params &p, const bool full = false) {
  if (p.verbose) { std::cout << "Computing " << which << " correlation matrix" << std::endl; }
  auto m = matrix_t(all_sites.size(), all_sites.size(), 0.0);
  for (const auto i: all_sites) {
    for (const auto j: all_sites) {
      if (full || i <= j) {
        if (which == "spin")  m(i-1, j-1) = calcSS(psi, i, j, p); // 0-based matrix indexing
        if (which == "density")   m(i-1, j-1) = calcCdagC(psi, i, j, p);
        if (p.debug) { std::cout << fmt::format("m({},{})={:18}\n", i, j, m(i-1, j-1)); }
      } else {
        m(i-1, j-1) = m(j-1, i-1);
      }
    }
  }
  return m;
}

// Spin correlation matrix
void MeasureSpinCorrelationMatrix(MPS &psi, H5Easy::File &file, std::string path, const params &p) {
  const auto m = calcMatrix("spin", psi, p.problem->all_indexes(), p);
  h5_dump_matrix(file, path + "/spin_correlation_matrix", m);
}

// Charge density matrix, <psi| c^\dag_i c_j |psi>
void MeasureChannelDensityMatrix(MPS &psi, H5Easy::File &file, std::string path, const params &p) {
  const auto m = calcMatrix("density", psi, p.problem->all_indexes(), p, true);
  h5_dump_matrix(file, path + "/channel_density_matrix", m);
}

auto calcPairCorrelation(MPS& psi, const ndx_t &bath_sites, const params &p) {
  std::vector<double> r;
  double tot = 0;
  auto impOp = op(p.sites, "Cup*Cdn", p.impindex);
  for(const auto j: bath_sites) {
    auto scOp = op(p.sites, "Cdagdn*Cdagup", j);
    double result = ImpurityCorrelator(psi, impOp, j, scOp, p);
    r.push_back(result);
    tot += result;
  }
  return std::make_pair(r, tot);
}

void MeasurePairCorrelation(MPS& psi, H5Easy::File & file, std::string path, const params &p) {
  const auto [r, tot] = calcPairCorrelation(psi, p.problem->bath_indexes(), p);
  if (p.stdout_verbosity >= 0){
    std::cout << "pair correlation = " << std::setprecision(full) << r << std::endl;
    std::cout << "pair correlation tot = " << tot << std::endl;}
  else std::cout << "pair correlations computed" << std::endl;
  H5Easy::dump(file, path + "/pair_correlation", r);
  H5Easy::dump(file, path + "/pair_correlation_total", tot);
}

//Prints <d^dag c_i + c_i^dag d> for each i. the sum of this expected value, weighted by 1/sqrt(N)
//gives <d^dag f_0 + f_0^dag d>, where f_0 = 1/sqrt(N) sum_i c_i. This is the expected value of hopping.
auto calcHopping(MPS& psi, const ndx_t &bath_sites, const params &p) {
  std::vector<double> rup, rdn;
  double totup = 0;
  double totdn = 0;
  auto impOpUp = op(p.sites, "Cup", p.impindex);
  auto impOpDagUp = op(p.sites, "Cdagup", p.impindex);
  auto impOpDn = op(p.sites, "Cdn", p.impindex);
  auto impOpDagDn = op(p.sites, "Cdagdn", p.impindex);
  // hopping expectation values for spin up
  for (const auto j : bath_sites) {
    auto scDagOp = op(p.sites, "Cdagup", j);
    auto scOp = op(p.sites, "Cup", j);
    auto result1 = ImpurityCorrelator(psi, impOpUp, j, scDagOp, p); // <d c_i^dag>
    auto result2 = ImpurityCorrelator(psi, impOpDagUp, j, scOp, p); // <d^dag c_i>
    auto sum = result1+result2;
    rup.push_back(sum);
    totup += sum;
  }
  // hopping expectation values for spin dn
  for (const auto j : bath_sites) {
    auto scDagOp = op(p.sites, "Cdagdn", j);
    auto scOp = op(p.sites, "Cdn", j);
    auto result1 = ImpurityCorrelator(psi, impOpDn, j, scDagOp, p);    // <d c_i^dag>
    auto result2 =  ImpurityCorrelator(psi, impOpDagDn, j, scOp, p); // <d^dag c_i>
    auto sum = result1+result2;
    rdn.push_back(sum);
    totdn += sum;
  }
  return std::make_tuple(rup, rdn, totup, totdn);
}

void MeasureHopping(MPS& psi, H5Easy::File & file, std::string path, const params &p) {
  const auto [rup, rdn, totup, totdn] = calcHopping(psi, p.problem->bath_indexes(), p);
  if (p.stdout_verbosity >= 0 ){
    std::cout << "hopping spin up = " << std::setprecision(full) << rup << std::endl;
    std::cout << "hopping correlation up tot = " << totup << std::endl;
    std::cout << "hopping spin down = " << std::setprecision(full) << rdn << std::endl;
    std::cout << "hopping correlation down tot = " << totdn << std::endl;
  }
  else std::cout << "hopping correlations computed" << std::endl;
  const auto tot = totup+totdn;
  std::cout << "total hopping correlation = " << tot << std::endl;
  H5Easy::dump(file, path + "/hopping/up", rup);
  H5Easy::dump(file, path + "/hopping/dn", rdn);
  H5Easy::dump(file, path + "/hopping_total/up",  totup);
  H5Easy::dump(file, path + "/hopping_total/dn",  totdn);
  H5Easy::dump(file, path + "/hopping_total/sum", tot);
}

//prints the occupation number Nup and Ndn at the impurity
auto calc_NUp_NDn(MPS& psi, int ndx, const params &p){
  psi.position(ndx);
  const auto valnup = psi.A(ndx) * p.sites.op("Nup",ndx) * dag(prime(psi.A(ndx),"Site"));
  const auto valndn = psi.A(ndx) * p.sites.op("Ndn",ndx) * dag(prime(psi.A(ndx),"Site"));
  return std::make_pair(std::real(valnup.cplx()), std::real(valndn.cplx()));
}

void MeasureImpurityUpDn(MPS& psi, auto &file, std::string path, const params &p){
  const auto [up, dn] = calc_NUp_NDn(psi, p.impindex, p);
  const auto sz = 0.5*(up-dn);
  std::cout << "impurity nup ndn = " << std::setprecision(full) << up << " " << dn << " sz = " << sz << std::endl;
  H5Easy::dump(file, path + "/impurity_Nup", up);
  H5Easy::dump(file, path + "/impurity_Ndn", dn);
  H5Easy::dump(file, path + "/impurity_Sz",  sz);
}

// total Sz of the state
auto calcTotalSpinz(MPS& psi, const ndx_t &all_sites, const params &p) {
  double totNup = 0.;
  double totNdn = 0.;
  for (const auto j: all_sites) {
    psi.position(j);
    auto Nupi = psi.A(j) * p.sites.op("Nup",j)* dag(prime(psi.A(j),"Site"));
    auto Ndni = psi.A(j) * p.sites.op("Ndn",j)* dag(prime(psi.A(j),"Site"));
    totNup += std::real(Nupi.cplx());
    totNdn += std::real(Ndni.cplx());
  }
  const auto totSz =  0.5*(totNup-totNdn);
  return std::make_tuple(totNup, totNdn, totSz);
}

void MeasureTotalSpinz(MPS& psi, H5Easy::File &file, std::string path, const params &p) {
  const auto [totNup, totNdn, totSz] = calcTotalSpinz(psi, p.problem->all_indexes(), p);
  std::cout << std::setprecision(full) << "Total spin z: " << " Nup = " << totNup << " Ndn = " << totNdn << " Sztot = " << totSz << std::endl;
  H5Easy::dump(file, path + "/total_Nup", totNup);
  H5Easy::dump(file, path + "/total_Ndn", totNdn);
  H5Easy::dump(file, path + "/total_Sz",  totSz);
}

// occupation numbers of levels 'sites'
std::vector<double> calcOccupancy(MPS &psi, const ndx_t &all_sites, const params &p) {
  std::vector<double> r;
  for(const auto i : all_sites) {
    // position call very important! otherwise one would need to contract the whole tensor network of <psi|O|psi> this way, only the local operator at site i is needed
    psi.position(i);
    const auto val = psi.A(i) * p.sites.op("Ntot",i) * dag(prime(psi.A(i),"Site"));
    r.push_back(std::real(val.cplx()));
  }
  return r;
}

void MeasureOccupancy(MPS& psi, auto & file, std::string path, const params &p) {
  const auto r = calcOccupancy(psi, p.problem->all_indexes(), p);
  const auto tot = std::accumulate(r.cbegin(), r.cend(), 0.0);
  if (p.stdout_verbosity >= 0) {
    std::cout << "site occupancies = " << std::setprecision(full) << r << std::endl;
    std::cout << "tot = " << tot << std::endl;
  }
  else std::cout << "site occupancies computed" << std::endl;
  H5Easy::dump(file, path + "/site_occupancies", r);
  H5Easy::dump(file, path + "/total_occupancy", tot);
}

// This is actually sqrt of local charge correlation, <n_up n_down>-<n_up><n_down>, summed over all bath levels.
// The sum (tot) corresponds to \bar{\Delta}', Eq. (4) in Braun, von Delft, PRB 50, 9527 (1999), first proposed by Dan Ralph.
// It reduces to Delta_BCS in the thermodynamic limit (if the impurity is decoupled, Gamma=0).
auto calcPairing(MPS &psi, const ndx_t &all_sites, const params &p) {
  std::vector<complex_t> r;
  complex_t tot = 0;
  for(const auto i : all_sites) {
    psi.position(i);
    auto val2  = psi.A(i) * p.sites.op("Cdagup*Cup*Cdagdn*Cdn", i) * dag(prime(psi.A(i),"Site"));
    auto val1u = psi.A(i) * p.sites.op("Cdagup*Cup", i) * dag(prime(psi.A(i),"Site"));
    auto val1d = psi.A(i) * p.sites.op("Cdagdn*Cdn", i) * dag(prime(psi.A(i),"Site"));
    // For Gamma>0, <C+CC+C>-<C+C><C+C> may be negative.
    auto diff = val2.cplx() - val1u.cplx() * val1d.cplx();
    auto sq = sqrt(diff) * p.sc->g();
    if (i != p.impindex){
      int bathndx = i < p.impindex ? i : i-1;
      sq *= pow(p.sc->y(bathndx), 2);
      if (bathndx == p.etasite){ //if the eta model is used!
        const double x = p.etarescale ? sqrt( (p.NBath-p.eta*p.eta)/(p.NBath-1.) ) : 1;
        sq *= pow(p.eta * x, 2);
      }
      tot += sq;
    }
    r.push_back(sq);
  }
  return std::make_pair(r, tot);
}

void MeasurePairing(MPS& psi, auto & file, std::string path, const params &p) {
  const auto [r, tot] = calcPairing(psi, p.problem->all_indexes(), p);
  
  if (p.stdout_verbosity >= 0){
  std::cout << "site pairing = " << std::setprecision(full) << r << std::endl;
  std::cout << "tot = " << tot << std::endl;}
  else std::cout << "pairing computed" << std::endl;
  dumpreal(file, path + "/pairing", r);
  dumpreal(file, path + "/pairing_total", tot);
}

// See von Delft, Zaikin, Golubev, Tichy, PRL 77, 3189 (1996)
// v = <c^\dag_up c^\dag_dn c_dn c_up>
// u = <c_dn c_up c^\dag_up c^\dag_dn>
auto calcAmplitudes(MPS &psi, const ndx_t &all_sites, const params &p) {
  std::vector<complex_t> rv, ru, rpdt;
  complex_t tot = 0;
  for(const auto i : all_sites) {
    psi.position(i);
    auto valv = psi.A(i) * p.sites.op("Cdagup*Cdagdn*Cdn*Cup", i) * dag(prime(psi.A(i),"Site"));
    auto valu = psi.A(i) * p.sites.op("Cdn*Cup*Cdagup*Cdagdn", i) * dag(prime(psi.A(i),"Site"));
    auto v = sqrt( std::real(valv.cplx()) ); // XXX
    auto u = sqrt( std::real(valu.cplx()) );
    auto pdt = v*u;
    ru.push_back(u);
    rv.push_back(v);
    rpdt.push_back(pdt);
    if (i!=p.impindex){
      int bathndx = i < p.impindex ? i : i-1;
      auto element = p.sc->g() * pow(p.sc->y(bathndx), 2) * pdt;
      if (bathndx == p.etasite){ //if the eta model is used!
        const double x = p.etarescale ? sqrt( (p.NBath-p.eta*p.eta)/(p.NBath-1.) ) : 1;
        element *= pow(p.eta * x, 2);
      }
      tot += element; // exclude the impurity site in the sum
    }
  }
  return std::make_tuple(rv, ru, rpdt, tot);
}

void MeasureAmplitudes(MPS& psi, auto & file, std::string path, const params &p) {
  const auto [rv, ru, rpdt, tot] = calcAmplitudes(psi, p.problem->all_indexes(), p);
  std::cout << "amplitudes vu = " << std::setprecision(full);
 
  if (p.stdout_verbosity >= 0){
  for (size_t i = 0; i < rv.size(); i++)
    std::cout << "[v=" << rv[i] << " u=" << ru[i] << " pdt=" << rpdt[i] << "] ";
  std::cout << std::endl << "tot = " << tot << std::endl;
  }
  else std::cout << "amplitudes computed" << std::endl;
  dumpreal(file, path + "/amplitudes/u", ru);
  dumpreal(file, path + "/amplitudes/v", rv);
  dumpreal(file, path + "/amplitudes/pdt", rpdt);
  dumpreal(file, path + "/amplitudes_total", tot);
}

// Computed entanglement/von Neumann entropy between the impurity and the system.
// Copied from https://www.itensor.org/docs.cgi?vers=cppv3&page=formulas/entanglement_mps
// von Neumann entropy at the bond between impurity and next site.
auto calcEntropy(MPS& psi, const int bond, const params &p) {
  psi.position(bond);
  // SVD this wavefunction to get the spectrum of density-matrix eigenvalues
  auto l = leftLinkIndex(psi, bond);
  auto s = siteIndex(psi, bond);
  auto [U,S,V] = svd(psi(bond), {l,s});
  auto u = commonIndex(U,S);
  //Apply von Neumann formula to the squares of the singular values
  double SvN = 0.;
  for(auto n : range1(dim(u))) {
    auto Sn = std::real(eltC(S,n,n));
    auto pp = sqr(Sn);
    if(pp > 1E-12) SvN += -pp*log(pp);
  }
  return SvN;
}

void MeasureEntropy(MPS& psi, auto & file, std::string path, const params &p) {
  Expects(p.impindex == 1); // Works as intended only if p.impindex=1.
  const auto SvN = calcEntropy(psi, p.impindex, p);
  std::cout << fmt::format("Entanglement entropy across impurity bond b={}, SvN = {:10}", p.impindex, SvN) << std::endl;
  H5Easy::dump(file, path + "/entanglement_entropy_imp", SvN);
}

void MeasureEntropy_beforeAfter(MPS& psi, auto & file, std::string path, const params &p) {
  Expects(p.impindex != 1); // Works as intended only if p.impindex=1.
  const auto SvN1 = calcEntropy(psi, p.impindex-1, p);
  const auto SvN2 = calcEntropy(psi, p.impindex, p);
  std::cout << fmt::format("Entanglement entropy before the impurity bond b={}, SvN = {:10}", p.impindex-1, SvN1) << std::endl;
  std::cout << fmt::format("Entanglement entropy after the impurity bond b={}, SvN = {:10}", p.impindex, SvN2) << std::endl;
  H5Easy::dump(file, path + "/entanglement_entropy_imp/before", SvN1);
  H5Easy::dump(file, path + "/entanglement_entropy_imp/after", SvN2);
}

// Contract all other tensors except one (site indexed by i). The diagonal terms are the squares of the amplitudes for the states |0>, |up>, |dn>, |2>.
auto calculate_density_matrix(MPS &psi, const int i, const params &p) {
  psi.position(i);
  auto psidag = dag(psi);
  auto psipsi = psi(i)*prime(psidag(i),"Site");
  return psipsi;
}

auto calculate_imp_density_matrix(MPS &psi, const params &p) {
  return calculate_density_matrix(psi, p.impindex, p);
}

auto to_real_matrix(auto &obj, const int dim1, const int dim2) {
  auto mat = matrix_t(dim1, dim2); // 0-based
  for (int i = 1; i <= dim1; i++) // 1-based
    for (int j = 1; j <= dim2; j++)
      mat(i-1, j-1) = std::real(obj.cplx(i,j));
  return mat;
}

void MeasureOnSiteDensityMatrices(MPS &psi, auto &file, std::string path, const params &p)  {
  const auto all_sites = p.problem->all_indexes();
  for (const auto i: all_sites) {
    const auto psipsi = calculate_density_matrix(psi, i, p);
    const auto P0 = std::real(psipsi.cplx(1,1));
    const auto Pu = std::real(psipsi.cplx(2,2));
    const auto Pd = std::real(psipsi.cplx(3,3));
    const auto P2 = std::real(psipsi.cplx(4,4));
    if (p.stdout_verbosity >= 2)
      std::cout << "site " << i <<" P: " << P0 << " " << Pu << " " << Pd << " " << P2 << "\n";
    H5Easy::dump(file, path + "/" + std::to_string(i) + "/P/0",    P0);
    H5Easy::dump(file, path + "/" + std::to_string(i) + "/P/up",   Pu);
    H5Easy::dump(file, path + "/" + std::to_string(i) + "/P/down", Pd);
    H5Easy::dump(file, path + "/" + std::to_string(i) + "/P/2",    P2);
    const auto mat = to_real_matrix(psipsi, 4, 4); // WARNING: in general, rho is complex!
    H5Easy::dump(file, path + "/" + std::to_string(i) + "/density_matrix", mat);
  }
  if (p.stdout_verbosity < 2) std::cout << "Ps computed" << std::endl;
}

void MeasureImpDensityMatrix(MPS &psi, auto &file, std::string path, const params &p) {
    const auto psipsi = calculate_imp_density_matrix(psi, p);
    const auto P0 = std::real(psipsi.cplx(1,1));
    const auto Pu = std::real(psipsi.cplx(2,2));
    const auto Pd = std::real(psipsi.cplx(3,3));
    const auto P2 = std::real(psipsi.cplx(4,4));
    if (p.stdout_verbosity >= 0) std::cout << "P_imp: " << P0 << " " << Pu  << " " << Pd << " " << P2 << "\n";
      else std::cout << "imp Ps computed" << std::endl;
    H5Easy::dump(file, path + "/P_imp/0",    P0);
    H5Easy::dump(file, path + "/P_imp/up",   Pu);
    H5Easy::dump(file, path + "/P_imp/down", Pd);
    H5Easy::dump(file, path + "/P_imp/2",    P2);
    const auto mat = to_real_matrix(psipsi, 4, 4); // WARNING: in general, rho is complex!
    H5Easy::dump(file, path + "/density_matrix_imp", mat);
}

ITensor swapF(const int i, const params & p){
  // This is a local operator wich gives 1 for single occupied levels and 0 elsewhere.
  // equivalent to N - 2*NupNdn
  // could be implemented also as: auto opi = op(p.sites, "Ntot", i) - ( 2 * op(p.sites, "Nupdn", i) );
  Index s = p.sites(i);
  Index sP = prime(s);
  ITensor swapF(dag(s),sP);
  swapF.set(s(2), sP(2), 1);
  swapF.set(s(3), sP(3), 1);
  return swapF;
}

void applyTwoSiteF(MPS &psi, const int i, const int j, const params &p){
  // applies the equivalent of 1 - 2 * swapF(i) * swapF(j)
  // this returns -1 only if the sites i and j are both singly occupied - as if they were swapped
  MPS newPsi = psi;
  // applying swapF(i) and swapF(j)
  newPsi.position(i);
  auto Ti = noPrime( swapF(i, p) * newPsi(i) );
  newPsi.set(i, Ti);
  newPsi.position(j);
  auto Tj = noPrime( swapF(j, p) * newPsi(j) );
  newPsi.set(j, Tj);
  psi.position(1); // for sum to work both MPS have to have the same orthogonality center
  newPsi.position(1);
  psi = sum(psi, -2.0 * newPsi, {"MaxDim",1000,"Cutoff",1E-9});
  psi.normalize();
  psi.orthogonalize();
}

void amplitudes(MPS &psi, const params &p){
  // Prints all large amplitudes of psi, but is hard coded for N=3!
  // Left here as might be useful for some other debugging.
  // 1 - 0, 2 - up, 3 - down, 4 - 2.
  Expects(p.N == 3);
  psi.position(1);
  auto tot = 0.0;
  std::cout << "all amplitudes: \n";
  for (int i=1; i<=4; i++){
    for (int j=1; j<=4; j++){
      for (int k=1; k<=4; k++){
        auto amp = psi.A(1)*dag(setElt(p.sites(1)(i)));
        amp *= psi.A(2)*dag(setElt(p.sites(2)(j)));
        amp *= psi.A(3)*dag(setElt(p.sites(3)(k)));
        //std::cout << "(" << i << ", " << j << ", " << k << "): \n";
        tot += pow(amp.real(), 2);
        if (abs(amp.real()) > 1e-8) std::cout << "(" << i << ", " << j << ", " << k << "): " << amp.real() << "\n";
      }
    }
  }
  std::cout << "total (has to be 1): " << tot << "\n";
}

MPS reversePsi(MPS psi, const params & p){ // psi MUST NOT BE PASSED AS REFERENCE HERE!
  // first take psi and apply all the fermionic swap gates to get the correct minus sign structure
  // to swap the channels, first permute site 1 past all other sites, then 2 past all sites but 1, then 3 past all but 1 and 2, ...
  for (int i = 1; i <= p.N; i++){
    for (int j = i+1; j <= p.N; j++){
      applyTwoSiteF(psi, i, j, p);
    }
  }
  // now swap the tensors
  MPS newPsi = MPS(length(psi));
    for (int i=1; i<=length(psi); i++){
    int j = length(psi)-(i-1);
    newPsi.ref(i) = psi(j);
  }
  // swap the indices
  for (int i=1; i<=length(psi); i++){
    int j = length(psi)-(i-1);
    if (i!=j){
      newPsi.ref(i) = newPsi(i) * delta(dag(p.sites(j)), p.sites(i));
    }
  }
  newPsi.position(1);
  return newPsi;
}

void MeasureParity(MPS &psi, auto &file, std::string path, const params &p) {
  // Measures the parity by reversing the tensors of the MPS and calculating the overlap.
  // Only works properly if p.reverse_second_channel_eps is turned on.
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  MPS reversedPsi = reversePsi(psi, p);
  auto parity = inner(psi, reversedPsi);
  std::cout << "parity = " << parity << "\n";
  H5Easy::dump(file, path + "/parity", parity);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Parity timing: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " s" << std::endl;
}

auto sweeps(params &p)
{
  auto inputsw = InputGroup(p.inputfn, "sweeps");
  auto sw_table = InputGroup(inputsw, "sweeps");
  return Sweeps(p.nrsweeps, sw_table);
}

// If a vector of MPOs is passed to the dmrg function, the dmrg will act as if these MPOs are sumed up.
// The idea is to optimize psi for <psi|H|psi> + spin_weight * (<S^2> - Sz(Sz+1)). This will tend to find a state with total spin close to initial Sz.
// This should for example minimize the admixture of the Sz=0 triplet state into the singlet at small Gammas.
// THIS HAS NOT BEEN TESTED (Jan 2023), AND ONLY WORKS FOR THE GROUND STATE.
// In May 2022 (Luka) asked a question regarding this on the iTensor discourse server - Miles' explanation should be there.
void generate_MPO_vector(std::vector<MPO> &MPOvec, const MPO &H, const state_t &st, params &p){
  MPOvec.push_back(H);
  if (p.enforce_total_spin){
    const auto [n, Sz, i] = st;
    double S = abs(Sz) * (abs(Sz) + 1);
    MPO S2_subtracted(p.sites);
    makeS2_MPO(S2_subtracted, p, -1*S, p.spin_enforce_weight);
    MPOvec.push_back(S2_subtracted);
  } 
}

void solve_gs(const state_t &st, store &s, params &p) {
  std::cout << "\nSweeping in the sector " << st <<  std::endl;
  auto H = p.problem->initH(st, p);
  auto state = p.problem->initState(st, p);
  MPS psi_init(state);
  for(auto i : range1(p.nrH)){
    psi_init = applyMPO(H,psi_init);
    psi_init.noPrime().normalize();
  }
  std::vector<MPO> MPOvec;
  generate_MPO_vector(MPOvec, H, st, p); // if p.enforce_total_spin the MPOvec contains H and S2, the MPO for the total spin, multiplied by the weight
  auto [GSenergy, psi] = dmrg(MPOvec, psi_init, sweeps(p),
                            {"Silent", p.Silent,
                             "Quiet", p.Quiet,
                             "EnergyErrgoal", p.EnergyErrgoal});
  if (p.enforce_total_spin) GSenergy = inner(psi, H, psi); // take out the spin part of the Hamiltonian
  s.eigen[st] = eigenpair(GSenergy, psi);
  s.stats[st] = psi_stats(psi, H);
}

void solve_es(const state_t &st, store &s, params &p) {
  std::cout << "\nSweeping in the sector " << st  << std::endl;
  const auto [n, Sz, i] = st;
  auto H = p.problem->initH(st, p);
  MPS psi_init_es = s.eigen[es({n, Sz}, i-1)].psi();
  std::vector<MPS> wfs(i);
  for (auto m = 0; m < i; m++)
    wfs[m] = s.eigen[es({n, Sz}, m)].psi();
  //std::vector<MPO> MPOvec;
  //generate_MPO_vector(MPOvec, H, st, p);
  auto [ESenergy, psi] = dmrg(H, wfs, psi_init_es, sweeps(p),
                            {"Silent", p.Silent,
                             "Quiet", p.Quiet,
                             "EnergyErrgoal", p.EnergyErrgoal,
                             "Weight", p.Weight});
  s.eigen[st] = eigenpair(ESenergy, psi);
  s.stats[st] = psi_stats(psi, H);
}

void save(const state_t &st, const eigenpair &ep, params &p)
{
  // SAVE p.sites AS WELL, CHECK load() FOR INFO.
  const auto &[n, S, i] = st;
  writeToFile(fmt::format("MPS_n{}_S{}_i{}", n, S, i), ep.psi());
  writeToFile(fmt::format("SITES_n{}_S{}_i{}", n, S, i), p.sites);
}

auto file_exists(const std::string &fn)
{
  std::ifstream F(fn);
  return bool(F);
}

std::optional<eigenpair> load(const state_t &st, params &p, const subspace_t &sub)
{
  const auto &[n, S, i] = st;
  const auto path = fmt::format("n{}_S{}_i{}", n, S, i);
  /*
  THIS IS CRITICAL!
  iTensor tensors (MPS, MPO, sites objects, ...) have indeces, labeled by QN etc, and an id - a random number attributed at creation.
  In order to be able to contract two tensors, their indeces must match, including the randomly generated id!
  To achieve this, along with psi, you save the p.sites object psi was created with, to match the indices. When loading, psi has to be
  initiated with the exact same sites object. Also, this exact object has to be used when applying operators etc., so the p.sites has
  to be overwritten with the loaded one.
  */
  const auto fn_sites = "SITES_" + path;
  Hubbard sites;
  if (!file_exists(fn_sites)) return std::nullopt;
  readFromFile(fn_sites, sites);
  p.sites = sites;
  const auto fn_mps = "MPS_" + path;
  if (!file_exists(fn_mps)) return std::nullopt;
  MPS psi(p.sites);
  readFromFile(fn_mps, psi);
  std::cout<<"psi loaded\n";
  auto H = p.problem->initH(st, p);
  auto E = inner(psi, H, psi);
  return eigenpair(E, psi);
}

void solve_state(const state_t &st, store &s, params &p)
{
  const auto [n, Sz, i] = st;
  if (i == 0)
    solve_gs(st, s, p);
  else
    solve_es(st, s, p);
  if (p.save) {
    save(st, s.eigen[st], p);
  }
}

void obtain_result(const subspace_t &sub, store &s, params &p)
{
  std::cout << "obtain result\n";
  for (int n = 0; n <= std::min(p.excited_states, p.stop_n); n++) {
    auto st = es(sub, n);
    try {
      const auto res = load(st, p, sub);
      if (res) {
        s.eigen[st] = res.value();
        std::cout << "load=" << sub << std::endl;
      }
    }
    catch (...) {}
    if (s.eigen.count(st) == 0) {
      std::cout << "solve=" << sub << std::endl;
      solve_state(st, s, p);   // fallback: compute
    }
  }
}

void solve(const std::vector<subspace_t> &l, store &s, params &p) {
  Expects(p.solve_ndx == -1 || (0 <= p.solve_ndx && p.solve_ndx < int(l.size())));
  auto obtain_result_l = [&s,&p](const auto &sub) { obtain_result(sub, s, p); };
  std::cout << "ndx=" << p.solve_ndx << std::endl;
  if (p.solve_ndx == -1) { // solve all
    if (p.parallel)
      std::for_each(std::execution::par, l.cbegin(), l.cend(), obtain_result_l);
    else
      std::for_each(std::execution::seq, l.cbegin(), l.cend(), obtain_result_l);
  } else { // solve one (ndx)
    obtain_result_l(l[p.solve_ndx]);
  }
}

void calc_properties(const state_t st, H5Easy::File &file, store &s, params &p)
{
  const auto [ntot, Sz, i] = st;
  const auto path = state_path(st);
  auto psi = s.eigen[st].psi();
  std::cout << fmt::format("\n\nRESULTS FOR THE SECTOR WITH {} PARTICLES, Sz {}, state {}:", ntot, Sz_string(Sz), i) << std::endl;
  const auto E = s.eigen[st].E();
  std::cout << fmt::format("Energy = {}", E) << std::endl;
  H5Easy::dump(file, path + "/E", E);
  p.problem->calc_properties(st, file, s, p); // also call the calc_properties function for the specified problem type
  MeasureTotalSpinz(psi, file, path, p);
  if (p.result_verbosity >= 0) MeasureOccupancy(psi, file, path, p);
  if (p.result_verbosity >= 0) MeasureTotalSpin(psi, file, path, p);
  if (p.measureAllDensityMatrices || p.result_verbosity >= 2) MeasureOnSiteDensityMatrices(psi, file, path, p);
  if (p.channelDensityMatrix      || p.result_verbosity >= 2) MeasureChannelDensityMatrix(psi, file, path, p);
  if (p.spinCorrelationMatrix     || p.result_verbosity >= 2) MeasureSpinCorrelationMatrix(psi, file, path, p);
  s.stats[st].dump();
}

void impurity_problem::calc_properties(const state_t st, H5Easy::File &file, store &s, params &p) {
  // these calculations that make sense only for the one impurity problem (one or two channels)
  const auto path = state_path(st);
  auto psi = s.eigen[st].psi();
  MeasureImpurityUpDn(psi, file, path, p);
  if (p.result_verbosity >= 0) MeasurePairing(psi, file, path, p);
  if (p.result_verbosity >= 0) MeasureAmplitudes(psi, file, path, p);
  if (p.result_verbosity >= 0) MeasureImpDensityMatrix(psi, file, path, p);
  if (p.chargeCorrelation         || p.result_verbosity >= 1) MeasureChargeCorrelation(psi, file, path, p);
  if (p.spinCorrelation           || p.result_verbosity >= 1) MeasureSpinCorrelation(psi, file, path, p);
  if (p.pairCorrelation           || p.result_verbosity >= 1) MeasurePairCorrelation(psi, file, path, p);
  if (p.hoppingExpectation        || p.result_verbosity >= 1) MeasureHopping(psi, file, path, p);
  if (p.computeEntropy) MeasureEntropy(psi, file, path, p);
  if (p.computeEntropy_beforeAfter) MeasureEntropy_beforeAfter(psi, file, path, p);
  if (numChannels() == 2){
      if (p.measureParity) MeasureParity(psi, file, path, p);
      if (p.measureChannelsEnergy) MeasureChannelsEnergy(psi, file, path, p);
      //if (p.measurePartialSumsOfSpinSpinMatrix) MeasurePartialSumsOfSpinSpinMatrix(psi, file, path, p); //here
  }
}

void chain_problem::calc_properties(const state_t st, H5Easy::File &file, store &s, params &p) {
  // here add calculations that make sense only for the chain problems
}

void two_impurity_problem::calc_properties(const state_t st, H5Easy::File &file, store &s, params &p) {
  // here add calculations that make sense only for the qd-sc-qd problem
  const auto path = state_path(st);
  auto psi = s.eigen[st].psi();
  MeasureImpImpSpinCorrelation(psi, file, path, p);
}

auto find_global_GS(store &s, auto & file) {
  auto m = std::min_element(begin(s.eigen), end(s.eigen), [](const auto &p1, const auto &p2) { return p1.second.E() < p2.second.E(); });
  state_t GS = m->first;
  double E_GS = m->second.E();
  const auto [N_GS, Sz_GS, i] = GS;
  //Expects(i == 0);
  if (i!=0) std::cout << "\nPossible problem! Global GS is an excited state!\n";
  std::cout << fmt::format("\nN_GS = {}\nSZ_GS = {}\nE_GS = {}\n",N_GS, Sz_string(Sz_GS), E_GS);
  H5Easy::dump(file, "/GS/N",  N_GS);
  H5Easy::dump(file, "/GS/Sz", Sz_GS);
  H5Easy::dump(file, "/GS/E",  E_GS);
  return std::make_pair(GS, E_GS);
}

void print_energies(store &s, double EGS, params &p) {
  skip_line();
  for (const auto &[state, ep] : s.eigen) {
    const auto [ntot, Sz, i] = state;
    const double E = ep.E();
    const double Ediff = E-EGS;
    std::cout << fmt::format(FMT_STRING("n = {:<5}  Sz = {:<4}  i = {:<3}  E = {:<22.15}  DeltaE = {:<22.15}"),
                             ntot, Sz_string(Sz), i, E, Ediff) << std::endl;
  }
}

// calculates <psi1|c_dag|psi2>, according to http://itensor.org/docs.cgi?vers=cppv3&page=formulas/mps_onesite_op
auto ExpectationValueAddEl(MPS psi1, MPS psi2, const std::string spin, const int position, const params &p){
  psi2.position(position);                                                      // set orthogonality center
  auto newTensor = noPrime(op(p.sites,"Cdag"+spin, position)*psi2(position)); // apply the local operator
  psi2.set(position,newTensor);                                                 // plug in the new tensor, with the operator applied
  return abs(innerC(psi1, psi2));
}

// calculates <psi1|c|psi2>
auto ExpectationValueTakeEl(MPS psi1, MPS psi2, const std::string spin, const int position, const params &p){
  psi2.position(position);
  auto newTensor = noPrime(op(p.sites,"C"+spin, position)*psi2(position));
  psi2.set(position,newTensor);
  return abs(innerC(psi1, psi2));
}

void calc_weight(store &s, state_t GS, state_t ES, int q, std::string sz, auto & file, params &p)
{
  double res;
  MPS & psiGS = s.eigen[GS].psi();
  if (s.eigen.count(ES)) {
    MPS & psiES = s.eigen[ES].psi();
    res = (q == +1 ? ExpectationValueAddEl(psiES, psiGS, sz, p.impindex, p) : ExpectationValueTakeEl(psiES, psiGS, sz, p.impindex, p));
    std::cout << "weight w" << (q == +1 ? "+" : "-") << " " << sz << ": " << res << std::endl;
  } else {
    res = std::numeric_limits<double>::quiet_NaN();
    std::cout <<  "ERROR: we don't have info about the state " << ES << std::endl;
  }
  H5Easy::dump(file, "weights/" + std::to_string(std::get<2>(ES)) + "/" + std::to_string(q) + "/" + sz, res);
}

void calculate_spectral_weights(store &s, state_t GS, auto &file, params &p, int excited) {
  skip_line();
  std::cout << "Spectral weights, " << (excited==0 ? "ground" : "excited") <<  " states:" << std::endl
    << "(Spectral weight is the square of the absolute value of the number.)" << std::endl;
  const auto [N_GS, Sz_GS, i] = GS;
  calc_weight(s, GS, {N_GS+1, Sz_GS+0.5, excited}, +1, "up", file, p);
  calc_weight(s, GS, {N_GS+1, Sz_GS-0.5, excited}, +1, "dn", file, p);
  calc_weight(s, GS, {N_GS-1, Sz_GS-0.5, excited}, -1, "up", file, p);
  calc_weight(s, GS, {N_GS-1, Sz_GS+0.5, excited}, -1, "dn", file, p);
}

auto calculate_overlap(const auto &psi1, const auto &psi2)
{
  return abs(innerC(psi1, psi2)); // abs because of random global phases!!
}

auto calculate_overlaps(store &s, auto &file, params &p) {
  skip_line();
  std::cout << "Overlaps:\n" ;
  for (const auto & [st1, st2] : itertools::product(s.eigen, s.eigen)) {
    const auto [ntot1, Sz1, i] = st1.first;
    const auto [ntot2, Sz2, j] = st2.first;
    if (ntot1 == ntot2 && Sz1 == Sz2 && i < j) {
      auto o = calculate_overlap(st1.second.psi(), st2.second.psi());
      std::cout << fmt::format(FMT_STRING("n = {:<5}  Sz = {:4}  i = {:<3}  j = {:<3}  |<i|j>| = {:<22.15}"),
                               ntot1, Sz_string(Sz1), i, j, o) << std::endl;
      H5Easy::dump(file, "overlaps/" + ij_path(ntot1, Sz1, i, j), o);
    }
  }
}

auto calculate_cdag_overlap(auto &psi1, auto &psi2, auto szchange, const params &p){
  // computes the overlap < psi1 | cdag_szchange | psi2 >
  std::vector<double> res;
  for (const int i : p.problem->all_indexes()) {
    auto r = abs(ExpectationValueAddEl(psi1, psi2, szchange, i, p));
    res.push_back(r);
  }
  return res;
}

void calculate_cdag_overlaps(store &s, auto &file, const params &p) {
  skip_line();
  std::cout << "<cdag_i>:\n" ;
  for (const auto & [st1, st2] : itertools::product(s.eigen, s.eigen)) {
    const auto [ntot1, Sz1, i] = st1.first;
    const auto [ntot2, Sz2, j] = st2.first;
    if ( ntot1 == ntot2 + 1 ){
      auto sz_change = Sz1 == Sz2 + 0.5 ? "up" : "dn";
      auto res = calculate_cdag_overlap(st1.second.psi(), st2.second.psi(), sz_change, p);
      double tot = std::accumulate(res.begin(), res.end(), 0.0);
      std::cout << fmt::format("<{}, {}, {}| cdag |{}, {}, {}>:\n", ntot1, Sz1, i, ntot2, Sz2, j);
      std::cout << "ci overlaps = " << std::setprecision(full) << res << std::endl;
      std::cout << "tot = " << tot << std::endl;
      H5Easy::dump(file, "cdag_overlaps/" + n1n2S1S2ij_path(ntot1, ntot2, Sz1, Sz2, i, j), res);
      H5Easy::dump(file, "cdag_overlaps/tot/" + n1n2S1S2ij_path(ntot1, ntot2, Sz1, Sz2, i, j), tot);
    }
  }
} 

// THIS IS FOR THE TWO CHANNEL MODEL ONLY - IT TAKES channelNum AS AN ARGUMENT FOR THE bath_indexes()!
auto one_channel_number_op(const int channelNum, auto &psi1, auto &psi2, const params &p){
  double res = 0.0;
  for (int i : p.problem->bath_indexes(channelNum)){
    MPS newpsi = psi2;
    newpsi.position(i);
    // Apply the n operator to a copy of psi2
    auto newT = p.sites.op("Ntot", i) * newpsi(i);
    newT.noPrime();
    newpsi.set(i, newT);
    psi1.position(i);
    res += abs(innerC(psi1, newpsi));
  }
  return res;
}

auto calculate_transition_dipole_moment(auto &psi1, auto &psi2, params &p){
  // < i | nsc1 | j > - < i | nsc2 | j >
  double res = 0.0;
  res += one_channel_number_op(1, psi1, psi2, p);
  res -= one_channel_number_op(2, psi1, psi2, p);
  return res;
}

auto calculate_transition_quadrupole_moment(auto &psi1, auto &psi2, params &p){
  // < i | nsc1 | j > + < i | nsc2 | j >
  double res = 0.0;
  res += one_channel_number_op(1, psi1, psi2, p);
  res += one_channel_number_op(2, psi1, psi2, p);
  return res;
}

void calculate_transition_dipole_moments(store &s, auto &file, params &p) {
  skip_line();
  std::cout << "Transition dipole moments:\n";
  for (const auto & [st1, st2] : itertools::product(s.eigen, s.eigen)) {
    const auto [ntot1, Sz1, i] = st1.first;
    const auto [ntot2, Sz2, j] = st2.first;
    if (ntot1 == ntot2 && Sz1 == Sz2 && i < j) { // for now only for Sz1==Sz2
      auto o = calculate_transition_dipole_moment(st1.second.psi(), st2.second.psi(), p);
      o = abs(o);
      std::cout << fmt::format(FMT_STRING("n = {:<5}  Sz = {:4}  i = {:<3}  j = {:<3}  |<i|nsc1 - nsc2|j>| = {:<22.15}"),
                               ntot1, Sz_string(Sz1), i, j, o) << std::endl;
      H5Easy::dump(file, "transition_dipole_moment/" + ij_path(ntot1, Sz1, i, j), o);
    }
  }
}

void calculate_transition_quadrupole_moments(store &s, auto &file, params &p) {
  skip_line();
  std::cout << "Transition quadrupole moments:\n";
  for (const auto & [st1, st2] : itertools::product(s.eigen, s.eigen)) {
    const auto [ntot1, Sz1, i] = st1.first;
    const auto [ntot2, Sz2, j] = st2.first;
    if (ntot1 == ntot2 && Sz1 == Sz2 && i < j) { // for now only for Sz1==Sz2
      auto o = calculate_transition_quadrupole_moment(st1.second.psi(), st2.second.psi(), p);
      o = abs(o);
      std::cout << fmt::format(FMT_STRING("n = {:<5}  Sz = {:4}  i = {:<3}  j = {:<3}  |<i|nsc1 + nsc2|j>| = {:<22.15}"),
                               ntot1, Sz_string(Sz1), i, j, o) << std::endl;
      H5Easy::dump(file, "transition_quadrupole_moment/" + ij_path(ntot1, Sz1, i, j), o);
    }
  }
}

auto calculate_charge_susceptibility(MPS &psi1, MPS &psi2, const params &p)
{
  auto psi2new = psi2;
  psi2new.position(p.impindex);                                                    // set orthogonality center
  auto newTensor = noPrime(op(p.sites,"Ntot", p.impindex)*psi2new(p.impindex));    // apply the local operator
  psi2new.set(p.impindex,newTensor);                                               // plug in the new tensor, with the operator applied
  return inner(psi1, psi2new);
}

void calculate_charge_susceptibilities(store &s, auto &file, params &p) {
  skip_line();
  std::cout << "Charge susceptibility:\n";
  for (const auto & [st1, st2] : itertools::product(s.eigen, s.eigen)) {
      const auto [ntot1, Sz1, i] = st1.first;
      const auto [ntot2, Sz2, j] = st2.first;
      if (ntot1 == ntot2 && Sz1 == Sz2 && i < j) {
        auto o = calculate_charge_susceptibility(st1.second.psi(), st2.second.psi(), p);
        o = abs(o); // ABSOLUTE VALUE! (because global signs of wavefunctions are arbitrary!)
        std::cout << fmt::format(FMT_STRING("n = {:<5}  Sz = {:4}  i = {:<3}  j = {:<3}  |<i|nimp|j>| = {:<22.15}"),
                                 ntot1, Sz_string(Sz1), i, j, o) << std::endl;
        H5Easy::dump(file, "charge_susceptibilty/" + ij_path(ntot1, Sz1, i, j), o);
    }
  }
}

void evolve(MPS &psiAtau, const MPO &H2, const int cnt, const double tau, params &p) {
  if (p.debug) std::cout << "*** evolve() starting, cnt=" << cnt << std::endl;
  if (cnt < p.evol_nr_expansion) {
    if (p.debug) std::cout << "*** addBasis()" << std::endl;
    std::vector<Real> epsilonK = { p.evol_epsilonK1, p.evol_epsilonK2 };
    addBasis(psiAtau, H2, epsilonK, {
        "Cutoff", 1E-8,                    // default is 1e-15
        "Method", "DensityMatrix",         // default is DensityMatrix
        "KrylovOrd", p.evol_krylovord,     // default is 3
        "DoNormalize", false,
        "Quiet", !p.debug
    });
  }
  if (p.debug) std::cout << "*** tdvp()" << std::endl;
  auto sweeps = Sweeps(p.evol_nrsweeps);
  sweeps.cutoff() = p.evol_sweeps_cutoff;  // default cutoff is 1e-8
  sweeps.maxdim() = p.evol_sweeps_maxdim;
  sweeps.niter()  = p.evol_sweeps_niter;
  const auto tdvp_t = -tau/p.evol_nrsweeps;
  tdvp(psiAtau, H2, tdvp_t, sweeps, {
      "DoNormalize", false,
      "Quiet", !p.debug,
      "Silent", !p.debug,
      "DebugLevel", (p.debug ? 2 : -1),
      "NumCenter", p.evol_numcenter        // default is 2
  });
  if (p.debug) std::cout << "*** evolve() done" << std::endl;
}

void calculate_dynamical_charge_susceptibility(store &s, const state_t GS, const double EGS, auto &file, params &p) {
  std::cout << "\nDynamical charge susceptibility:" << std::endl;
  
  const auto psi0 = s.eigen[GS].psi();
  if (p.debug) {
    const auto norm = inner(psi0, psi0);
    std::cout << "norm=" << norm << std::endl;
  }

  const auto id = MPO(p.sites); // identity operator
  if (p.debug) {
    const auto norm = inner(psi0, id, psi0);
    std::cout << "norm=" << norm << std::endl;
  }

  auto psiA = psi0;
  psiA.position(p.impindex);                                                    // set orthogonality center
  auto newTensor = noPrime(op(p.sites, "Ntot", p.impindex) * psiA(p.impindex)); // apply the local operator
  psiA.set(p.impindex, newTensor);
  psiA.position(1);
  if (p.debug) {
    const auto AA = inner(psiA, psiA);
    std::cout << "<A|A>=<psi|n^2|psi>=" << AA << std::endl;
  }

  const auto w0 = EGS + p.omega_r;
  const auto w0p = EGS - p.omega_r;
  if (p.debug)
    std::cout << "omega_0=" << w0 << " omega'_0=" << w0p << std::endl;

  auto H = p.problem->initH(GS, p);
  if (p.debug) {
    const auto E0 = inner(psi0, H, psi0);
    std::cout << "E0=" << E0 << " EGS=" << EGS << std::endl;
  }

  auto psiB = applyMPO(H, psiA);
  psiB.noPrime();
  psiB *= -1.0;
  psiB.plusEq(w0*psiA);
  if (p.debug) {
    const auto BA = inner(psiB, psiA);
    std::cout << "<B|A>=" << BA << std::endl;
  }

  auto psiBp = applyMPO(H, psiA);
  psiBp.noPrime();
  psiBp *= -1.0;
  psiBp.plusEq(w0p*psiA);
  if (p.debug) {
    const auto BpA = inner(psiBp, psiA);
    std::cout << "<B'|A>=" << BpA << std::endl;
  }

  auto minus_w0id = id;
  minus_w0id *= -w0;
  auto H1 = H;
  H1.plusEq(minus_w0id);             // H1 = H-w0*I
  auto H2 = nmultMPO(prime(H1), H1); // H2 = H1^2
  H2.mapPrime(2,1);
  if (p.debug) {
    const auto AH2A = inner(psiA, H2, psiA);
    std::cout << "<A|(H-omega_0)^2|A>=" << AH2A << std::endl;
  }

  auto minus_w0pid = id;
  minus_w0pid *= -w0p;
  auto H1p = H;
  H1p.plusEq(minus_w0pid);
  auto H2p = nmultMPO(prime(H1p), H1p);
  H2p.mapPrime(2,1);
  if (p.debug) {
    const auto AH2pA = inner(psiA, H2p, psiA);
    std::cout << "<A|(H-omega'_0)^2|A>=" << AH2pA << std::endl;
  }
  
  std::cout << std::endl;
  
  auto psiAtau = psiA;
  auto psiAptau = psiA;

  int cnt = 0;
  std::vector<double> table;
  const double tau_max = p.tau_max * (1.0 + 1.0e-8); // avoid round-off problems
  for (double tau = 0; tau <= tau_max; tau += p.tau_step, cnt++) {
    const auto scpdt = inner(psiB, psiAtau) + inner(psiBp, psiAptau);
    const auto intg = exp(-pow(p.eta_r,2) * tau) * scpdt;
    table.push_back(intg);
    std::cout << fmt::format(FMT_STRING("cnt = {:<5}  tau = {:<10.5}  scpdt = {:<22.15}  intg = {:<22.15}"),
                             cnt, tau, scpdt, intg) << std::endl;
    H5Easy::dump(file, "dynamical_charge_susceptibilty/tau/"   + std::to_string(cnt), tau);
    H5Easy::dump(file, "dynamical_charge_susceptibilty/scpdt/" + std::to_string(cnt), scpdt);
    H5Easy::dump(file, "dynamical_charge_susceptibilty/intg/"  + std::to_string(cnt), intg);

    if (tau + p.tau_step <= tau_max) {
      evolve(psiAtau, H2, cnt, p.tau_step, p);
      if (p.debug) {
        const auto AtauAtau = inner(psiAtau, psiAtau);
        std::cout << "<A(tau)|A(tau)>=" << AtauAtau << std::endl;
        const auto BAtau = inner(psiB, psiAtau);
        std::cout << "<B|A(tau)>=" << BAtau << std::endl;
      }
      evolve(psiAptau, H2p, cnt, p.tau_step, p);
      if (p.debug) {
        const auto AptauAptau = inner(psiAptau, psiAptau);
        std::cout << "<A'(tau)|A'(tau)>=" << AptauAptau << std::endl;
        const auto BpAptau = inner(psiBp, psiAptau);
        std::cout << "<B'|A'(tau)>=" << BpAptau << std::endl;
      }
    }
  }
  H5Easy::dump(file, "dynamical_charge_susceptibilty/nr", cnt);
  
  boost::math::interpolators::cardinal_cubic_b_spline<double> spline(table.begin(), table.end(), 0.0, p.tau_step);
  if (p.debug)
    std::cout << "intg(" << p.tau_max/2 << ")=" << spline(p.tau_max/2) << std::endl;
  
  double error {};
  const auto rechi = boost::math::quadrature::gauss_kronrod<double, 15>::integrate(spline, 0, p.tau_max, 5, 1e-9, &error);

  std::cout << fmt::format(FMT_STRING("\nRe[chi] = {:<22.15}   error = {:10.5}"), rechi, error) << std::endl;
  H5Easy::dump(file, "dynamical_charge_susceptibilty/rechi",       rechi);
  H5Easy::dump(file, "dynamical_charge_susceptibilty/rechi_error", error);
}

void process_and_save_results(store &s, params &p, std::string h5_filename) {
  H5Easy::File file(h5_filename, H5Easy::File::Overwrite);
  for(const auto & [st, e]: s.eigen)
    calc_properties(st, file, s, p);
  const auto [GS, EGS] = find_global_GS(s, file);
  print_energies(s, EGS, p);
  if (p.calcweights) {
    calculate_spectral_weights(s, GS, file, p, 0);
    if (p.excited_state) calculate_spectral_weights(s, GS, file, p, 1);
  }
  if (p.transition_dipole_moment)
    calculate_transition_dipole_moments(s, file, p);
  if (p.transition_quadrupole_moment)
    calculate_transition_quadrupole_moments(s, file, p);
  if (p.overlaps)
    calculate_overlaps(s, file, p);
  if (p.cdag_overlaps)
    calculate_cdag_overlaps(s, file, p);
  if (p.charge_susceptibility)
    calculate_charge_susceptibilities(s, file, p);
  if (p.chi)
    calculate_dynamical_charge_susceptibility(s, GS, EGS, file, p);
}
