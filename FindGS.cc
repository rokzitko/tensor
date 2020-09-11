#include "FindGS.h"

// Make an array centered at nref
auto n_list(int nref, int nrange) {
  std::vector<int> n;
  n.push_back(nref);
  for (auto i : range1(nrange)) {
    n.push_back(nref+i);
    n.push_back(nref-i);
  }
  return n;
}

void init_subspace_lists(params &p)
{
  const int nhalf = p.N;  // total nr of electrons at half-filling
  int nref = nhalf;       // default
  if (p.nref >= 0) {      // override
    nref = p.nref;
  } else if (p.refisn0) { // adaptable
    nref = round(p.sc->n0() + 0.5 - (p.qd->eps()/p.qd->U())); // calculation of the energies is centered around this n
  }
  p.numPart = n_list(nref, p.nrange);

  bool magnetic_field = ((p.qd->EZ()!=0 || p.sc->EZ()!=0) ? true : false); // true if there is magnetic field, implying that Sz=0.5 states are NOT degenerate
  // Sz values for n are in Szs[n]
  for (auto ntot : p.numPart) {
    std::vector<spin> sz_list;
    if (even(ntot))
      sz_list.push_back(spin0);
    else {
      sz_list.push_back(spinp);
      if (magnetic_field) sz_list.push_back(spinm);
    }
    p.Szs[ntot] = sz_list;
    for (auto sz : sz_list) 
      p.iterateOver.push_back(subspace(ntot, sz));
  }
}

InputGroup parse_cmd_line(int argc, char *argv[], params &p) {
  if (argc != 2)
    throw std::runtime_error("Please provide input file. Usage: executable <input file>");
  p.inputfn = { argv[1] };                                 // read parameters from the input file
  auto input = InputGroup{p.inputfn, "params"};            // get input parameters using InputGroup from itensor
  p.problem = set_problem(input.getString("MPO", "std"));  // problem type
  p.NImp = 1;
  p.N = input.getInt("N", 0);
  if (p.N != 0)
    p.NBath = p.N-p.NImp;
  else { // N not specified, try NBath
    p.NBath = input.getInt("NBath", 0);
    if (p.NBath == 0) 
      throw std::runtime_error("specify either N or NBath!");
    p.N = p.NBath+p.NImp;
  }
  p.impindex = p.problem->imp_index(p.NBath);
  std::cout << "N=" << p.N << " NBath=" << p.NBath << " impindex=" << p.impindex << std::endl;
  // sites is an ITensor thing. It defines the local hilbert space and operators living on each site of the lattice.
  // For example sites.op("N",1) gives the pariticle number operator on the first site.
  p.sites = Hubbard(p.N);

  // parameters entering the problem definition
  double U = input.getReal("U", 0); // need to parse it first because it enters the default value for epsimp just below
  p.qd = std::make_unique<imp>(U, input.getReal("epsimp", -U/2.), input.getReal("EZ_imp", 0.));
  p.sc = std::make_unique<SCbath>(p.NBath, input.getReal("alpha", 0), input.getReal("Ec", 0), input.getReal("n0", p.N-1), input.getReal("EZ_bulk", 0.));
  p.Gamma = std::make_unique<hyb>(input.getReal("gamma", 0));
  p.V12 = input.getReal("V", 0); // handled in a special way
  p.band_level_shift = input.getYesNo("band_level_shift", false);

  // parameters for the 2-channel problem
  p.sc1 = std::make_unique<SCbath>(p.NBath/2, input.getReal("alpha1", 0), input.getReal("Ec1", 0), input.getReal("n01", (p.N-1)/2), input.getReal("EZ_bulk1", 0));
  p.sc2 = std::make_unique<SCbath>(p.NBath/2, input.getReal("alpha2", 0), input.getReal("Ec2", 0), input.getReal("n02", (p.N-1)/2), input.getReal("EZ_bulk2", 0));
  p.Gamma1 = std::make_unique<hyb>(input.getReal("gamma1", 0));
  p.Gamma2 = std::make_unique<hyb>(input.getReal("gamma2", 0));

  // parameters controlling the calculation targets
  p.nref = input.getInt("nref", -1);
  p.nrange = input.getInt("nrange", 1);
  p.refisn0 = input.getYesNo("refisn0", false);
  p.excited_state = input.getYesNo("excited_state", false);

  // parameters controlling the postprocessing and output
  p.computeEntropy = input.getYesNo("computeEntropy", false);
  p.impNupNdn = input.getYesNo("impNupNdn", false);
  p.chargeCorrelation = input.getYesNo("chargeCorrelation", false);
  p.spinCorrelation = input.getYesNo("spinCorrelation", false);
  p.pairCorrelation = input.getYesNo("pairCorrelation", false);
  p.hoppingExpectation = input.getYesNo("hoppingExpectation", false);
  p.calcweights = input.getYesNo("calcweights", false);
  p.printTotSpinZ = input.getYesNo("printTotSpinZ", false);

  // parameters controlling the calculation
  p.printDimensions = input.getYesNo("printDimensions", false);
  p.parallel = input.getYesNo("parallel", false);
  p.verbose = input.getYesNo("verbose", false);
  p.EnergyErrgoal = input.getReal("EnergyErrgoal", 1e-16);
  p.nrH = input.getInt("nrH", 5);
  p.sc_only = input.getYesNo("sc_only", false);
  p.randomMPSb = input.getYesNo("randomMPS", false);
  p.Weight = input.getReal("Weight", 11.0);

  init_subspace_lists(p);
  return input;
}

// Initialize the MPS in a product state with ntot electrons.
// Sz is the z-component of the total spin.
// Electron is added on the impurity site only if sc_only=false.
MPS initPsi(charge ntot, spin Sz, const auto &sites, int impindex, bool sc_only, bool randomMPSb) {
  my_assert(ntot >= 0);
  my_assert(Sz == -1 || Sz == -0.5 || Sz == 0 || Sz == +0.5 || Sz == +1);
  int tot = 0;      // electron counter, for assertion test
  double Sztot = 0; // SZ counter, for assertion test
  auto state = InitState(sites);
  const auto nimp = sc_only ? 0 : 1;        // number of electrons in the impurity level
  const auto nsc = sc_only ? ntot : ntot-1; // number of electrons in the bath
  // ** Add electron to the impurity site
  if (nimp) {
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
  // ** Add electrons to the bath
  if (nsc) {
    const int npair = nsc/2;                // number of pairs in the bath
    auto j = 0;                             // counts added pairs
    auto i = 1;                             // site index (1 based)
    for(; j < npair; i++)
      if (i != impindex) {                  // skip impurity site
        j++;
        state.set(i, "UpDn");
        tot += 2;                           // Sztot does not change!
      }
    if (odd(nsc)) {                         // if ncs is odd (implies even ntot and sz=0), add spin-down electron
      if (i == impindex)
        i++;
      if (Sztot + 0.5 == Sz) {              // need to add spin-up electron
        state.set(i, "Up");
        Sztot += 0.5;
      } else if (Sztot - 0.5 == Sz) {       // need to add spin-down electron
        state.set(i, "Dn");
        Sztot -= 0.5;
      } else throw std::runtime_error("oops! should not happen!");
      tot++;
    }
  }
  my_assert(tot == ntot);
  my_assert(Sztot == Sz);
  MPS psi(state);
  if (randomMPSb)
    psi = randomMPS(state);
  return psi;
}

// computes the correlation between operator impOp on impurity and operator opj on j
double ImpurityCorrelator(MPS& psi, auto impOp, int j, auto opj, const params &p) {
  my_assert(p.impindex == 1); // !!!
  psi.position(p.impindex);
  MPS psidag = dag(psi);
  psidag.prime("Link");
  auto li_1 = leftLinkIndex(psi,p.impindex);
  auto C = prime(psi(p.impindex),li_1)*impOp;
  C *= prime(psidag(p.impindex),"Site");
  for (int k = p.impindex+1; k < j; ++k){
    C *= psi(k);
    C *= psidag(k);
  }
  auto lj = rightLinkIndex(psi,j);
  C *= prime(psi(j),lj)*opj;
  C *= prime(psidag(j),"Site");
  return elt(C);
}

//CORRELATION FUNCTIONS BETWEEN THE IMPURITY AND ALL SC LEVELS:
//according to: http://www.itensor.org/docs.cgi?vers=cppv3&page=formulas/correlator_mps

// <n_imp n_i>
auto calcChargeCorrelation(MPS& psi, const params &p) {
  std::vector<double> r;
  double tot = 0;
  auto impOp = op(p.sites, "Ntot", p.impindex);
  for (auto j: range1(1, length(psi))) {
    if (j != p.impindex) { // skip impurity site!
      auto scOp = op(p.sites, "Ntot", j);
      double result = ImpurityCorrelator(psi, impOp, j, scOp, p);
      r.push_back(result);
      tot += result;
    }
  }
  return std::make_pair(r, tot);
}

void ChargeCorrelation(MPS& psi, auto & file, std::string path, const params &p) {
  auto [r, tot] = calcChargeCorrelation(psi, p);
  std::cout << "charge correlation = " << std::setprecision(full) << r << std::endl;
  std::cout << "charge correlation tot = " << tot << std::endl;
  dump(file, path + "/charge_correlation", r);
  dump(file, path + "/charge_correlation_total", tot);
}

// <S_imp S_i> = <Sz_imp Sz_i> + 1/2 ( <S+_imp S-_i> + <S-_imp S+_i> )
auto calcSpinCorrelation(MPS& psi, const params &p) {
  std::vector<double> rzz, rpm, rmp;
  //squares of the impurity operators; for on-site terms
  double tot = 0;
    
  //impurity spin operators
  auto impSz = 0.5*( op(p.sites, "Nup", p.impindex) - op(p.sites, "Ndn", p.impindex) );
  auto impSp = op(p.sites, "Cdagup*Cdn", p.impindex);
  auto impSm = op(p.sites, "Cdagdn*Cup", p.impindex);
  auto SzSz = 0.25 * ( op(p.sites, "Nup*Nup", p.impindex) - op(p.sites, "Nup*Ndn", p.impindex) - op(p.sites, "Ndn*Nup", p.impindex) + op(p.sites, "Ndn*Ndn", p.impindex));
  //SzSz term
  psi.position(p.impindex);
  //on site term
  auto onSiteSzSz = elt(psi(p.impindex) * SzSz *  dag(prime(psi(p.impindex),"Site")));
  tot += onSiteSzSz;
  for(auto j: range1(1, length(psi))) {
    if (j != p.impindex) {
      auto scSz = 0.5*( op(p.sites, "Nup", j) - op(p.sites, "Ndn", j) );
      auto result = ImpurityCorrelator(psi, impSz, j, scSz, p);
      rzz.push_back(result);
      tot += result;
    }
  }
  //S+S- term
  psi.position(p.impindex);
  //on site term
  auto onSiteSpSm = elt(psi(p.impindex) * op(p.sites, "Cdagup*Cdn*Cdagdn*Cup", p.impindex) * dag(prime(psi(p.impindex),"Site")));
  tot += 0.5*onSiteSpSm; 
  for(auto j: range1(1, length(psi))) {
    if (j != p.impindex) {
      auto scSm = op(p.sites, "Cdagdn*Cup", j);
      auto result = ImpurityCorrelator(psi, impSp, j, scSm, p);
      rpm.push_back(result);
      tot += 0.5*result;
    }
  }
  //S- S+ term
  psi.position(p.impindex);
  //on site term
  auto onSiteSmSp = elt(psi(p.impindex) * op(p.sites, "Cdagdn*Cup*Cdagup*Cdn", p.impindex) * dag(prime(psi(p.impindex),"Site")));
  tot += 0.5*onSiteSmSp; 
  for(auto j: range1(1, length(psi))) {
    if (j != p.impindex) {
      auto scSp = op(p.sites, "Cdagup*Cdn", j);
      auto result = ImpurityCorrelator(psi, impSm, j, scSp, p);
      rmp.push_back(result);
      tot += 0.5*result;
    }
  }
  return std::make_tuple(onSiteSzSz, onSiteSpSm, onSiteSmSp, rzz, rpm, rmp, tot);
}

void SpinCorrelation(MPS& psi, File & file, std::string path, const params &p) {
  auto [onSiteSzSz, onSiteSpSm, onSiteSmSp, rzz, rpm, rmp, tot] = calcSpinCorrelation(psi, p);
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
  dump(file, path + "/spin_correlation_imp/zz", onSiteSzSz);
  dump(file, path + "/spin_correlation_imp/pm", onSiteSpSm);
  dump(file, path + "/spin_correlation_imp/mp", onSiteSmSp);
  dump(file, path + "/spin_correlation/zz", rzz);
  dump(file, path + "/spin_correlation/pm", rpm);
  dump(file, path + "/spin_correlation/mp", rmp);
  dump(file, path + "/spin_correlation_total", tot);
}

auto calcPairCorrelation(MPS& psi, const params &p) {
  std::vector<double> r;
  double tot = 0;
  auto impOp = op(p.sites, "Cup*Cdn", p.impindex);
  for(auto j: range1(2, length(psi))) {
    auto scOp = op(p.sites, "Cdagdn*Cdagup", j);
    double result = ImpurityCorrelator(psi, impOp, j, scOp, p);
    r.push_back(result);
    tot += result;
  }
  return std::make_pair(r, tot);
}

void PairCorrelation(MPS& psi, File & file, std::string path, const params &p) {
  auto [r, tot] = calcPairCorrelation(psi, p);
  std::cout << "pair correlation = " << std::setprecision(full) << r << std::endl;
  std::cout << "pair correlation tot = " << tot << std::endl;
  dump(file, path + "/pair_correlation", r);
  dump(file, path + "/pair_correlation_total", tot);
}

//Prints <d^dag c_i + c_i^dag d> for each i. the sum of this expected value, weighted by 1/sqrt(N)
//gives <d^dag f_0 + f_0^dag d>, where f_0 = 1/sqrt(N) sum_i c_i. This is the expected value of hopping.
auto calcexpectedHopping(MPS& psi, const params &p) {
  std::vector<double> rup, rdn;
  double totup = 0;
  double totdn = 0;
  auto impOpUp = op(p.sites, "Cup", p.impindex);
  auto impOpDagUp = op(p.sites, "Cdagup", p.impindex);
  auto impOpDn = op(p.sites, "Cdn", p.impindex);
  auto impOpDagDn = op(p.sites, "Cdagdn", p.impindex);
  // hopping expectation values for spin up
  for (auto j : range1(1, length(psi))) {
    if (j != p.impindex) {
      auto scDagOp = op(p.sites, "Cdagup", j);
      auto scOp = op(p.sites, "Cup", j);
      auto result1 = ImpurityCorrelator(psi, impOpUp, j, scDagOp, p); // <d c_i^dag>
      auto result2 = ImpurityCorrelator(psi, impOpDagUp, j, scOp, p); // <d^dag c_i>
      auto sum = result1+result2;
      rup.push_back(sum);
      totup += sum;
    }
  }
  // hopping expectation values for spin dn
  for (auto j : range1(1, length(psi))) {
    if (j != p.impindex) {
      auto scDagOp = op(p.sites, "Cdagdn", j);
      auto scOp = op(p.sites, "Cdn", j);
      auto result1 = ImpurityCorrelator(psi, impOpDn, j, scDagOp, p);    // <d c_i^dag>
      auto result2 =  ImpurityCorrelator(psi, impOpDagDn, j, scOp, p); // <d^dag c_i>
      auto sum = result1+result2;
      rdn.push_back(sum);
      totdn += sum;
    }
  }
  return std::make_tuple(rup, rdn, totup, totdn);
}

void expectedHopping(MPS& psi, File & file, std::string path, const params &p) {
  auto [rup, rdn, totup, totdn] = calcexpectedHopping(psi, p);
  std::cout << "hopping spin up = " << std::setprecision(full) << rup << std::endl;
  std::cout << "hopping correlation up tot = " << totup << std::endl;
  std::cout << "hopping spin down = " << std::setprecision(full) << rdn << std::endl;
  std::cout << "hopping correlation down tot = " << totdn << std::endl;
  auto tot = totup+totdn;
  std::cout << "total hopping correlation = " << tot << std::endl;
  dump(file, path + "/hopping/up", rup);
  dump(file, path + "/hopping/dn", rdn);
  dump(file, path + "/hopping_total/up",  totup);
  dump(file, path + "/hopping_total/dn",  totdn);
  dump(file, path + "/hopping_total/sum", tot);
}

//prints the occupation number Nup and Ndn at the impurity
auto calcImpurityUpDn(MPS& psi, const params &p){
  psi.position(p.impindex);
  auto valnup = psi.A(p.impindex) * p.sites.op("Nup",p.impindex)* dag(prime(psi.A(p.impindex),"Site"));
  auto valndn = psi.A(p.impindex) * p.sites.op("Ndn",p.impindex)* dag(prime(psi.A(p.impindex),"Site"));
  return std::make_pair(std::real(valnup.cplx()), std::real(valndn.cplx()));
}

void ImpurityUpDn(MPS& psi, auto &file, std::string path, const params &p){
  auto [up, dn] = calcImpurityUpDn(psi, p);
  auto sz = 0.5*(up-dn);
  std::cout << "impurity nup ndn = " << std::setprecision(full) << up << " " << dn << " sz = " << sz << std::endl;
  dump(file, path + "/impurity_Nup", up);
  dump(file, path + "/impurity_Ndn", dn);
  dump(file, path + "/impurity_Sz",  sz);
}

// total Sz of the state
auto calcTotalSpinz(MPS& psi, const params &p) {
  double totNup = 0.;
  double totNdn = 0.;
  for (auto j: range1(length(psi))) {
    psi.position(j);
    auto Nupi = psi.A(j) * p.sites.op("Nup",j)* dag(prime(psi.A(j),"Site"));
    auto Ndni = psi.A(j) * p.sites.op("Ndn",j)* dag(prime(psi.A(j),"Site"));
    totNup += std::real(Nupi.cplx());
    totNdn += std::real(Ndni.cplx());
  }
  auto totSz =  0.5*(totNup-totNdn);
  return std::make_tuple(totNup, totNdn, totSz);
}

void TotalSpinz(MPS& psi, File &file, std::string path, const params &p) {
  auto [totNup, totNdn, totSz] = calcTotalSpinz(psi, p);
  std::cout << std::setprecision(full) << "Total spin z: " << " Nup = " << totNup << " Ndn = " << totNdn << " Sztot = " << totSz << std::endl;
  dump(file, path + "/total_Nup", totNup);
  dump(file, path + "/total_Ndn", totNdn);
  dump(file, path + "/total_Sz",  totSz);
}

// occupation numbers of all levels in the problem
auto calcOcc(MPS &psi, const params &p) {
  std::vector<double> r;
  for(auto i : range1(length(psi)) ) {
    // position call very important! otherwise one would need to contract the whole tensor network of <psi|O|psi> this way, only the local operator at site i is needed
    psi.position(i);
    auto val = psi.A(i) * p.sites.op("Ntot",i)* dag(prime(psi.A(i),"Site"));
    r.push_back(std::real(val.cplx()));
  }
  return r;
}

void MeasureOcc(MPS& psi, auto & file, std::string path, const params &p) {
  auto r = calcOcc(psi, p);
  auto tot = std::accumulate(r.begin(), r.end(), 0.0);
  std::cout << "site occupancies = " << std::setprecision(full) << r << std::endl;
  std::cout << "tot = " << tot << std::endl;
  dump(file, path + "/site_occupancies", r);
  dump(file, path + "/total_occupancy", tot);
}

// This is actually sqrt of local charge correlation, <n_up n_down>-<n_up><n_down>, summed over all bath levels.
// The sum (tot) corresponds to \bar{\Delta}', Eq. (4) in Braun, von Delft, PRB 50, 9527 (1999), first proposed by Dan Ralph.
// It reduces to Delta_BCS in the thermodynamic limit (if the impurity is decoupled, Gamma=0).
auto calcPairing(MPS &psi, const params &p) {
  std::vector<complex_t> r;
  complex_t tot = 0;
  for(auto i : range1(length(psi))) {
    psi.position(i);
    auto val2  = psi.A(i) * p.sites.op("Cdagup*Cup*Cdagdn*Cdn", i) * dag(prime(psi.A(i),"Site"));
    auto val1u = psi.A(i) * p.sites.op("Cdagup*Cup", i) * dag(prime(psi.A(i),"Site"));
    auto val1d = psi.A(i) * p.sites.op("Cdagdn*Cdn", i) * dag(prime(psi.A(i),"Site"));
    // For Gamma>0, <C+CC+C>-<C+C><C+C> may be negative.
    auto diff = val2.cplx() - val1u.cplx() * val1d.cplx();
    auto sq = p.sc->g() * sqrt(diff); // XXX: only meaningful for single channel!
    r.push_back(sq);
    if (i != p.impindex) tot += sq; // exclude the impurity site in the sum
  }
  return std::make_pair(r, tot);
}

void MeasurePairing(MPS& psi, auto & file, std::string path, const params &p) {
  auto [r, tot] = calcPairing(psi, p);
  std::cout << "site pairing = " << std::setprecision(full) << r << std::endl;
  std::cout << "tot = " << tot << std::endl;
  dumpreal(file, path + "/pairing", r);
  dumpreal(file, path + "/pairing_total", tot);
}

// See von Delft, Zaikin, Golubev, Tichy, PRL 77, 3189 (1996)
// v = <c^\dag_up c^\dag_dn c_dn c_up>
// u = <c_dn c_up c^\dag_up c^\dag_dn>
auto calcAmplitudes(MPS &psi, const params &p) {
  std::vector<complex_t> rv, ru, rpdt;
  complex_t tot = 0;
  for(auto i : range1(length(psi)) ) {
    psi.position(i);
    auto valv = psi.A(i) * p.sites.op("Cdagup*Cdagdn*Cdn*Cup", i) * dag(prime(psi.A(i),"Site"));
    auto valu = psi.A(i) * p.sites.op("Cdn*Cup*Cdagup*Cdagdn", i) * dag(prime(psi.A(i),"Site"));
    auto v = sqrt( std::real(valv.cplx()) ); // XXX
    auto u = sqrt( std::real(valu.cplx()) );
    auto pdt = v*u;
    auto element = p.sc->g() * pdt;
    ru.push_back(u);
    rv.push_back(v);
    rpdt.push_back(pdt);
    if (i != p.impindex) tot += element; // exclude the impurity site in the sum
  }
  return std::make_tuple(rv, ru, rpdt, tot);
}

void MeasureAmplitudes(MPS& psi, auto & file, std::string path, const params &p) {
  auto [rv, ru, rpdt, tot] = calcAmplitudes(psi, p);
  std::cout << "amplitudes vu = " << std::setprecision(full);
  for (size_t i = 0; i < rv.size(); i++)
    std::cout << "[v=" << rv[i] << " u=" << ru[i] << " pdt=" << rpdt[i] << "] ";
  std::cout << std::endl << "tot = " << tot << std::endl;
  dumpreal(file, path + "/amplitudes/u", ru);
  dumpreal(file, path + "/amplitudes/v", rv);
  dumpreal(file, path + "/amplitudes/pdt", rpdt);
  dumpreal(file, path + "/amplitudes_total", tot);
}

// Computed entanglement/von Neumann entropy between the impurity and the system.
// Copied from https://www.itensor.org/docs.cgi?vers=cppv3&page=formulas/entanglement_mps
// von Neumann entropy at the bond between impurity and next site.
auto calcEntropy(MPS& psi, const params &p) {
  my_assert(p.impindex == 1); // Works as intended only if p.impindex=1.
  psi.position(p.impindex);
  // SVD this wavefunction to get the spectrum of density-matrix eigenvalues
  auto l = leftLinkIndex(psi, p.impindex);
  auto s = siteIndex(psi, p.impindex);
  auto [U,S,V] = svd(psi(p.impindex), {l,s});
  auto u = commonIndex(U,S);
  //Apply von Neumann formula to the squares of the singular values
  double SvN = 0.;
  for(auto n : range1(dim(u))) {
    auto Sn = elt(S,n,n);
    auto pp = sqr(Sn);
    if(pp > 1E-12) SvN += -pp*log(pp);
  }
  return SvN;
}

void PrintEntropy(MPS& psi, auto & file, std::string path, const params &p) {
  auto SvN = calcEntropy(psi, p);
  std::cout << fmt::format("Entanglement entropy across impurity bond b={}, SvN = {:10}", p.impindex, SvN) << std::endl;
  dump(file, path + "/entanglement_entropy_imp", SvN);
}

//calculates the groundstates and the energies of the relevant particle number sectors
void FindGS(InputGroup &input, store &s, params &p){
  auto inputsw = InputGroup(p.inputfn,"sweeps");
  auto sw_table = InputGroup(inputsw,"sweeps");
  int nrsweeps = input.getInt("nrsweeps", 15);
  auto sweeps = Sweeps(nrsweeps,sw_table);

#pragma omp parallel for if(p.parallel) 
  for (size_t i=0; i<p.iterateOver.size(); i++){
    auto sub = p.iterateOver[i];
    auto [ntot, Sz] = sub;
    std::cout << "\nSweeping in the sector with " << ntot << " particles, Sz = " << Sz << std::endl;
    auto [H, Eshift] = p.problem->initH(ntot, p);
    auto psi_init = initPsi(ntot, Sz, p.sites, p.impindex, p.sc_only, p.randomMPSb);
    Args args; // args is used to store and transport parameters between various functions
    // Apply the MPO a couple of times to get DMRG started, otherwise it might not converge.
    for(auto i : range1(p.nrH)){
      psi_init = applyMPO(H,psi_init,args);
      psi_init.noPrime().normalize();
    }
    auto [E, psi] = dmrg(H, psi_init, sweeps, {"Silent", p.parallel, 
                                               "Quiet", !p.printDimensions, 
                                               "EnergyErrgoal", p.EnergyErrgoal});
    double GSenergy = E+Eshift;
    s.eigen0[sub] = eigenpair(GSenergy, psi);
    s.stats0[sub] = psi_stats(E, psi, H);
    if (p.excited_state) {
      auto wfs = std::vector<MPS>(1);
      wfs.at(0) = psi;
      auto [E1, psi1] = dmrg(H, wfs, psi, sweeps, {"Silent", p.parallel,
                                                   "Quiet", !p.printDimensions,
                                                   "Weight", p.Weight});
      double ESenergy = E1+Eshift;
      s.eigen1[sub] = eigenpair(ESenergy, psi1);
    }
  }
}

// calculates <psi1|c_dag|psi2>, according to http://itensor.org/docs.cgi?vers=cppv3&page=formulas/mps_onesite_op
auto ExpectationValueAddEl(MPS psi1, MPS psi2, std::string spin, const params &p){
  psi2.position(p.impindex);                                                      // set orthogonality center
  auto newTensor = noPrime(op(p.sites,"Cdag"+spin, p.impindex)*psi2(p.impindex)); // apply the local operator
  psi2.set(p.impindex,newTensor);                                                 // plug in the new tensor, with the operator applied
  return inner(psi1, psi2);
}

// calculates <psi1|c|psi2>
auto ExpectationValueTakeEl(MPS psi1, MPS psi2, std::string spin, const params &p){
  psi2.position(p.impindex);
  auto newTensor = noPrime(op(p.sites,"C"+spin, p.impindex)*psi2(p.impindex));
  psi2.set(p.impindex,newTensor);
  return inner(psi1, psi2);
}

void calc_weight(store &s, subspace subGS, subspace subES, int q, std::string sz, auto & file, params &p)
{
  double res;
  MPS & psiGS = s.eigen0[subGS].psi();
  if (s.eigen0.find(subES) != s.eigen0.end()) {
    MPS & psiES = s.eigen0[subES].psi();
    res = (q == +1 ? ExpectationValueAddEl(psiES, psiGS, sz, p) : ExpectationValueTakeEl(psiES, psiGS, sz, p));
    std::cout << "weight w" << (q == +1 ? "+" : "-") << " " << sz << ": " << res << std::endl;
  } else {
    res = std::numeric_limits<double>::quiet_NaN();
    std::cout <<  "ERROR: we don't have info about the sector " << subES << std::endl;
  }
  dump(file, "weights/" + std::to_string(q) + "/" + sz, res);
}

void calculate_spectral_weights(store &s, subspace subGS, auto & file, params &p) {
  std::cout << std::endl << "Spectral weights:" << std::endl 
    << "(Spectral weight is the square of the absolute value of the number.)" << std::endl;
  auto [N_GS, Sz_GS] = subGS;
  calc_weight(s, subGS, subspace(N_GS+1, Sz_GS+0.5), +1, "up", file, p);
  calc_weight(s, subGS, subspace(N_GS+1, Sz_GS-0.5), +1, "dn", file, p);
  calc_weight(s, subGS, subspace(N_GS-1, Sz_GS-0.5), -1, "up", file, p);
  calc_weight(s, subGS, subspace(N_GS-1, Sz_GS+0.5), -1, "dn", file, p);
}

void print_energies(store &s, double EGS, params &p) {
  std::cout << std::endl;
  for(auto ntot: p.numPart)
    for(auto Sz: p.Szs[ntot]) {
      auto E0 = s.eigen0[subspace(ntot, Sz)].E();
      std::cout << fmt::format("n = {:5}  Sz = {:4}  E = {22:17}  DeltaE = {22:17}", ntot, Sz, E0, E0-EGS) << std::endl;
      if (p.excited_state) {
        auto E1 = s.eigen1[subspace(ntot, Sz)].E();
        std::cout << fmt::format(" 1st excited state    E = {22:17}  DeltaE = {22:17}", E1, E1-EGS) << std::endl;
      }
    }
}

auto find_global_GS_subspace(store &s, auto & file) {
  subspace subGS;
  double EGS = std::numeric_limits<double>::infinity();
  for(const auto & [sub, eig] : s.eigen0) {
    auto E0 = eig.E();
    if (E0 < EGS) {
      EGS = E0;
      subGS = sub;
    }
  }
  auto [N_GS, Sz_GS] = subGS;
  std::cout << fmt::format("\nN_GS = {}\nSZ_GS = {}\n",N_GS, Sz_GS);
  dump(file, "/GS/N",  N_GS);
  dump(file, "/GS/Sz", Sz_GS);
  return subGS;
}

// Loops over all particle sectors and prints relevant quantities
void calculateAndPrint(InputGroup &input, store &s, params &p) {
  File file("solution.h5", File::Overwrite);
  for(auto ntot: p.numPart) {
    for (auto Sz: p.Szs[ntot]) {
      auto sub = subspace(ntot, Sz);
      auto E = s.eigen0[sub].E();
      dump(file, str(sub, "0/E"), E);
      MPS & GS = s.eigen0[sub].psi();
      std::cout << fmt::format("\n\nRESULTS FOR THE SECTOR WITH {} PARTICLES, Sz {}:", ntot, Sz) << std::endl
        << fmt::format("Ground state energy = {}", E) << std::endl
        << fmt::format("norm = {}", s.stats0[sub].norm()) << std::endl;
      auto path0 = str(sub, "0");
      MeasureOcc(GS, file, path0, p);
      MeasurePairing(GS, file, path0, p);
      MeasureAmplitudes(GS, file, path0, p);
      if (p.computeEntropy) PrintEntropy(GS, file, path0, p);
      if (p.impNupNdn) ImpurityUpDn(GS, file, path0, p);
      if (p.chargeCorrelation) ChargeCorrelation(GS, file, path0, p);
      if (p.spinCorrelation) SpinCorrelation(GS, file, path0, p);
      if (p.pairCorrelation) PairCorrelation(GS, file, path0, p);
      if (p.hoppingExpectation) expectedHopping(GS, file, path0, p);
      if (p.printTotSpinZ) TotalSpinz(GS, file, path0, p);
      // various measures of convergence (energy deviation, residual value)
      std::cout << fmt::format("Eigenvalue(bis): <GS|H|GS> = {}", s.stats0[sub].Ebis()) << std::endl
        << fmt::format("diff: E_GS - <GS|H|GS> = {}", E-s.stats0[sub].Ebis()) << std::endl // TODO: remove this
        << fmt::format("deltaE: sqrt(<GS|H^2|GS> - <GS|H|GS>^2) = {}", s.stats0[sub].deltaE()) << std::endl
        << fmt::format("residuum: <GS|H|GS> - E_GS*<GS|GS> = {}", s.stats0[sub].residuum()) << std::endl;
      s.stats0[sub].dump(file, path0);
      if (p.excited_state){
        double E1 = s.eigen1[sub].E();
        MPS & ES = s.eigen1[sub].psi();
        dump(file, str(sub, "1/E"), E1);
        MeasureOcc(ES, file, str(sub, "1"), p);
        std::cout << fmt::format("Excited state energy = {}", E1) << std::endl;
       }
    } //end of Sz for loop 
  } //end of ntot for loop
  subspace subGS = find_global_GS_subspace(s, file);
  auto EGS = s.eigen0[subGS].E();
  print_energies(s, EGS, p);
  if (p.calcweights) 
    calculate_spectral_weights(s, subGS, file, p);
}
