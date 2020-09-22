#include "FindGS.h"

std::vector<subspace_t> init_subspace_lists(params &p)
{
  const int nhalf = p.N;  // total nr of electrons at half-filling
  int nref = nhalf;       // default
  if (p.nref >= 0)        // override
    nref = p.nref;
  else if (p.refisn0)     // adaptable
    nref = round(p.sc->n0() + 0.5 - (p.qd->eps()/p.qd->U())); // calculation of the energies is centered around this n

  std::vector<subspace_t> l;
  for (const auto &ntot : n_list(nref, p.nrange)) {
    spin szmax = even(ntot) ? (p.spin1 ? 1 : 0) : 0.5;
    spin szmin = p.magnetic_field() ? -szmax : (even(ntot) ? 0 : 0.5);
    for (spin sz = szmin; sz <= szmax; sz += 1.0)
      l.push_back({ntot, sz});
  }
  return l;
}

void parse_cmd_line(int argc, char *argv[], params &p) {
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
  const double U = input.getReal("U", 0); // need to parse it first because it enters the default value for epsimp just below
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
  p.spin1 = input.getYesNo("spin1", false);
  p.excited_state = input.getYesNo("excited_state", false);
  p.excited_states = input.getInt("excited_states", 0);
  if (p.excited_states >= 1)
    p.excited_state = true;
  if (p.excited_state && p.excited_states == 0)
    p.excited_states = 1; // override

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
  p.nrsweeps = input.getInt("nrsweeps", 15);
  p.parallel = input.getYesNo("parallel", true); // parallel by default!
  p.Quiet = input.getYesNo("Quiet", true);
  p.Silent = input.getYesNo("Silent", p.parallel);
  p.verbose = input.getYesNo("verbose", !p.parallel);
  p.EnergyErrgoal = input.getReal("EnergyErrgoal", 1e-16);
  p.nrH = input.getInt("nrH", 5);
  p.sc_only = input.getYesNo("sc_only", false);
  p.randomMPS_GS = input.getYesNo("randomMPS_GS", false);
  p.randomMPS_ES = input.getYesNo("randomMPS_ES", false);
  p.Weight = input.getReal("Weight", 11.0);
  p.overlaps = input.getYesNo("overlaps", false);
}

// computes the correlation between operator impOp on impurity and operator opj on j
/*double ImpurityCorrelator(MPS& psi, auto impOp, int j, auto opj, const params &p) {
  Expects(p.impindex == 1);
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
*/

double ImpurityCorrelator(MPS& psi, auto impOp, int j, auto opj, const params &p) {
  //Expects(p.impindex == 1);
  psi.position(p.impindex);
  MPS psidag = dag(psi);
  psidag.prime("Link");
  

  int first = std::min(p.impindex, j);
  int second = std::max(p.impindex, j);
  // apply the operator to the first site (wheter impurity or j)
  auto li_1 = leftLinkIndex(psi,first);
  auto C = prime(psi(first),li_1)*(first==j ? opj : impOp) ;
  C *= prime(psidag(first),"Site");
  
  for (int k = first+1; k < second; ++k){
    C *= psi(k);
    C *= psidag(k);
  }
  auto lj = rightLinkIndex(psi,second);
  C *= prime(psi(second),lj)*(first==j ? impOp : opj);
  C *= prime(psidag(second),"Site");
  return elt(C);
}




//CORRELATION FUNCTIONS BETWEEN THE IMPURITY AND ALL SC LEVELS:
//according to: http://www.itensor.org/docs.cgi?vers=cppv3&page=formulas/correlator_mps

// <n_imp n_i>
auto calcChargeCorrelation(MPS& psi, const ndx_t sites, const params &p) {
  std::vector<double> r;
  double tot = 0;
  auto impOp = op(p.sites, "Ntot", p.impindex);
  for (const auto j: sites) {
    if (j != p.impindex) { // skip impurity site!
      auto scOp = op(p.sites, "Ntot", j);
      double result = ImpurityCorrelator(psi, impOp, j, scOp, p);
      r.push_back(result);
      tot += result;
    }
  }
  return std::make_pair(r, tot);
}

void MeasureChargeCorrelation(MPS& psi, auto & file, std::string path, const params &p) {
  const auto [r, tot] = calcChargeCorrelation(psi, range(1, p.N), p);
  std::cout << "charge correlation = " << std::setprecision(full) << r << std::endl;
  std::cout << "charge correlation tot = " << tot << std::endl;
  dump(file, path + "/charge_correlation", r);
  dump(file, path + "/charge_correlation_total", tot);
}

// <S_imp S_i> = <Sz_imp Sz_i> + 1/2 ( <S+_imp S-_i> + <S-_imp S+_i> )
auto calcSpinCorrelation(MPS& psi, const ndx_t &sites, const params &p) {
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
  for(const auto j: sites) {
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
  for(const auto j: sites) {
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
  for(const auto j: sites) {
    if (j != p.impindex) {
      auto scSp = op(p.sites, "Cdagup*Cdn", j);
      auto result = ImpurityCorrelator(psi, impSm, j, scSp, p);
      rmp.push_back(result);
      tot += 0.5*result;
    }
  }
  return std::make_tuple(onSiteSzSz, onSiteSpSm, onSiteSmSp, rzz, rpm, rmp, tot);
}

void MeasureSpinCorrelation(MPS& psi, File & file, std::string path, const params &p) {
  const auto [onSiteSzSz, onSiteSpSm, onSiteSmSp, rzz, rpm, rmp, tot] = calcSpinCorrelation(psi, range(1, p.N), p);
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

auto calcPairCorrelation(MPS& psi, const ndx_t &sites, const params &p) {
  std::vector<double> r;
  double tot = 0;
  auto impOp = op(p.sites, "Cup*Cdn", p.impindex);
  for(const auto j: sites) {
    if (j != p.impindex) {
      auto scOp = op(p.sites, "Cdagdn*Cdagup", j);
      double result = ImpurityCorrelator(psi, impOp, j, scOp, p);
      r.push_back(result);
      tot += result;
    }
  }
  return std::make_pair(r, tot);
}

void MeasurePairCorrelation(MPS& psi, File & file, std::string path, const params &p) {
  const auto [r, tot] = calcPairCorrelation(psi, range(1, p.N), p);
  std::cout << "pair correlation = " << std::setprecision(full) << r << std::endl;
  std::cout << "pair correlation tot = " << tot << std::endl;
  dump(file, path + "/pair_correlation", r);
  dump(file, path + "/pair_correlation_total", tot);
}

//Prints <d^dag c_i + c_i^dag d> for each i. the sum of this expected value, weighted by 1/sqrt(N)
//gives <d^dag f_0 + f_0^dag d>, where f_0 = 1/sqrt(N) sum_i c_i. This is the expected value of hopping.
auto calcHopping(MPS& psi, const ndx_t &sites, const params &p) {
  std::vector<double> rup, rdn;
  double totup = 0;
  double totdn = 0;
  auto impOpUp = op(p.sites, "Cup", p.impindex);
  auto impOpDagUp = op(p.sites, "Cdagup", p.impindex);
  auto impOpDn = op(p.sites, "Cdn", p.impindex);
  auto impOpDagDn = op(p.sites, "Cdagdn", p.impindex);
  // hopping expectation values for spin up
  for (const auto j : sites) {
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
  for (const auto j : sites) {
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

void MeasureHopping(MPS& psi, File & file, std::string path, const params &p) {
  const auto [rup, rdn, totup, totdn] = calcHopping(psi, range(1, p.N), p);
  std::cout << "hopping spin up = " << std::setprecision(full) << rup << std::endl;
  std::cout << "hopping correlation up tot = " << totup << std::endl;
  std::cout << "hopping spin down = " << std::setprecision(full) << rdn << std::endl;
  std::cout << "hopping correlation down tot = " << totdn << std::endl;
  const auto tot = totup+totdn;
  std::cout << "total hopping correlation = " << tot << std::endl;
  dump(file, path + "/hopping/up", rup);
  dump(file, path + "/hopping/dn", rdn);
  dump(file, path + "/hopping_total/up",  totup);
  dump(file, path + "/hopping_total/dn",  totdn);
  dump(file, path + "/hopping_total/sum", tot);
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
  dump(file, path + "/impurity_Nup", up);
  dump(file, path + "/impurity_Ndn", dn);
  dump(file, path + "/impurity_Sz",  sz);
}

// total Sz of the state
auto calcTotalSpinz(MPS& psi, const ndx_t &sites, const params &p) {
  double totNup = 0.;
  double totNdn = 0.;
  for (const auto j: sites) {
    psi.position(j);
    auto Nupi = psi.A(j) * p.sites.op("Nup",j)* dag(prime(psi.A(j),"Site"));
    auto Ndni = psi.A(j) * p.sites.op("Ndn",j)* dag(prime(psi.A(j),"Site"));
    totNup += std::real(Nupi.cplx());
    totNdn += std::real(Ndni.cplx());
  }
  const auto totSz =  0.5*(totNup-totNdn);
  return std::make_tuple(totNup, totNdn, totSz);
}

void MeasureTotalSpinz(MPS& psi, File &file, std::string path, const params &p) {
  const auto [totNup, totNdn, totSz] = calcTotalSpinz(psi, range(1, p.N), p);
  std::cout << std::setprecision(full) << "Total spin z: " << " Nup = " << totNup << " Ndn = " << totNdn << " Sztot = " << totSz << std::endl;
  dump(file, path + "/total_Nup", totNup);
  dump(file, path + "/total_Ndn", totNdn);
  dump(file, path + "/total_Sz",  totSz);
}

// occupation numbers of levels 'sites'
auto calcOccupancy(MPS &psi, const ndx_t &sites, const params &p) {
  std::vector<double> r;
  for(const auto i : sites) {
    // position call very important! otherwise one would need to contract the whole tensor network of <psi|O|psi> this way, only the local operator at site i is needed
    psi.position(i);
    const auto val = psi.A(i) * p.sites.op("Ntot",i) * dag(prime(psi.A(i),"Site"));
    r.push_back(std::real(val.cplx()));
  }
  return r;
}

void MeasureOccupancy(MPS& psi, auto & file, std::string path, const params &p) {
  const auto r = calcOccupancy(psi, range(1, p.N), p);
  const auto tot = std::accumulate(r.cbegin(), r.cend(), 0.0);
  std::cout << "site occupancies = " << std::setprecision(full) << r << std::endl;
  std::cout << "tot = " << tot << std::endl;
  dump(file, path + "/site_occupancies", r);
  dump(file, path + "/total_occupancy", tot);
}

// This is actually sqrt of local charge correlation, <n_up n_down>-<n_up><n_down>, summed over all bath levels.
// The sum (tot) corresponds to \bar{\Delta}', Eq. (4) in Braun, von Delft, PRB 50, 9527 (1999), first proposed by Dan Ralph.
// It reduces to Delta_BCS in the thermodynamic limit (if the impurity is decoupled, Gamma=0).
auto calcPairing(MPS &psi, const ndx_t &sites, const params &p) {
  std::vector<complex_t> r;
  complex_t tot = 0;
  for(const auto i : sites) {
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
  const auto [r, tot] = calcPairing(psi, range(1, p.N), p);
  std::cout << "site pairing = " << std::setprecision(full) << r << std::endl;
  std::cout << "tot = " << tot << std::endl;
  dumpreal(file, path + "/pairing", r);
  dumpreal(file, path + "/pairing_total", tot);
}

// See von Delft, Zaikin, Golubev, Tichy, PRL 77, 3189 (1996)
// v = <c^\dag_up c^\dag_dn c_dn c_up>
// u = <c_dn c_up c^\dag_up c^\dag_dn>
auto calcAmplitudes(MPS &psi, const ndx_t &sites, const params &p) {
  std::vector<complex_t> rv, ru, rpdt;
  complex_t tot = 0;
  for(const auto i : sites) {
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
  const auto [rv, ru, rpdt, tot] = calcAmplitudes(psi, range(1, p.N), p);
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
  Expects(p.impindex == 1); // Works as intended only if p.impindex=1.
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

void MeasureEntropy(MPS& psi, auto & file, std::string path, const params &p) {
  const auto SvN = calcEntropy(psi, p);
  std::cout << fmt::format("Entanglement entropy across impurity bond b={}, SvN = {:10}", p.impindex, SvN) << std::endl;
  dump(file, path + "/entanglement_entropy_imp", SvN);
}

auto sweeps(params &p)
{
  auto inputsw = InputGroup(p.inputfn, "sweeps");
  auto sw_table = InputGroup(inputsw, "sweeps");
  return Sweeps(p.nrsweeps, sw_table);
}

void solve_subspace(const subspace_t &sub, store &s, params &p) {
  std::cout << "\nSweeping in the sector " << sub << ", ground state" << std::endl;
  auto [H, Eshift] = p.problem->initH(sub, p);
  auto state = p.problem->initState(sub, p);
  MPS psi_init(state);
  if (p.randomMPS_GS)
    psi_init = randomMPS(state);
  Args args; // args is used to store and transport parameters between various functions
  // Apply the MPO a couple of times to get DMRG started, otherwise it might not converge.
  for(auto i : range1(p.nrH)){
    psi_init = applyMPO(H,psi_init,args);
    psi_init.noPrime().normalize();
  }
  auto [E, psi] = dmrg(H, psi_init, sweeps(p), {"Silent", p.Silent, 
      "Quiet", p.Quiet, "EnergyErrgoal", p.EnergyErrgoal});
  const double GSenergy = E+Eshift;
  s.eigen[gs(sub)] = eigenpair(GSenergy, psi);
  s.stats[gs(sub)] = psi_stats(psi, H);
  if (p.excited_state) {
    std::vector<MPS> wfs;
    MPS psi_prev = psi;
    for (auto n = 1; n <= p.excited_states; n++) {
      std::cout << "\nSweeping in the sector " << sub << ", excited state n=" << n << std::endl;
      wfs.push_back(psi_prev);
      MPS psi_init_es = psi_prev;
      if (p.randomMPS_ES) {
        psi_init_es = randomMPS(state);
        psi_init_es.normalize();
      }
      auto [E_n, psi_n] = dmrg(H, wfs, psi_init_es, sweeps(p), {"Silent", p.Silent,
          "Quiet", p.Quiet, "Weight", p.Weight});
      const double ESenergy = E_n+Eshift;
      s.eigen[es(sub, n)] = eigenpair(ESenergy, psi_n);
      s.stats[es(sub, n)] = psi_stats(psi_n, H);
      psi_prev = psi_n;
    }
  }
}

void solve_all(const std::vector<subspace_t> &l, store &s, params &p) {
  if (p.parallel)
    std::for_each(std::execution::par, l.cbegin(), l.cend(), 
                  [&s,&p](const auto  &sub) { solve_subspace(sub, s, p); });
  else
    std::for_each(std::execution::seq, l.cbegin(), l.cend(), 
                  [&s,&p](const auto  &sub) { solve_subspace(sub, s, p); });
}
  
void calc_properties(const state_t st, File &file, store &s, params &p)
{
  const auto [ntot, Sz, i] = st;
  const auto path = state_path(st);
  std::cout << fmt::format("\n\nRESULTS FOR THE SECTOR WITH {} PARTICLES, Sz {}, state {}:", ntot, Sz_string(Sz), i) << std::endl;
  const auto E = s.eigen[st].E();
  std::cout << fmt::format("Energy = {}", E) << std::endl;
  dump(file, path + "/E", E);
  auto psi = s.eigen[st].psi();
  MeasureOccupancy(psi, file, path, p);
  MeasurePairing(psi, file, path, p);
  MeasureAmplitudes(psi, file, path, p);
  if (p.computeEntropy) MeasureEntropy(psi, file, path, p);
  if (p.impNupNdn) MeasureImpurityUpDn(psi, file, path, p);
  if (p.chargeCorrelation) MeasureChargeCorrelation(psi, file, path, p);
  if (p.spinCorrelation) MeasureSpinCorrelation(psi, file, path, p);
  if (p.pairCorrelation) MeasurePairCorrelation(psi, file, path, p);
  if (p.hoppingExpectation) MeasureHopping(psi, file, path, p);
  if (p.printTotSpinZ) MeasureTotalSpinz(psi, file, path, p);
  s.stats[st].dump();
}

auto find_global_GS(store &s, auto & file) {
  auto m = std::min_element(begin(s.eigen), end(s.eigen), [](const auto &p1, const auto &p2) { return p1.second.E() < p2.second.E(); });
  state_t GS = m->first;
  double E_GS = m->second.E();
  const auto [N_GS, Sz_GS, i] = GS;
  Expects(i == 0);
  std::cout << fmt::format("\nN_GS = {}\nSZ_GS = {}\nE_GS = {}\n",N_GS, Sz_string(Sz_GS), E_GS);
  dump(file, "/GS/N",  N_GS);
  dump(file, "/GS/Sz", Sz_GS);
  dump(file, "/GS/E",  E_GS);
  return std::make_pair(GS, E_GS);
}

void print_energies(store &s, double EGS, params &p) {
  skip_line();
  for (const auto &st : s.eigen) {
    const auto [ntot, Sz, i] = st.first;
    const double E = st.second.E();
    const double Ediff = E-EGS;
    std::cout << fmt::format(FMT_STRING("n = {:<5}  Sz = {:<4}  i = {:<3}  E = {:<22.15}  DeltaE = {:<22.15}"), 
                             ntot, Sz_string(Sz), i, E, Ediff) << std::endl;
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

void calc_weight(store &s, state_t GS, state_t ES, int q, std::string sz, auto & file, params &p)
{
  double res;
  MPS & psiGS = s.eigen[GS].psi();
  if (s.eigen.count(ES)) {
    MPS & psiES = s.eigen[ES].psi();
    res = (q == +1 ? ExpectationValueAddEl(psiES, psiGS, sz, p) : ExpectationValueTakeEl(psiES, psiGS, sz, p));
    std::cout << "weight w" << (q == +1 ? "+" : "-") << " " << sz << ": " << res << std::endl;
  } else {
    res = std::numeric_limits<double>::quiet_NaN();
    std::cout <<  "ERROR: we don't have info about the state " << ES << std::endl;
  }
  dump(file, "weights/" + std::to_string(std::get<2>(ES)) + "/" + std::to_string(q) + "/" + sz, res);
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
  return inner(psi1, psi2);
}

void calculate_overlaps(store &s, auto &file, params &p) {
  skip_line();
  for (const auto & [st1, st2] : itertools::product(s.eigen, s.eigen)) {
    const auto [ntot1, Sz1, i] = st1.first;
    const auto [ntot2, Sz2, j] = st2.first;
    if (ntot1 == ntot2 && Sz1 == Sz2 && i < j) {
      auto o = calculate_overlap(st1.second.psi(), st2.second.psi());
      std::cout << fmt::format(FMT_STRING("n = {:<5}  Sz = {:4}  i = {:<3}  j = {:<3}  <i|j> = {:<22.15}"),
                               ntot1, Sz_string(Sz1), i, j, o) << std::endl;
      dump(file, "overlaps/" + ij_path(ntot1, Sz1, i, j), o);
    }
  }
}

void process_and_save_results(store &s, params &p, std::string h5_filename) {
  File file(h5_filename, File::Overwrite);
  for(const auto & [st, e]: s.eigen)
    calc_properties(st, file, s, p);
  const auto [GS, EGS] = find_global_GS(s, file);
  print_energies(s, EGS, p);
  if (p.calcweights) {
    calculate_spectral_weights(s, GS, file, p, 0);
    if (p.excited_state) calculate_spectral_weights(s, GS, file, p, 1);
  }  
  if (p.overlaps)
    calculate_overlaps(s, file, p);
}
