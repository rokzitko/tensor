#include "FindGS.h"

#include "SC_BathMPO.h"
#include "SC_BathMPO_MiddleImp.h"
#include "SC_BathMPO_Ec.h"
#include "SC_BathMPO_Ec_V.h"
#include "SC_BathMPO_MiddleImp_TwoChannel.h"

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
  const int nhalf = p.N; // total nr of electrons at half-filling
  const int nref = (p.refisn0 ? ( round(p.sc->n0() + 0.5 - (p.qd->eps()/p.qd->U())) ) : nhalf); //calculation of the energies is centered around this n
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
    throw std::runtime_error("Please provide input file. Usage: ./executable inputfile.txt");

  //read parameters from the input file
  p.inputfn = {argv[1]};
  auto input = InputGroup{p.inputfn, "params"}; //get input parameters using InputGroup from itensor

  p.MPO = input.getString("MPO", "std"); // problem type
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
  if (p.MPO == "middle" || p.MPO == "middle_2channel") {
    assert(even(p.NBath));
    p.impindex = 1+p.NBath/2;
  } else if (p.MPO == "std" || p.MPO == "Ec" || p.MPO == "Ec_V") {
    p.impindex = 1;
  } else
    throw std::runtime_error("Unknown MPO type");
  std::cout << "N=" << p.N << " NBath=" << p.NBath << " impindex=" << p.impindex << std::endl;

  double U = input.getReal("U", 0); // need to parse it first because it enters the default value for epsimp just below
  p.qd = std::make_unique<imp>(U, input.getReal("epsimp", -U/2.), input.getReal("EZ_imp", 0.));
  p.sc = std::make_unique<SCbath>(p.NBath, input.getReal("alpha", 0), input.getReal("Ec", 0), input.getReal("n0", p.N-1), input.getReal("EZ_bulk", 0.));
  p.Gamma = std::make_unique<hyb>(input.getReal("gamma", 0));
  p.V12 = input.getReal("V", 0); // handled in a special way

  // sites is an ITensor thing. It defines the local hilbert space and operators living on each site of the lattice.
  // For example sites.op("N",1) gives the pariticle number operator on the first site.
  p.sites = Hubbard(p.N);

  p.computeEntropy = input.getYesNo("computeEntropy", false);
  p.impNupNdn = input.getYesNo("impNupNdn", false);

  p.excited_state = input.getYesNo("excited_state", false);
  p.printDimensions = input.getYesNo("printDimensions", false);
  p.calcweights = input.getYesNo("calcweights", false);
  p.refisn0 = input.getYesNo("refisn0", false);
  p.parallel = input.getYesNo("parallel", false);
  p.verbose = input.getYesNo("verbose", false);
  p.band_level_shift = input.getYesNo("band_level_shift", false);
  p.printTotSpinZ = input.getYesNo("printTotSpinZ", false);

  p.chargeCorrelation = input.getYesNo("chargeCorrelation", false);
  p.pairCorrelation = input.getYesNo("pairCorrelation", false);
  p.spinCorrelation = input.getYesNo("spinCorrelation", false);
  p.hoppingExpectation = input.getYesNo("hoppingExpectation", false);

  p.EnergyErrgoal = input.getReal("EnergyErrgoal", 1e-16);
  p.nrH = input.getInt("nrH", 5);
  p.nrange = input.getInt("nrange", 1);

  // TWO CHANNEL PARAMETERS
  p.alpha1 = input.getReal("alpha1", 0);
  p.alpha2 = input.getReal("alpha2", 0);
  p.n01 = input.getReal("n01", (p.N-1)/2);
  p.n02 = input.getReal("n02", (p.N-1)/2);
  p.gamma1 = input.getReal("gamma1", 0);
  p.gamma2 = input.getReal("gamma2", 0);
  p.Ec1 = input.getReal("Ec1", 0);
  p.Ec2 = input.getReal("Ec2", 0);
  p.EZ_bulk1 = input.getReal("EZ_bulk1", 0);
  p.EZ_bulk2 = input.getReal("EZ_bulk2", 0);
  // TO DO
  // p.sc1 = std::make_unique<SCbath>(p.NBath, input.getReal("alpha1", 0), input.getReal("Ec1", 0), input.getReal("n01", p.N-1), input.getReal("EZ_bulk1", 0));
  // p.sc2 = std::make_unique<SCbath>(p.NBath, input.getReal("alpha2", 0), input.getReal("Ec2", 0), input.getReal("n02", p.N-1), input.getReal("EZ_bulk2", 0));
  p.SCSCinteraction = input.getReal("SCSCinteraction", 0);

  init_subspace_lists(p);
  return input;
}

//initialize the Hamiltonian
std::tuple<MPO, double> initH(int ntot, params &p){
  auto eps = p.sc->eps(p.band_level_shift);
  auto V = p.Gamma->V(p.sc->Nbath());
  if (p.verbose) {
    std::cout << "eps=" << eps << std::endl;
    std::cout << "V=" << V << std::endl;
  }

  double Eshift = 0;  // constant term in the Hamiltonian
  MPO H(p.sites); // MPO is the hamiltonian in "MPS-form" after this line it is still a trivial operator
  if (p.MPO == "std") {
    assert(p.V12 != 0.0);
    Eshift = p.sc->Ec()*pow(ntot-p.sc->n0(),2); // occupancy dependent effective energy shift
    double epseff = p.qd->eps() - 2.*p.sc->Ec()*(ntot-p.sc->n0()) + p.sc->Ec();
    double Ueff = p.qd->U() + 2.*p.sc->Ec();
    Fill_SCBath_MPO(H, eps, V, epseff, Ueff, p); // defined in SC_BathMPO.h, fills the MPO with the necessary entries
  } else if (p.MPO == "middle") {
    assert(p.V12 != 0.0);
    Eshift = p.sc->Ec()*pow(ntot-p.sc->n0(),2); // occupancy dependent effective energy shift
    double epseff = p.qd->eps() - 2.*p.sc->Ec()*(ntot-p.sc->n0()) + p.sc->Ec();
    double Ueff = p.qd->U() + 2.*p.sc->Ec();
    Fill_SCBath_MPO_MiddleImp(H, eps, V, epseff, Ueff, p);
  } else if (p.MPO == "Ec") {
    assert(p.V12 != 0.0);
    Eshift = p.sc->Ec()*pow(p.sc->n0(), 2);
    Fill_SCBath_MPO_Ec(H, eps, V, p);
  } else if (p.MPO == "Ec_V") {
    Eshift = p.sc->Ec()*pow(p.sc->n0(), 2) + p.V12 * p.sc->n0() * p.qd->nu();
    double epseff = p.qd->eps() - p.V12 * p.sc->n0();
    double epsishift = -p.V12 * p.qd->nu();
    Fill_SCBath_MPO_Ec_V(H, eps, V, epseff, epsishift, p);
  } else if (p.MPO == "middle_2channel") {
    Eshift = p.Ec1*pow(p.n01, 2) + p.Ec2*pow(p.n02, 2);
    Fill_SCBath_MPO_MiddleImp_TwoChannel(H, eps, V, p);
  } else
    throw std::runtime_error("Unknown MPO type " + p.MPO);
  Eshift += p.qd->U()/2.; // RZ, for convenience
  return std::make_tuple(H, Eshift);
}

// Initialize the MPS in a product state with ntot electrons.
// Sz is the spin of the electron on the impurity site.
MPS initPsi(int ntot, float Sz, const auto &sites, int impindex, bool randomMPSb) {
  assert(ntot >= 0);
  assert(Sz == -0.5 || Sz == 0 || Sz == +0.5);
  int tot = 0;   // electron counter, for assertion test
  int SZtot = 0; // SZ counter, for assertion test
  auto state = InitState(sites);
  // ** Add electron to the impurity site
  if (ntot > 0) {
    if (Sz == -0.5) {
      state.set(impindex, "Dn");
      SZtot -= 0.5;
    }
    if (Sz == 0 || Sz == +0.5) {
      state.set(impindex, "Up");
      SZtot += 0.5;
    }
    tot++;
  }
  // ** Add electrons to the bath
  const int nsc = ntot-1;    // number of electrons in the bath
  if (nsc > 0) {
    const int npair = nsc/2; // number of pairs in the bath
    int j = 0;               // counts added pairs
    int i = 1;               // site index (1 based)
    for(; j < npair; i++)
      if (i != impindex) {   // skip impurity site
        j++;
        state.set(i, "UpDn");
        tot += 2;            // SZtot does not change!
      }
    if (odd(nsc)) {          // if ncs is odd (implies even ntot and sz=0), add spin-down electron
      if (i == impindex) 
        i++;
      state.set(i,"Dn");
      SZtot -= 0.5;
      tot++;
    }
  }
  assert(tot == n);
  assert(SZtot == Sz);
  MPS psi(state);
  if (randomMPSb) 
    psi = randomMPS(state);
  return psi;
}

//calculates <psi1|c_dag|psi2>, according to http://itensor.org/docs.cgi?vers=cppv3&page=formulas/mps_onesite_op
void ExpectationValueAddEl(MPS psi1, MPS psi2, std::string spin, const params &p){
  psi2.position(p.impindex); //set orthogonality center
  auto newTensor = noPrime(op(p.sites,"Cdag"+spin, p.impindex)*psi2(p.impindex)); //apply the local operator
  psi2.set(p.impindex,newTensor); //plug in the new tensor, with the operator applied
  auto res = inner(psi1, psi2);
  std::cout << "weight w+ " << spin << ": " << res << "\n";
}

//calculates <psi1|c|psi2>
void ExpectationValueTakeEl(MPS psi1, MPS psi2, std::string spin, const params &p){
  psi2.position(p.impindex);
  auto newTensor = noPrime(op(p.sites,"C"+spin, p.impindex)*psi2(p.impindex));
  psi2.set(p.impindex,newTensor);
  auto res = inner(psi1, psi2);
  std::cout << "weight w- " << spin << ": " << res << "\n";
}

//computes the correlation between operator impOp on impurity and operator opj on j
double ImpurityCorrelator(MPS& psi, auto impOp, int j, auto opj, const params &p){
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
void ChargeCorrelation(MPS& psi, const params &p){
  std::cout << "charge correlation = ";
  auto impOp = op(p.sites, "Ntot", p.impindex);
  double tot = 0;
  for(auto j: range1(2, length(psi))) {
    auto scOp = op(p.sites, "Ntot", j);
    double result = ImpurityCorrelator(psi, impOp, j, scOp, p);
    std::cout << std::setprecision(full) << result << " ";
    tot += result;
  }
  std::cout << std::endl;
  std::cout << "charge correlation tot = " << tot << "\n"; 
}

// <S_imp S_i> = <Sz_imp Sz_i> + 1/2 ( <S+_imp S-_i> + <S-_imp S+_i> )
void SpinCorrelation(MPS& psi, const params &p){
  std::cout << "spin correlations:\n";
  //impurity spin operators
  auto impSz = 0.5*( op(p.sites, "Nup", p.impindex) - op(p.sites, "Ndn", p.impindex) );
  auto impSp = op(p.sites, "Cdagup*Cdn", p.impindex);
  auto impSm = op(p.sites, "Cdagdn*Cup", p.impindex);
  auto SzSz = 0.25 * ( op(p.sites, "Nup*Nup", p.impindex) - op(p.sites, "Nup*Ndn", p.impindex) - op(p.sites, "Ndn*Nup", p.impindex) + op(p.sites, "Ndn*Ndn", p.impindex));
  //squares of the impurity operators; for on-site terms
  double tot = 0;
  //SzSz term
  std::cout << "SzSz correlations: ";
  psi.position(p.impindex);
  //on site term
  auto onSiteSzSz = elt(psi(p.impindex) * SzSz *  dag(prime(psi(p.impindex),"Site")));
  std::cout << std::setprecision(full) << onSiteSzSz << " ";
  tot+=onSiteSzSz;
  for(auto j: range1(2, length(psi))) {
    auto scSz = 0.5*( op(p.sites, "Nup", j) - op(p.sites, "Ndn", j) );
    double result = ImpurityCorrelator(psi, impSz, j, scSz, p);
    std::cout << std::setprecision(full) << result << " ";
    tot += result;
  }
  std::cout << std::endl;
  //S+S- term
  std::cout << "S+S- correlations: ";
  psi.position(p.impindex);
  //on site term
  auto onSiteSpSm = elt(psi(p.impindex) * op(p.sites, "Cdagup*Cdn*Cdagdn*Cup", p.impindex) *  dag(prime(psi(p.impindex),"Site")));
  std::cout << std::setprecision(17) << onSiteSpSm << " ";
  tot += 0.5*onSiteSpSm; 
  for(auto j: range1(2, length(psi))) {
    auto scSm = op(p.sites, "Cdagdn*Cup", j);
    double result = ImpurityCorrelator(psi, impSp, j, scSm, p);
    std::cout << std::setprecision(full) << result << " ";
    tot += 0.5*result;
  }
  std::cout << std::endl;
  //S- S+ term
  std::cout << "S-S+ correlations: ";
  psi.position(p.impindex);
  //on site term
  auto onSiteSmSp = elt(psi(p.impindex) * op(p.sites, "Cdagdn*Cup*Cdagup*Cdn", p.impindex) *  dag(prime(psi(p.impindex),"Site")));
  std::cout << std::setprecision(full) << onSiteSmSp << " ";
  tot += 0.5*onSiteSmSp; 
  for(auto j: range1(2, length(psi))) {
    auto scSp = op(p.sites, "Cdagup*Cdn", j);
    double result = ImpurityCorrelator(psi, impSm, j, scSp, p);
    std::cout << std::setprecision(full) << result << " ";
    tot += 0.5*result;
  }
  std::cout << std::endl;
  std::cout << "spin correlation tot = " << tot << "\n";
}

void PairCorrelation(MPS& psi, const params &p){
  std::cout << "pair correlation = ";
  auto impOp = op(p.sites, "Cup*Cdn", p.impindex);
  double tot = 0;
  for(auto j: range1(2, length(psi))) {
    auto scOp = op(p.sites, "Cdagdn*Cdagup", j);
    double result = ImpurityCorrelator(psi, impOp, j, scOp, p);
    std::cout << std::setprecision(full) << result << " ";
    tot+=result;
  }
  std::cout << std::endl;
  std::cout << "pair correlation tot = " << tot << "\n";
}

//Prints <d^dag c_i + c_i^dag d> for each i. the sum of this expected value, weighted by 1/sqrt(N)
//gives <d^dag f_0 + f_0^dag d>, where f_0 = 1/sqrt(N) sum_i c_i. This is the expected value of hopping.
void expectedHopping(MPS& psi, const params &p){
  auto impOpUp = op(p.sites, "Cup", p.impindex);
  auto impOpDagUp = op(p.sites, "Cdagup", p.impindex);
  auto impOpDn = op(p.sites, "Cdn", p.impindex);
  auto impOpDagDn = op(p.sites, "Cdagdn", p.impindex);
  double totup = 0;
  double totdn = 0;
  // hopping expectation values for spin up
  std::cout << "hopping spin up = ";
  for (auto j : range1(2, length(psi))){
    auto scDagOp = op(p.sites, "Cdagup", j);
    auto scOp = op(p.sites, "Cup", j);
    double result = ImpurityCorrelator(psi, impOpUp, j, scDagOp, p);    // <d c_i^dag>
    double resultdag = ImpurityCorrelator(psi, impOpDagUp, j, scOp, p); // <d^dag c_i>
    std::cout << std::setprecision(full) << result << " " << resultdag << " ";
    std::cout << std::setprecision(full) << result+resultdag << " ";
    totup += result+resultdag;
  }
  std::cout << std::endl;
  std::cout << "hopping correlation up tot = " << totup << "\n";
  // hopping expectation values for spin up
  std::cout << "hopping spin down = ";
  for (auto j : range1(2, length(psi))){
    auto scDagOp = op(p.sites, "Cdagdn", j);
    auto scOp = op(p.sites, "Cdn", j);
    double result = ImpurityCorrelator(psi, impOpDn, j, scDagOp, p);    // <d c_i^dag>
    double resultdag =  ImpurityCorrelator(psi, impOpDagDn, j, scOp, p); // <d^dag c_i>
    std::cout << std::setprecision(full) << result << " " << resultdag << " ";
    //std::cout << std::setprecision(full) << result+resultdag << " ";
    totdn+=result+resultdag;
  }
  std::cout << std::endl;
  std::cout << "hopping correlation down tot = " << totdn << "\n";
  std::cout << "total hopping correlation = " << totup + totdn << "\n";
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
  std::cout << "impurity nup ndn = " << std::setprecision(full) << up << " " << dn << "\n";
  dump(file, path + "/impurity_n_up", up);
  dump(file, path + "/impurity_n_dn", dn);
}

//prints total Sz of the state
void TotalSpinz(MPS& psi, const params &p){
  double totNup = 0.;
  double totNdn = 0.;
  for(auto j: range1(length(psi))) {
    psi.position(j);
    auto Nupi = psi.A(j) * p.sites.op("Nup",j)* dag(prime(psi.A(j),"Site"));
    auto Ndni = psi.A(j) * p.sites.op("Ndn",j)* dag(prime(psi.A(j),"Site"));
    totNup += std::real(Nupi.cplx());
    totNdn += std::real(Ndni.cplx());
  }
  std::cout << std::setprecision(full) << "Total spin z: " << " Nup = " << totNup << " Ndn = " << totNdn << " Sztot = " << 0.5*(totNup-totNdn) <<  "\n";
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
  std::cout << "site occupancies = " << std::setprecision(full) << r << std::endl;
  auto tot = std::accumulate(r.begin(), r.end(), 0.0);
  Print(tot);
  dump(file, path + "/site_occupancies", r);
  dump(file, path + "/total_occupancy", tot);
}

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
    auto sq = p.sc->g() * sqrt(diff);
    r.push_back(sq);
    if (i != p.impindex) tot += sq; // exclude the impurity site in the sum
  }
  return std::make_pair(r, tot);
}

void MeasurePairing(MPS& psi, auto & file, std::string path, const params &p) {
  auto [r, tot] = calcPairing(psi, p);
  std::cout << "site pairing = " << std::setprecision(full) << r << std::endl;
  Print(tot);
  dumpreal(file, path + "/pairing", r);
  dumpreal(file, path + "/total_pairing", tot);
}

// See von Delft, Zaikin, Golubev, Tichy, PRL 77, 3189 (1996)
auto calcAmplitudes(MPS &psi, const params &p) {
  std::vector<complex_t> r;
  complex_t tot = 0;
  for(auto i : range1(length(psi)) ) {
    psi.position(i);
    auto valv = psi.A(i) * p.sites.op("Cdagup*Cdagdn*Cdn*Cup", i) * dag(prime(psi.A(i),"Site"));
    auto valu = psi.A(i) * p.sites.op("Cdn*Cup*Cdagup*Cdagdn", i) * dag(prime(psi.A(i),"Site"));
    auto v = sqrt( std::real(valv.cplx()) ); // XXX
    auto u = sqrt( std::real(valu.cplx()) );
    auto pdt = v*u;
    auto element = p.sc->g() * pdt;
    std::cout << "[v=" << v << " u=" << u << " pdt=" << pdt << "] ";
    if (i != p.impindex) tot += element; // exclude the impurity site in the sum
  }
  return std::make_pair(r, tot);
}

void MeasureAmplitudes(MPS& psi, auto & file, std::string path, const params &p) {
  auto [r, tot] = calcAmplitudes(psi, p);
  std::cout << "amplitudes vu = " << std::setprecision(full) << r << std::endl;
  Print(tot);
  dumpreal(file, path + "/amplitudes", r);
  dumpreal(file, path + "/total_amplitude", tot);
}

// Computed entanglement/von Neumann entropy between the impurity and the system.
// Copied from https://www.itensor.org/docs.cgi?vers=cppv3&page=formulas/entanglement_mps
// von Neumann entropy at the bond between impurity and next site.
auto calcEntropy(MPS& psi, const params &p) {
  assert(p.impindex == 1); // Works as intended only if p.impindex=1.
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
  printfln("Entanglement entropy across impurity bond b=%d, SvN = %.10f", p.impindex, SvN);
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

    //initialize H and psi
    auto [H, Eshift] = initH(ntot, p);
    auto psi_init = initPsi(ntot, Sz, p.sites, p.impindex, input.getYesNo("randomMPS", false)); 

    Args args; //args is used to store and transport parameters between various functions

    //Apply the MPO a couple of times to get DMRG started, otherwise it might not converge.
    for(auto i : range1(p.nrH)){
      psi_init = applyMPO(H,psi_init,args);
      psi_init.noPrime().normalize();
    }

    auto [E, psi] = dmrg(H,psi_init,sweeps,{"Silent",p.parallel, "Quiet",!p.printDimensions, "EnergyErrgoal",p.EnergyErrgoal}); // call itensor dmrg
    double GSenergy = E+Eshift;
    s.eigen0[sub] = eigenpair(GSenergy, psi);
    s.stats0[sub] = psi_stats(E, psi, H);

    if (p.excited_state) {
      auto wfs = std::vector<MPS>(1);
      wfs.at(0) = psi;
      auto [E1, psi1] = dmrg(H,wfs,psi,sweeps,{"Silent",p.parallel,"Quiet",true,"Weight",11.0});
      double ESenergy = E1+Eshift;
      s.eigen1[sub] = eigenpair(ESenergy, psi1);
    }
  }
}//end FindGS

void calculate_spectral_weights(store &s, subspace subGS, params &p) {
  MPS & psiGS = s.eigen0[subGS].psi();
  auto [N_GS, Sz_GS] = subGS;

  printfln(""); 
  printfln("Spectral weights:");
  printfln("(Spectral weight is the square of the absolute value of the number.)");

  auto cp = subspace(N_GS+1, Sz_GS+0.5);
  if (s.eigen0.find(cp) != s.eigen0.end()) { // if the N_GS+1 state was computed, print the <N+1|c^dag|N> terms
    MPS & psiNp = s.eigen0[cp].psi();
    ExpectationValueAddEl(psiNp, psiGS, "up", p);
  }  
  else printfln("ERROR: we don't have info about the N_GS+1, Sz_GS+0.5 occupancy sector.");

  auto cm = subspace(N_GS+1, Sz_GS-0.5);
  if (s.eigen0.find(cm) != s.eigen0.end()) { // if the N_GS+1 state was computed, print the <N+1|c^dag|N> terms
    MPS & psiNp = s.eigen0[cm].psi();
    ExpectationValueAddEl(psiNp, psiGS, "dn", p);
  }  
  else printfln("ERROR: we don't have info about the N_GS+1, Sz_GS-0.5 occupancy sector.");

  auto am = subspace(N_GS-1, Sz_GS-0.5);
  if (s.eigen0.find(am) != s.eigen0.end()) { // if the N_GS-1 state was computed, print the <N-1|c|N> terms
    MPS & psiNm = s.eigen0[am].psi();
    ExpectationValueTakeEl(psiNm, psiGS, "up", p);
  }
  else printfln("ERROR: we don't have info about the N_GS-1, Sz_GS+0.5 occupancy sector.");
  
  auto ap = subspace(N_GS-1, Sz_GS+0.5);
  if (s.eigen0.find(ap) != s.eigen0.end()) { // if the N_GS-1 state was computed, print the <N-1|c|N> terms
    MPS & psiNm = s.eigen0[ap].psi();
    ExpectationValueTakeEl(psiNm, psiGS, "dn", p);
  }
  else printfln("ERROR: we don't have info about the N_GS-1, Sz_GS-0.5 occupancy sector.");
}

void print_energies(store &s, params &p) {
  printfln("");
  for(auto ntot: p.numPart)
    for(auto Sz: p.Szs[ntot])
      printfln("n = %.17g  Sz = %.17g  E = %.17g", ntot, Sz, s.eigen0[subspace(ntot,Sz)].E());
}

auto find_global_GS_subspace(store &s) {
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
  printfln("N_GS = %i",N_GS);
  printfln("Sz_GS = %i",Sz_GS);
  return subGS;
}

//Loops over all particle sectors and prints relevant quantities.
void calculateAndPrint(InputGroup &input, store &s, params &p) {
  File file("solution.h5", File::Overwrite);
  for(auto ntot: p.numPart) {
    for (double Sz: p.Szs[ntot]) {
      auto sub = subspace(ntot, Sz);
      auto E = s.eigen0[sub].E();
      dump(file, str(sub, "0/E"), E);
      MPS & GS = s.eigen0[sub].psi();
      printfln("\n\nRESULTS FOR THE SECTOR WITH %i PARTICLES, Sz %i:", ntot, Sz);
      printfln("Ground state energy = %.17g", E);
      printfln("norm = %.17g", s.stats0[sub].norm());

      MeasureOcc(GS, file, str(sub, "0"), p);
      MeasurePairing(GS, file, str(sub, "0"), p);
      MeasureAmplitudes(GS, file, str(sub, "0"), p);

      if (p.computeEntropy) PrintEntropy(GS, file, str(sub, "0"), p);
      if (p.impNupNdn) ImpurityUpDn(GS, file, str(sub, "0"), p);
      if (p.chargeCorrelation) ChargeCorrelation(GS, p);
      if (p.spinCorrelation) SpinCorrelation(GS, p);
      if (p.pairCorrelation) PairCorrelation(GS, p);
      if (p.hoppingExpectation) expectedHopping(GS, p);
      if (p.printTotSpinZ) TotalSpinz(GS, p);

      //various measures of convergence (energy deviation, residual value)
      printfln("Eigenvalue(bis): <GS|H|GS> = %.17g", s.stats0[sub].Ebis());
      printfln("diff: E_GS - <GS|H|GS> = %.17g", E-s.stats0[sub].Ebis()); // TODO: remove this
      printfln("deltaE: sqrt(<GS|H^2|GS> - <GS|H|GS>^2) = %.17g", s.stats0[sub].deltaE());
      printfln("residuum: <GS|H|GS> - E_GS*<GS|GS> = %.17g", s.stats0[sub].residuum());
      s.stats0[sub].dump(file, str(sub, "0"));

      if (p.excited_state){
        double E1 = s.eigen1[sub].E();
        MPS & ES = s.eigen1[sub].psi();
        dump(file, str(sub, "1/E"), E1);
        MeasureOcc(ES, file, str(sub, "1"), p);
        printfln("Excited state energy = %.17g", E1);
       }
    } //end of Sz for loop 
  } //end of ntot for loop

  print_energies(s, p);

  subspace subGS = find_global_GS_subspace(s);
  if (p.calcweights) 
    calculate_spectral_weights(s, subGS, p);
}
