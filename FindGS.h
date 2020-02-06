#ifndef _calcGS_h_
#define _calcGS_h_

using namespace itensor;

struct params {
  string inputfn;       // filename of the input file
  InputGroup input;     // itensor input parser

  int N;                // number of sites
  int NImp;             // number of impurity orbitals
  int NBath;            // number of bath sites
  int impindex;         // impurity position in the chain (1 based)

  Hubbard sites;        // itensor object

  //all bools have default value false
  bool excited_state;   //if true computes the first excited state
  bool randomMPSb;      //it true sets the initial state to random
  bool printDimensions; //if true prints dmrg() prints info during the sweep
  bool calcweights;     //if true calculates the spectral weights of the two closes spectroscopically availabe excitations
  bool refisn0;         //if true the energies will be computed in the sectors centered around the one with n = round(n0) + 1
  bool parallel;        //if true enables openMP parallel calculation of the for loop in findGS()
  bool verbose;         // verbosity level

  double EnergyErrgoal; //the convergence value at which dmrg() will stop the sweeps; default = machine precision
  int nrH;              //number of times to apply H to psi before comencing the sweep - akin to a power method; default = 5
  int nrange;           //the number of energies computed is 2*nrange + 1

  double alpha;         // e-e coupling
  double n0;            // charge offset
  double d;             // d=2D/NBath, level spacing
  double g;             // strength of the SC coupling

  double U;             // e-e on impurity site
  double gamma;         // hybridisation
  double Ec;            // charging energy
  double epsimp;        // impurity level
  double Ueff;          // effective e-e on impurity site (after mapping)

  std::vector<int> numPart; // range of total occupancies of interest
};

void FindGS(InputGroup &input,
  std::map<int, MPS>& psiStore, std::map<int, double>& GSEstore, std::map<int, MPS> ESpsiStore, std::map<int, double>& ESEstore,
  std::map<int, double>& GS0bisStore, std::map<int, double>& deltaEStore, std::map<int, double>& residuumStore, const params &p);
void calculateAndPrint(InputGroup &input,
  std::map<int, MPS> psiStore, std::map<int, double> GSEstore,std::map<int, MPS> ESpsiStore, std::map<int, double> ESEstore,
  std::map<int, double> GS0bisStore, std::map<int, double> deltaEStore, std::map<int, double> residuumStore, const params &p);
void GetBathParams(double epseff, std::vector<double>& eps, std::vector<double>& V, const params &p);
MPO initH(std::vector<double> eps, std::vector<double> V, const params &p);
MPS initPsi(int ntot, const params &p);
double ExpectationValueAddEl(MPS psi1, MPS psi2, std::string spin, const params &p);
double ExpectationValueTakeEl(MPS psi1, MPS psi2, std::string spin, const params &p);
void MyDMRG(MPS& psi, MPO& H, double& energy, Args args);
void MeasureOcc(MPS& psi, const params &);
void MeasurePairing(MPS& psi, const params &);
void MeasureAmplitudes(MPS& psi, const params &);
InputGroup parse_cmd_line(int, char * [], params &p);

#endif
