#include <itensor/all.h>
#include <itensor/util/args.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <chrono>

#include "FindGS.h"


double get_channel_occupancy(MPS &psi, params &p){
  // sums up the channel occupancies (ONLY USEFUL FOR THE SINGLE CHANNEL PROBLEMS!) - for two channel, change bath_indexes() to bath_indexes(1) or (2)
  auto r = calcOccupancy(psi, p.problem->bath_indexes(), p);
  double nSC = std::accumulate(r.begin(), r.end(), 0.0); // the total occupancy

  return nSC;
}

// fill up the map of nSCs with initial guesses - these are n-nu, as nu is the approximate occupation of imp
void initial_guess_nSCs(const std::vector<subspace_t> &l, params &p){
  // for all subspaces, for all excited states (that will be computed), insert the initial guess
  for (const auto [n,  Sz] : l){
    for (int i = 0; i <= std::min(p.excited_states, p.stop_n); i++) {
      state_t key = {n, Sz, i};
      p.MFnSCs.insert({key, n - (0.5 - (p.qd->eps()/p.qd->U())) });
    }
  }
}

void get_and_append_nSCs_Es(auto &nSCs, auto &Es, store &s, params &p){
  for (auto &[state, ep] : s.eigen) {
    double nSC = get_channel_occupancy(ep.psi(), p);
    nSCs[state].push_back(nSC);
    Es[state].push_back(ep.E());
  }
}

void update_MFnSCs(auto &nSCs, store &s, params &p){
  for (const auto &[state, ep] : s.eigen) {
    p.MFnSCs[state] = nSCs[state].back();
  }
}


bool test_convergence(auto oldnSCs, auto newnSCs, store &s, params &p){
  // compares old and new values of nSC, only if they are converged for all states gives true

  for (const auto &[state, ep] : s.eigen) {
    double dif = std::abs( (oldnSCs[state] - newnSCs[state].back()) / newnSCs[state].back() );

    if (dif > p.MF_precision) {
      return false;      
    }
  }
  return true;
}

void print_iteration_results(auto Es, auto nSCs, store &s){

  H5Easy::File file("solution.h5", H5Easy::File::ReadWrite);

  std::cout << "\nITERATION RESULTS: \n";
  for (const auto &[state, ep] : s.eigen) {
    const auto [n, Sz, i] = state;

    std::cout << "n = " << n << ", Sz = " << Sz << ", i = " << i << ": \n";  
    std::cout << "SC occupancies = " << std::setprecision(full) << nSCs[state] << std::endl;
    std::cout << "energies = " << std::setprecision(full) << Es[state] << std::endl;
    std::cout << "\n";

    const auto path = state_path(state);
    H5Easy::dump(file, path + "/iteration/nSCs", nSCs[state]);
    H5Easy::dump(file, path + "/iteration/Es", Es[state]);

  }
}


int main(int argc, char* argv[]) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  
  params p;
  store s;

  parse_cmd_line(argc, argv, p);
  auto l = init_subspace_lists(p);
  
  initial_guess_nSCs(l, p);
   
  std::map< state_t, std::vector<double> > Es;   // The actual values of energy for all states 
  std::map< state_t, std::vector<double> > nSCs; // The actual values of nSC for all states 

  bool converged = false;
  int iter_count = 0;
  while ( !converged ) {
    iter_count ++;

    std::cout << "\n\n ITERATION NUMBER: " << iter_count << "\n\n";
    
    solve(l, s, p);

    get_and_append_nSCs_Es(nSCs, Es, s, p);

    converged = test_convergence(p.MFnSCs, nSCs, s, p);

    update_MFnSCs(nSCs, s, p);
  }

  process_and_save_results(s, p);
  
  print_iteration_results(Es, nSCs, s);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << std::endl << "Wall time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " s" << std::endl;
}
