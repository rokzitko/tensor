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
  for (const auto & [n,  Sz] : l){
    for (int i = 0; i <= std::min(p.excited_states, p.stop_n); i++) {
      
      double guess = n - (0.5 - (p.qd->eps()/p.qd->U()));
      
      //THIS INITIAL guess DOES NOT WORK AS WELL
      //if (p.qd->U()/2 > p.sc->Ec()) guess = n - (0.5 - (p.qd->eps()/p.qd->U()));
      //else guess = p.sc->n0();


      state_t key = {n, Sz, i};
      p.MFnSCs.insert({key, guess });
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


void update_state_convergence_map(auto &oldnSCs, auto &newnSCs, auto &state_convergence_map, store &s, params &p){
  // compares old and new values of nSC, saves a map of state : true/false telling wether the state is converged or not

  for (const auto &[state, ep] : s.eigen) {
    double dif = std::abs( (oldnSCs[state] - newnSCs[state].back()) / newnSCs[state].back() );

    if (dif > p.MF_precision) {
       state_convergence_map[state] = false;      
    }
    else {
      state_convergence_map[state] = true; 
    }
  }
}

void update_s_struct(auto &state_convergence_map, auto &s){
  // if the state is not conserved, its data has to be removed from s as it has to be computed in the next iteration of solve()


  for ( const auto & state : state_convergence_map ){
    state_t st = state.first;
    bool converged = state.second;

    if (!converged){
      s.eigen.erase(st);
      s.stats.erase(st);
    }
  }
}

double get_fullH_energy(const MPS &psi, const state_t &st, params &p){
  // computes the expected value of energy < psi | H | psi >, where H is the real Hamiltonian without the mean field approximation

  // create the instace of the full problem
  auto fullProblem = set_problem("Ec", p);  // problem type with hard-coded type "Ec", gives the full Hamiltonian

  MPO fullH = fullProblem->initH(st, p);

  return inner(psi, fullH, psi);
} 

void print_iteration_results(auto Es, auto nSCs, store &s, params &p){

  H5Easy::File file("solution.h5", H5Easy::File::ReadWrite);

  std::cout << "\nITERATION RESULTS: \n";
  for (const auto &[state, ep] : s.eigen) {
    const auto [n, Sz, i] = state;

    std::cout << "n = " << n << ", Sz = " << Sz << ", i = " << i << ": \n";  
    std::cout << "SC occupancies = " << std::setprecision(full) << nSCs[state] << std::endl;
    std::cout << "energies = " << std::setprecision(full) << Es[state] << std::endl;

    // PRINT THE ENERGY OBTAINED BY FULL H
    double fullE = get_fullH_energy(ep.psi(), state, p);
    std::cout << "full H energy = " << std::setprecision(full) << fullE << std::endl;

    std::cout << "\n";

    const auto path = state_path(state);
    H5Easy::dump(file, path + "/iteration/nSCs", nSCs[state]);
    H5Easy::dump(file, path + "/iteration/Es", Es[state]);
    H5Easy::dump(file, path + "/iteration/fullHE", fullE);

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

  bool stop = false;
  std::map<state_t, bool> state_convergence_map;
  int iter_count = 0;
  
  // The iteration will stop if it performs max_iter steps, but also if all states converge. This is because after each step the entries of non-converged states are removed from the stats s structure. 
  // If the state is there (meaning it has converged), the obtain_result() function does not run the calculation. 
  while ( iter_count <= p.max_iter ) {
    iter_count ++;

    std::cout << "\n\n ITERATION NUMBER: " << iter_count << "\n\n";

    solve(l, s, p);
  
    get_and_append_nSCs_Es(nSCs, Es, s, p);
    update_state_convergence_map(p.MFnSCs, nSCs, state_convergence_map, s, p);
    update_MFnSCs(nSCs, s, p);

    // do not delete the stats of non converged states in the last or second to last iterations  
    if ( iter_count < p.max_iter-1 ) {
      update_s_struct(state_convergence_map, s);
    }

  }

  process_and_save_results(s, p);
  
  print_iteration_results(Es, nSCs, s, p);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << std::endl << "Wall time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " s" << std::endl;
}
