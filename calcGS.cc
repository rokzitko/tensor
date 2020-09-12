#include <itensor/all.h>
#include <itensor/util/args.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>
#include <chrono>

#include "FindGS.h"

int main(int argc, char* argv[]) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  params p;
  store s;
  parse_cmd_line(argc, argv, p);
  auto l = init_subspace_lists(p);
  solve_all(l, s, p);
  process_and_save_results(s, p);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << std::endl << "Wall time: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " s" << std::endl;
}
