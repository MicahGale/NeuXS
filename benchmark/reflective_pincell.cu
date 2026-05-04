// neuxs includes
#include "cross_section_reader.h"
#include "geometry.cuh"
#include "material.cuh"
#include "timer.cuh"
#include "transport.cuh"

#include "reflective_pincell.cuh"

/* Usage:
 *   ./this_executable(I need to create another cmake for benchmark)
 * <aos|soa|log> <double|single>
 */

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr
        << "Usage: " << (argc > 0 ? argv[0] : "./app")
        << " <aos|soa|log> <double|single> <n_particles> <threads_per_block>\n";
    return 1;
  }

  const std::string_view xs_type = argv[1];
  const std::string_view precision = argv[2];
  const int number_of_particles = std::atoi(argv[3]);
  const int threads_per_block = std::atoi(argv[4]);

  if (number_of_particles < 0 or threads_per_block < 0) {
    std::cerr << ("number_of_particles and  number of threads_per_block"
                  "must be more than 1\n");
    return 1;
  }
  neuxs::MemoryManager().printDeviceInfo();
  if (precision == "double")
    return pincell::dispatch_xs<double>(xs_type, number_of_particles,
                                        threads_per_block);
  if (precision == "single")
    return pincell::dispatch_xs<float>(xs_type, number_of_particles,
                                       threads_per_block);

  std::cerr << "Invalid precision: " << precision
            << " (expected double|single)\n";
  return 1;
}
