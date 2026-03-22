#include <openmc/particle.h>
#include <iostream>

int main() {
    openmc::Particle neuxs_demo_particle;
    int64_t epic_particle_id = neuxs_demo_particle.id();

    printf("neuxs epic particle id %d\n",static_cast<int>(epic_particle_id));
    // by default id should be -1
    return 0;
}