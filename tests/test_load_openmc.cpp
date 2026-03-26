#include <openmc/particle.h>
#include <iostream>

int main() {
    openmc::Particle neuxs_demo_particle;
    int64_t epic_particle_id = neuxs_demo_particle.id();

    if (static_cast<int>(epic_particle_id) == -1)
        return 0;
    return 1;
}