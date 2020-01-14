/*
 * Ngrain V1
 * Nbody simulation on GPU/MPI with Kokkos
 *
 * Ben Prather
 */

#include "decs.h"

#include <cmath>
#include <iostream>
#include <random>
#include <sstream>

void accel(const KScalar mass, const KVector pos, const KVector vel,
           const index_t nbody, KVector &acc);
void accel_2layer(const KScalar mass, const KVector pos, const KVector vel,
                  const index_t nbody, KVector &acc);
void accel_outer(const KScalar mass, const KVector pos, const KVector vel,
                  const index_t nbody, KVector &acc);

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  // TODO params struct to pass to update routines
  const unsigned int nbody = 16384;
  // const unsigned int nbody = 4096;
  // const unsigned int NDIM = 3;  //This is very hard to declare at runtime


  //const double G = 0.1;

  const double tf = 10.0;
  const double dt = 0.1;

  // Random initial dist parameaters
  const double xmin = 0.0;
  const double xmax = 1.0e6;
  const double massmin = 0.0;
  const double massmax = 10.0;

  KScalar mass("mass", nbody);
  KVector pos("position", nbody, NDIM);
  KVector vel("velocity", nbody, NDIM);
  KVector acc("acceleration", nbody, NDIM);

  KScalar::HostMirror h_mass = Kokkos::create_mirror_view(mass);
  KVector::HostMirror h_pos = Kokkos::create_mirror_view(pos);
  KVector::HostMirror h_vel = Kokkos::create_mirror_view(vel);
  KVector::HostMirror h_acc = Kokkos::create_mirror_view(acc);

  // TODO only works for CUDA?
  // ExecSpace::print_configuration(std::cout);

  // Fill arrays from the host for standard random numbers
  std::default_random_engine generator;
  std::uniform_real_distribution<double> pos_dist(xmin, xmax);
  std::uniform_real_distribution<double> mass_dist(massmin, massmax);
  for (index_t i = 0; i < nbody; ++i) {
    h_mass(i) = mass_dist(generator);
    for (index_t j = 0; j < NDIM; ++j) {
      h_pos(i, j) = pos_dist(generator);
      h_vel(i, j) = 0;
    }
  }

  Kokkos::deep_copy(mass, h_mass);
  Kokkos::deep_copy(pos, h_pos);
  Kokkos::deep_copy(vel, h_vel);

  std::cout << "Beginning simulation:\n";

  auto start_chrono = std::chrono::high_resolution_clock::now();

  double t = 0.0;
  while (t < tf) {

    accel(mass, pos, vel, nbody, acc);

    // Update velocity and position of objects
    Kokkos::parallel_for(nbody, KOKKOS_LAMBDA(const int i) {
      FOR_NDIM vel(i, dim) += dt * acc(i, dim);
      FOR_NDIM pos(i, dim) += dt * vel(i, dim);
    });

    // std::cout << t << "\n";

    t += dt;
  }

  auto end_chrono = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed_chrono = end_chrono - start_chrono;

  std::cout << "Elapsed time: " << elapsed_chrono.count() << " secs"
            << std::endl;
  std::cout << "Interactions per second: "
            << ( ( (tf/dt) / elapsed_chrono.count() ) * nbody) * (nbody - 1)
            << std::endl;

  Kokkos::deep_copy(h_pos, pos);

  //	for (index_t i = 0; i < nbody; ++i) {
  //		std::cout << "(";
  //		for (index_t j = 0; j < NDIM; ++j) {
  //			std::cout << h_pos(i, j) << ",";
  //		}
  //		std::cout << ")" << std::endl;
  //	}

  // TODO set so this gets called on exceptions?
  Kokkos::finalize();

  return 0;
}

void accel(const KScalar mass, const KVector pos, const KVector vel,
           const index_t nbody, KVector &acc) {
  // Calculate the new acceleration for each object
  Kokkos::parallel_for(nbody, KOKKOS_LAMBDA(const index_t i) {
    double acc_new[NDIM];
    FOR_NDIM acc_new[dim] = 0;

    // TODO how to ND?
    for (index_t j = 0; j < nbody; ++j) {
			if (i == j) continue;

      double r[NDIM];
      FOR_NDIM r[dim] = pos(j, dim) - pos(i, dim);

      double dist_sum = 0;
      FOR_NDIM dist_sum += r[dim] * r[dim];

      // TODO pass G
      double acc_mag =
          (0.01) * mass(j) / std::pow(dist_sum, 3 / 2); // this includes the 1/r

      FOR_NDIM acc_new[dim] += r[dim] * acc_mag;
    }

    FOR_NDIM acc(i, dim) = acc_new[dim];
  });
}

// TODO all this is really slow.
#if 1

typedef Kokkos::View<
	double *[NDIM], Kokkos::DefaultExecutionSpace::scratch_memory_space,
	Kokkos::MemoryTraits<Kokkos::Unmanaged>>
ScratchVector;
typedef Kokkos::View<
	double *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
	Kokkos::MemoryTraits<Kokkos::Unmanaged>>
ScratchScalar;

// This is cringeworthy but I want to try threading vs vectorizing
#define InnerSplit Kokkos::TeamThreadRange

// TODO this fails on CUDA due to large allocation.  Ditch it?
void accel_2layer(const KScalar mass, const KVector pos, const KVector vel,
                  const index_t nbody, KVector &acc) {
  // Calculate the new acceleration for each object

	int scratch_size = ScratchVector::shmem_size(nbody) +
		ScratchScalar::shmem_size(nbody);

  Kokkos::parallel_for(team_policy(nbody, Kokkos::AUTO).set_scratch_size( 0, Kokkos::PerTeam( scratch_size ) ),
      KOKKOS_LAMBDA(const member_type &member) {
        const int i = member.league_rank();

        VecReduce acc_new;
        FOR_NDIM acc_new.vector[dim] = 0;

				ScratchVector vscratch( member.team_scratch( 0 ), nbody);
				ScratchScalar sscratch(member.team_scratch( 0 ), nbody );

				Kokkos::parallel_for(InnerSplit(member, nbody), [&] (const int j) {
						FOR_NDIM vscratch(j, dim) = pos(j, dim) - pos(i, dim);
					});

				Kokkos::parallel_for(InnerSplit(member, nbody), [&] (const int j) {
						sscratch(j) = 0;
						FOR_NDIM sscratch(j) += vscratch(j, dim)*vscratch(j,dim);
					});

       Kokkos::parallel_for(InnerSplit(member, nbody), [&] (const int j) {
              // TODO pass G
						sscratch(j) = (0.01) * mass(j) /
							std::pow(sscratch(j), 3 / 2); // this includes the 1/r
					});

			// This be some shitty memory access
				FOR_NDIM {
				  Kokkos::parallel_reduce(InnerSplit(member, nbody), [&] (const int j, double &acc_part) {
							if (i != j) acc_part = vscratch(j,dim) * sscratch(j);
						}, acc_new.vector[dim]);
				}

				Kokkos::single( Kokkos::PerTeam( member ), [&] () {
          FOR_NDIM acc(i, dim) = acc_new.vector[dim];
				});
      });
}

void accel_outer(const KScalar mass, const KVector pos, const KVector vel,
                  const index_t nbody, KVector &acc) {
  // Calculate the new acceleration for each object

	FOR_NDIM {
  Kokkos::parallel_for(team_policy(nbody, Kokkos::AUTO),
      KOKKOS_LAMBDA(const member_type &member) {
        const int i = member.league_rank();

				double acc_new_dim;

				Kokkos::parallel_reduce(InnerSplit(member, nbody), [&] (const int j, double &acc_part) {
						double r = pos(j, dim) - pos(i, dim);
						double rmag = 0;
						for (int k = 0; k < NDIM; ++k) {
							double dist = pos(j,k) - pos(i,k);
  						rmag += dist*dist;
						}
            // TODO pass G
						double acc_mag = (0.01) * mass(j) /
							std::pow(rmag, 3 / 2); // this includes the 1/r
						if (i != j) acc_part = r * acc_mag;
					}, acc_new_dim);

				Kokkos::single( Kokkos::PerTeam( member ), [&] () {
          acc(i, dim) = acc_new_dim;
				});
      });
	}
}
#endif
