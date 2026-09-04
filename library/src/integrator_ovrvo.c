#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mp_structs.h"
#include "constants.h"
#include "mpi_base.h"

// Generated independent random numbers from a uniform distribution using
// Box-Muller transform.
// https://literateprograms.org/box-muller_transform__c_.html
double randn() {
    double U1, U2, R, mult;
    static double X2;
    static int call = 0;

    if (call == 1) {
        call = !call;
        return X2;
    }

    do {
        U1 = 2.0 * rand () / RAND_MAX - 1;
        U2 = 2.0 * rand () / RAND_MAX - 1;
        R = pow (U1, 2) + pow (U2, 2);
    }
    while (R >= 1 || R == 0);

    mult = sqrt(-2 * log (R) / R);

    X2 = U2 * mult;

    call = !call;

    return U1 * mult;
}

void ovrvo_integrator_init(integrator *integrator) {
    integrator->part1 = ovrvo_integrator_part1;
    integrator->part2 = ovrvo_integrator_part2;
    integrator->init_thermostat = ovrvo_integrator_init_thermostat;
    integrator->stop_thermostat = ovrvo_integrator_stop_thermostat;
}

void o_block(integrator *integrator, particles *p) {
    // This function needs to be MPI aware as it is possible for every process to generate
    // different random numbers leading to very process working with desynchronized velocities/positions.
    int rank = get_rank();
    long int ni;
    double dt = integrator->dt;
    double T = integrator->T;
    double var1, var2;
    
    int n_p = p->n_p;
    double *vel = p->vel;
    double *masses = p->mass;
    int *types = p->types;

    if (rank == 0) {
        // The original call to multivariate_normal had a diagonal covariance so we are fine
        // with using randn for each component to generate 3 independent random numbers with the respective
        // mean = 0.0 and variance = 1.0.
        #pragma omp parallel for private(ni, var1, var2)
        for (int i = 0; i < n_p; i++) {
            double c1 = integrator->c1_by_type == NULL
                ? integrator->c1 : integrator->c1_by_type[types[i]];
            double c1_sqrt = sqrt(c1);
            ni = i * 3;
            var2 = (1 - c1) * kB * T;
            var1 = sqrt(var2 / masses[i]);
            for (int j = 0; j < 3; j++) {
                vel[ni + j] *= c1_sqrt;
                vel[ni + j] += var1 * randn();
            }
        }
    }

    bcast_double(vel, n_p * 3, 0);
}

void v_block(integrator *integrator, particles *p) {
    double dt = integrator->dt;
    int n_p = p->n_p;
    double *vel = p->vel;
    double *forces = p->fcs_tot;
    double *masses = p->mass;
    int *types = p->types;

    long int ni;
    #pragma omp parallel for private(ni)
    for (int i = 0; i < n_p; i++) {
        double c2 = integrator->c2_by_type == NULL
            ? integrator->c2 : integrator->c2_by_type[types[i]];
        ni = i * 3;
        for (int j = 0; j < 3; j++) {
            vel[ni + j] += 0.5 * dt * c2 * forces[ni + j] / masses[i];
        }
    }
}

void r_block(integrator *integrator, particles *p) {
    int n_p = p->n_p;
    double dt = integrator->dt;

    double *pos = p->pos;
    double *vel = p->vel;
    int *types = p->types;
    double L = p->L;

    long int ni;
    double app;
    #pragma omp parallel for private(ni, app)
    for (int i = 0; i < n_p; i++) {
        double c2 = integrator->c2_by_type == NULL
            ? integrator->c2 : integrator->c2_by_type[types[i]];
        ni = i * 3;
        for (int j = 0; j < 3; j++) {
            app = pos[ni + j] + c2 * dt * vel[ni + j];
            if (app < 0) {
                pos[ni + j] = app + L;
            } else if (app >= L) {
                pos[ni + j] = app - L;
            } else {
                pos[ni + j] = app;
            }
        }
    }
}

void ovrvo_integrator_part1(integrator *integrator, particles *p) {
    if (integrator->enabled == INTEGRATOR_ENABLED) {
        o_block(integrator, p);
    }
    v_block(integrator, p);
    r_block(integrator, p);
}

void ovrvo_integrator_part2(integrator *integrator, particles *p) {
    v_block(integrator, p);
    if (integrator->enabled == INTEGRATOR_ENABLED) {
        o_block(integrator, p);
    }
}

void ovrvo_integrator_init_thermostat(integrator *integrator, double *params) {
    integrator->T = params[0];
    double gamma = params[1];

    integrator->enabled = INTEGRATOR_ENABLED;

    integrator->c1 = exp(-gamma * integrator->dt);
    integrator->c2 = sqrt(2 / (gamma * integrator->dt) * tanh(0.5 * gamma * integrator->dt));
}

void ovrvo_integrator_init_type_thermostat(integrator *integrator, double T,
                                           int n_typ, const double *gammas) {
    integrator->T = T;
    integrator->enabled = INTEGRATOR_ENABLED;
    integrator->n_typ = n_typ;
    integrator->c1_by_type = malloc(n_typ * sizeof(double));
    integrator->c2_by_type = malloc(n_typ * sizeof(double));
    for (int type = 0; type < n_typ; type++) {
        double gamma_dt = gammas[type] * integrator->dt;
        integrator->c1_by_type[type] = exp(-gamma_dt);
        integrator->c2_by_type[type] = sqrt(2.0 / gamma_dt * tanh(0.5 * gamma_dt));
    }
}

void ovrvo_integrator_stop_thermostat(integrator *integrator) {
    integrator->enabled = INTEGRATOR_DISABLED;
    integrator->c1 = 1.0;
    integrator->c2 = 1.0;
    free(integrator->c1_by_type);
    free(integrator->c2_by_type);
    integrator->c1_by_type = NULL;
    integrator->c2_by_type = NULL;
    integrator->n_typ = 0;
}
