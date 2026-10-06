#include <math.h>

#include "mp_structs.h"

void verlet_integrator_init(integrator *integrator) {
    integrator->part1 = verlet_integrator_part1;
    integrator->part2 = verlet_integrator_part2;
    integrator->init_thermostat = verlet_integrator_init_thermostat;
    integrator->stop_thermostat = verlet_integrator_stop_thermostat;
}

void verlet_integrator_part1(integrator *integrator, particles *p) {
    double target_T = integrator->T;
    double current_T;

    long int ni, nj;
    double app;
    double dt = integrator->dt;

    double L = p->L;
    double *pos = p->pos;
    double *vel = p->vel;
    double *forces = p->fcs_tot;
    double *masses = p->mass;
    
    if (integrator->enabled == INTEGRATOR_ENABLED) {
        current_T = particles_get_temperature(p);
        double scale = sqrt(target_T / current_T);
        for (int i = 0; i < p->n_p; i++) {
            if (p->fixed[i]) continue;
            ni = i * 3;
            vel[ni]     *= scale;
            vel[ni + 1] *= scale;
            vel[ni + 2] *= scale;
        }
    }

    #pragma omp parallel for private(ni,nj,app)
    for (int i = 0; i < p->n_p; i++) {
        if (p->fixed[i]) continue;
        ni = i * 3;
        for (int j = 0; j < 3; j++) {
            nj = ni + j;
            vel[nj] += 0.5 * dt * forces[nj] / masses[i];
            app = pos[nj] + dt * vel[nj];
            if (app < 0) {
                pos[nj] = app + L;
            } else if (app >= L) {
                pos[nj] = app - L;
            } else {
                pos[nj] = app;
            }
        }
    }
}

void verlet_integrator_part2(integrator *integrator, particles *p) {
    long int ni, nj;
    double dt = integrator->dt;

    double *vel = p->vel;
    double *forces = p->fcs_tot;
    double *masses = p->mass;

    #pragma omp parallel for private(ni,nj)
    for (int i = 0; i < p->n_p; i++) {
        if (p->fixed[i]) continue;
        ni = i * 3;
        for (int j = 0; j < 3; j++) {
            nj = ni + j;
            vel[nj] += 0.5 * dt * forces[nj] / masses[i];
        }
    }
}

void verlet_integrator_init_thermostat(integrator *integrator, double *params) {
    integrator->T = params[0];
    integrator->enabled = INTEGRATOR_ENABLED;
}

void verlet_integrator_stop_thermostat(integrator *integrator) {
    integrator->enabled = INTEGRATOR_DISABLED;
}