#include <stdlib.h>
#include <math.h>

#include "mp_structs.h"
#include "constants.h"

char integrator_type_str[2][16] = {"OVRVO", "VERLET"};

int get_integrator_type_num() {
    return INTEGRATOR_TYPE_NUM;
}

char *get_integrator_type_str(int n) {
    return integrator_type_str[n];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// General integrator functions
integrator * integrator_init(int n_p, double dt, int type) {
    integrator *new = (integrator *)malloc(sizeof(integrator));
    new->type = type;
    new->n_p = n_p;
    new->dt = dt;

    new->enabled = INTEGRATOR_DISABLED;
    new->c1 = 1.0;
    new->c2 = 1.0;
    new->n_typ = 0;
    new->c1_by_type = NULL;
    new->c2_by_type = NULL;

    switch (type) {
        case INTEGRATOR_TYPE_OVRVO:
            ovrvo_integrator_init(new);
            break;
        case INTEGRATOR_TYPE_VERLET:
            verlet_integrator_init(new);
            break;
    }

    new->free = integrator_free;

    return new;
}

void integrator_free(integrator *integrator) {
    free(integrator->c1_by_type);
    free(integrator->c2_by_type);
    free(integrator);
}
