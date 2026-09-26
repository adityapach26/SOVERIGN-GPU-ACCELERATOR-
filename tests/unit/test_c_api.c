#include <stdio.h>
#include <stdlib.h>
#include "sankhya/sankhya.h"

int main(void) {
    sankhya_env_t* env = NULL;
    sankhya_model_t* model = NULL;
    double objval = 0.0;
    int rc;

    printf("Starting C API test...\n");

    /* 1. Create environment */
    rc = sankhya_create_env(&env);
    if (rc != SANKHYA_SUCCESS) {
        printf("FAILED: sankhya_create_env returned %d\n", rc);
        return 1;
    }
    if (env == NULL) {
        printf("FAILED: env is NULL after success\n");
        return 1;
    }

    /* 2. Load model (using the custom fixture) */
    rc = sankhya_load_model(env, "fixture", &model);
    if (rc != SANKHYA_SUCCESS) {
        printf("FAILED: sankhya_load_model returned %d\n", rc);
        sankhya_free_env(env);
        return 1;
    }
    if (model == NULL) {
        printf("FAILED: model is NULL after success\n");
        sankhya_free_env(env);
        return 1;
    }

    /* Test invalid argument */
    if (sankhya_load_model(env, NULL, &model) != SANKHYA_ERROR_INVALID_ARGUMENT) {
        printf("FAILED: expected INVALID_ARGUMENT\n");
        return 1;
    }

    /* 3. Solve the model */
    rc = sankhya_solve(model);
    if (rc != SANKHYA_SUCCESS) {
        printf("FAILED: sankhya_solve returned %d\n", rc);
        sankhya_free_model(model);
        sankhya_free_env(env);
        return 1;
    }

    /* 4. Get objective value */
    rc = sankhya_get_objval(model, &objval);
    if (rc != SANKHYA_SUCCESS) {
        printf("FAILED: sankhya_get_objval returned %d\n", rc);
        sankhya_free_model(model);
        sankhya_free_env(env);
        return 1;
    }

    /* Verify objective value (our fixture min -x1 - x2 => -2.0) */
    if (objval < -2.001 || objval > -1.999) {
        printf("FAILED: expected objective roughly -2.0, got %f\n", objval);
        sankhya_free_model(model);
        sankhya_free_env(env);
        return 1;
    }

    printf("C API test passed!\n");

    /* 5. Cleanup */
    sankhya_free_model(model);
    sankhya_free_env(env);
    
    return 0;
}
