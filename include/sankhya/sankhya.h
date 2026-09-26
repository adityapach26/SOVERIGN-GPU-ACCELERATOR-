#ifndef SANKHYA_C_API_H
#define SANKHYA_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sankhya_env_s sankhya_env_t;
typedef struct sankhya_model_s sankhya_model_t;

/* Error codes */
#define SANKHYA_SUCCESS 0
#define SANKHYA_ERROR_INVALID_ARGUMENT 1
#define SANKHYA_ERROR_LOAD_FAILED 2
#define SANKHYA_ERROR_SOLVE_FAILED 3
#define SANKHYA_ERROR_NOT_AVAILABLE 4
#define SANKHYA_ERROR_INTERNAL 5

int sankhya_create_env(sankhya_env_t** env);

/* 
 * Loads a model from the specified filename.
 * If filename is "fixture", it constructs a small built-in test model 
 * because the core does not yet have a file reader.
 */
int sankhya_load_model(
    sankhya_env_t* env,
    const char* filename,
    sankhya_model_t** model
);

int sankhya_solve(sankhya_model_t* model);

int sankhya_get_objval(
    sankhya_model_t* model,
    double* objval
);

/* Optional cleanup functions to prevent memory leaks */
void sankhya_free_model(sankhya_model_t* model);
void sankhya_free_env(sankhya_env_t* env);

#ifdef __cplusplus
}
#endif

#endif /* SANKHYA_C_API_H */
