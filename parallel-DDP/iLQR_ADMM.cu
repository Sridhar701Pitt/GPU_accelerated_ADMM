/***
nvcc -std=c++11 -o iLQR.exe WAFR_iLQR_examples.cu utils/cudaUtils.cu utils/threadUtils.cpp -gencode arch=compute_61,code=sm_61 -rdc=true -O3
***/
#define EE_COST 0
#define TOL_COST 0.0

#include "DDPHelpers.cuh"
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

#define ROLLOUT_FLAG 0
#define RANDOM_MEAN 0.0

// pendulum
//#define PLANT == 1
#define RANDOM_STDEV 0.001
#define GOAL_T 3.1416
#define GOAL_O 0.0

char errMsg[]  = "Error: Unkown code - usage is [C]PU or [G]PU with [CS] for serial line search\n";
char tot[]  = " TOT";	char init[] = "INIT";	char fp[]   = "  FP";	char fs[]   = "  FS";	char bp[]   = "  BP";	char nis[]  = " NIS";
double tTime[ADMM_MAX_ITERS];	double fsimTime[ADMM_MAX_ITERS*MAX_ITER];	double fsweepTime[ADMM_MAX_ITERS*MAX_ITER];	double bpTime[ADMM_MAX_ITERS*MAX_ITER];
double nisTime[ADMM_MAX_ITERS*MAX_ITER];	double initTime[ADMM_MAX_ITERS];	algType Jout[ADMM_MAX_ITERS*(MAX_ITER+1)];	int alphaOut[ADMM_MAX_ITERS*(MAX_ITER+1)];
std::default_random_engine randEng(time(0)); //seed
std::normal_distribution<double> randDist(RANDOM_MEAN, RANDOM_STDEV); //mean followed by stdiv

template <typename T>
__host__ __forceinline__
void loadXU(T *x, T *u, T *xGoal, int ld_x, int ld_u){
	for (int k=0; k<NUM_TIME_STEPS; k++){
		T *xk = x + k*ld_x;
		#if PLANT == 1 // pend
			xk[0] = 0.0;	xk[1] = (T)randDist(randEng);
		#endif
	}
	for (int k=0; k<NUM_TIME_STEPS; k++){
		T *uk = u + k*ld_u;
		#if PLANT == 1 || PLANT == 2 // pend and cart
			uk[0] = 0.01;
		#endif
	}
  #if PLANT == 1 // pend
    const T temp[] = {GOAL_T,GOAL_O};
  #endif
  for (int i=0; i < STATE_SIZE; i++){xGoal[i] = temp[i];}
}

template <typename T>
__host__
void testGPU(){
	// GPU VARS	
	// first integer constants for the leading dimmensions of allocaitons
	int ld_x, ld_u, ld_P, ld_p, ld_AB, ld_H, ld_g, ld_KT, ld_du, ld_d, ld_A;
	// then vars for stream handles
	cudaStream_t *streams;
	// algorithm hyper parameters
	T *alpha, *d_alpha;
	int *alphaIndex;
	// then variables defined by blocks for backward pass
	T *d_P, *d_p, *d_Pp, *d_pp, *d_AB, *d_H, *d_g, *d_KT, *d_du;
	// variables for forward pass
	T **d_x, **d_u, **h_d_x, **h_d_u, *d_xp, *d_xp2, *d_up, *d_JT, *J;
	// variables for forward sweep
	T **d_d, **h_d_d, *d_dp, *d_dT, *d, *d_ApBK, *d_Bdu, *d_dM;
	// for checking inversion errors
	int *err, *d_err;
	// for expected cost reduction
	T *dJexp, *d_dJexp;
	// goal point
	T *xGoal, *d_xGoal;
	// Inertias and Tbodybase
	T *d_I, *d_Tbody;

	// Allocate space and initialize the variables
	allocateMemory_GPU<T>(&d_x, &h_d_x, &d_xp, &d_xp2, &d_u, &h_d_u, &d_up, &d_xGoal, &xGoal,
				&d_P, &d_Pp, &d_p, &d_pp, &d_AB, &d_H, &d_g, &d_KT, &d_du,
				&d_d, &h_d_d, &d_dp, &d_dT, &d_dM, &d, &d_ApBK, &d_Bdu,
				&d_JT, &J, &d_dJexp, &dJexp, &alpha, &d_alpha, &alphaIndex, &d_err, &err, 
				&ld_x, &ld_u, &ld_P, &ld_p, &ld_AB, &ld_H, &ld_g, &ld_KT, &ld_du, &ld_d, &ld_A,
						&streams, &d_I, &d_Tbody);

	T *x0 = (T *)malloc(ld_x*NUM_TIME_STEPS*sizeof(T));
	T *u0 = (T *)malloc(ld_u*NUM_TIME_STEPS*sizeof(T));

	// Define the ADMM primal and dual variables
	T *x_bar = (T *)malloc(ld_x*NUM_TIME_STEPS*sizeof(T));
	T *u_bar = (T *)malloc(ld_u*NUM_TIME_STEPS*sizeof(T));
	T *x_lambda = (T *)malloc(ld_x*NUM_TIME_STEPS*sizeof(T));
	T *u_lambda = (T *)malloc(ld_u*NUM_TIME_STEPS*sizeof(T));

  	// Initialise primal,dual variables with random variables from -2*PI to 2*PI
	#define PIRAND ((T) 4*rand()*PI/RAND_MAX - 2*PI)
	for (int k = 0; k < NUM_TIME_STEPS*ld_x; k++){
		x_bar[k] = PIRAND;
		x_lambda[k] = PIRAND;
	}

	for (int k = 0; k < NUM_TIME_STEPS*ld_u; k++){
		u_bar[k] = PIRAND;
		u_lambda[k] = PIRAND;
	}

  	// ADMM for loop starts here **************************************************************************************************
	for (int i=0; i<ADMM_MAX_ITERS; i++)
	{
		// TODO: STEP 1: 1st ADMM sub-block solved by performing ADMM; x0,u0 correspond to x_new,u_new in the ADMM code
		// *** Need to pass x_bar, u_bar, x_lambda, u_lambda to iLQR_GPU for use in cost calculation
		// Pass x_bar, u_bar, x_lambda, u_lambda to runiLQR_GPU -> Pass to forwardSimGPU()
		//                                                      -> Pass to costKern<<>>
		//                                                      -> Pass to costFunc() for cost calc with Augmented Lagrangian
		loadXU<T>(x0,u0,xGoal,ld_x,ld_u);
		runiLQR_GPU<T>(x0, u0, nullptr, nullptr, nullptr, nullptr, xGoal, &Jout[i*(MAX_ITER+1)], &alphaOut[i*(MAX_ITER+1)], ROLLOUT_FLAG, 1,  1,
			&tTime[i], &fsimTime[i*MAX_ITER], &fsweepTime[i*MAX_ITER], &bpTime[i*MAX_ITER], &nisTime[i*MAX_ITER], &initTime[i], streams,
			d_x, h_d_x, d_xp, d_xp2, d_u, h_d_u, d_up, d_P, d_p, d_Pp, d_pp, d_AB, d_H, d_g, d_KT, d_du,
			d_d, h_d_d, d_dp, d_dT, d, d_ApBK, d_Bdu, d_dM, alpha, d_alpha, alphaIndex, d_JT, J, dJexp, d_dJexp, d_xGoal,
			err, d_err, ld_x, ld_u, ld_P, ld_p, ld_AB, ld_H, ld_g, ld_KT, ld_du, ld_d, ld_A,
      x_bar, u_bar, x_lambda, u_lambda,
      d_I, d_Tbody);

		// TODO: STEP 2: Update x_lambda, u_lambda
    // x_lambda = x_lambda + x0 - x_bar;
    // u_lambda = u_lambda + u0 - u_bar;

		// TODO: STEP 3: 2nd ADMM sub-block project x_bar, u_bar into valid control limits

		// TODO: STEP 4: Calculate residuals for x_0, u_0, x_lambda, u_lambda
    // res_u = norm(unew - u_bar);
    // res_x = norm(xnew - x_bar);
    // res_ulambda = roll*norm(u_bar - u_bar_old);
    // res_xlambda = roll*norm(x_bar - x_bar_old);

	}
  // ADMM for loop ends here *************************************************************
  
	// print final state
	printf("Final state:\n");	for (int i = 0; i < STATE_SIZE; i++){printf("%15.5f ",x0[(NUM_TIME_STEPS-2)*ld_x + i]);}	printf("\n");
	
	printf("Final xtraj:\n");  
	for (int i = 0; i < NUM_TIME_STEPS; i++){
		printMat<T,1,DIM_x_r>(&x0[i*ld_x],1,0,1);
	}

	printf("Final control:\n");  
	for (int i = 0; i < NUM_TIME_STEPS; i++){
		printMat<T,1,DIM_u_r>(&u0[i*ld_u],1,0,1);
	}

	// free those vars
	freeMemory_GPU<T>(d_x, h_d_x, d_xp, d_xp2, d_u, h_d_u, d_up, xGoal, d_xGoal,  d_P, d_Pp, d_p, d_pp, d_AB, d_H, d_g, d_KT, d_du, 
				   d_d, h_d_d, d_dp, d_dM, d_dT, d,  d_ApBK, d_Bdu, d_JT, J, d_dJexp, dJexp, alpha, d_alpha, alphaIndex, d_err, err, 
                   streams, d_I, d_Tbody);
	
	free(x0);
	free(u0);
	free(x_bar);
	free(u_bar);
	free(x_lambda);
	free(u_lambda);
}

int main(int argc, char *argv[])
{
	srand(time(NULL));

	testGPU<algType>();
  
	return 0;
}