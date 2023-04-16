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
#include <string>

#define ROLLOUT_FLAG 0
#define RANDOM_MEAN 0.0

// pendulum
//#define PLANT == 1
#define RANDOM_STDEV 0.5
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
void loadXU(T *x, T *u, T *xGoal, int ld_x, int ld_u,
			T *x_bar, T *u_bar, T *x_lambda, T *u_lambda){
	for (int k=0; k<NUM_TIME_STEPS; k++){
		T *xk = x + k*ld_x;
		
		T *xk_bar = x_bar + k*ld_x;
		T *xk_lambda = x_lambda + k*ld_x;
		
		#if PLANT == 1 // pend
			xk[0] = 0.0;	xk[1] = (T)randDist(randEng);
			xk_bar[0] =  (T)randDist(randEng);	xk_bar[1] = (T)randDist(randEng);
			xk_lambda[0] = (T)randDist(randEng);	xk_lambda[1] = (T)randDist(randEng);
		#endif
	}
	for (int k=0; k<NUM_TIME_STEPS; k++){
		T *uk = u + k*ld_u;

		T *uk_bar = u_bar + k*ld_u;
		T *uk_lambda = u_lambda + k*ld_u;
		
		#if PLANT == 1 || PLANT == 2 // pend and cart
			uk[0] = 0.01;
			uk_bar[0] = (T)randDist(randEng);
			uk_lambda[0] = (T)randDist(randEng);
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

	// global copies and dual variables
	T **u_bar, **h_u_bar;
	T **x_bar, **h_x_bar;
	T **u_lambda, **h_u_lambda;
	T **x_lambda, **h_x_lambda;

	// Allocate space and initialize the variables
	allocateMemory_GPU<T>(&d_x, &h_d_x, &d_xp, &d_xp2, &d_u, &h_d_u, &d_up, &d_xGoal, &xGoal,
				&d_P, &d_Pp, &d_p, &d_pp, &d_AB, &d_H, &d_g, &d_KT, &d_du,
				&d_d, &h_d_d, &d_dp, &d_dT, &d_dM, &d, &d_ApBK, &d_Bdu,
				&d_JT, &J, &d_dJexp, &dJexp, &alpha, &d_alpha, &alphaIndex, &d_err, &err, 
				&ld_x, &ld_u, &ld_P, &ld_p, &ld_AB, &ld_H, &ld_g, &ld_KT, &ld_du, &ld_d, &ld_A,
				&u_bar, &h_u_bar, &x_bar, &h_x_bar, &u_lambda, &h_u_lambda, &x_lambda, &h_x_lambda, 
				&streams, &d_I, &d_Tbody);

	// Variables for storing the quantities corresponding to the selected alpha for each iLQR iteration
	T *x0 = (T *)malloc(ld_x*NUM_TIME_STEPS*sizeof(T));
	T *u0 = (T *)malloc(ld_u*NUM_TIME_STEPS*sizeof(T));
	T *x_bar_0 = (T *)malloc(ld_x*NUM_TIME_STEPS*sizeof(T));
	T *u_bar_0 = (T *)malloc(ld_u*NUM_TIME_STEPS*sizeof(T));
	T *x_lambda_0 = (T *)malloc(ld_x*NUM_TIME_STEPS*sizeof(T));
	T *u_lambda_0 = (T *)malloc(ld_u*NUM_TIME_STEPS*sizeof(T));
	T *x_bar_old = (T *)malloc(ld_x*NUM_TIME_STEPS*sizeof(T));
	T *u_bar_old = (T *)malloc(ld_u*NUM_TIME_STEPS*sizeof(T));

	// state and control limits 
	T x_lims[2][2] = {{-1.57, 4.71}, 	// assume theta -> [-pi/2 , 3pi /2]
					  {-10.0, 10.0}};	// assume theta_dot -> [-10 , 10]
	T u_lims[2] = {-1.0, 1.0};			// assume u -> [-10 , 10]

	// Initialise primal and dual variables with values
	loadXU<T>(x0,u0,xGoal,ld_x,ld_u, x_bar_0, u_bar_0, x_lambda_0, u_lambda_0);

	// Copy contents from x_bar,u_bar into x_bar_old,u_bar_old
	memcpy(x_bar_old, x_bar_0, ld_x*NUM_TIME_STEPS*sizeof(T));
	memcpy(u_bar_old, u_bar_0, ld_u*NUM_TIME_STEPS*sizeof(T));

	// Open files for writing plot data
	fclose(fopen("metric_plots/plot_res_u.txt","w"));
	FILE *file_res_u = fopen("metric_plots/plot_res_u.txt","a");

	fclose(fopen("metric_plots/plot_res_x.txt","w"));
	FILE *file_res_x = fopen("metric_plots/plot_res_x.txt","a");

	fclose(fopen("metric_plots/plot_res_x_lambda.txt","w"));
	FILE *file_res_x_lambda = fopen("metric_plots/plot_res_x_lambda.txt","a");

	fclose(fopen("metric_plots/plot_res_u_lambda.txt","w"));
	FILE *file_res_u_lambda = fopen("metric_plots/plot_res_u_lambda.txt","a");

	fclose(fopen("metric_plots/config_constants.txt","w"));
	FILE *file_config_constants = fopen("metric_plots/config_constants.txt","a");

  	// ADMM for loop starts here **************************************************************************************************
	for (int i=0; i<ADMM_MAX_ITERS; i++)
	{
		// *********************************************************************************************************************************
		// TODO: STEP 1: 1st ADMM sub-block solved by performing ADMM; x0,u0 correspond to x_new,u_new in the ADMM code
		// *** Need to pass x_bar, u_bar, x_lambda, u_lambda to iLQR_GPU for use in cost calculation
		// Pass x_bar, u_bar, x_lambda, u_lambda to runiLQR_GPU -> Pass to forwardSimGPU()
		//                                                      -> Pass to costKern<<>>
		//                                                      -> Pass to costFunc() for cost calc with Augmented Lagrangian
		runiLQR_GPU<T>(x0, u0, 
			x_bar_0, u_bar_0, x_lambda_0, u_lambda_0,
			nullptr, nullptr, nullptr, nullptr, xGoal, &Jout[i*(MAX_ITER+1)], &alphaOut[i*(MAX_ITER+1)], ROLLOUT_FLAG, 1,  1,
			&tTime[i], &fsimTime[i*MAX_ITER], &fsweepTime[i*MAX_ITER], &bpTime[i*MAX_ITER], &nisTime[i*MAX_ITER], &initTime[i], streams,
			d_x, h_d_x, d_xp, d_xp2, d_u, h_d_u, d_up, d_P, d_p, d_Pp, d_pp, d_AB, d_H, d_g, d_KT, d_du,
			d_d, h_d_d, d_dp, d_dT, d, d_ApBK, d_Bdu, d_dM, alpha, d_alpha, alphaIndex, d_JT, J, dJexp, d_dJexp, d_xGoal,
			err, d_err, ld_x, ld_u, ld_P, ld_p, ld_AB, ld_H, ld_g, ld_KT, ld_du, ld_d, ld_A,
			x_bar, u_bar, x_lambda, u_lambda,
			h_x_bar, h_u_bar, h_x_lambda, h_u_lambda,
			d_I, d_Tbody);
		// *********************************************************************************************************************************

		// *********************************************************************************************************************************
		// TODO: STEP 2: Update x_lambda, u_lambda
		// x_lambda = x_lambda + x0 - x_bar;
		// u_lambda = u_lambda + u0 - u_bar;
		
		// Updating x_lambda
		for (int k=0; k<NUM_TIME_STEPS; k++)
		{	
			T *xk = x0 + k*ld_x;
			T *xk_lambda = x_lambda_0 + k*ld_x;
			T *xk_bar = x_bar_0 + k*ld_x;
			
			xk_lambda[0] = xk_lambda[0] + xk[0] - xk_bar[0];
			xk_lambda[1] = xk_lambda[1] + xk[1] - xk_bar[1];
		}
		
		// Updating u_lambda
		for (int k=0; k<NUM_TIME_STEPS; k++)
		{	
			T *uk = u0 + k*ld_u;
			T *uk_lambda = u_lambda_0 + k*ld_u;
			T *uk_bar = u_bar_0 + k*ld_u;
			
			uk_lambda[0] = uk_lambda[0] + uk[0] - uk_bar[0];
		}
		// *********************************************************************************************************************************

		// *********************************************************************************************************************************
		// TODO: STEP 3: 2nd ADMM sub-block project x_bar, u_bar into valid control limits
		// We need to project the (x0 + x_lambda),(u0 + u_lambda) variable to the valid state and control set
		
		// x0 + x_lambda Projection
		for (int k=0; k<NUM_TIME_STEPS; k++)
		{	
			T *xk = x0 + k*ld_x;
			T *xk_lambda = x_lambda_0 + k*ld_x;
			T *xk_bar = x_bar_0 + k*ld_x;

			// Update x_bar[-]
			if (xk[0] + xk_lambda[0] < x_lims[0][0])
				{xk_bar[0] = x_lims[0][0]; }
			
			else if (xk[0] + xk_lambda[0] > x_lims[0][1])
				{xk_bar[0] = x_lims[0][1];	}
			
			else
				{xk_bar[0] = xk[0] + xk_lambda[0];	}
			
			// Update x_bar[1]
			if (xk[1] + xk_lambda[1] < x_lims[1][0])
				{xk_bar[1] = x_lims[1][0]; }
			
			else if (xk[1] + xk_lambda[1] > x_lims[1][1])
				{xk_bar[1] = x_lims[1][1];	}
			
			else
				{xk_bar[1] = xk[1] + xk_lambda[1];	}
		}

		// u0 + u_lambda Projection
		for (int k=0; k<NUM_TIME_STEPS; k++)
		{	
			T *uk = u0 + k*ld_u;
			T *uk_lambda = u_lambda_0 + k*ld_u;
			T *uk_bar = u_bar_0 + k*ld_u;

			// Update u_bar
			if (uk[0] + uk_lambda[0] < u_lims[0])
				{uk_bar[0] = u_lims[0]; }
			
			else if (uk[0] + uk_lambda[0] > u_lims[1])
				{uk_bar[0] = u_lims[1];	}
			
			else
				{uk_bar[0] = uk[0] + uk_lambda[0];	}
		}
		// CLOCK_DELAY();

		// printf("*************************************************\n");
		// *********************************************************************************************************************************

		// *********************************************************************************************************************************
		// TODO: STEP 4: Calculate residuals for x_0, u_0, x_lambda, u_lambda
		// res_u = norm(unew - u_bar);
		// res_x = norm(xnew - x_bar);
		// res_ulambda = RHO_ADMM * norm(u_bar - u_bar_old);
		// res_xlambda = RHO_ADMM * norm(x_bar - x_bar_old);
		
		// Calc residual_x
		float res_x = 0;
		float element_diff;
		for (int k=0; k<NUM_TIME_STEPS; k++)
		{	
			T *xk = x0 + k*ld_x;
			T *xk_bar = x_bar_0 + k*ld_x;

			element_diff = xk[0] - xk_bar[0];
			res_x += pow(element_diff,2);
			element_diff = xk[1] - xk_bar[1];
			res_x += pow(element_diff,2);
		}

		// Calc residual_u
		float res_u = 0;
		for (int k=0; k<NUM_TIME_STEPS; k++)
		{	
			T *uk = u0 + k*ld_u;
			T *uk_bar = u_bar_0 + k*ld_u;
			
			element_diff = uk[0] - uk_bar[0];
			res_u += pow(element_diff,2);
		}

		// Calc residual_x_lambda
		float res_x_lambda = 0;
		for (int k=0; k<NUM_TIME_STEPS; k++)
		{	
			
			T *xk_bar = x_bar_0 + k*ld_x;
			T *xk_bar_old = x_bar_old + k*ld_x;

			element_diff = xk_bar[0] - xk_bar_old[0];
			res_x_lambda += pow(element_diff,2);
			element_diff = xk_bar[1] - xk_bar_old[1];
			res_x_lambda += pow(element_diff,2);
		}

		// Calc residual_u_lambda
		float res_u_lambda = 0;
		for (int k=0; k<NUM_TIME_STEPS; k++)
		{	
			T *uk_bar = u_bar_0 + k*ld_u;
			T *uk_bar_old = u_bar_old + k*ld_u;
			
			element_diff = uk_bar[0] - uk_bar_old[0];
			res_u_lambda += pow(element_diff,2);
		}

		// Copy contents from x_bar,u_bar into x_bar_old,u_bar_old
		memcpy(x_bar_old, x_bar_0, ld_x*NUM_TIME_STEPS*sizeof(T));
		memcpy(u_bar_old, u_bar_0, ld_u*NUM_TIME_STEPS*sizeof(T));

		fprintf(file_res_u,"%.4f\n", res_u);
		fprintf(file_res_x,"%.4f\n", res_x);
		fprintf(file_res_x_lambda,"%.4f\n", res_x_lambda);
		fprintf(file_res_u_lambda,"%.4f\n", res_u_lambda);
		// *********************************************************************************************************************************
	}

	// Experiment Setting Params: ADMM_MAX_ITERS, RHO_ADMM, MAX_ITER, M, TOTAL_TIME, u_lims[0], u_lims[1]
	fprintf(file_config_constants,"%d\n%f\n%d\n%d\n%f\n%f\n%f\n", ADMM_MAX_ITERS, RHO_ADMM, MAX_ITER, M, TOTAL_TIME, u_lims[0], u_lims[1]);
		
	fclose(file_res_u);
	fclose(file_res_x);
	fclose(file_res_x_lambda);
	fclose(file_res_u_lambda);
	fclose(file_config_constants);
  // ADMM for loop ends here *************************************************************
  
	// print final state
	printf("Final state:\n");	for (int i = 0; i < STATE_SIZE; i++){printf("%15.5f ",x0[(NUM_TIME_STEPS-2)*ld_x + i]);}	printf("\n");
	
	printf("Final xtraj:\n"); 
	fclose(fopen("mat.txt","w"));
	FILE *f = fopen("mat.txt","a");
	for (int i = 0; i < NUM_TIME_STEPS; i++){
		printMat<T,1,DIM_x_r>(&x0[i*ld_x],1,0,1,f);
	}
	fclose(f);

	printf("Final control:\n");  
	for (int i = 0; i < NUM_TIME_STEPS; i++){
		printMat<T,1,DIM_u_r>(&u0[i*ld_u],1,0,0);
	}

	// free those vars
	freeMemory_GPU<T>(d_x, h_d_x, d_xp, d_xp2, d_u, h_d_u, d_up, xGoal, d_xGoal,  d_P, d_Pp, d_p, d_pp, d_AB, d_H, d_g, d_KT, d_du, 
				   d_d, h_d_d, d_dp, d_dM, d_dT, d,  d_ApBK, d_Bdu, d_JT, J, d_dJexp, dJexp, alpha, d_alpha, alphaIndex, d_err, err, 
                   u_bar, h_u_bar, x_bar, h_x_bar, u_lambda, h_u_lambda, x_lambda, h_x_lambda,
				   streams, d_I, d_Tbody);
	
	free(x0);
	free(u0);
	free(x_bar_0);
	free(u_bar_0);
	free(x_lambda_0);
	free(u_lambda_0);
}

int main(int argc, char *argv[])
{
	srand(time(NULL));

	testGPU<algType>();
  
	return 0;
}