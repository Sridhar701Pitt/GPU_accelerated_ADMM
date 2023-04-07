/*****************************************************************
 * DDP MPC Helper Functions
 * (currently only supports iLQR - UDP in future release)
 *
 * 1: Variable Wrappers (structs) and memory allocators/freers
 *  algTrace
 *  trajVars
 *  CPUVars
 *  GPUVars
 *  matDimms
 *  allocateMemory_(C/G)PU_MPC
 *  freeMemory_(C/G)PU_MPC
 *
 * 2: MPC Alg Helpers
 *  shiftAndCopy(Kern)
 *  computeControlSimple
 *  rolloutMPC(Kern)
 *  rolloutMPC2(Kern)
 *  loadVars(C/G)PU_MPC
 *  storeVars(C/G)PU_MPC
 *
 * 3: MPC Main Algorithm Wrappers
 *   runiLQR_MPC_(C/G)PU
 *
 * 4: LCM helpers
 *   LCM_MPCLoop_Handler
 *   LCM_TrajRunner_Handler
 *   
 *
 *****************************************************************/

#include <lcm/lcm-cpp.hpp>
#include "../lcmtypes/drake/lcmt_iiwa_status.hpp"
#include "../lcmtypes/drake/lcmt_iiwa_command.hpp"
#include "../lcmtypes/kuka/lcmt_target_twist.hpp"

const char *GOAL_CHANNEL        = "GOAL_CHANNEL";
const char *ARM_STATUS_CHANNEL  = "IIWA_STATUS";//"ARM_STATUS_CHANNEL";
const char *ARM_COMMAND_CHANNEL = "IIWA_COMMAND";//ARM_COMMAND_CHANNEL";
#define GOAL_PUBLISHER_RATE_MS 30
#define TEST_DELTA 0 // random small delta to keep things interesting (in ms) for tests

#define TIME_STEP_LENGTH_IN_ms (TIME_STEP*1000.0)
#define TIME_STEP_LENGTH_IN_us (TIME_STEP_LENGTH_IN_ms*1000.0)
#define get_time_us_i64(time) (static_cast<int64_t>(std::ceil(get_time_us(time))))
#define get_time_ms_i64(time) (static_cast<int64_t>(std::ceil(get_time_ms(time))))
#define us_to_ms_i64(time) (static_cast<int64_t>(std::ceil(static_cast<double>(time)/1000.0))
#define ms_to_us_i64(time) (time*1000)
#define get_time_steps_us_d(start,end) (static_cast<double>(end - start)/TIME_STEP_LENGTH_IN_us)
#define get_time_steps_ms_d(start,end) (static_cast<double>(end - start)/TIME_STEP_LENGTH_IN_ms)
#define get_time_steps_us_f(start,end) (static_cast<int>(std::floor(get_time_steps_us_d(start,end))))
#define get_time_steps_ms_f(start,end) (static_cast<int>(std::floor(get_time_steps_ms_d(start,end))))
#define get_time_steps_us_c(start,end) (static_cast<int>(std::ceil(get_time_steps_us_d(start,end))))
#define get_time_steps_ms_c(start,end) (static_cast<int>(std::ceil(get_time_steps_ms_d(start,end))))
#define get_steps_us_d(delta) (delta/TIME_STEP_LENGTH_IN_us)
#define get_steps_ms_d(delta) (delta/TIME_STEP_LENGTH_IN_ms)
#define get_steps_us_f(delta) (static_cast<int>(std::floor(get_steps_us_d(delta))))
#define get_steps_ms_f(delta) (static_cast<int>(std::floor(get_steps_ms_d(delta))))
#define get_steps_us_c(delta) (static_cast<int>(std::ceil(get_steps_us_d(delta))))
#define get_steps_ms_c(delta) (static_cast<int>(std::ceil(get_steps_ms_d(delta))))

#include <mutex>
#include <vector>

// 1: Variable Wrappers (structs) and memory allocators/freers //
    template <typename T>
    struct algTrace{
        std::vector<T> J;             std::vector<int> alpha;
        std::vector<double> tTime;    std::vector<double> simTime;    std::vector<double> sweepTime;
        std::vector<double> initTime; std::vector<double> bpTime;     std::vector<double> nisTime;
    };
    template <typename T>
    struct trajVars{
        T *x;       T *u;       T *KT;
        int ld_x;   int ld_u;   int ld_KT;
        int64_t t0_plant;       int64_t t0_sys; // the plant (arm) may have its own clock which may differ than my system time
        std::mutex *lock;       bool first_pass;
        int last_successful_solve;
    };

    template <typename T>
    struct CPUVars{
        T *x;       T *xp;  T *xp2; T *x_old; 
        T *u;       T *up;          T *u_old; 
        T *P;       T *p;   T *Pp;  T *pp; 
        T *AB;      T *H;   T *g;   T *KT;   T *KT_old;   T *du; 
        T *d;       T *dp; 
        T *ApBK;    T *Bdu;
        T *JT;      T *dJexp;
        T *alpha;   int *err;
        T *I;       T *Tbody;
        T *xGoal;   T *xActual;
        std::thread *threads;
        std::mutex *lock;
    };

    template <typename T>
    struct GPUVars{
        T **d_x;   T **h_d_x;   T *d_xp;   T *d_xp2;  T *d_x_old; 
        T **d_u;   T **h_d_u;   T *d_up;              T *d_u_old;
        T *d_P;    T *d_p;      T *d_Pp;   T *d_pp; 
        T *d_AB;   T *d_H;      T *d_g;    T *d_KT;   T *d_KT_old;  T *d_du;
        T **d_d;   T **h_d_d;   T *d_dT;   T *d_dp;   T *d_dM;      T *d;
        T *d_ApBK; T *d_Bdu;
        T *d_JT;   T *J;        T *dJexp;  T *d_dJexp; 
        T *alpha;  T *d_alpha;  int *alphaIndex;   int *err;  int *d_err;
        T *d_I;    T *d_Tbody;
        T *xGoal;  T *d_xGoal;  T *xActual;  T *d_xActual;
        cudaStream_t *streams;
        std::mutex *lock;
    };

    struct matDimms{
        int ld_x;   int ld_u;   int ld_P;   int ld_p; 
        int ld_AB;  int ld_H;   int ld_g;
        int ld_KT;  int ld_du;  int ld_d;   int ld_A;
    };
    //int 64_t t_actual; is the only other var we will ever need
    template <typename T>
    __host__ __forceinline__
    void allocateTrajVars(trajVars<T> *tv, matDimms *md){
        tv->x =  (T *)malloc((md->ld_x)*NUM_TIME_STEPS*sizeof(T));
        tv->u =  (T *)malloc((md->ld_u)*NUM_TIME_STEPS*sizeof(T));
        tv->KT = (T *)malloc((md->ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));
        tv->ld_x = md->ld_x;    tv->ld_u = md->ld_u;    tv->ld_KT = md->ld_KT;
        tv->t0_plant = 0;       tv->t0_sys = 0;         tv->first_pass = true;
        tv->lock = new std::mutex; 
        tv->last_successful_solve = 0;
    }

    template <typename T>
    __host__ __forceinline__
    void freeTrajVars(trajVars<T> *tv){
        free(tv->x);    free(tv->u);    free(tv->KT);
        delete tv->lock;
    }
    template <typename T>
    __host__ __forceinline__
    void allocateMemory_GPU_MPC(GPUVars<T> *gv, matDimms *md, trajVars<T> *tv){
        // first use the old memory allocator which we know works
        // note on device x,u is [NUM_ALPHA][SIZE*NUM_TIME_STEPS]
        //      on host   x,u is [SIZE*NUM_TIME_STEPS]
        md->ld_u = DIM_u_r;    md->ld_x = DIM_x_r;    // we should use pitched malloc but for now just set to DIM_<>_r
        gv->h_d_x = (T **)malloc(NUM_ALPHA*sizeof(T*));
        gv->h_d_u = (T **)malloc(NUM_ALPHA*sizeof(T*));
        gpuErrchk(cudaMalloc((void**)&(gv->d_x), NUM_ALPHA*sizeof(T*))); 
        gpuErrchk(cudaMalloc((void**)&(gv->d_u), NUM_ALPHA*sizeof(T*)));
        for (int i=0; i<NUM_ALPHA; i++){
            gpuErrchk(cudaMalloc((void**)&((gv->h_d_x)[i]),(md->ld_x)*NUM_TIME_STEPS*sizeof(T)));
            gpuErrchk(cudaMalloc((void**)&((gv->h_d_u)[i]),(md->ld_u)*NUM_TIME_STEPS*sizeof(T)));
        }
        gpuErrchk(cudaMalloc((void**)&(gv->d_xp), (md->ld_x)*NUM_TIME_STEPS*sizeof(T))); 
        gpuErrchk(cudaMalloc((void**)&(gv->d_xp2), (md->ld_x)*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_x_old), (md->ld_x)*NUM_TIME_STEPS*sizeof(T))); 
        gpuErrchk(cudaMalloc((void**)&(gv->d_up), (md->ld_u)*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_u_old), (md->ld_u)*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMemcpy(gv->d_x, gv->h_d_x, NUM_ALPHA*sizeof(T*), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(gv->d_u, gv->h_d_u, NUM_ALPHA*sizeof(T*), cudaMemcpyHostToDevice));

        // and for the xGoal
        int goalSize = EE_COST ? 6 : STATE_SIZE;
        gv->xGoal = (T *)malloc(goalSize*sizeof(T));
        gpuErrchk(cudaMalloc((void**)&(gv->d_xGoal),goalSize*sizeof(T)));

        // allocate memory with pitched malloc and thus collect the lds (for now just set to DIM_<>_r)
        md->ld_P = DIM_P_r;    md->ld_p = DIM_p_r;    md->ld_AB = DIM_AB_r;  md->ld_H = DIM_H_r;    md->ld_g = DIM_g_r;
        md->ld_KT = DIM_KT_r;  md->ld_du = DIM_du_r;  md->ld_d = DIM_d_r;    md->ld_A = DIM_A_r;
        gpuErrchk(cudaMalloc((void**)&(gv->d_AB),(md->ld_AB)*DIM_AB_c*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_P),(md->ld_P)*DIM_P_c*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_Pp),(md->ld_P)*DIM_P_c*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_p),(md->ld_p)*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_pp),(md->ld_p)*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_H),(md->ld_H)*DIM_H_c*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_g),(md->ld_g)*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_KT),(md->ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_KT_old),(md->ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_du),(md->ld_du)*NUM_TIME_STEPS*sizeof(T)));
        gv->h_d_d = (T **)malloc(NUM_ALPHA*sizeof(T*));
        gpuErrchk(cudaMalloc((void**)&(gv->d_d), NUM_ALPHA*sizeof(T*)));
        for (int i=0; i<NUM_ALPHA; i++){
            gpuErrchk(cudaMalloc((void**)&((gv->h_d_d)[i]),(md->ld_d)*NUM_TIME_STEPS*sizeof(T)));  
        } 
        gpuErrchk(cudaMemcpy(gv->d_d, gv->h_d_d, NUM_ALPHA*sizeof(T*), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc((void**)&(gv->d_dp), (md->ld_d)*NUM_TIME_STEPS*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_dT), NUM_ALPHA*sizeof(T)));
        gpuErrchk(cudaMalloc((void**)&(gv->d_dM), sizeof(T))); 
        gv->d = (T *)malloc(NUM_ALPHA*sizeof(T));
        gpuErrchk(cudaMalloc((void**)&(gv->d_Bdu), (md->ld_d)*NUM_TIME_STEPS*sizeof(T)));  
        gpuErrchk(cudaMalloc((void**)&(gv->d_ApBK), (md->ld_A)*DIM_A_c*NUM_TIME_STEPS*sizeof(T)));  

        // then for cost, alpha, rho, and errors
        gpuErrchk(cudaMalloc((void**)&(gv->d_JT), (EE_COST ? max(M_F*NUM_ALPHA,NUM_TIME_STEPS) : NUM_ALPHA)*sizeof(T)));
        gv->J = (T *)malloc(NUM_ALPHA*sizeof(T));
        gpuErrchk(cudaMalloc((void**)&(gv->d_dJexp),2*M_B*sizeof(T)));
        gv->dJexp = (T *)malloc(2*M_B*sizeof(T));
        gv->alphaIndex = (int *)malloc(sizeof(int));    *(gv->alphaIndex) = 0;
        gv->alpha = (T *)malloc(NUM_ALPHA*sizeof(T));
        gpuErrchk(cudaMalloc((void**)&(gv->d_alpha), NUM_ALPHA*sizeof(T)));
        for (int i=0; i<NUM_ALPHA; i++){(gv->alpha)[i] = pow(ALPHA_BASE,i);}
        gpuErrchk(cudaMemcpy(gv->d_alpha, gv->alpha, NUM_ALPHA*sizeof(T), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc((void**)&(gv->d_err),M_B*sizeof(int)));
        gv->err = (int *)malloc(M_B*sizeof(int));

        // put streams in order of priority
        int priority, minPriority, maxPriority;
        gpuErrchk(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));
        gv->streams = (cudaStream_t *)malloc(NUM_STREAMS*sizeof(cudaStream_t));
        for(int i=0; i<NUM_STREAMS; i++){
            priority = min(minPriority+i,maxPriority);
            gpuErrchk(cudaStreamCreateWithPriority(&((gv->streams)[i]),cudaStreamNonBlocking,priority));
        }

        // load in the Inertia and Tbody
        T *I = (T *)malloc(36*NUM_POS*sizeof(T));   initI<T>(I);
        gpuErrchk(cudaMalloc((void**)&(gv->d_I),36*NUM_POS*sizeof(T)));    
        gpuErrchk(cudaMemcpy(gv->d_I, I, 36*NUM_POS*sizeof(T), cudaMemcpyHostToDevice));   free(I);
        T *Tbody = (T *)malloc(36*NUM_POS*sizeof(T));   initT<T>(Tbody);
        gpuErrchk(cudaMalloc((void**)&(gv->d_Tbody),36*NUM_POS*sizeof(T)));
        gpuErrchk(cudaMemcpy(gv->d_Tbody, Tbody, 36*NUM_POS*sizeof(T), cudaMemcpyHostToDevice));   free(Tbody);

        // set shared memory banks to T precision
        //gpuErrchk(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
        // not using cudaFuncSetCacheConfig because documentation says may induce syncs which we don't want

        // the additionally allocate xActual / d_xActual
        gv->xActual = (T *)malloc(STATE_SIZE*sizeof(T));
        gpuErrchk(cudaMalloc((void**)&(gv->d_xActual), STATE_SIZE*sizeof(T)));
        // and the current trajVars
        allocateTrajVars<T>(tv,md);
        gpuErrchk(cudaDeviceSynchronize());
        gv->lock = new std::mutex;
    }

    template <typename T>
    __host__ __forceinline__
    void freeMemory_GPU_MPC(GPUVars<T> *gv){
        for (int i=0; i<NUM_ALPHA; i++){
            gpuErrchk(cudaFree((gv->h_d_x)[i]));  gpuErrchk(cudaFree((gv->h_d_u)[i]));  gpuErrchk(cudaFree((gv->h_d_d)[i]));
        }
        gpuErrchk(cudaFree(gv->d_x));           free(gv->h_d_x);        gpuErrchk(cudaFree(gv->d_xp));  gpuErrchk(cudaFree(gv->d_x_old));   gpuErrchk(cudaFree(gv->d_xp2)); 
        gpuErrchk(cudaFree(gv->d_u));           free(gv->h_d_u);        gpuErrchk(cudaFree(gv->d_up));  gpuErrchk(cudaFree(gv->d_u_old));
        gpuErrchk(cudaFree(gv->d_xGoal));       free(gv->xGoal);   
        gpuErrchk(cudaFree(gv->d_P));   gpuErrchk(cudaFree(gv->d_Pp));  gpuErrchk(cudaFree(gv->d_p));   gpuErrchk(cudaFree(gv->d_pp));  
        gpuErrchk(cudaFree(gv->d_AB));  gpuErrchk(cudaFree(gv->d_H));   gpuErrchk(cudaFree(gv->d_g));   gpuErrchk(cudaFree(gv->d_KT));      gpuErrchk(cudaFree(gv->d_KT_old));  gpuErrchk(cudaFree(gv->d_du));
        gpuErrchk(cudaFree(gv->d_d));           free(gv->h_d_d);        gpuErrchk(cudaFree(gv->d_dp));  gpuErrchk(cudaFree(gv->d_dT));      gpuErrchk(cudaFree(gv->d_dM));  
        free(gv->d);                    gpuErrchk(cudaFree(gv->d_Bdu)); gpuErrchk(cudaFree(gv->d_ApBK));
        gpuErrchk(cudaFree(gv->d_JT));          free(gv->J);            gpuErrchk(cudaFree(gv->d_dJexp));   free(gv->dJexp);    
        free(gv->alpha);                gpuErrchk(cudaFree(gv->d_alpha));   free(gv->alphaIndex);
        gpuErrchk(cudaFree(gv->d_err));         free(gv->err);  
        gpuErrchk(cudaFree(gv->d_I));   gpuErrchk(cudaFree(gv->d_Tbody));   free(gv->xActual);          gpuErrchk(cudaFree(gv->d_xActual));
        for(int i=0; i<NUM_STREAMS; i++){gpuErrchk(cudaStreamDestroy((gv->streams)[i]));}     free(gv->streams);
        gpuErrchk(cudaDeviceSynchronize());
        delete gv->lock;
    }

    template <typename T>
    __host__ __forceinline__
    void allocateMemory_CPU_MPC(CPUVars<T> *cv, matDimms *md, trajVars<T> *tv){
        md->ld_x = DIM_x_r;    md->ld_u = DIM_u_r;
        cv->x = (T *)malloc((md->ld_x)*NUM_TIME_STEPS*sizeof(T));
        cv->xp = (T *)malloc((md->ld_x)*NUM_TIME_STEPS*sizeof(T)); 
        cv->xp2 = (T *)malloc((md->ld_x)*NUM_TIME_STEPS*sizeof(T));
        cv->x_old = (T *)malloc((md->ld_x)*NUM_TIME_STEPS*sizeof(T));
        cv->u = (T *)malloc((md->ld_u)*NUM_TIME_STEPS*sizeof(T));
        cv->up = (T *)malloc((md->ld_u)*NUM_TIME_STEPS*sizeof(T));
        cv->u_old = (T *)malloc((md->ld_u)*NUM_TIME_STEPS*sizeof(T));
        int goalSize = EE_COST ? 6 : STATE_SIZE;    cv->xGoal = (T *)malloc(goalSize*sizeof(T));
        // allocate memory for vars with pitched malloc and thus collect the lds (for now just set ld = DIM_<>_r)
        md->ld_AB = DIM_AB_r;  md->ld_P = DIM_P_r;    md->ld_p = DIM_p_r;    md->ld_H = DIM_H_r;
        md->ld_g = DIM_g_r;    md->ld_KT = DIM_KT_r;  md->ld_du = DIM_du_r;  md->ld_d = DIM_d_r;    md->ld_A = DIM_A_r;
        cv->P = (T *)malloc((md->ld_P)*DIM_P_c*NUM_TIME_STEPS*sizeof(T));
        cv->Pp = (T *)malloc((md->ld_P)*DIM_P_c*NUM_TIME_STEPS*sizeof(T));
        cv->p = (T *)malloc((md->ld_p)*NUM_TIME_STEPS*sizeof(T));
        cv->pp = (T *)malloc((md->ld_p)*NUM_TIME_STEPS*sizeof(T));
        cv->AB = (T *)malloc((md->ld_AB)*DIM_AB_c*NUM_TIME_STEPS*sizeof(T));
        cv->H = (T *)malloc((md->ld_H)*DIM_H_c*NUM_TIME_STEPS*sizeof(T));
        cv->g = (T *)malloc((md->ld_g)*NUM_TIME_STEPS*sizeof(T));
        cv->KT = (T *)malloc((md->ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));
        cv->KT_old = (T *)malloc((md->ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));
        cv->du = (T *)malloc((md->ld_du)*NUM_TIME_STEPS*sizeof(T));
        cv->d = (T *)malloc((md->ld_d)*NUM_TIME_STEPS*sizeof(T));
        cv->dp = (T *)malloc((md->ld_d)*NUM_TIME_STEPS*sizeof(T));
        cv->Bdu = (T *)malloc((md->ld_d)*NUM_TIME_STEPS*sizeof(T));  
        cv->ApBK = (T *)malloc((md->ld_A)*DIM_A_c*NUM_TIME_STEPS*sizeof(T));  
        // could have FSIM or COST THREADS cost comps
        cv->JT = (T *)malloc(max(FSIM_THREADS,COST_THREADS)*sizeof(T));
        cv->dJexp = (T *)malloc(2*M_B*sizeof(T));
        // allocate and init alpha
        cv->alpha = (T *)malloc(NUM_ALPHA*sizeof(T));
        for (int i=0; i<NUM_ALPHA; i++){(cv->alpha)[i] = pow(ALPHA_BASE,i);}
        cv->err = (int *)malloc(M_B*sizeof(int));
        // load in the Inertia and Tbody if requested
        cv->I = (T *)malloc(36*NUM_POS*sizeof(T));   initI<T>(cv->I);
        cv->Tbody = (T *)malloc(36*NUM_POS*sizeof(T));   initT<T>(cv->Tbody);
        // the additionally allocate xActual
        cv->xActual = (T *)malloc(STATE_SIZE*sizeof(T));
        // and the current trajVars
        allocateTrajVars<T>(tv,md);
        // and the threads
        cv->threads = new std::thread[MAX_CPU_THREADS];
        cv->lock = new std::mutex;
    }
    
    template <typename T>
    __host__ __forceinline__
    void freeMemory_CPU_MPC(CPUVars<T> *cv){
        // first use the old memory free which we know works
        free(cv->x);        free(cv->xp);       free(cv->x_old);    free(cv->xp2);      
        free(cv->u);        free(cv->up);       free(cv->u_old);
        free(cv->P);        free(cv->Pp);       free(cv->p);        free(cv->pp);    
        free(cv->AB);       free(cv->H);        free(cv->g);        free(cv->KT);       free(cv->KT_old);   free(cv->du);   
        free(cv->d);        free(cv->dp);       free(cv->Bdu);      free(cv->ApBK); 
        free(cv->dJexp);    free(cv->err);      free(cv->alpha);    free(cv->JT);   
        free(cv->I);        free(cv->Tbody);    free(cv->xGoal);    free(cv->xActual);
        delete[] cv->threads;
        delete cv->lock;
    }

// 1: Variable Wrappers (structs) and memory allocators/freers //

// 2: MPC Alg Helpers //
    template <typename T>
    __global__
    void costKern_MPC(T *d_x, T *d_u, T *d_xg, int ld_x, int ld_u, T *d_Tbody, T *d_JT){
        __shared__ T s_x[STATE_SIZE];   __shared__ T s_u[NUM_POS];
        __shared__ T s_sinq[NUM_POS];   __shared__ T s_cosq[NUM_POS];
        __shared__ T s_Tb[36*NUM_POS];  __shared__ T s_T[36*NUM_POS];
        __shared__ T s_eePos[6];        d_JT[blockIdx.x] = 0;
        for (int k = blockIdx.x; k < NUM_TIME_STEPS; k += gridDim.x){
            T *xk = &d_x[k*ld_x];           T *uk = &d_u[k*ld_u];
            // load in the states and controls
            #pragma unroll
            for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < STATE_SIZE; ind += blockDim.x*blockDim.y){
                s_x[ind] = xk[ind];     if (ind < NUM_POS){s_u[ind] = uk[ind];}
            }
            __syncthreads();
            // then compute the end effector position
            compute_eePos<T>(s_eePos,s_T,s_Tb,s_sinq,s_cosq,s_x,d_Tbody);
            __syncthreads();
            // then compute the cost
            if (threadIdx.x == 0 && threadIdx.y == 0){
                d_JT[blockIdx.x] += costFunc(s_eePos,d_xg,s_x,s_u,k);
            }
        }
    }

    template <typename T, int DIM_R, int DIM_C, int DIM_N>
    __host__ __device__ __forceinline__
    void reset(T *A, int ld_A, T *val){
        int start, delta; singleLoopVals(&start,&delta);
        int size = (DIM_N-1)*DIM_C*ld_A;
        #pragma unroll
        for (int ind = start; ind < size; ind += delta){
            A[ind] = val[ind%(DIM_C*ld_A)];
        }
    }
    template <typename T, int DIM_R, int DIM_C, int DIM_N>
    __host__ __device__ __forceinline__
    void reset(T *A, int ld_A, T val){
        int start, delta; singleLoopVals(&start,&delta);
        int size = (DIM_N-1)*DIM_C*ld_A;
        #pragma unroll
        for (int ind = start; ind < size; ind += delta){
            A[ind] = val;
        }
    }
    template <typename T, int DIM_R, int DIM_C, int DIM_N>
    __global__
    void resetKern(T *A, int ld_A, T *val){
        reset<T,DIM_R,DIM_C,DIM_N>(A,ld_A,val);
    }
    template <typename T, int DIM_R, int DIM_C, int DIM_N>
    __global__
    void resetKern(T *A, int ld_A, T val){
        reset<T,DIM_R,DIM_C,DIM_N>(A,ld_A,val);
    }

    template <typename T, int DIM_R, int DIM_C, int DIM_N, bool FLAG = 0>
    __host__ __device__ __forceinline__
    void shiftAndCopy(T *A, int shiftVal, int ld_A, T *B = nullptr){
        if (shiftVal == 0){
            if (B == nullptr){return;}// early exit if we don't have to do anything
            else{memcpy(B,A,ld_A*DIM_C*NUM_TIME_STEPS*sizeof(T)); return;} // just a memcpy so do that
        } 
        // else do the whole shift and copy thing
        int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
        int ksrc = shiftVal;
        #pragma unroll
        for (int k = 0; k < DIM_N - 1; k++){
            int dstOffset = ld_A*DIM_C*k;
            T *Adst = A + dstOffset;
            T *Bdst = B + dstOffset;
            T *Asrc = A + ld_A*DIM_C*ksrc;
            #pragma unroll
            for (int c = starty; c < DIM_C; c += dy){
                #pragma unroll
                for (int r = startx; r < DIM_R; r += dx){
                    int i = c*ld_A + r;
                    T val = (FLAG && ksrc >= DIM_N - 1) ? 0.0 : Asrc[i];
                    Adst[i] = val;
                    if(B != nullptr){Bdst[i] = val;}
                }
            }
            if (ksrc < DIM_N - 1){ksrc++;}
        }
    }

    template <typename T, int DIM_R, int DIM_C, int DIM_N, bool FLAG = 0>
    __global__
    void shiftAndCopyKern(T *A, int shiftVal, int ld_A, T *B = nullptr){
        shiftAndCopy<T,DIM_R,DIM_C,DIM_N,FLAG>(A,shiftVal,ld_A,B);
    }

    template <typename T>
    __global__
    void shiftAndCopyKern2(T *dst, T *src, int size, int count, T *cpy = nullptr){
        int start, delta; singleLoopVals(&start,&delta);
        #pragma unroll
        for (int i = start; i < size*count; i += delta){
            T val = src[i%size];
            dst[i] = val; if(cpy != nullptr){cpy[i] = val;}

        }
    }

    template <typename T, int SIZE>
    __host__ __device__ __forceinline__
    void finiteCheck(T *var, T val){
        int start, delta; singleLoopVals(&start,&delta);
        #pragma unroll
        for(int i = 0; i < SIZE; i++){if(!isfinite(var[i])){var[i] = val;}}
    }

    template <typename T, int SIZE>
    __host__ __device__ __forceinline__
    void finiteCheck(T *var, T *vals){
        int start, delta; singleLoopVals(&start,&delta);
        #pragma unroll
        for(int i = 0; i < SIZE; i++){if(!isfinite(var[i])){var[i] = vals[i];}}
    }

    template <typename T>
    __host__ __device__ __forceinline__
    T clip(T var, T min, T max){
        return var < min ? min : var > max ? max : var;
    }

    template <typename T, int SIZE>
    __host__ __device__ __forceinline__
    void clipVals(T *var, T min, T max){
        int start, delta; singleLoopVals(&start,&delta);
        #pragma unroll
        for(int i = 0; i < SIZE; i++){clip<T>(var[i]);}
    }

    template <typename T>
    __host__ __device__ __forceinline__
    void computeControlSimple(T *u, T *x, T *xdes, T *KT, T *s_dx, int ld_KT){
        int start, delta; singleLoopVals(&start,&delta);
        #pragma unroll
        for (int ind = start; ind < STATE_SIZE; ind += delta){s_dx[ind] = x[ind]-xdes[ind];}
        hd__syncthreads();
        // compute the new control: u = u - K(x-xdes) -- but note we have KT not K
        #pragma unroll
        for (int r = start; r < CONTROL_SIZE; r += delta){
            // get the Kdx for this row
            T Kdx = 0;
            #pragma unroll
            for (int c = 0; c < STATE_SIZE; c++){Kdx += KT[c + r*ld_KT]*s_dx[c];}
            // and then get this control with it
            u[r] -= Kdx; // clip<T>(Kdx,0.5*u[r],1.5*u[r]);
        }
    }

    // for first block of traj
    template <typename T, int N_ROLLOUT>
    __host__ __device__ __forceinline__
    void rolloutMPC(T *x0, T *u0, T *KT0, T *d0, T *xprev, T *xActual, T *s_dx, T *s_qdd, T dt, int shiftAmount,
                    int ld_x, int ld_u, int ld_d, int ld_KT, T *d_I = nullptr, T *d_Tbody = nullptr){
        // now we only need to roll out the first block from xActual
        int start, delta; singleLoopVals(&start,&delta);
        for (int ind = start; ind < STATE_SIZE; ind += delta){x0[ind] = xActual[ind];}
        T *xk = &x0[0];    T *xkp1 = &x0[ld_x];    T *xpk = &xprev[0];     T *uk = &u0[0];     T *KTk = &KT0[0];   T *dk = &d0[0];
        // T *xk = &x0[0];                     T *xkp1 = &xk[ld_x];            T *dk = &d0[0];
        // T *xpk = &xprev[ld_x*shiftAmount];  T *uk = &u0[ld_u*shiftAmount];  T *KTk = &KT0[ld_KT*DIM_KT_c*shiftAmount];      
        #pragma unroll
        // simulate forward the first block
        for(int k = 0; k < N_ROLLOUT-1; k++){
            // load in the x and u and compute controls
            // computeControlSimple<T>(uk,xk,xpk,KTk,s_dx,ld_KT);
            // hd__syncthreads();
            // clipVals<T,CONTROL_SIZE>(uk,-10,10);
            // hd__syncthreads();
            // finiteCheck<T,CONTROL_SIZE>(uk,0.01);
            // hd__syncthreads();
            // then use this control to compute the new state
            T *s_xkp1 = s_dx; // re-use this shared mem as we are done with it for this loop
            _integrator<T>(s_xkp1,xk,uk,s_qdd,d_I,d_Tbody,dt,nullptr);
            hd__syncthreads();
            // finiteCheck<T,STATE_SIZE>(s_xkp1,xk);
            // hd__syncthreads();
            // then write to global memory unless "final" state where we just use for defect on boundary
            #pragma unroll
            for (int ind = start; ind < STATE_SIZE; ind += delta){
                //if(abs((s_xkp1[ind] - xk[ind])/xk[ind]) > 0.5){s_xkp1[ind] = xk[ind];}
                if (k < N_ROLLOUT-1 || k == NUM_TIME_STEPS - 2){xkp1[ind] = s_xkp1[ind];}
                else {dk[ind] = s_xkp1[ind] - xkp1[ind];}
            }
            // update the offsets for the next pass or break to zoh
            xk = xkp1;      xkp1 += ld_x;   dk += ld_d;     
            uk += ld_u;     xpk += ld_x;    KTk += ld_KT*DIM_KT_c;
            // if (k + shiftAmount < NUM_TIME_STEPS-2){uk += ld_u; xpk += ld_x; KTk += ld_KT*DIM_KT_c;}
            hd__syncthreads();
        }
    }

    template <typename T, int N_ROLLOUT>
    __global__
    void rolloutMPCKern(T *x0, T *u0, T *KT0, T *d0, T *xprev, T *xActual, T dt, int shiftAmount,
                        int ld_x, int ld_u, int ld_d, int ld_KT, T *d_I = nullptr, T *d_Tbody = nullptr){
        __shared__ T s_dx[STATE_SIZE];    __shared__ T s_qdd[NUM_POS];
        rolloutMPC<T,N_ROLLOUT>(x0,u0,KT0,d0,xprev,xActual,s_dx,s_qdd,dt,shiftAmount,ld_x,ld_u,ld_d,ld_KT,d_I,d_Tbody);
    }

    // for shiftAmount to end of traj
    template <typename T>
    __host__ __device__ __forceinline__
    void rolloutMPC2(T *x0, T *u0, T *KT0, T *xprev, T *s_dx, T *s_qdd, int shiftAmount, T dt, 
                     int ld_x, int ld_u, int ld_KT,T *d_I = nullptr, T *d_Tbody = nullptr){
        if (shiftAmount == 0){return;} // early exit if we don't have to do anything
         // now we rollout the N-shiftAmount
        int start, delta; singleLoopVals(&start,&delta);    
        T *xk = &x0[(NUM_TIME_STEPS-1-shiftAmount)*ld_x];    T *xkp1 = xk + ld_x;    T *xpk = &xprev[(NUM_TIME_STEPS-1-shiftAmount)*ld_x];
        T *uk = &u0[(NUM_TIME_STEPS-1-shiftAmount)*ld_u];    T *KTk = &KT0[(NUM_TIME_STEPS-1-shiftAmount)*ld_KT*DIM_KT_c];
        // simulate forward the first block
        for(int k = 0; k < shiftAmount; k++){
            // load in the x and u and compute controls
            computeControlSimple<T>(uk,xk,xpk,KTk,s_dx,ld_KT);
            hd__syncthreads();
            // finiteCheck<T,CONTROL_SIZE>(uk,0.001);
            // hd__syncthreads();
            // then use this control to compute the new state
            _integrator<T>(xkp1,xk,uk,s_qdd,d_I,d_Tbody,dt,nullptr);
            hd__syncthreads();
            // finiteCheck<T,STATE_SIZE>(xkp1,xk);
            // update the offsets for the next pass or break to zoh
            xk = xkp1;   xkp1 += ld_x;  uk += ld_u;     xpk += ld_x;    KTk += ld_KT*DIM_KT_c;
            hd__syncthreads();
        }
    }

    template <typename T>
    __global__
    void rolloutMPCKern2(T *x0, T *u0, T *KT0, T *xprev, int shiftAmount, T dt, 
                        int ld_x, int ld_u, int ld_KT, T *d_I = nullptr, T *d_Tbody = nullptr){
        __shared__ T s_dx[STATE_SIZE];    __shared__ T s_qdd[NUM_POS];
        rolloutMPC2<T>(x0,u0,KT0,xprev,s_dx,s_qdd,shiftAmount,dt,ld_x,ld_u,ld_KT,d_I,d_Tbody);
    }

    template <typename T>
    __host__ __forceinline__
    void loadVarsGPU_MPC(T *x_old, T *u_old, T *KT_old, T **h_d_x, T *d_xp, T **h_d_u, T *d_up, T **h_d_d, T *d_KT, T *d_xGoal, T *xGoal, T *d_xActual, T *xActual,
                         T *d_P, T *d_Pp, T *d_p, T *d_pp, T *d_du, T *d_AB, T *d_H, T *d_dT, int *d_err, int *alphaIndex, T dt, cudaStream_t *streams, dim3 dynDimms,
                         int shiftAmount, int ld_x, int ld_u, int ld_d, int ld_KT, int ld_P, int ld_p, int ld_du, int ld_AB,
                         T *d_I = nullptr, T *d_Tbody = nullptr){
        // note since we assume that we will reach the desired states to start later blocks the
        // control will remain the same as will the states outside of the first block
        // therefore we are simply copying things over in later blocks so first execute a shift copy zoh for everything
        // note in <x/u/d>[*alphaIndex], P, p, KT are all of the ones from the last run
        int goalSize;   if (!EE_COST){goalSize = STATE_SIZE;}   else{goalSize = 6;}
        gpuErrchk(cudaMemcpyAsync(d_xGoal, xGoal, goalSize*sizeof(T), cudaMemcpyHostToDevice, streams[0]));
        gpuErrchk(cudaMemcpyAsync(d_xActual, xActual, STATE_SIZE*sizeof(T), cudaMemcpyHostToDevice, streams[1]));
        shiftAndCopyKern<T,DIM_x_r,DIM_x_c,NUM_TIME_STEPS><<<1,DIM_x_r,0,streams[2]>>>(h_d_x[*alphaIndex],shiftAmount,ld_x,d_xp);
        shiftAndCopyKern<T,DIM_u_r,DIM_u_c,(NUM_TIME_STEPS-1),1><<<1,DIM_u_r,0,streams[3]>>>(h_d_u[*alphaIndex],shiftAmount,ld_u,d_up);
        shiftAndCopyKern<T,DIM_d_r,DIM_d_c,NUM_TIME_STEPS><<<1,DIM_d_r,0,streams[4]>>>(h_d_d[*alphaIndex],shiftAmount,ld_d);
        shiftAndCopyKern<T,DIM_KT_r,DIM_KT_c,NUM_TIME_STEPS-1,1><<<1,DIM_KT_r*DIM_KT_c,0,streams[5]>>>(d_KT,shiftAmount,ld_KT);
        shiftAndCopyKern<T,DIM_P_r,DIM_P_c,NUM_TIME_STEPS><<<1,DIM_P_r*DIM_P_c,0,streams[6]>>>(d_P,shiftAmount,ld_P);
        shiftAndCopyKern<T,DIM_p_r,DIM_p_c,NUM_TIME_STEPS><<<1,DIM_p_r,0,streams[7]>>>(d_p,shiftAmount,ld_p);
        shiftAndCopyKern<T,DIM_P_r,DIM_P_c,NUM_TIME_STEPS><<<1,DIM_P_r*DIM_P_c,0,streams[6]>>>(d_Pp,shiftAmount,ld_P);
        shiftAndCopyKern<T,DIM_p_r,DIM_p_c,NUM_TIME_STEPS><<<1,DIM_p_r,0,streams[7]>>>(d_pp,shiftAmount,ld_p);     
        gpuErrchk(cudaMemsetAsync(d_du,0,ld_du*NUM_TIME_STEPS*sizeof(T),streams[8]));
        gpuErrchk(cudaMemsetAsync(d_err,0,M_B*sizeof(int),streams[9]));
        gpuErrchk(cudaMemsetAsync(d_dT,0,NUM_ALPHA*sizeof(T),streams[10]));
        T *ABN = d_AB + ld_AB*DIM_AB_c*(NUM_TIME_STEPS-2);
        gpuErrchk(cudaMemsetAsync(ABN,0,ld_AB*DIM_AB_c*sizeof(T),streams[11]));
        // sync on the ones we need for the rollout
        gpuErrchk(cudaStreamSynchronize(streams[0]));   gpuErrchk(cudaStreamSynchronize(streams[1]));
        gpuErrchk(cudaStreamSynchronize(streams[2]));   gpuErrchk(cudaStreamSynchronize(streams[3]));
        gpuErrchk(cudaStreamSynchronize(streams[4]));   gpuErrchk(cudaStreamSynchronize(streams[5]));
        // do the rollout
        #define FULL_ROLLOUT 1
        #if FULL_ROLLOUT
            rolloutMPCKern<T,NUM_TIME_STEPS><<<1,dynDimms,0,streams[0]>>>(h_d_x[*alphaIndex],h_d_u[*alphaIndex],d_KT,h_d_d[*alphaIndex],d_xp,d_xActual,dt,shiftAmount,ld_x,ld_u,ld_d,ld_KT,d_I,d_Tbody);
        #else    
            rolloutMPCKern<T,N_F><<<1,dynDimms,0,streams[0]>>>(h_d_x[*alphaIndex],h_d_u[*alphaIndex],d_KT,h_d_d[*alphaIndex],d_xp,d_xActual,dt,shiftAmount,ld_x,ld_u,ld_d,ld_KT,d_I,d_Tbody);
            if (M_F > 1){rolloutMPCKern2<T><<<1,dynDimms,0,streams[1]>>>(h_d_x[*alphaIndex],h_d_u[*alphaIndex],d_KT,d_xp,shiftAmount,dt,ld_x,ld_u,ld_KT,d_I,d_Tbody);}
        #endif
        // save the shifted into tvars in case all iters fail
        gpuErrchk(cudaMemcpyAsync(x_old, d_xp, ld_x*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost, streams[1]));
        gpuErrchk(cudaMemcpyAsync(u_old, d_up, ld_u*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost, streams[2]));
        gpuErrchk(cudaMemcpyAsync(KT_old, d_KT, ld_KT*DIM_KT_c*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost, streams[3]));
        gpuErrchk(cudaDeviceSynchronize()); // sync and exit
    }

    template <typename T>
    __host__ __forceinline__
    void loadVarsCPU_MPC(T *x_old, T *u_old, T *KT_old, T *x, T *xp, T *u, T *up, T *d, T *KT, T *xGoal, T *xActual, T *P, T *Pp, 
                         T *p, T *pp, T *du, T *AB, int *err, int *alphaIndex, T dt, std::thread *threads, int shiftAmount, 
                         int ld_x, int ld_u, int ld_d, int ld_KT, int ld_P, int ld_p, int ld_du, int ld_AB,
                         T *I = nullptr, T *Tbody = nullptr){
        // note since we assume that we will reach the desired states to start later blocks the
        // control will remain the same as will the states outside of the first block
        // therefore we are simply copying things over in later blocks so first execute a shift copy zoh for everything
        // note in <x/u/d>, P, p, KT are all of the ones from the last run
        threads[0] = std::thread(&shiftAndCopy<T,DIM_x_r,DIM_x_c,NUM_TIME_STEPS>, std::ref(x), shiftAmount, ld_x, std::ref(xp));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 0);}
        threads[1] = std::thread(&shiftAndCopy<T,DIM_u_r,DIM_u_c,NUM_TIME_STEPS-1>, std::ref(u), shiftAmount, ld_u, std::ref(up));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 1);}
        threads[2] = std::thread(&shiftAndCopy<T,DIM_d_r,DIM_d_c,NUM_TIME_STEPS>, std::ref(d), shiftAmount, ld_d, nullptr);
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 2);}
        threads[3] = std::thread(&shiftAndCopy<T,DIM_KT_r,DIM_KT_c,NUM_TIME_STEPS-1>, std::ref(KT), shiftAmount, ld_KT, nullptr);
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 3);}
        
        // as always clear the things we can clear and shiftcopy P,p
        threads[4] = std::thread(&shiftAndCopy<T,DIM_P_r,DIM_P_c,NUM_TIME_STEPS>, std::ref(P), shiftAmount, ld_P, nullptr);
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 4);}
        threads[5] = std::thread(&shiftAndCopy<T,DIM_p_r,DIM_p_c,NUM_TIME_STEPS>, std::ref(p), shiftAmount, ld_p, nullptr);
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 5);}
        threads[6] = std::thread(&shiftAndCopy<T,DIM_P_r,DIM_P_c,NUM_TIME_STEPS>, std::ref(Pp), shiftAmount, ld_P, nullptr);
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 6);}
        threads[7] = std::thread(&shiftAndCopy<T,DIM_p_r,DIM_p_c,NUM_TIME_STEPS>, std::ref(pp), shiftAmount, ld_p, nullptr);
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 7);}
        threads[8] = std::thread(memset, std::ref(du), 0, ld_du*NUM_TIME_STEPS*sizeof(T));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 8);}
        threads[9] = std::thread(memset, std::ref(err), 0, BP_THREADS*sizeof(int));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 9);}
        T *ABN = AB + ld_AB*DIM_AB_c*(NUM_TIME_STEPS-3);
        threads[10] = std::thread(memset, std::ref(ABN), 0, ld_AB*DIM_AB_c*sizeof(T));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 10);}

        // sync on x,u,d,KT and then rollout first block
        threads[0].join();  threads[1].join();  threads[2].join();  threads[3].join();
        T s_dx[STATE_SIZE];  T s_qdd[NUM_POS];
        #define FULL_ROLLOUT 1
        #if FULL_ROLLOUT
            threads[0] = std::thread(&rolloutMPC<T,NUM_TIME_STEPS>, x, u, KT, d, xp, xActual, s_dx, s_qdd, shiftAmount, dt, ld_x, ld_u, ld_d, ld_KT, I, Tbody);
            if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 0);}
        #else
            T s_dx2[STATE_SIZE];  T s_qdd2[NUM_POS];
            threads[0] = std::thread(&rolloutMPC<T,N_F>, x, u, KT, d, xp, xActual, s_dx, s_qdd, shiftAmount, dt, ld_x, ld_u, ld_d, ld_KT, I, Tbody);
            if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 0);}
            if (M_F > 1){
                threads[1] = std::thread(&rolloutMPC2<T>, x, u, KT, xp, s_dx2, s_qdd2, shiftAmount, dt, ld_x, ld_u, ld_KT, I, Tbody);
                if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 1);}
            }
        #endif
        // save the shifted into tvars in case all iters fail
        threads[2] = std::thread(memcpy, std::ref(x_old), std::ref(xp), ld_x*NUM_TIME_STEPS*sizeof(T));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 2);}
        threads[3] = std::thread(memcpy, std::ref(u_old), std::ref(up), ld_u*NUM_TIME_STEPS*sizeof(T));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 3);}
        threads[11] = std::thread(memcpy, std::ref(KT_old), std::ref(KT), ld_KT*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 11);}
        // join threads
        threads[0].join(); if(!FULL_ROLLOUT && M_F > 1){threads[1].join();} threads[2].join(); threads[3].join(); threads[11].join();
        threads[4].join(); threads[5].join(); threads[6].join(); threads[7].join(); threads[8].join(); threads[9].join(); threads[10].join();
    }

    // store vars to CPU and compute total max defect if requested for debug / printing
    template <typename T>
    __host__ __forceinline__
    void storeVarsGPU_MPC(GPUVars<T> *gv, trajVars<T> *tv, matDimms *md, int64_t tActual_sys, int64_t tActual_plant, T finalJ, int defectFlag){
        (tv->lock)->lock();             tv->t0_sys = tActual_sys;         tv->t0_plant = tActual_plant;
        tv->last_successful_solve++;    // increment failure counter by 1 since now last solve at 1 iter ago minimum
        // if it was successful at all then copy out else reset to previous successful solve
        if (tv->last_successful_solve == 1){
            if (defectFlag){defectKern<<<NUM_ALPHA,NUM_TIME_STEPS,0,(gv->streams)[0]>>>(gv->d_d,gv->d_dT,md->ld_d); gpuErrchk(cudaPeekAtLastError());}
            gpuErrchk(cudaMemcpyAsync(tv->x, gv->h_d_x[*(gv->alphaIndex)], (md->ld_x)*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost, (gv->streams)[1]));
            gpuErrchk(cudaMemcpyAsync(tv->u, gv->h_d_u[*(gv->alphaIndex)], (md->ld_u)*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost, (gv->streams)[2]));
            gpuErrchk(cudaMemcpyAsync(tv->KT, gv->d_KT, (md->ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToHost, (gv->streams)[3]));
            if (defectFlag){gpuErrchk(cudaStreamSynchronize((gv->streams)[0]));}
            if (defectFlag){gpuErrchk(cudaMemcpyAsync(gv->d, gv->d_dT, NUM_ALPHA*sizeof(T), cudaMemcpyDeviceToHost, (gv->streams)[0]));}
        }
        else{
            gpuErrchk(cudaMemcpyAsync(gv->h_d_x[*(gv->alphaIndex)], gv->d_x_old, (md->ld_x)*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToDevice, (gv->streams)[1]));
            gpuErrchk(cudaMemcpyAsync(gv->h_d_u[*(gv->alphaIndex)], gv->d_u_old, (md->ld_u)*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToDevice, (gv->streams)[2]));
            gpuErrchk(cudaMemcpyAsync(gv->d_KT, gv->d_KT_old, (md->ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T), cudaMemcpyDeviceToDevice, (gv->streams)[3]));
        }
        gpuErrchk(cudaDeviceSynchronize()); // sync to be done
        (tv->lock)->unlock();
    }

    template <typename T>
    __host__ __forceinline__
    void storeVarsCPU_MPC(CPUVars<T> *cv, trajVars<T> *tv, matDimms *md, int64_t tActual_sys, int64_t tActual_plant, int defectFlag, T *maxd){
        (tv->lock)->lock();             tv->t0_sys = tActual_sys;         tv->t0_plant = tActual_plant;
        tv->last_successful_solve++;    // increment failure counter by 1 since now last solve at 1 iter ago minimum
        if (tv->last_successful_solve == 1){
            if (defectFlag){*maxd = defectComp(cv->d,md->ld_d);}
            (cv->threads)[0] = std::thread(memcpy, std::ref(tv->x), std::ref(cv->x), (md->ld_x)*NUM_TIME_STEPS*sizeof(T));
            if(FORCE_CORE_SWITCHES){setCPUForThread(cv->threads, 0);}
            (cv->threads)[1] = std::thread(memcpy, std::ref(tv->u), std::ref(cv->u), (md->ld_u)*NUM_TIME_STEPS*sizeof(T));
            if(FORCE_CORE_SWITCHES){setCPUForThread(cv->threads, 1);}
            (cv->threads)[2] = std::thread(memcpy, std::ref(tv->KT), std::ref(cv->KT), (md->ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));
            if(FORCE_CORE_SWITCHES){setCPUForThread(cv->threads, 2);}
        }
        else{
            if (defectFlag){*maxd = defectComp(cv->d,md->ld_d);}
            (cv->threads)[0] = std::thread(memcpy, std::ref(cv->x), std::ref(cv->x_old), (md->ld_x)*NUM_TIME_STEPS*sizeof(T));
            if(FORCE_CORE_SWITCHES){setCPUForThread(cv->threads, 0);}
            (cv->threads)[1] = std::thread(memcpy, std::ref(cv->u), std::ref(cv->u_old), (md->ld_u)*NUM_TIME_STEPS*sizeof(T));
            if(FORCE_CORE_SWITCHES){setCPUForThread(cv->threads, 1);}
            (cv->threads)[2] = std::thread(memcpy, std::ref(cv->KT), std::ref(cv->KT_old), (md->ld_KT)*DIM_KT_c*NUM_TIME_STEPS*sizeof(T));
            if(FORCE_CORE_SWITCHES){setCPUForThread(cv->threads, 2);}
        }
        (cv->threads)[0].join();
        (cv->threads)[1].join();
        (cv->threads)[2].join();
        (tv->lock)->unlock();
    }
                

    template <typename T>
    __host__ __forceinline__
    int getHardwareControls(double *q_out, double *u_out, T *x, T *u, T *KT, double t0, 
                            const double *qActual, const double *qdActual, double tActual, int ld_x, int ld_u, int ld_KT){
        int start, delta; singleLoopVals(&start,&delta);    T dx[STATE_SIZE];
        // compute index in current traj we want to use (both round down for zoh and fraction for foh)
        double dt = get_time_steps_us_d(t0,tActual); int ind_rd = static_cast<int>(dt); double fraction = dt - static_cast<double>(ind_rd);
        // printf("Sending ind_rd[%d] w/ fraction[%f] for t0[%f] vs tActual[%f]\n",ind_rd,fraction,t0,tActual);
        // see if beyond bounds and fail
        if (ind_rd >= NUM_TIME_STEPS-1){return 1;}
        // u,KT do zoh so take rd
        T *uk = &u[ind_rd*ld_u];             T *KTk = &KT[ind_rd*ld_KT*DIM_KT_c];
        // foh for xk
        T *xk_u = &x[(ind_rd+1)*ld_x];       T *xk_d = &x[ind_rd*ld_x];
        // then compute the state delta
        #pragma unroll
        for (int ind = start; ind < STATE_SIZE; ind += delta){
            T val = (static_cast<T>(1.0-fraction)*xk_d[ind] + static_cast<T>(fraction)*xk_u[ind]);
            dx[ind] = (ind < NUM_POS ? qActual[ind] : qdActual[ind-NUM_POS]) - val;
            // printf("dx[%f] for val[%f] vs prev[%f]\n",dx[ind],(ind < NUM_POS ? qActual[ind] : qdActual[ind-NUM_POS]),val);
            if(ind < NUM_POS){q_out[ind] = val; if(!isfinite(q_out[ind])){q_out[ind] = 0;}}    //else{qd_out[ind-NUM_POS] = val;}
        }
        hd__syncthreads();
        // and formulate the control from that delta (and save out KT -> K)
        #pragma unroll
        for (int r = start; r < CONTROL_SIZE; r += delta){
            T val = uk[r];
            #pragma unroll
            for (int c = 0; c < STATE_SIZE; c++){val -= KTk[c + r*ld_KT]*dx[c];}
            u_out[r] = static_cast<double>(val);
            if(!isfinite(u_out[r])){u_out[r] = 0;}
            // printf("u_out[%f] vs. u_in[%f] for ind[%d]\n",u_out[r],uk[r],r);
        }
        return 0;
    }
// 2: MPC Alg Helpers //

// 3: MPC Main Algorithm Wrappers //
    template <typename T>
    __host__ __forceinline__
    void runiLQR_MPC_GPU(trajVars<T> *tv, GPUVars<T> *gv, matDimms *md, algTrace<T> *data, 
                         int64_t tActual_sys, int64_t tActual_plant, int ignoreFirstDefectFlag, 
                         int max_iter = MAX_ITER, double time_budget = MAX_SOLVER_TIME){
        // INITIALIZE THE ALGORITHM //
            struct timeval start, end, start2, end2;    gettimeofday(&start,NULL);  gettimeofday(&start2,NULL);
            T prevJ, dJ, z;     int iter = 1;   T rho = RHO_INIT;   T drho = 1.0;   //*(gv->alphaIndex) = 0;
            int shiftAmount = get_time_steps_us_f(tv->t0_plant,tActual_plant);   T Jout[MAX_ITER+1];   int alphaOut[MAX_ITER+1];
            // printf("current t0_plant entering MPC: %ld\n",tv->t0_plant);

            // define kernel dimms
            dim3 ADimms(DIM_A_r,1);//DIM_A_r,DIM_A_c);
            dim3 bpDimms(8,7);              dim3 dynDimms(8,7);//(36,7);
            dim3 FPBlocks(M_F,NUM_ALPHA);   dim3 gradBlocks(DIM_AB_c,NUM_TIME_STEPS-1);     dim3 intDimms(NUM_TIME_STEPS-1,1);
            if(USE_FINITE_DIFF){intDimms.y = STATE_SIZE + CONTROL_SIZE;}

            // load and clear variables as requested and init the alg
            loadVarsGPU_MPC<T>(gv->d_x_old,gv->d_u_old,gv->d_KT_old,gv->h_d_x, gv->d_xp, gv->h_d_u, gv->d_up, gv->h_d_d, gv->d_KT, gv->d_xGoal, gv->xGoal, gv->d_xActual, gv->xActual,
                               gv->d_P, gv->d_Pp, gv->d_p, gv->d_pp, gv->d_du, gv->d_AB, gv->d_H, gv->d_dT, gv->d_err, gv->alphaIndex,(T)TIME_STEP, gv->streams, dynDimms,
                               shiftAmount, md->ld_x, md->ld_u, md->ld_d, md->ld_KT, md->ld_P, md->ld_p, md->ld_du, md->ld_AB,
                               gv->d_I, gv->d_Tbody);
            gettimeofday(&end2,NULL);
            (data->initTime).push_back(time_delta_ms(start2,end2));
            // do initial "next iteration setup"
            gettimeofday(&start2,NULL);
            initAlgGPU<T>(gv->d_x,gv->h_d_x,gv->d_xp,gv->d_xp2,gv->d_u,gv->h_d_u,gv->d_up,gv->d_d,gv->h_d_d,gv->d_dp,gv->d_dT,gv->d_AB,gv->d_H,
                          gv->d_g,gv->d_KT,gv->d_du,gv->d_JT,&prevJ,gv->d_xGoal,gv->d_alpha,gv->alphaIndex,alphaOut,Jout,gv->streams,dynDimms,
                          intDimms,0,md->ld_x,md->ld_u,md->ld_d,md->ld_AB,md->ld_H,md->ld_g,md->ld_KT,md->ld_du,gv->d_I,gv->d_Tbody);
            gettimeofday(&end2,NULL);
            (data->nisTime).push_back(time_delta_ms(start2,end2));
        // INITIALIZE THE ALGORITHM //

        // debug print
        if (DEBUG_SWITCH){
            gpuErrchk(cudaMemcpy(&prevJ, &((gv->d_JT)[*(gv->alphaIndex)]), sizeof(T), cudaMemcpyDeviceToHost));
            T xPrint[STATE_SIZE];
            gpuErrchk(cudaMemcpy(xPrint, ((gv->h_d_x)[*(gv->alphaIndex)]) + (md->ld_x)*(NUM_TIME_STEPS-1), STATE_SIZE*sizeof(T), cudaMemcpyDeviceToHost));
            printf("Iter[0] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] Rho[%f]\n",xPrint[0],xPrint[1],prevJ,*(gv->alphaIndex),rho);
        }

        // now start computing iterates
        while(1){
            gettimeofday(&end,NULL); if(time_delta_ms(start,end) > time_budget){break;};
            // BACKWARD PASS //
                gettimeofday(&start2,NULL);
                // run full backward pass if it fails we have maxed our regularizer and need to exit
                if (backwardPassGPU<T>(gv->d_AB,gv->d_P,gv->d_p,gv->d_Pp,gv->d_pp,gv->d_H,gv->d_g,gv->d_KT,gv->d_du,
                                       (gv->h_d_d)[*(gv->alphaIndex)],gv->d_ApBK,gv->d_Bdu,(gv->h_d_x)[*(gv->alphaIndex)],
                                       gv->d_xp2,gv->d_dJexp,gv->err,gv->d_err,&rho,&drho,gv->streams,bpDimms,
                                       md->ld_AB,md->ld_P,md->ld_p,md->ld_H,md->ld_g,md->ld_KT,md->ld_du,md->ld_A,md->ld_d,md->ld_x)){
                    if (DEBUG_SWITCH){printf("Exiting for maxRho\n");}
                    break;
                }
                // make sure everything that was supposed to finish did by now (incuding previous NIS stuff)
                gpuErrchk(cudaDeviceSynchronize());
                gettimeofday(&end2,NULL);
                (data->bpTime).push_back(time_delta_ms(start2,end2));
            // BACKWARD PASS //

            gettimeofday(&end,NULL); if(time_delta_ms(start,end) > time_budget){break;};
            // FORWARD PASS //
                // FORWARD SWEEP //
                    gettimeofday(&start2,NULL);
                    // Sweep forward with all alpha in parallel if applicable
                    if (M_F > 1){
                        forwardSweepKern<T><<<NUM_ALPHA,ADimms,0,(gv->streams)[0]>>>(gv->d_x,gv->d_ApBK,gv->d_Bdu,gv->h_d_d[*(gv->alphaIndex)],
                                                                               gv->d_xp,gv->d_alpha,md->ld_x,md->ld_d,md->ld_A);
                        gpuErrchk(cudaPeekAtLastError());   gpuErrchk(cudaDeviceSynchronize());
                    }
                    gettimeofday(&end2,NULL);
                    (data->sweepTime).push_back(time_delta_ms(start2,end2));
                // FORWARD SWEEP //

                // FORWARD SIM //
                    gettimeofday(&start2,NULL);
                    // Simulate forward with all alpha in parallel with MS, compute costs and line search
                    forwardSimGPU<T>(gv->d_x,gv->d_xp,gv->d_xp2,gv->d_u,gv->d_KT,gv->d_du,gv->alpha,gv->d_alpha,gv->d,gv->d_d,gv->d_dT,
                                     gv->dJexp,gv->d_dJexp,gv->J,gv->d_JT,gv->d_xGoal,&dJ,&z,prevJ,gv->streams,dynDimms,FPBlocks,
                                     gv->alphaIndex,&ignoreFirstDefectFlag,md->ld_x,md->ld_u,md->ld_KT,md->ld_du,md->ld_d,gv->d_I,gv->d_Tbody);
                    gettimeofday(&end2,NULL);
                    (data->simTime).push_back(time_delta_ms(start2,end2));
                // FORWARD SIM //
            // FORWARD PASS //

            // NEXT ITERATION SETUP //
                gettimeofday(&start2,NULL);
                // process accept or reject of traj and test for exit
                bool exitFlag = acceptRejectTrajGPU<T>(gv->h_d_x,gv->d_xp,gv->h_d_u,gv->d_up,gv->h_d_d,gv->d_dp,gv->J,
                                                       &prevJ,&dJ,&rho,&drho,gv->alphaIndex,alphaOut,Jout,&iter,gv->streams,
                                                       md->ld_x,md->ld_u,md->ld_d,max_iter);
                if(alphaOut[iter - !exitFlag] > 0){if (DEBUG_SWITCH){printf("successful iter w/ alpha[%d]\n",alphaOut[iter - !exitFlag]);}tv->last_successful_solve = 0;} // note that we were able to take a step
                if (exitFlag){
                    gettimeofday(&end2,NULL);
                    (data->nisTime).push_back(time_delta_ms(start2,end2));
                    break;
                }

                // if we have gotten here then prep for next pass
                gettimeofday(&end,NULL); if(time_delta_ms(start,end) > time_budget){break;};
                nextIterationSetupGPU<T>(gv->d_x,gv->h_d_x,gv->d_xp,gv->d_u,gv->h_d_u,gv->d_up,gv->d_d,gv->h_d_d,gv->d_dp,gv->d_AB,gv->d_H,
                                         gv->d_g,gv->d_P,gv->d_p,gv->d_Pp,gv->d_pp,gv->d_xGoal,gv->alphaIndex,gv->streams,dynDimms,intDimms,
                                         md->ld_x,md->ld_u,md->ld_d,md->ld_AB,md->ld_H,md->ld_g,md->ld_P,md->ld_p,gv->d_I,gv->d_Tbody);                
                gettimeofday(&end2,NULL);
                (data->nisTime).push_back(time_delta_ms(start2,end2));
            // NEXT ITERATION SETUP //
            // debug print
            if (DEBUG_SWITCH){
                T xPrint[STATE_SIZE];
                gpuErrchk(cudaMemcpy(xPrint, ((gv->h_d_x)[*(gv->alphaIndex)]) + (md->ld_x)*(NUM_TIME_STEPS-1), STATE_SIZE*sizeof(T), cudaMemcpyDeviceToHost));
                printf("Iter[%d] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] rho[%f] dJ[%f] z[%f] max_d[%f]\n",
                        iter-1,xPrint[0],xPrint[1],prevJ,*(gv->alphaIndex),rho,dJ,z,gv->d[*(gv->alphaIndex)]);
            }
        }

        // EXIT Handling
            // on exit make sure everything finishes
            gpuErrchk(cudaDeviceSynchronize());
            if (DEBUG_SWITCH){
                T xPrint[STATE_SIZE];
                gpuErrchk(cudaMemcpy(xPrint, ((gv->h_d_x)[*(gv->alphaIndex)]) + (md->ld_x)*(NUM_TIME_STEPS-1), STATE_SIZE*sizeof(T), cudaMemcpyDeviceToHost));
                printf("Exit with Iter[%d] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] rho[%f] dJ[%f] z[%f] max_d[%f]\n",
                    iter,xPrint[0],xPrint[1],prevJ,*(gv->alphaIndex),rho,dJ,z,gv->d[*(gv->alphaIndex)]);
            }

            // Bring back the final state and control (and save trace)
            gettimeofday(&start2,NULL);
            storeVarsGPU_MPC<T>(gv,tv,md,tActual_sys,tActual_plant,prevJ,0);
            for (int i=0; i <= iter; i++){(data->alpha).push_back(alphaOut[i]);   (data->J).push_back(Jout[i]);}
            gettimeofday(&end2,NULL);
            gettimeofday(&end,NULL);
            (data->initTime).back() += time_delta_ms(start2,end2);
            (data->tTime).push_back(time_delta_ms(start,end));
    }
        
    template <typename T>
    __host__ __forceinline__
    void runiLQR_MPC_CPU(trajVars<T> *tv, CPUVars<T> *cv, matDimms *md, algTrace<T> *data, 
                         int64_t tActual_sys, int64_t tActual_plant, int ignoreFirstDefectFlag, 
                         double time_budget = MAX_SOLVER_TIME, int max_iter = MAX_ITER){
        // INITIALIZE THE ALGORITHM //
            struct timeval start, end, start2, end2;    gettimeofday(&start,NULL);  gettimeofday(&start2,NULL);
            T prevJ, dJ, J, z;      T maxd = 0; int iter = 1;
            T rho = RHO_INIT;       T drho = 1.0;   int alphaIndex = 0;
            int shiftAmount = get_time_steps_us_f(tv->t0_plant,tActual_plant);   int alphaOut[MAX_ITER+1];   T Jout[MAX_ITER+1];
            
            // load in vars and init the alg
            loadVarsCPU_MPC<T>(cv->x_old,cv->u_old,cv->KT_old,cv->x,cv->xp,cv->u,cv->up,cv->d,cv->KT,cv->xGoal,cv->xActual,cv->P,cv->Pp,
                               cv->p,cv->pp,cv->du,cv->AB,cv->err,&alphaIndex,(T)TIME_STEP,cv->threads,shiftAmount,
                               md->ld_x,md->ld_u,md->ld_d,md->ld_KT,md->ld_P,md->ld_p,md->ld_du,md->ld_AB,cv->I,cv->Tbody);
            gettimeofday(&end2,NULL);
            (data->initTime).push_back(time_delta_ms(start2,end2));

            // do initial "next iteration setup"
            gettimeofday(&start2,NULL);
            initAlgCPU<T>(cv->x,cv->xp,cv->xp2,cv->u,cv->up,cv->AB,cv->H,cv->g,cv->KT,cv->du,cv->d,cv->JT,Jout,&prevJ,cv->alpha,alphaOut,
                          cv->xGoal,cv->threads,0,md->ld_x,md->ld_u,md->ld_AB,md->ld_H,md->ld_g,md->ld_KT,md->ld_du,md->ld_d,cv->I,cv->Tbody);
            gettimeofday(&end2,NULL);
            (data->nisTime).push_back(time_delta_ms(start2,end2));
        // INITIALIZE THE ALGORITHM //
        
        // debug print -- so ready to start
        if (DEBUG_SWITCH){
            printf("Iter[0] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] Rho[%f]\n",
                        cv->x[(md->ld_x)*(NUM_TIME_STEPS-1)],cv->x[(md->ld_x)*(NUM_TIME_STEPS-1)+1],prevJ,alphaIndex,rho);
        }

        while(1){
            gettimeofday(&end,NULL); if(time_delta_ms(start,end) > time_budget){break;};
            // BACKWARD PASS //
                gettimeofday(&start2,NULL);
                backwardPassCPU<T>(cv->AB,cv->P,cv->p,cv->Pp,cv->pp,cv->H,cv->g,cv->KT,cv->du,cv->d,cv->dp,
                                   cv->ApBK,cv->Bdu,cv->x,cv->xp2,cv->dJexp,cv->err,&rho,&drho,cv->threads,
                                   md->ld_AB,md->ld_P,md->ld_p,md->ld_H,md->ld_g,md->ld_KT,md->ld_du,md->ld_A,md->ld_d,md->ld_x);
                gettimeofday(&end2,NULL);
                (data->bpTime).push_back(time_delta_ms(start2,end2));
            // BACKWARD PASS //

            gettimeofday(&end,NULL); if(time_delta_ms(start,end) > time_budget){break;};
            // FORWARD PASS //
                dJ = -1.0;  alphaIndex = 0; (data->sweepTime).push_back(0.0);    (data->simTime).push_back(0.0);
                while(1){
                    // FORWARD SWEEP //
                        gettimeofday(&start2,NULL);
                        // Do the forward sweep if applicable
                        if (M_F > 1){forwardSweep<T>(cv->x,cv->ApBK,cv->Bdu,cv->d,cv->xp,(cv->alpha)[alphaIndex],md->ld_x,md->ld_d,md->ld_A);}
                        gettimeofday(&end2,NULL);
                        (data->sweepTime).back() += time_delta_ms(start2,end2);
                    // FORWARD SWEEP //

                    // FORWARD SIM //
                        gettimeofday(&start2,NULL);
                        int err = forwardSimCPU<T>(cv->x,cv->xp,cv->xp2,cv->u,cv->up,cv->KT,cv->du,cv->d,cv->dp,cv->dJexp,cv->JT,
                                                   (cv->alpha)[alphaIndex],cv->xGoal,&J,&dJ,&z,prevJ,&ignoreFirstDefectFlag,&maxd,
                                                   cv->threads,md->ld_x,md->ld_u,md->ld_KT,md->ld_du,md->ld_d,cv->I,cv->Tbody);
                        gettimeofday(&end2,NULL);   
                        (data->simTime).back() += time_delta_ms(start2,end2);
                        if(err){if (alphaIndex < NUM_ALPHA - 1){(alphaIndex)++; continue;} else{alphaIndex = -1; break;}} else{break;}
                    // FORWARD SIM //
                }
            // FORWARD PASS //

            // NEXT ITERATION SETUP //
                gettimeofday(&start2,NULL);    
                // process accept or reject of traj and test for exit
                bool exitFlag = acceptRejectTrajCPU<T>(cv->x,cv->xp,cv->u,cv->up,cv->d,cv->dp,J,&prevJ,&dJ,&rho,&drho,&alphaIndex,
                                                       alphaOut,Jout,&iter,cv->threads,md->ld_x,md->ld_u,md->ld_d,max_iter);
                if(alphaOut[iter - !exitFlag] > 0){if (DEBUG_SWITCH){printf("successful iter w/ alpha[%d]\n",alphaOut[iter - !exitFlag]);}tv->last_successful_solve = 0;} // note that we were able to take a step
                if (exitFlag){
                    gettimeofday(&end2,NULL);
                    (data->nisTime).push_back(time_delta_ms(start2,end2));
                    break;
                }
                // if we have gotten here then prep for next pass
                gettimeofday(&end,NULL); if(time_delta_ms(start,end) > time_budget){break;};
                nextIterationSetupCPU<T>(cv->x,cv->xp,cv->u,cv->up,cv->d,cv->dp,cv->AB,cv->H,cv->g,cv->P,cv->p,cv->Pp,cv->pp,cv->xGoal,cv->threads,
                                         md->ld_x,md->ld_u,md->ld_d,md->ld_AB,md->ld_H,md->ld_g,md->ld_P,md->ld_p,cv->I,cv->Tbody);
                gettimeofday(&end2,NULL);
                (data->nisTime).push_back(time_delta_ms(start2,end2));
            // NEXT ITERATION SETUP //
      
            if (DEBUG_SWITCH){
                printf("Iter[%d] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] rho[%f] dJ[%f] z[%f] max_d[%f]\n",
                            iter-1,cv->x[(md->ld_x)*(NUM_TIME_STEPS-1)],cv->x[(md->ld_x)*(NUM_TIME_STEPS-1)+1],prevJ,alphaIndex,rho,dJ,z,maxd);
            }
        }

        // EXIT Handling
            if (DEBUG_SWITCH){
                printf("Exit with Iter[%d] Xf[%.4f, %.4f] Cost[%.4f] AlphaIndex[%d] rho[%f] dJ[%f] z[%f] max_d[%f]\n",
                            iter,cv->x[(md->ld_x)*(NUM_TIME_STEPS-1)],cv->x[(md->ld_x)*(NUM_TIME_STEPS-1)+1],prevJ,alphaIndex,rho,dJ,z,maxd);
            }
            // Bring back the final state and control (and compute final d if needed)
            gettimeofday(&start2,NULL);
            storeVarsCPU_MPC<T>(cv,tv,md,tActual_sys,tActual_plant,0,&maxd);
            for (int i=0; i <= iter; i++){(data->alpha).push_back(alphaOut[i]);   (data->J).push_back(Jout[i]);}
            gettimeofday(&end2,NULL);
            gettimeofday(&end,NULL);
            (data->initTime).back() += time_delta_ms(start2,end2);
            (data->tTime).push_back(time_delta_ms(start,end));
        // EXIT Handling
    }
// 3: MPC Main Algorithm Wrappers //

// 4: LCM helpers// TODO: currently unstable need to update
    // template <typename T>
    // class LCM_TrajRunner_Handler {
    //     public:
    //         trajVars<T> *vars; // local pointer to the global traj variables
    //         lcm::LCM *lcm_ptr; // ptr to LCM object for publish ability
    //         int first_pass;

    //         // init and store the global location and lock and lcm ptr
    //         LCM_TrajRunner_Handler(trajVars<T> *vars_in, lcm::LCM *lcm_in, int fp_in) : vars(vars_in), lcm_ptr(lcm_in), first_pass(fp_in) {} 
    //         ~LCM_TrajRunner_Handler(){} // do nothing in the destructor
        
    //         // lcm callback function
    //         void handleMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){
    //             (vars->lock)->lock(); // wait for any updates to current desired traj to finish being written
    //             //construct output msg container and begin to load it with data
    //             //printf("Current x is %f, %f, %f, %f, %f, %f, %f\n",msg->joint_position_measured[0],msg->joint_position_measured[1],msg->joint_position_measured[2],msg->joint_position_measured[3],msg->joint_position_measured[4],msg->joint_position_measured[5],msg->joint_position_measured[6]);
    //             drake::lcmt_iiwa_command dataOut;   
    //             dataOut.num_joints = static_cast<int32_t>(NUM_POS);   dataOut.num_torques = static_cast<int32_t>(CONTROL_SIZE);
    //             dataOut.utime = static_cast<int64_t>(msg->utime);     
    //             if(first_pass){vars->t0_plant = msg->utime;    first_pass = 0;}// printf("first pass traj runner\n");}
    //             dataOut.joint_position.resize(dataOut.num_joints);      dataOut.joint_torque.resize(dataOut.num_torques);
    //             // get the correct controls for this time
    //             // printf("getting controls for time %ld vs. t0 %ld\n",dataOut.utime,vars->t0_plant);
    //             int err = getHardwareControls<T>(&(dataOut.joint_position[0]), &(dataOut.joint_torque[0]), 
    //                                              vars->x, vars->u, vars->KT, static_cast<double>(vars->t0_plant), 
    //                                              &(msg->joint_position_measured[0]), &(msg->joint_velocity_estimated[0]),  
    //                                              static_cast<double>(msg->utime), vars->ld_x, vars->ld_u, vars->ld_KT);
    //             //printf("controls comped\n");
    //             (vars->lock)->unlock(); // release the lock after computations are done and then publish the output
    //             if (!err){lcm_ptr->publish(ARM_COMMAND_CHANNEL,&dataOut);}// printf("data sent\n");}
    //             else{printf("[!]CRITICAL ERROR: Asked to execute beyond bounds of current traj.\n");}
    //         }
    // };

    // template <typename T>
    // __host__
    // void runTrajRunner(lcm::LCM *lcm_ptr, trajVars<T> *tvars, lcm::Subscription *sub){
    //     // launch the thread running the trajRunner
    //     LCM_TrajRunner_Handler<T> handler = LCM_TrajRunner_Handler<T>(tvars, lcm_ptr, 1);
    //     sub = lcm_ptr->subscribe(ARM_STATUS_CHANNEL, &LCM_TrajRunner_Handler<T>::handleMessage, &handler);
    //     sub->setQueueCapacity(1);
    //     // poll the fd for updates
    //     //int lcm_fd = lcm_ptr->getFileno();    fd_set fds;     FD_ZERO(&fds);  FD_SET(lcm_fd, &fds);   struct timeval timeout = {1,0};
    //     while(0 == lcm_ptr->handle());
    //     // while(1){    
    //     //      if (select(lcm_fd + 1, &fds, 0, 0, &timeout)) {if (FD_ISSET(lcm_fd, &fds)){lcm_ptr->handle();}}
    //     //      //struct timeval start, end;    gettimeofday(&start,NULL);  while(1){gettimeofday(&end,NULL);   if (time_delta_ms(start,end) >= 50){break;}}
    //     // }
    // }

    // template <typename T>
    // class LCM_MPCLoop_Handler {
    //     public:
    //         GPUVars<T> *gvars; // local pointer to the global algorithm variables
    //         CPUVars<T> *cvars; // local pointer to the global algorithm variables
    //         matDimms *dimms; // pointer to mat dimms
    //         trajVars<T> *tvars; // local pointer to the global traj variables
    //         algTrace<T> *data; // local pointer to the global algorithm trace data stuff
    //         int iterLimit; int timeLimit; bool mode; // limits for solves and cpu/gpu mode

    //         // init and store the global location
    //         LCM_MPCLoop_Handler(GPUVars<T> *avIn, trajVars<T> *tvarsIn, matDimms *dimmsIn, algTrace<T> *dataIn, int iL, int tL) : 
    //                                                 gvars(avIn), tvars(tvarsIn), dimms(dimmsIn), data(dataIn), iterLimit(iL), timeLimit(tL) {cvars = nullptr; mode = 1;}
    //         LCM_MPCLoop_Handler(CPUVars<T> *avIn, trajVars<T> *tvarsIn, matDimms *dimmsIn, algTrace<T> *dataIn, int iL, int tL) : 
    //                                                 cvars(avIn), tvars(tvarsIn), dimms(dimmsIn), data(dataIn), iterLimit(iL), timeLimit(tL) {gvars = nullptr; mode = 0;}
    //         ~LCM_MPCLoop_Handler(){} // do nothing in the destructor
        
    //         // lcm callback function
    //         void handleMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){
    //             // determine if GPU or CPU mode and save down time and state pointers and try to get the 
    //             //    global lock and compute a new MPC iter if ready else return
    //             if(mode){if (!((gvars->lock)->try_lock())){return;}}             
    //                else {if (!((cvars->lock)->try_lock())){return;}}       
    //             int64_t tActual_plant = msg->utime;           
    //             struct timeval sys_t; gettimeofday(&sys_t,NULL);        int64_t tActual_sys = get_time_us_i64(sys_t);
                
    //             // if first load initialize timers and keep inital traj (so set xActual to x)
    //             T *xActual_load = mode == 1 ? gvars->xActual : cvars->xActual;
    //             if (tvars->first_pass){
    //                 //printf("fist pass MPC loop\n");
    //                 tvars->t0_plant = tActual_plant;    tvars->t0_sys = tActual_sys;     tvars->first_pass = false;
    //                 #pragma unroll
    //                 for (int i=0; i<STATE_SIZE; i++){xActual_load[i] = (T)(tvars->x)[i];}
    //             }
    //             // else update xActual
    //             else{
    //                 #pragma unroll
    //                 for (int i=0; i<NUM_POS; i++){xActual_load[i]         = (T)(msg->joint_position_measured)[i]; 
    //                                               xActual_load[i+NUM_POS] = (T)(msg->joint_velocity_estimated)[i];}
    //                 if(mode){gpuErrchk(cudaMemcpy(gvars->d_xActual, gvars->xActual, STATE_SIZE*sizeof(T), cudaMemcpyHostToDevice));}
    //             }
                
    //             // run iLQR
    //             if(mode){runiLQR_MPC_GPU(tvars,gvars,dimms,data,tActual_sys,tActual_plant,0,iterLimit,timeLimit);  (gvars->lock)->unlock();}
    //             else{    runiLQR_MPC_CPU(tvars,cvars,dimms,data,tActual_sys,tActual_plant,0,iterLimit,timeLimit);  (cvars->lock)->unlock();}
    //         }      
    // };

    // template <typename T>
    // __host__
    // void runMPCHandler(lcm::LCM *lcm_ptr, LCM_MPCLoop_Handler<T> *handler, lcm::Subscription *sub){
    //     sub = lcm_ptr->subscribe(ARM_STATUS_CHANNEL, &LCM_MPCLoop_Handler<T>::handleMessage, handler);
    //     sub->setQueueCapacity(1);
    //     while(0 == lcm_ptr->handle());
    // }

    // template <typename T>
    // class LCM_IIWA_STATUS_printer {
    //     public:
    //         // T *xGoal; 
    //         // LCM_IIWA_STATUS_printer(GPUVars<T> *avIn){xGoal = gvars->xGoal;}
    //         // LCM_IIWA_STATUS_printer(CPUVars<T> *avIn){xGoal = cvars->xGoal;}
    //         LCM_IIWA_STATUS_printer(){}
    //         ~LCM_IIWA_STATUS_printer(){}

    //         void handleMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_status *msg){                
    //             double eePos[NUM_POS];   compute_eePos_scratch<double>((double *)&(msg->joint_position_measured[0]), &eePos[0]);
    //             printf("[%ld] eePos: [%f %f %f]\n",msg->utime,eePos[0],eePos[1],eePos[2]);
    //             // T eNorm = static_cast<T>(sqrt(pow(eePos[0]-xGoal[0],2) + pow(eePos[1]-xGoal[1],2) + pow(eePos[2]-xGoal[2],2)));
    //             // printf("[%ld] eePos: [%f %f %f] w/ error:[%f]\n",msg->utime,eePos[0],eePos[1],eePos[2],eNorm);
    //         }
    // };

    // template <typename T>
    // __host__
    // void run_IIWA_STATUS_printer(lcm::LCM *lcm_ptr, LCM_IIWA_STATUS_printer<T> *handler, lcm::Subscription *sub){
    //     sub = lcm_ptr->subscribe(ARM_STATUS_CHANNEL, &LCM_IIWA_STATUS_printer<T>::handleMessage, handler);
    //     // sub->setQueueCapacity(1);
    //     while(0 == lcm_ptr->handle());
    // }

    // template <typename T>
    // class LCM_IIWA_COMMAND_printer {
    //     public:
    //         LCM_IIWA_COMMAND_printer(){}
    //         ~LCM_IIWA_COMMAND_printer(){}

    //         void handleMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_command *msg){                
    //             double eePos[NUM_POS];   compute_eePos_scratch<double>((double *)&(msg->joint_position[0]), &eePos[0]);
    //             printf("[%ld] eePosDes: [%f %f %f] control: [%f %f %f %f %f %f %f ]\n",msg->utime,eePos[0],eePos[1],eePos[2],
    //                                         msg->joint_torque[0],msg->joint_torque[1],msg->joint_torque[2],msg->joint_torque[3],
    //                                         msg->joint_torque[4],msg->joint_torque[5],msg->joint_torque[6]);
    //         }
    // };

    // template <typename T>
    // __host__
    // void run_IIWA_COMMAND_printer(lcm::LCM *lcm_ptr, LCM_IIWA_COMMAND_printer<T> *handler, lcm::Subscription *sub){
    //     sub = lcm_ptr->subscribe(ARM_COMMAND_CHANNEL, &LCM_IIWA_COMMAND_printer<T>::handleMessage, handler);
    //     // sub->setQueueCapacity(1);
    //     while(0 == lcm_ptr->handle());
    // }

    // struct SimVars{
    //     double uCom[CONTROL_SIZE];  
    //     double xCom[STATE_SIZE];
    //     bool flag;
    //     std::mutex *lock;
    // };

    // class LCM_Simulator_Handler {
    //     public:
    //         SimVars *svars;

    //         LCM_Simulator_Handler(SimVars *svin) : svars(svin){}
    //         ~LCM_Simulator_Handler(){}

    //         // lcm callback function to update the svars
    //         void handleMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const drake::lcmt_iiwa_command *msg){
    //             while(!(svars->lock)->try_lock());
    //             for(int i = 0; i < NUM_POS; i++){(svars->xCom)[i] = msg->joint_position[i]; (svars->uCom)[i] = msg->joint_torque[i];}
    //             if(!(svars->flag)){svars->flag = 1;}
    //             (svars->lock)->unlock();
    //         }
    // };

    // __host__
    // void runSimulatorHandler(SimVars *svars, lcm::LCM *lcm_ptr, lcm::Subscription *sub){
    //     // launch the thread running the trajRunner
    //     LCM_Simulator_Handler handler = LCM_Simulator_Handler(svars);
    //     sub = lcm_ptr->subscribe(ARM_COMMAND_CHANNEL, &LCM_Simulator_Handler::handleMessage, &handler);
    //     sub->setQueueCapacity(1);
    //     while(0 == lcm_ptr->handle());
    // }

    // __host__
    // void runSimulator(lcm::LCM *lcm_ptr, lcm::Subscription *sub, double *xInit, int dt_us = 1000){
    //     double currX[STATE_SIZE];       double nextX[STATE_SIZE];   double qdd[NUM_POS];
    //     for(int i=0; i<STATE_SIZE; i++){currX[i] = xInit[i];}
    //     double Tbody[36*NUM_POS];       double I[36*NUM_POS];       
    //     initI<double>(I);               initT<double>(Tbody);
    //     int64_t time = 0;               double dt = static_cast<double>(dt_us)/1000.0/1000.0;    struct timeval start, end;
    //     SimVars *svars = new SimVars;   svars->flag = 0;    svars->lock = new std::mutex(); 
    //     // launch the thread running the trajRunner
    //     std::thread lcmThread = std::thread(runSimulatorHandler, svars, lcm_ptr, sub);
    //     lcmThread.detach(); // how do I kill later?
    //     lcm::LCM lcm_ptr2;  if(!lcm_ptr2.good()){printf("LCM Failed to Init\n");}
    //     // simulator and publisher
    //     while(1){
    //         if (svars->flag){
    //             gettimeofday(&start,NULL);   time += dt_us;
    //             //construct output msg container and begin to load it with data
    //             drake::lcmt_iiwa_status dataOut; dataOut.utime = time;
    //             dataOut.num_joints = static_cast<int32_t>(NUM_POS); 
    //             dataOut.joint_position_measured.resize(dataOut.num_joints);      
    //             dataOut.joint_velocity_estimated.resize(dataOut.num_joints);
    //             dataOut.joint_position_commanded.resize(dataOut.num_joints);  
    //             dataOut.joint_position_ipo.resize(dataOut.num_joints);  
    //             dataOut.joint_torque_measured.resize(dataOut.num_joints);  
    //             dataOut.joint_torque_commanded.resize(dataOut.num_joints);
    //             dataOut.joint_torque_external.resize(dataOut.num_joints);
    //             while(!(svars->lock)->try_lock());
    //             for(int i = 0; i < NUM_POS; i++){
    //                 dataOut.joint_position_commanded[i] = (svars->xCom)[i];  dataOut.joint_position_ipo[i] = 0;
    //                 double val = (svars->uCom)[i];                           dataOut.joint_torque_external[i] = 0;
    //                 dataOut.joint_torque_measured[i] = val;                  dataOut.joint_torque_commanded[i] = val;
    //             }
    //             (svars->lock)->unlock();
    //             // run integration in a bunch of steps
    //             int numSteps = 1; dt /= numSteps;
    //             for (int i = 0; i < numSteps; i++){
    //                 _integrator<double>(nextX,currX,&(dataOut.joint_torque_measured[0]),qdd,I,Tbody,dt);
    //                 for(int i = 0; i < STATE_SIZE; i++){currX[i] = nextX[i];}
    //             }
    //             for(int i = 0; i < STATE_SIZE; i++){
    //                 if (i < NUM_POS){dataOut.joint_position_measured[i] = currX[i];}
    //                 else{dataOut.joint_velocity_estimated[i-NUM_POS] = currX[i];}
    //             }
    //             lcm_ptr2.publish(ARM_STATUS_CHANNEL,&dataOut);
    //         }
    //         while(1){gettimeofday(&end,NULL);   if (time_delta_ms(start,end) >= dt_us){break;}}
    //     }
    // }

    // template <typename T>
    // class LCM_UpdateGoal_Handler {
    //     public:
    //         T **xGoal; // local pointer to the global location

    //         LCM_UpdateGoal_Handler(float **xg) : xGoal(xg) {} // init and store the global location
    //         ~LCM_UpdateGoal_Handler(){} // do nothing in the destructor

    //         // lcm callback function to update the current goal
    //         void handleMessage(const lcm::ReceiveBuffer *rbuf, const std::string &chan, const kuka::lcmt_target_twist *msg){
    //             #pragma unroll
    //             for (int i = 0; i < 3; i++){(*xGoal)[i] = (T)msg->position[i];   (*xGoal)[i+3] = (T)msg->velocity[i];} // update global goal
    //         }
    // };

// // 4: LCM call wrappers //