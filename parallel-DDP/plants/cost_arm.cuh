/*****************************************************************
 * Kuka Arm Cost Funcs
 *
 * TBD NEED TO ADD DOC HERE
 *****************************************************************/

#define Q1 0.1 // q
#define Q2 0.001 // qd
#define R  0.0001
#define QF1 1000.0 // q
#define QF2 1000.0 // qd

#define USE_SMOOTH_ABS 0
#define SMOOTH_ABS_ALPHA 0.2

#ifndef MPC_MODE
	#if USE_SMOOTH_ABS
		#define Q_HAND1 0.1		//2.0 // xyz
		#define Q_HAND2 0.001		//2.0 // rpy
		#define R_HAND 0.0001		//0.0001
		#define QF_HAND1 1000000.0	//20000.0 // xyz
		#define QF_HAND2 10000.0	//20000.0 // rpy
		#define Q_xdHAND 0.1		//1.0//0.1
		#define QF_xdHAND 10000.0	//10.0//100.0
		#define Q_xHAND 0.0		//0.0//0.001//1.0
		#define QF_xHAND 0.0		//0.0//1.0
	#else
		#define Q_HAND1 0.1		//1.0 // xyz
		#define Q_HAND2 0//0.001		//1.0 // rpy
		#define R_HAND 0.0001		//0.001
		#define QF_HAND1 1000.0		//5000.0 // xyz
		#define QF_HAND2 0//10.0		//5000.0 // rpy
		#define Q_xdHAND 0.1		//1.0//0.1
		#define QF_xdHAND 1000.0	//10.0//100.0
		#define Q_xHAND 0.0		//0.0//0.001//1.0
		#define QF_xHAND 0.0		//0.0//1.0
	#endif
#else
	#define Q_HAND1 0.1		//1.0 // xyz
	#define Q_HAND2 0		//1.0 // rpy
	#define R_HAND 0.0001		//0.001
	#define QF_HAND1 1000.0		//5000.0 // xyz
	#define QF_HAND2 0		//5000.0 // rpy
	#define Q_xdHAND 0.1		//1.0//0.1
	#define QF_xdHAND 10.0	//10.0//100.0
	#define Q_xHAND 0.0		//0.0//0.001//1.0
	#define QF_xHAND 0.0		//0.0//1.0
#endif

// Also define limits on torque, pos, velocity and Q/Rs for those
#define SAFETY_FACTOR_P 0.8
#define SAFETY_FACTOR_V 0.8
#define SAFETY_FACTOR_T 0.8
// From URDF
#define TORQUE_LIMIT (300.0 * SAFETY_FACTOR_T)
#define R_TL 0.1
#define POS_LIMIT_024 (2.96705972839 * SAFETY_FACTOR_P)
#define POS_LIMIT_135 (2.09439510239 * SAFETY_FACTOR_P)
#define POS_LIMIT_6   (3.05432619099 * SAFETY_FACTOR_P)
#define Q_PL 100.0
// From KukaSim
#define VEL_LIMIT_01  (1.483529 * SAFETY_FACTOR_V) // 85°/s in rad/s
#define VEL_LIMIT_2   (1.745329 * SAFETY_FACTOR_V) // 100°/s in rad/s
#define VEL_LIMIT_3   (1.308996 * SAFETY_FACTOR_V) // 75°/s in rad/s
#define VEL_LIMIT_4   (2.268928 * SAFETY_FACTOR_V) // 130°/s in rad/s
#define VEL_LIMIT_56  (2.356194 * SAFETY_FACTOR_V) // 135°/s in rad/s
#define Q_VL 100.0

template <typename T>
__host__ __device__ __forceinline__
T getPosLimit(int ind){return (T)(ind == 6 ? POS_LIMIT_6 : (ind % 2 ? POS_LIMIT_135 : POS_LIMIT_024));}

template <typename T>
__host__ __device__ __forceinline__
T getVelLimit(int ind){return (T)(ind > 4 ? VEL_LIMIT_56 : (ind == 4 ? VEL_LIMIT_4 : (ind == 3 ? VEL_LIMIT_3 : (ind == 2 ? VEL_LIMIT_2 : VEL_LIMIT_01))));}

template <typename T>
__host__ __device__ __forceinline__
T getTorqueLimit(int ind){return (T)TORQUE_LIMIT;}

template <typename T, int dLevel>
__host__ __device__ __forceinline__
T pieceWiseQuadratic(T val, T qStart){
	T delta = abs(val) - qStart; 	bool flag = delta > 0;
	#if dLevel == 0
		if (flag) {return ((T)0.5)*delta*delta;}
	#elif dLevel == 1
		if (flag) {return sgn(val)*delta;}
	#elif dLevel == 2
		if (flag) {return (T)1.0;}
	#else
		#error "Derivative of pieceWiseQuadratic is not implemented beyond level 2\n"
	#endif
	return (T)0.0;
}

template <typename T>
__host__ __device__ __forceinline__
void getLimitVars(T *s_x, T *s_u, T *qr, T *val, T *limit, int ind, int k){
	if (ind < NUM_POS){	         *qr = Q_PL;		*val = s_x[ind];				*limit = getPosLimit<T>(ind);}
	else if (ind < STATE_SIZE){  *qr = Q_VL;		*val = s_x[ind];				*limit = getVelLimit<T>(ind-NUM_POS);}
	else{                        *qr = R_TL;		*val = s_u[ind-STATE_SIZE]; 	*limit = getTorqueLimit<T>(ind-STATE_SIZE);}
}

template <typename T, int dLevel>
__host__ __device__ __forceinline__
T limitCosts(T *s_x, T *s_u, int ind, int k){
	T qr;	T val;	T limit;	T cost = 0.0;
	getLimitVars(s_x,s_u,&qr,&val,&limit,ind,k);	cost += qr*pieceWiseQuadratic<T,dLevel>(val,limit);
	#if dLevel == 0 // at level 0 we only call this once and want all 3 vs at higher levels we call it once per variable
		getLimitVars(s_x,s_u,&qr,&val,&limit,ind+NUM_POS,k);	cost += qr*pieceWiseQuadratic<T,dLevel>(val,limit);
		getLimitVars(s_x,s_u,&qr,&val,&limit,ind+STATE_SIZE,k);	cost += qr*pieceWiseQuadratic<T,dLevel>(val,limit);
	#endif
	return cost;
}

template <typename T>
__host__ __device__ __forceinline__
T eeCost(T *s_eePos, T *d_eeGoal, int k){
	T cost = 0.0;
 	for (int i = 0; i < 6; i ++){
    	T delta = s_eePos[i] - d_eeGoal[i];
    	cost += 0.5*(k == NUM_TIME_STEPS-1 ? (i < 3 ? QF_HAND1 : QF_HAND2) : (i < 3 ? Q_HAND1 : Q_HAND2))*delta*delta;
 	}
 	if (USE_SMOOTH_ABS){
    	cost = (T) sqrt(2*cost + SMOOTH_ABS_ALPHA*SMOOTH_ABS_ALPHA) - SMOOTH_ABS_ALPHA;
 	}
 	return cost;
}

template <typename T>
__host__ __device__ __forceinline__
T deeCost(T *s_eePos, T *s_deePos, T *d_eeGoal, int k, int r){
	T val = 0.0;
 	#pragma unroll
 	for (int i = 0; i < 6; i++){
 		T delta = s_eePos[i]-d_eeGoal[i];
 		T deePos = s_deePos[r*6+i];
    	val += (k == NUM_TIME_STEPS - 1 ? (i < 3 ? QF_HAND1 : QF_HAND2) : (i < 3 ? Q_HAND1 : Q_HAND2))*delta*deePos;
 	}
 	if (USE_SMOOTH_ABS){
		T val2 = 0.0;
    	#pragma unroll
    	for (int i = 0; i < 6; i++){
    		T delta = s_eePos[i]-d_eeGoal[i];
       		val2 += (k == NUM_TIME_STEPS - 1 ? (i < 3 ? QF_HAND1 : QF_HAND2) : (i < 3 ? Q_HAND1 : Q_HAND2))*delta*delta;
    	}
    	val2 += SMOOTH_ABS_ALPHA*SMOOTH_ABS_ALPHA;
    	val /= sqrt(val2);
 	}
 	return val;
}

// eeCost Func to split shared mem
template <typename T>
__host__ __device__ __forceinline__
void costFunc(T *s_cost, T *s_eePos, T *d_eeGoal, T *s_x, T *s_u, int k){
	int start, delta; singleLoopVals(&start,&delta);
	#pragma unroll
    for (int ind = start; ind < NUM_POS; ind += delta){
    	T cost = 0.0;
    	if(ind == 0){cost += eeCost(s_eePos,d_eeGoal,k);} // compute in one thread for smooth abs
    	// in all cases add on u cost and state cost
      	cost += 0.5*(k == NUM_TIME_STEPS-1 ? 0.0 : R_HAND)*s_u[ind]*s_u[ind]; // add on input cost
      	cost += 0.5*(k == NUM_TIME_STEPS-1 ? QF_xHAND : Q_xHAND)*s_x[ind]*s_x[ind]; // add on the state tend to zero cost            
      	cost += 0.5*(k == NUM_TIME_STEPS-1 ? QF_xdHAND : Q_xdHAND)*s_x[ind + NUM_POS]*s_x[ind + NUM_POS]; // add on the state tend to zero cost  
      	// add on any limit costs if needed
      	#if USE_LIMITS_FLAG
      		cost += limitCosts<T,0>(s_x,s_u,ind,k);
  		#endif
      	s_cost[ind] += cost;
   	}
}

// eeCost Func returns single val
template <typename T>
__host__ __device__ __forceinline__
T costFunc(T *s_eePos, T *d_eeGoal, T *s_x, T *s_u, int k){
	T cost = 0.0;
	#pragma unroll
    for (int ind = 0; ind < NUM_POS; ind ++){
    	if(ind == 0){cost += eeCost(s_eePos,d_eeGoal,k);} // compute in one thread for smooth abs
    	// in all cases add on u cost and state cost
      	cost += 0.5*(k == NUM_TIME_STEPS-1 ? 0.0 : R_HAND)*s_u[ind]*s_u[ind]; // add on input cost
      	cost += 0.5*(k == NUM_TIME_STEPS-1 ? QF_xHAND : Q_xHAND)*s_x[ind]*s_x[ind]; // add on the state tend to zero cost            
      	cost += 0.5*(k == NUM_TIME_STEPS-1 ? QF_xdHAND : Q_xdHAND)*s_x[ind + NUM_POS]*s_x[ind + NUM_POS]; // add on the state tend to zero cost  
      	// add on any limit costs if needed
      	#if USE_LIMITS_FLAG
      		cost += limitCosts<T,0>(s_x,s_u,ind,k);
  		#endif
   	}
   	return cost;
}

// eeCost Grad
template <typename T>
__host__ __device__ __forceinline__
void costGrad(T *Hk, T*gk, T *s_eePos, T *s_deePos, T *d_eeGoal, T *s_x, T *s_u, int k, int ld_H, T *d_JT = nullptr, int tid = -1){
	// then to get the gradient and Hessian we need to compute the following for the state block (and also standard control block)
	// J = \sum_i Q_i*pow(hand_delta_i,2) + other stuff
	// dJ/dx = g = \sum_i Q_i*hand_delta_i*dh_i/dx + other stuff
	int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
	int start, delta; singleLoopVals(&start,&delta);
	#pragma unroll
	for (int r = start; r < DIM_g_r; r += delta){
	  	T val = 0.0;
	  	if (r < NUM_POS){val += deeCost(s_eePos,s_deePos,d_eeGoal,k,r);}
	  	// add on the joint level state cost (tend to zero regularizer) and control cost
	  	if (r < NUM_POS){val += (k == NUM_TIME_STEPS - 1 ? QF_xHAND : Q_xHAND)*s_x[r];}
	  	else if (r < STATE_SIZE){val += (k == NUM_TIME_STEPS - 1 ? QF_xdHAND : Q_xdHAND)*s_x[r];}
	  	else{val += (k == NUM_TIME_STEPS - 1 ? 0 : R_HAND)*s_u[r-STATE_SIZE];}
	  	// add on any limit costs if needed
	  	#if USE_LIMITS_FLAG
      		val += limitCosts<T,1>(s_x,s_u,r,k);
  		#endif
	  	gk[r] = val;
	}
	hd__syncthreads();
	// d2J/dx2 = H \approx dh_i/dx'*dh_i/dx + other stuff
	#pragma unroll
	for (int c = starty; c < DIM_H_c; c += dy){
	  	T *H = &Hk[c*ld_H];
	  	#pragma unroll
	  	for (int r= startx; r<DIM_H_r; r += dx){
	     	T val = 0.0;
	     	// multiply two columns for pseudo-Hessian (dropping d2q/dh2 term)
	     	if (c < NUM_POS && r < NUM_POS){
	        	#pragma unroll
	        	for (int j = 0; j < 6; j++){
	           		val += s_deePos[r*6+j]*s_deePos[c*6+j];
	        	}
	     	}
		    // if applicable add on the joint level state cost (tend to zero regularizer) and control cost
		    if (r == c){
	        	if (r < NUM_POS){val += (k == NUM_TIME_STEPS - 1 ? QF_xHAND : Q_xHAND);}
	        	else if (r < STATE_SIZE){val += (k == NUM_TIME_STEPS - 1 ? QF_xdHAND : Q_xdHAND);}
	        	else {val += (k== NUM_TIME_STEPS - 1) ? 0.0 : R_HAND;}
	        	// add on any limit costs if needed
	        	#if USE_LIMITS_FLAG
      				val += limitCosts<T,2>(s_x,s_u,r,k);
  				#endif
	     	}
	     	H[r] = val;//s_g[k]*s_g[i]; // before we multiplied gradient but that isn't correct
	  	}
	}
	//if cost asked for compute it
	bool flag = d_JT != nullptr; int ind = (tid != -1 ? tid : k);
	#ifdef __CUDA_ARCH__
		if(threadIdx.x != 0 || threadIdx.y != 0){flag = 0;}
		if (flag){d_JT[ind] = costFunc(s_eePos,d_eeGoal,s_x,s_u,k);}
	#else
		if (flag){d_JT[ind] += costFunc(s_eePos,d_eeGoal,s_x,s_u,k);}
	#endif
	
}

// joint level cost func returns single val
template <typename T>
__host__ __device__ __forceinline__
T costFunc(T *xk, T *uk, T *xgk, int k){
	T cost = 0.0;
	if (k == NUM_TIME_STEPS - 1){
		#pragma unroll
    	for (int i=0; i<STATE_SIZE; i++){T delta = xk[i]-xgk[i]; cost += (T) (i < NUM_POS ? QF1 : QF2)*delta*delta;}
    }
    else{
    	#pragma unroll
        for (int i=0; i<STATE_SIZE; i++){T delta = xk[i]-xgk[i]; cost += (T) (i < NUM_POS ? Q1 : Q2)*delta*delta;}
    	#pragma unroll
        for (int i=0; i<CONTROL_SIZE; i++){cost += (T) R*uk[i]*uk[i];}
	}
	return 0.5*cost;
}

// joint level cost grad
template <typename T>
__host__ __device__ __forceinline__
void costGrad(T *Hk, T *gk, T *xk, T *uk, T *xgk, int k, int ld_H){
	if (k == NUM_TIME_STEPS - 1){
		#pragma unroll
      	for (int i=0; i<STATE_SIZE; i++){
      		#pragma unroll
         	for (int j=0; j<STATE_SIZE; j++){
            	Hk[i*ld_H + j] = (i != j) ? 0.0 : (i < NUM_POS ? QF1 : QF2);
         	}  
      	}
      	#pragma unroll
      	for (int i=0; i<STATE_SIZE; i++){
         	gk[i] = (i < NUM_POS ? QF1 : QF2)*(xk[i]-xgk[i]);
      	}
      	#pragma unroll
      	for (int i=0; i<CONTROL_SIZE; i++){
         	gk[i+STATE_SIZE] = 0.0;
      	}
   	}
   	else{
      	#pragma unroll
      	for (int i=0; i<STATE_SIZE+CONTROL_SIZE; i++){
      		#pragma unroll
         	for (int j=0; j<STATE_SIZE+CONTROL_SIZE; j++){
            	Hk[i*ld_H + j] = (i != j) ? 0.0 : (i < NUM_POS ? Q1 : (i < STATE_SIZE ? Q2 : R));
         	}  
      	}
      	#pragma unroll
      	for (int i=0; i<STATE_SIZE; i++){
         	gk[i] = (i < NUM_POS ? Q1 : Q2)*(xk[i]-xgk[i]);
      	}
      	#pragma unroll
      	for (int i=0; i<CONTROL_SIZE; i++){
         	gk[i+STATE_SIZE] = R*uk[i];
      	}
   	}
}