/*****************************************************************
 * DDP Forward Pass Helper Functions
 * (currently only supports iLQR - UDP in future release)
 *
 * forwardSweep(Kern)
 *   forwardSweepInner
 * forwardSim(C/G)PU
 *   cost(Threaded/Kern)
 *   defect(Comp/Kern)
 *   forwardSim(Kern)
 *     forwardSimInner
 *       computeControlKT
 *       integrator (from integrators)
 *         dynamics (from dynamics)
 *****************************************************************/

template <typename T>
__host__ __device__ __forceinline__
void forwardSweepInner(T *s_ApBK, T *Bk, T *dk, T *s_dx, T *xk, T *xpk, T alpha, int ld_x, int ld_d, int ld_A, T *Ak = nullptr){
	// loop forward and compute new states and controls with the sweep (using the AB linearization)
	T *xkp1 = xk + ld_x;
	int start, delta; singleLoopVals(&start,&delta);
   	for(int k=0; k<NUM_TIME_STEPS-1; k++){
		// compute the new state: xkp1 = xkp1 + (A - B*K)(xnew-x) - alpha*B*du + d
		// we can do these in a series of parallel computations on seperate threads
		// stage 1: ApBK = A - BK, dpBdu = d - B*du computed in backward pass so load them in
		//          actually only load ApBKK because dpBdu only used once
		if (Ak != nullptr){loadMatToShared<T,DIM_A_r,DIM_A_c>(s_ApBK,Ak,ld_A);}
		// stage 1: compute dx = x - xp
		loadDeltaV<T,DIM_x_r>(s_dx, xk, xpk);
		hd__syncthreads();

		// stage 2: compute xkp1 += dM + XM*xp
		#pragma unroll
		for (int kx = start; kx < DIM_x_r; kx += delta){
			T val = 0;
			// multiply row kx of XM by xp
			#pragma unroll
			for (int i=0; i<DIM_x_r; i++){val += s_ApBK[kx + DIM_A_r*i]*s_dx[i];}
			// and then add d and the previos value and save to global memory
			xkp1[kx] += alpha*Bk[kx] + val + (onDefectBoundary(k) ? dk[kx] : 0.0);
		}

		// then update the offsets for the next pass
		xk = xkp1;
		xpk += ld_x;
		xkp1 += ld_x;
		dk += ld_d;
		Bk += ld_d;
		if (Ak != nullptr){Ak += ld_A*DIM_A_c;} else{s_ApBK += ld_A*DIM_A_c;}
		hd__syncthreads();
   	}
}
		
template <typename T>
__global__
void forwardSweepKern(T **d_x, T *d_ApBK, T *d_Bdu, T *d_d, T *d_xp, T *d_alpha, int ld_x, int ld_d, int ld_A){
	__shared__ T s_ApBK[DIM_A_r*DIM_A_c];
	__shared__ T s_dx[DIM_x_r];
	T alpha = d_alpha[blockIdx.x];
	T *xk = d_x[blockIdx.x];
	forwardSweepInner(s_ApBK,d_Bdu,d_d,s_dx,xk,d_xp,alpha,ld_x,ld_d,ld_A,d_ApBK);
}

template <typename T>
__host__ __forceinline__
void forwardSweep(T *x, T *ApBK, T *Bdu, T *d, T *xp, T alpha, int ld_x, int ld_d, int ld_A){
	T s_dx[DIM_x_r];
	forwardSweepInner(ApBK,Bdu,d,s_dx,x,xp,alpha,ld_x,ld_d,ld_A);
}

template <typename T>
__host__ __forceinline__
void forwardSweepThreaded(threadDesc_t desc, T **xs, T *ApBK, T *Bdu, T **ds, T *xp, T *alphas, int ld_x, int ld_d, int ld_A){
	T s_dx[DIM_x_r];
	for (unsigned int i=0; i<desc.reps; i++){
		int k = desc.tid + i*desc.dim;
		forwardSweepInner(ApBK,Bdu,ds[k],s_dx,xs[k],xp,alphas[k],ld_x,ld_d,ld_A);
	}
}
template <typename T>
__host__ __forceinline__
void forwardSweep2(T **xs, T *ApBK, T *Bdu, T **ds, T *xp, T *alphas, int alphaStart, std::thread *threads, int ld_x, int ld_d, int ld_A){
	T **xstart = &xs[alphaStart];	T **dstart = &ds[alphaStart];	T *astart = &alphas[alphaStart];
	threadDesc_t desc; 	desc.dim = FSIM_ALPHA_THREADS;	desc.reps = 1;
	for (unsigned int thread_i = 0; thread_i < FSIM_ALPHA_THREADS; thread_i++){
		desc.tid = thread_i;
		threads[thread_i] = std::thread(&forwardSweepThreaded<T>,desc,std::ref(xstart),std::ref(ApBK),std::ref(Bdu),std::ref(dstart),std::ref(xp),std::ref(astart),ld_x,ld_d,ld_A);
		if(FORCE_CORE_SWITCHES){setCPUForThread(threads, thread_i);}
	}
	for (unsigned int thread_i = 0; thread_i < FSIM_ALPHA_THREADS; thread_i++){threads[thread_i].join();}
}

template <typename T>
__global__
void defectKern(T **d_d, T *d_dT, int ld_d){
	__shared__ T s_d[NUM_TIME_STEPS];
  	// compute the defect in shared memory
    s_d[threadIdx.x] = 0.0;
    if (onDefectBoundary(threadIdx.x)){
      	T *dk = &d_d[blockIdx.x][threadIdx.x*ld_d];
		for (int c = 0; c < DIM_d_r; c ++){
			s_d[threadIdx.x] += abs(dk[c]);
		}
    }
    __syncthreads();
	// then max it all up per alpha with a reduce pattern
    reduceMax<T>(s_d);
    __syncthreads();
	if (threadIdx.x == 0){d_dT[blockIdx.x] = s_d[0];}
}

template <typename T>
__host__ 
T defectComp(T *d, int ld_d){
	T maxD = 0;
	#pragma unroll
	for (int i = 0; i < NUM_TIME_STEPS; i++){
		if (onDefectBoundary(i)){
			T currD = 0;
			#pragma unroll
			for (int j = 0; j < DIM_d_r; j++){currD += abs(d[i*ld_d + j]);}
			if(currD > maxD){currD = maxD;}
		}
	}
	return maxD;
}

// cost kern using external costFunc
template <typename T>
__global__
void costKern(T **d_x, T **d_u, T *d_JT, T *d_xg, int ld_x, int ld_u){
   	auto s_J = shared_memory_proxy<T>();
   	#pragma unroll
   	for (int a = blockIdx.x; a < NUM_ALPHA; a += gridDim.x){
   		// compute cost load into shared memory for reduction
      	s_J[threadIdx.x] = 0.0; T *xa = d_x[a]; T *ua = d_u[a];
      	#pragma unroll
      	for (int k = threadIdx.x; k < NUM_TIME_STEPS; k += blockDim.x){
      		T *xk = &xa[k*ld_x]; 	T *uk = &ua[k*ld_u];
      		s_J[threadIdx.x] += costFunc(xk,uk,d_xg,k);
      	}
      	__syncthreads();
   		// then sum it all up per alpha with a reduce pattern
      	reduceSum<T>(s_J);
   		if (threadIdx.x == 0){d_JT[a] = s_J[0];}
   		__syncthreads();
   	}
}

// cost kern just for summing over M_F or NUM_TIME_STEPS
template <typename T, int MODE>
__global__
void costKern(T *d_JT){
	auto s_JT = shared_memory_proxy<T>();
	// mode 0 implies that we have computed the cost in the forward pass and need to sum
	//        up per block across M_F shooting intervals (extern shared mem is size 0)
	if (MODE == 0){
		int alpha = threadIdx.x;
		T J = 0;
		#pragma unroll
		for (int i=0; i<M_F; i++){J += d_JT[alpha*M_F + i];}
		__syncthreads();
		d_JT[alpha] = J;
	}
	else if (MODE == 1){
	// mode 1 implies that we have computed the cost in the gradient/Hessian comp and need
	//        to sum up per block across all NUM_TIME_STEPS (must launch with extern shared mem)
	//        note this also only happens in the init so only one alpha block
		s_JT[threadIdx.x] = 0.0;
		for (int k = threadIdx.x; k < NUM_TIME_STEPS; k += blockDim.x){s_JT[threadIdx.x] += d_JT[k];}
		__syncthreads();
		// sum it all up per alpha with a reduce pattern
		reduceSum<T>(s_JT);
		__syncthreads();
		if (threadIdx.x == 0){d_JT[0] = s_JT[0];}
	}
	// else bad mode
	else{
		printf("ERROR: invalid mode for costKern_v2 usage is 0:M_F sum 1:NUM_TIME_STEPS sum 2:NUM_TIME_STEPS comp. Note: need extern shared mem for 1,2.");
	}
}

template <typename T>
__host__
void costThreaded(threadDesc_t desc, T *x, T *u, T *JT, T *xg, int ld_x, int ld_u){
	JT[desc.tid] = 0;
	for (unsigned int i=0; i<desc.reps; i++){
		int k = desc.tid + i*desc.dim;
		T *xk = &x[k*ld_x]; 	T *uk = &u[k*ld_u];
  		JT[desc.tid] += costFunc(xk,uk,xg,k);
  	}
}

template <typename T>
__host__ __device__ __forceinline__
void computeControlKT(T *u, T *x, T *xp, T *KT, T *du, T alpha, T *s_dx, int ld_KT, T *s_u = nullptr, T *s_x = nullptr){
	int start, delta; singleLoopVals(&start,&delta);
	#pragma unroll
	for (int ind = start; ind < STATE_SIZE; ind += delta){
		T val = x[ind]; s_dx[ind] = val-xp[ind];
		if (s_x != nullptr){s_x[ind] = val;}
	}
	hd__syncthreads();
	// compute the new control: u = u - alpha*du - K(xnew-x) -- but note we have KT not K
	#pragma unroll
	for (int r = start; r < CONTROL_SIZE; r += delta){
		// get the Kdx for this row
		T Kdx = 0;
		#pragma unroll
		for (int c = 0; c < STATE_SIZE; c++){Kdx += KT[c + r*ld_KT]*s_dx[c];}
		// and then get this control with it
		u[r] -= alpha*du[r] + Kdx;
		if (s_u != nullptr){s_u[r] = u[r];}
	}
}

template <typename T>
__host__ __device__ __forceinline__
void forwardSimInner(T *x, T *u, T *KT, T *du, T *d, T alpha, T *xp, T *s_dx, T *s_qdd, T dt, int bInd, int ld_x, int ld_u, int ld_KT, int ld_du, int ld_d, \
					 T *d_I = nullptr, T *d_Tbody = nullptr, T* s_x = nullptr, T *s_u = nullptr, T *s_cost = nullptr, T *s_eePos = nullptr, T *d_xGoal = nullptr){
	int start, delta; singleLoopVals(&start,&delta);
	int kStart = bInd * N_F;
	unsigned int iters = EE_COST ? N_F : (bInd < M_F - 1 ? N_F : N_F - 1);
	T *xk = &x[kStart*ld_x];
	T *xkp1 = xk + ld_x;
	T *uk = &u[kStart*ld_u];
	T *xpk = &xp[kStart*ld_x];
	T *KTk = &KT[kStart*ld_KT*DIM_KT_c];
	T *duk = &du[kStart*ld_du];
	T *dk = &d[((bInd+1)*N_F-1)*ld_d]; // will only be used once on boundary so lock in
	bool tempFlag = (s_x != nullptr) && (s_u != nullptr);
	for(unsigned int k=0; k < iters; k++){
		// load in the x and u and compute controls
		computeControlKT<T>(uk,xk,xpk,KTk,duk,alpha,s_dx,ld_KT,s_u,s_x);
		hd__syncthreads();
		// then use this control to compute the new state
		T *s_xkp1 = s_dx; // re-use this shared mem as we are done with it for this loop
		if(tempFlag){_integrator<T>(s_xkp1,s_x,s_u,s_qdd,d_I,d_Tbody,dt,s_eePos);}
		else        {_integrator<T>(s_xkp1,xk,uk,s_qdd,d_I,d_Tbody,dt,s_eePos);}
		hd__syncthreads();
		// then write to global memory unless "final" state where we just use for defect on boundary
		#pragma unroll
		for (int ind = start; ind < STATE_SIZE; ind += delta){
			if (k < N_F - 1){xkp1[ind] = s_xkp1[ind];}
			else if (bInd < M_F - 1){dk[ind] = s_xkp1[ind] - xkp1[ind];}
		}
		#if EE_COST
			// Also compute running / final cost if needed (note not on defect "final" states)
			if (k < N_F - 1 || bInd == M_F - 1){
				if(tempFlag){costFunc(s_cost,s_eePos,d_xGoal,s_x,s_u,bInd*N_F+k);}
				else{		 costFunc(s_cost,s_eePos,d_xGoal,xk,uk,bInd*N_F+k);}
			}
		#endif
		// update the offsets for the next pass
		xk    = xkp1;
		xkp1 += ld_x;
		uk   += ld_u;
		xpk  += ld_x;
		KTk  += ld_KT*DIM_KT_c;
		duk  += ld_du;
		hd__syncthreads();
	}
}

template <typename T>
__global__
void forwardSimKern(T **d_x, T **d_u, T *d_KT, T *d_du, T **d_d, T *d_alpha, \
					T *d_xp, int ld_x, int ld_u, int ld_KT, int ld_du, int ld_d, \
					T *d_I = nullptr, T *d_Tbody = nullptr, T *d_xGoal = nullptr, T *d_JT = nullptr){
	__shared__ T s_x[STATE_SIZE];	__shared__ T s_u[NUM_POS];	__shared__ T s_qdd[NUM_POS];	__shared__ T s_dx[STATE_SIZE];
	int alphaInd = blockIdx.y;		int bInd = blockIdx.x;
	// zero out the cost and set up to track eePos if in EE_COST scenario
	#if EE_COST
		__shared__ T s_cost[NUM_POS];	__shared__ T s_eePos[6];	zeroSharedMem<T,NUM_POS>(s_cost);
	#else
		T *s_cost = nullptr;	T *s_eePos = nullptr;
	#endif
	// loop forward and compute new states and controls
	forwardSimInner(d_x[alphaInd],d_u[alphaInd],d_KT,d_du,d_d[alphaInd],d_alpha[alphaInd],d_xp,s_dx,s_qdd,
					(T)TIME_STEP,bInd,ld_x,ld_u,ld_KT,ld_du,ld_d,d_I,d_Tbody,s_x,s_u,s_cost,s_eePos,d_xGoal);
	#if EE_COST // sum up the total cost
		if (threadIdx.y == 0 && threadIdx.x == 0){d_JT[bInd + alphaInd*M_F] = s_cost[0]+s_cost[1]+s_cost[2]+s_cost[3]+s_cost[4]+s_cost[5]+s_cost[6];}
	#endif
}

template <typename T>
__host__
void forwardSim(threadDesc_t desc, T *x, T *u, T *KT, T *du, T *d, T alpha, T *xp, \
				int ld_x, int ld_u, int ld_KT, int ld_du, int ld_d, \
				T *I = nullptr, T *Tbody = nullptr, T *xGoal = nullptr, T *JT = nullptr){
	T *s_x = nullptr;		T *s_u = nullptr;		T s_qdd [NUM_POS];		T s_dx[STATE_SIZE];
	// zero out the cost and set up to track eePos if EE_COST scenario
	#if EE_COST
		T s_cost[NUM_POS];	T s_eePos[6];	zeroSharedMem<T,NUM_POS>(s_cost);
	#else
		T *s_cost = nullptr; 	T *s_eePos = nullptr;
	#endif
	// loop forward and coT mpute new states and controls
	for (unsigned int i=0; i<desc.reps; i++){
  		int bInd = (desc.tid+i*desc.dim);
		forwardSimInner(x,u,KT,du,d,alpha,xp,s_dx,s_qdd,(T)TIME_STEP,bInd,ld_x,ld_u,ld_KT,ld_du,ld_d,I,Tbody,s_x,s_u,s_cost,s_eePos,xGoal);
	}
	#if EE_COST // sum up the total cost
		JT[desc.tid] = s_cost[0]+s_cost[1]+s_cost[2]+s_cost[3]+s_cost[4]+s_cost[5]+s_cost[6];
	#endif
}

// template <typename T>
// __host__
// void forwardSim(threadDesc_t desc, T *x, T *u, T *KT, T *du, T *d, T alpha, T *xp, 
// 				int ld_x, int ld_u, int ld_KT, int ld_du, int ld_d, 
// 				T *I = nullptr, T *Tbody = nullptr, T *xGoal = nullptr, T *JT = nullptr){
// 	T *s_x = nullptr;		T *s_u = nullptr;		T *tempMem;
// 	// zero out the cost and set up to track eePos if EE_COST scenario
// 	#if EE_COST
// 		tempMem = (T *)malloc((NUM_POS+STATE_SIZE+NUM_POS+6)*sizeof(T));
// 		T *s_qdd = &tempMem[0]						T *s_dx = &tempMem[NUM_POS];	
// 		T *s_cost = &tempMem[NUM_POS+STATE_SIZE];	T *s_eePos&tempMem[NUM_POS+STATE_SIZE+NUM_POS];		zeroSharedMem<T,NUM_POS>(s_cost);
// 	#else
// 		tempMem = (T *)malloc((NUM_POS+STATE_SIZE)*sizeof(T));
// 		T *s_qdd = &tempMem[0];		T *s_dx = &tempMem[NUM_POS];	
// 		T *s_cost = nullptr; 		T *s_eePos = nullptr;
// 	#endif
// 	// loop forward and coT mpute new states and controls
// 	for (int i=0; i<desc.reps; i++){
//   		int bInd = (desc.tid+i*desc.dim);
// 		forwardSimInner(x,u,KT,du,d,alpha,xp,s_dx,s_qdd,(T)TIME_STEP,bInd,ld_x,ld_u,ld_KT,ld_du,ld_d,I,Tbody,s_x,s_u,s_cost,s_eePos,xGoal);
// 	}
// 	#if EE_COST // sum up the total cost
// 		JT[desc.tid] = s_cost[0]+s_cost[1]+s_cost[2]+s_cost[3]+s_cost[4]+s_cost[5]+s_cost[6];
// 	#endif
// 	free(tempMem);
// }

template <typename T>
__host__ __forceinline__
void forwardSimGPU(T **d_x, T *d_xp, T *d_xp2, T **d_u, T *d_KT, T *d_du, T *alpha, T *d_alpha, T *d, T **d_d, T *d_dT, T *dJexp, T *d_dJexp, \
                   T *J, T *d_JT, T *d_xGoal, T *dJ, T *z, T prevJ, cudaStream_t *streams, dim3 dynDimms, dim3 FPBlocks, int *alphaIndex, \
                   int *ignore_defect, int ld_x, int ld_u, int ld_KT, int ld_du, int ld_d, T *d_I = nullptr, T *d_Tbody = nullptr){
	// ACTUAL FORWARD SIM //
	forwardSimKern<T><<<FPBlocks,dynDimms,0,streams[0]>>>(d_x,d_u,d_KT,d_du,d_d,d_alpha,d_xp,ld_x,ld_u,ld_KT,ld_du,ld_d,d_I,d_Tbody,d_xGoal,d_JT);
	gpuErrchk(cudaPeekAtLastError());

	// while we are doing these (very expensive) operation save xp into xp2 which we'll need for the backward pass linear transform on the next iter
	gpuErrchk(cudaMemcpyAsync(d_xp2,d_xp,ld_x*DIM_x_c*NUM_TIME_STEPS*sizeof(T),cudaMemcpyDeviceToDevice,streams[3]));

	// also while waiting on the GPU memcpy the expected reduction back that was computed during the back pass and sum
	gpuErrchk(cudaMemcpyAsync(dJexp, d_dJexp, 2*M_B*sizeof(T), cudaMemcpyDeviceToHost, streams[2]));
	gpuErrchk(cudaStreamSynchronize(streams[2]));
	for (int i=1; i<M_B; i++){dJexp[0] += dJexp[2*i]; dJexp[1] += dJexp[2*i+1];}
	gpuErrchk(cudaDeviceSynchronize());
	// ACTUAL FORWARD SIM //

	// LINE SEARCH //
	// then compute the cost and defect once the forward simulation finishes
	#if !EE_COST
		costKern<T><<<NUM_ALPHA,NUM_TIME_STEPS,NUM_TIME_STEPS*sizeof(T),streams[0]>>>(d_x,d_u,d_JT,d_xGoal,ld_x,ld_u);
	#else
		costKern<T,0><<<1,NUM_ALPHA,0,streams[0]>>>(d_JT);
	#endif
	gpuErrchk(cudaPeekAtLastError());
	if (M_F > 1){defectKern<<<NUM_ALPHA,NUM_TIME_STEPS,0,streams[1]>>>(d_d,d_dT,ld_d);	gpuErrchk(cudaPeekAtLastError());}

	// then find the best J that shows improvement
	// since NUM_ALPHA <= 32 (usually) the overhead in launching a kernel will outweigh the gains of logN comps vs N serial comps
	gpuErrchk(cudaStreamSynchronize(streams[0]));
	gpuErrchk(cudaMemcpyAsync(J, d_JT, NUM_ALPHA*sizeof(T), cudaMemcpyDeviceToHost, streams[0]));
	if (M_F > 1){gpuErrchk(cudaStreamSynchronize(streams[1])); gpuErrchk(cudaMemcpyAsync(d, d_dT, NUM_ALPHA*sizeof(T), cudaMemcpyDeviceToHost, streams[1]));}
	*dJ = -1.0; *z = 0.0; T cdJ = -1.0; T cz = 0.0; bool JFlag, zFlag, dFlag;
	gpuErrchk(cudaDeviceSynchronize());
	for (int i=0; i<NUM_ALPHA; i++){
		cdJ = prevJ - J[i]; JFlag = cdJ >= 0.0 && cdJ > *dJ;
		cz = cdJ / (alpha[i]*dJexp[0] + alpha[i]*alpha[i]/2.0*dJexp[1]); zFlag = USE_EXP_RED ? (EXP_RED_MIN < cz && cz < EXP_RED_MAX) : 1;
		dFlag = M_F == 1 || *ignore_defect ? 1 :  d[i] < MAX_DEFECT_SIZE;
		//printf("Alpha[%f] -> dJ[%f] -> z[%f], d[%f] so flags are J[%d]z[%d]f[%d] vs bdJ[%f]\n",alpha[i],cdJ,cz,d[i],JFlag,zFlag,dFlag,*dJ);
		if(JFlag && zFlag && dFlag){
			if (d[i] < USE_MAX_DEFECT){*ignore_defect = 0;} // update the ignore defect
			*alphaIndex = i; *dJ = cdJ; *z = cz; // update current best index, dJ, z
			if (!ALPHA_BEST_SWITCH){break;} // pick first alpha strategy   
		}
	}
	// LINE SEARCH //
}

template <typename T>
__host__ __forceinline__
int forwardSimCPU(T *x, T *xp, T *xp2, T *u, T *up, T *KT, T *du, T *d, T *dp, T *dJexp, T *JT, T alpha, \
                  T *xGoal, T *J, T *dJ, T *z, T prevJ, int *ignore_defect, T *maxd, std::thread *threads, \
                  int ld_x, int ld_u, int ld_KT, int ld_du, int ld_d, T *I = nullptr, T *Tbody = nullptr){
	// ACTUAL FORWARD SIM //
	// printf("computing alpha[%f]\n",alpha);
	*J = 0; 	threadDesc_t desc; 		desc.dim = FSIM_THREADS;
	for (unsigned int thread_i = 0; thread_i < FSIM_THREADS; thread_i++){
        desc.tid = thread_i;   desc.reps = compute_reps(thread_i,FSIM_THREADS,M_F);
    	threads[thread_i] = std::thread(&forwardSim<T>, desc, std::ref(x), std::ref(u), std::ref(KT), std::ref(du), std::ref(d), alpha, std::ref(xp), 
    													ld_x, ld_u, ld_KT, ld_du, ld_d, std::ref(I), std::ref(Tbody), std::ref(xGoal), std::ref(JT));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, thread_i);}
    }
    // while we are doing these (very expensive) operation save xp into xp2 which we'll need for the backward pass linear transform
    // and sum expCost -- note: only do this the first time
    if (alpha == 1.0){
        threads[max(FSIM_THREADS,COST_THREADS)] = std::thread(memcpy, std::ref(xp2), std::ref(xp), ld_x*NUM_TIME_STEPS*sizeof(T));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 0);} 
        for (unsigned int i=1; i<BP_THREADS; i++){dJexp[0] += dJexp[2*i]; dJexp[1] += dJexp[2*i+1];}
    }
	for (unsigned int thread_i = 0; thread_i < FSIM_THREADS; thread_i++){threads[thread_i].join();}
	// ACTUAL FORWARD SIM //

	// LINE SEARCH //
    // then compute the cost once the forward simulation finishes (already comped in FSIM if EE cost)
    bool JFlag, zFlag, dFlag;
    // if not ee cost need to launch cost threaded and sum across threads
    #if !EE_COST
        desc.dim = COST_THREADS;
        for (unsigned int thread_i = 0; thread_i < COST_THREADS; thread_i++){
            desc.tid = thread_i;   desc.reps = compute_reps(thread_i,COST_THREADS,NUM_TIME_STEPS);
            threads[thread_i] = std::thread(&costThreaded<T>, desc, std::ref(x), std::ref(u), std::ref(JT), std::ref(xGoal), ld_x, ld_u);
            if(FORCE_CORE_SWITCHES){setCPUForThread(threads, thread_i);}
        }     
        for (unsigned int thread_i = 0; thread_i < COST_THREADS; thread_i++){threads[thread_i].join(); *J += JT[thread_i];}
	#else
		for (unsigned int thread_i = 0; thread_i < FSIM_THREADS; thread_i++){*J += JT[thread_i];}
	#endif
    // don't forget to join the xp2 copy if applicable
    if (alpha == 1.0){threads[max(FSIM_THREADS,COST_THREADS)].join();}
        
	// if J satisfies line search criteria
    *dJ = prevJ - *J; 											JFlag = *dJ >= 0.0;
	*z = *dJ / (alpha*dJexp[0] + alpha*alpha/2.0*dJexp[1]); 	zFlag = !USE_EXP_RED || (EXP_RED_MIN < *z && *z < EXP_RED_MAX);
    if (M_F > 1){*maxd = defectComp(d,ld_d); dFlag = *maxd < MAX_DEFECT_SIZE;} else{dFlag = 1;}
    // printf("Alpha[%f] -> J[%f]dJ[%f] -> z[%f], d[%f] so flags are J[%d]z[%d]f[%d]\n",alpha,*J,*dJ,*z,*maxd,JFlag,zFlag,dFlag);
    if(JFlag && zFlag && dFlag){
		if (*ignore_defect && *maxd < USE_MAX_DEFECT){*ignore_defect = 0;}
		return 0;
    }
    // else fails so need to restore x to xp and u to up (and d to dp if M_F > 1)
    else{
        threads[0] = std::thread(memcpy, std::ref(x), std::ref(xp), ld_x*NUM_TIME_STEPS*sizeof(T));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 0);}
        threads[1] = std::thread(memcpy, std::ref(u), std::ref(up), ld_u*NUM_TIME_STEPS*sizeof(T));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 1);}
        if (M_F > 1){
            threads[2] = std::thread(memcpy, std::ref(d), std::ref(dp), ld_d*NUM_TIME_STEPS*sizeof(T));
            if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 2);}
        }
        threads[0].join();
        threads[1].join();
        if (M_F > 1){threads[2].join();}
        return 1;
    }
    // LINE SEARCH //
}

// xs, us, ds as ** and JT longer (alpha*max(cost,fsim)threads in size) also pass in all alphas and startAlpha
template <typename T>
__host__ __forceinline__
int forwardSimCPU2(T **xs, T *xp, T *xp2, T **us, T *up, T *KT, T *du, T **ds, T *dp, T *dJexp, T **JTs, T *alphas, int startAlpha, \
				   T *xGoal, T *J, T *dJ, T *z, T prevJ, int *ignore_defect, T *maxd, std::thread *threads, \
				   int ld_x, int ld_u, int ld_KT, int ld_du, int ld_d, T *I = nullptr, T *Tbody = nullptr){
	// ACTUAL FORWARD SIM //
	*J = 0; 	T Js[FSIM_ALPHA_THREADS];		for(unsigned int i = 0; i < FSIM_ALPHA_THREADS; i++){Js[i] = 0;}
	threadDesc_t desc; 		desc.dim = FSIM_ALPHA_THREADS;	unsigned int threads_launched = 0;
	for (unsigned int alpha_i = 0; alpha_i < FSIM_ALPHA_THREADS; alpha_i++){
		unsigned int alphaInd = startAlpha + alpha_i; 	if(alphaInd >= NUM_ALPHA){break;}
		for (unsigned int thread_i = 0; thread_i < FSIM_ALPHA_THREADS; thread_i++){
	        desc.tid = thread_i;   desc.reps = compute_reps(thread_i,FSIM_ALPHA_THREADS,M_F);
	    	threads[alpha_i*FSIM_ALPHA_THREADS+thread_i] = std::thread(&forwardSim<T>, desc, std::ref(xs[alphaInd]), std::ref(us[alphaInd]), std::ref(KT), std::ref(du), std::ref(ds[alphaInd]), alphas[alphaInd], std::ref(xp), 
	    													ld_x, ld_u, ld_KT, ld_du, ld_d, std::ref(I), std::ref(Tbody), std::ref(xGoal), std::ref(JTs[alphaInd]));
	        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, alpha_i*FSIM_ALPHA_THREADS+thread_i);}	threads_launched++;
	    }
	}
    // while we are doing these (very expensive) operation save xp into xp2 which we'll need for the backward pass linear transform
    // and sum expCost -- note: only do this the first time
    if (startAlpha == 0){
        threads[FSIM_ALPHA_THREADS*max(FSIM_ALPHA_THREADS,COST_THREADS)] = std::thread(memcpy, std::ref(xp2), std::ref(xp), ld_x*NUM_TIME_STEPS*sizeof(T));
        if(FORCE_CORE_SWITCHES){setCPUForThread(threads, 0);} 
        for (unsigned int i=1; i<BP_THREADS; i++){dJexp[0] += dJexp[2*i]; dJexp[1] += dJexp[2*i+1];}
    }
	for (unsigned int thread_i = 0; thread_i < threads_launched; thread_i++){threads[thread_i].join();}
	// ACTUAL FORWARD SIM //

	// LINE SEARCH //
    // then compute the cost once the forward simulation finishes (already comped in FSIM if EE cost)
    // if not EE cost need to launch cost threaded and sum across threads
    #if !EE_COST
		for (unsigned int alpha_i = 0; alpha_i < FSIM_ALPHA_THREADS; alpha_i++){
			unsigned int alphaInd = startAlpha + alpha_i;	if(alphaInd >= NUM_ALPHA){break;}
	        desc.dim = COST_THREADS;
	        for (unsigned int thread_i = 0; thread_i < COST_THREADS; thread_i++){
	            desc.tid = thread_i;   desc.reps = compute_reps(thread_i,COST_THREADS,NUM_TIME_STEPS);
	            threads[alpha_i*COST_THREADS+thread_i] = std::thread(&costThreaded<T>, desc, std::ref(xs[alphaInd]), std::ref(us[alphaInd]), std::ref(JTs[alphaInd]), std::ref(xGoal), ld_x, ld_u);
	            if(FORCE_CORE_SWITCHES){setCPUForThread(threads, alpha_i*COST_THREADS+thread_i);}
	        }     
	    }
	    for (unsigned int alpha_i = 0; alpha_i < FSIM_ALPHA_THREADS; alpha_i++){
	    	unsigned int alphaInd = startAlpha + alpha_i;	if(alphaInd >= NUM_ALPHA){break;}	T *JT = JTs[alphaInd];
	    	for (unsigned int thread_i = 0; thread_i < COST_THREADS; thread_i++){threads[alpha_i*COST_THREADS+thread_i].join(); Js[alpha_i] += JT[thread_i];}
        }
	#else
		for (unsigned int alpha_i = 0; alpha_i < FSIM_ALPHA_THREADS; alpha_i++){
			unsigned int alphaInd = startAlpha + alpha_i;	if(alphaInd >= NUM_ALPHA){break;}	T *JT = JTs[alphaInd];
			for (unsigned int thread_i = 0; thread_i < FSIM_ALPHA_THREADS; thread_i++){Js[alpha_i] += JT[thread_i];}
		}
	#endif
    // don't forget to join the xp2 copy if applicable
    if (startAlpha == 0){threads[FSIM_ALPHA_THREADS*max(FSIM_ALPHA_THREADS,COST_THREADS)].join();}     

	// if J satisfies line search criteria
	T cdJ = -1.0; T cz = 0.0; T cd = 0.0; int alphaIndex = -1; bool JFlag, zFlag, dFlag;
	for (unsigned int alpha_i = 0; alpha_i < FSIM_ALPHA_THREADS; alpha_i++){
		unsigned int alphaInd = startAlpha + alpha_i;	if(alphaInd >= NUM_ALPHA){break;}	T alpha = alphas[alphaInd];					
		T cJ = Js[alpha_i];		cdJ = prevJ - cJ;					JFlag = cdJ >= 0.0 && cdJ > *dJ;
		cz = cdJ / (alpha*dJexp[0] + alpha*alpha/2.0*dJexp[1]); 	zFlag = !USE_EXP_RED || (EXP_RED_MIN < cz && cz < EXP_RED_MAX);
	    if (M_F > 1){cd = defectComp(ds[alphaInd],ld_d); dFlag = cd < MAX_DEFECT_SIZE;} else{dFlag = 1;}
	    // printf("Alpha[%f] -> J[%f]dJ[%f] -> z[%f], d[%f] so flags are J[%d]z[%d]f[%d]\n",alpha,cJ,cdJ,cz,cd,JFlag,zFlag,dFlag);
	    if(JFlag && zFlag && dFlag){
			if (*ignore_defect && cd < USE_MAX_DEFECT){*ignore_defect = 0;} // update the ignore defect
			alphaIndex = alphaInd; *dJ = cdJ; *z = cz; *J = Js[alpha_i]; *maxd = cd; // update current best index, dJ, z, J, maxd
			if (!ALPHA_BEST_SWITCH){break;} // pick first alpha strategy   
	    }
	}
    // else failed
    return alphaIndex; //-1 on failure
    // LINE SEARCH //
}