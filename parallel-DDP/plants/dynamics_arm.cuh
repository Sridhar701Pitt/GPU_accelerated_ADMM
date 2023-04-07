/****************************************************************
 * CUDA Rigid Body DYNamics
 *
 * Based on the Joint Space Inversion Algorithm
 * currently special cased for 7dof Kuka Arm
 *
 * initI(T *s_I), initT(T *s_T)
 *
 * compute_eePos(T *s_T, T *s_eePos, T *s_dT, T *s_deePos, T *s_sinq, T *s_Tb, T *s_dTb, T *s_x, T *s_cosq, T *d_Tbody);
 *   (load_Tb, compute_T_TA_J, compute_dT_dTA_dJ)
 *
 * dynamics(T *s_qdd, T *s_x, T *s_u, T *d_I, T *d_Tbody, T *s_eePos = nullptr, int reps = 1);
 *    load_Tb, load_I
 *    compute_T_TA_J
 *      loadAdjoint
 *    (compute_eePos)
 *    compute_Iw_Icrbs_twist
 *    compute_JdotV
 *      crfm
 *    compute_M_Tau
 *      crfm
 *    invertMatrix
 *    compute_qdd
 *
 * dynamicsGradient(T *s_dqdd, T *s_qdd, T *s_x, T *s_u, T *d_I, T *d_Tbody);
 *    load_Tb, load_I
 *    compute_T_TA_J, compute_dT_dTA_dJ
 *      loadAdjoint
 *    (compute_eePos)
 *    compute_Iw_Icrbs_twist
 *    compute_JdotV
 *      crfm
 *    compute_M_Tau
 *      crfm
 *    invertMatrix, compute_qdd
 *    compute_dM, compute_dqdd_dM
 *    compute_dtwist, compute_dJdotV, compute_dWb
 *       crfm, crfmz
 *    compute_dTau, finish_dqdd
 *****************************************************************/

#ifdef MPC_MODE
   #define GRAVITY 0.0 // Kuka arm does automatic gravity compensation on top of torques sent so assume 0
#else
   #define GRAVITY 9.81 // in full iLQR sim we are trying to come up with something assuming no gravity comp or anything hardware related
#endif
/*** PLANT SPECIFIC RBDYN HELPERS ***/
#define USE_OLD 1
template <typename T>
__host__ __device__ 
void initI(T *s_I){
    s_I[0] = 0.121128;
    s_I[4] = -0.6912;
    s_I[5] = -0.1728;
    s_I[7] = 0.116244;
    s_I[8] = 0.025623;
    s_I[9] = 0.6912;
    s_I[13] = 0.025623;
    s_I[14] = 0.017484;
    s_I[15] = 0.1728;
    s_I[19] = 0.6912;
    s_I[20] = 0.1728;
    s_I[21] = 5.76;
    s_I[24] = -0.6912;
    s_I[28] = 5.76;
    s_I[30] = -0.1728;
    s_I[35] = 5.76;
    s_I[36] = 0.06380575;
    s_I[37] = -0.000112395;
    s_I[38] = -0.00008001;
    s_I[40] = -0.2667;
    s_I[41] = 0.37465;
    s_I[42] = -0.000112395;
    s_I[43] = 0.0416019715;
    s_I[44] = -0.0108483;
    s_I[45] = 0.2667;
    s_I[47] = -0.001905;
    s_I[48] = -0.00008001;
    s_I[49] = -0.0108483;
    s_I[50] = 0.0331049215;
    s_I[51] = -0.37465;
    s_I[52] = 0.001905;
    s_I[55] = 0.2667;
    s_I[56] = -0.37465;
    s_I[57] = 6.35;
    s_I[60] = -0.2667;
    s_I[62] = 0.001905;
    s_I[66] = 0.37465;
    s_I[67] = -0.001905;
    s_I[71] = 6.35;
    s_I[72] = 0.0873;
    s_I[76] = -0.455;
    s_I[77] = 0.105;
    s_I[79] = 0.08295;
    s_I[80] = -0.00878;
    s_I[81] = 0.455;
    s_I[85] = -0.00878;
    s_I[86] = 0.01075;
    s_I[87] = -0.105;
    s_I[64] = 6.35;
    s_I[91] = 0.455;
    s_I[92] = -0.105;
    s_I[93] = 3.5;
    s_I[96] = -0.455;
    s_I[100] = 3.5;
    s_I[102] = 0.105;
    s_I[107] = 3.5;
    s_I[108] = 0.0367575;
    s_I[112] = -0.119;
    s_I[113] = 0.2345;
    s_I[115] = 0.020446;
    s_I[116] = -0.005133;
    s_I[117] = 0.119;
    s_I[121] = -0.005133;
    s_I[122] = 0.0217115;
    s_I[123] = -0.2345;
    s_I[127] = 0.119;
    s_I[128] = -0.2345;
    s_I[129] = 3.5;
    s_I[132] = -0.119;
    s_I[136] = 3.5;
    s_I[138] = 0.2345;
    s_I[143] = 3.5;
    s_I[144] = 0.0317595;
    s_I[145] = -0.00000735;
    s_I[146] = -0.0000266;
    s_I[148] = -0.266;
    s_I[149] = 0.0735;
    s_I[150] = -0.00000735;
    s_I[151] = 0.028916035;
    s_I[152] = -0.002496;
    s_I[153] = 0.266;
    s_I[155] = -0.00035;
    s_I[156] = -0.0000266;
    s_I[157] = -0.002496;
    s_I[158] = 0.006033535;
    s_I[159] = -0.0735;
    s_I[160] = 0.00035;
    s_I[163] = 0.266;
    s_I[164] = -0.0735;
    s_I[165] = 3.5;
    s_I[168] = -0.266;
    s_I[170] = 0.00035;
    s_I[172] = 3.5;
    s_I[174] = 0.0735;
    s_I[175] = -0.00035;
    s_I[179] = 3.5;
    s_I[180] = 0.004900936;
    s_I[184] = -0.00072;
    s_I[185] = 0.00108;
    s_I[187] = 0.004700288;
    s_I[188] = 0.000245568;
    s_I[189] = 0.00072;
    s_I[193] = 0.000245568;
    s_I[194] = 0.003600648;
    s_I[195] = -0.00108;
    s_I[199] = 0.00072;
    s_I[200] = -0.00108;
    s_I[201] = 1.8;
    s_I[204] = -0.00072;
    s_I[208] = 1.8;
    s_I[210] = 0.00108;
    s_I[215] = 1.8;
    s_I[216] = 0.10732607081630547718464896433943;
    s_I[217] = 0.00019989199349797965904636243283932;
    s_I[218] = -0.0053829202963780343332844680048765;
    s_I[219] = -0.000000000000000000000000086572214906444569626737234912971;
    s_I[220] = -0.4471958408698656350921396551712;
    s_I[221] = -0.00000026790984812577454527075348882093;
    s_I[222] = 0.00019989199349797960483625380856409;
    s_I[223] = 0.098340754312488648514190003879776;
    s_I[224] = 0.000029124896308221664471436659904491;
    s_I[225] = 0.4471958408698656350921396551712;
    s_I[226] = -0.000000000000000000000037314314248140254118040921485142;
    s_I[227] = 0.012494963104108957122062584232935;
    s_I[228] = -0.005382920296378033465922730016473;
    s_I[229] = 0.000029124896308221881311871157005378;
    s_I[230] = 0.12444928521768228169008807526552;
    s_I[231] = 0.00000026790984812750932168628949930911;
    s_I[232] = -0.012494963104108946713721728372093;
    s_I[233] = -0.0000000000000000000033881317890178754317538255352671;
    s_I[234] = -0.0000000000000000000016941561609528859931138471397541;
    s_I[235] = 0.4471958408698656350921396551712;
    s_I[236] = 0.00000026790984812750932168628949930911;
    s_I[237] = 6.4;
    s_I[238] = 0.00000000000000000000000091178226550731791047478487008133;
    s_I[239] = 0.00000000000000000010842021729662189555389373538375;
    s_I[240] = -0.4471958408698656350921396551712;
    s_I[241] = -0.000000000000000000000051018957034122961580356685370858;
    s_I[242] = -0.0124949631041089484484452043489;
    s_I[243] = 0.00000000000000000000000087446681723681373628459234511013;
    s_I[244] = 6.4;
    s_I[245] = -0.00000000000000000000084498302793689761931173291132555;
    s_I[246] = -0.00000026790984812924404516226630640352;
    s_I[247] = 0.012494963104108955387339108256128;
    s_I[248] = -0.0000000000000000000000000000000004;
    s_I[249] = 0.00000000000000000011;
    s_I[250] = -0.0000000000000000000007;
    s_I[251] = 6.4;
}

template <typename T>
__host__ __device__ 
void initT(T *s_T){
    #pragma unroll
    for (int i = 0; i < 36*7; i++){s_T[i] = 0.0;}
    s_T[10] = 1.0;
    s_T[14] = 0.1575;
    s_T[15] = 1.0;
    s_T[44] = -0.00000000000020682;
    s_T[45] = 1.0;
    s_T[46] = 0.0000000000048966;
    s_T[50] = 0.2025;
    s_T[51] = 1.0;
    s_T[80] = -0.00000000000020682;
    s_T[81] = 1.0;
    s_T[82] = 0.0000000000048966;
    s_T[85] = 0.2045;
    s_T[87] = 1.0;
    s_T[117] = -1.0;
    s_T[118] = 0.0000000000048966;
    s_T[122] = 0.2155;
    s_T[123] = 1.0;
    s_T[152] = -0.0000000000000000000000010127;
    s_T[153] = 1.0;
    s_T[154] = -0.0000000000048966;
    s_T[157] = 0.1845;
    s_T[159] = 1.0;
    s_T[189] = -1.0;
    s_T[190] = 0.0000000000048966;
    s_T[194] = 0.2155;
    s_T[195] = 1.0;
    s_T[224] = -0.0000000000000000000000010127;
    s_T[225] = 1.0;
    s_T[226] = -0.0000000000048966;
    s_T[229] = 0.081;
    s_T[231] = 1.0;
}

template <typename T>
__host__ __device__ 
void updateT(T *s_T, T *s_cosx, T *s_sinx){
    s_T[0] = s_cosx[0];
    s_T[1] = s_sinx[0];
    s_T[4] = -s_sinx[0];
    s_T[5] = s_cosx[0];
    s_T[36] = 0.0000000000000000000000010127*s_sinx[1]-s_cosx[1];
    s_T[37] = -0.00000000000020682*s_cosx[1]-0.0000000000048966*s_sinx[1];
    s_T[38] = s_sinx[1];
    s_T[40] = 0.0000000000000000000000010127*s_cosx[1]+s_sinx[1];
    s_T[41] = 0.00000000000020682*s_sinx[1]-0.0000000000048966*s_cosx[1];
    s_T[42] = s_cosx[1];
    s_T[44] = -0.00000000000020682;
    s_T[72] = 0.0000000000000000000000010127*s_sinx[2]-s_cosx[2];
    s_T[73] = -0.00000000000020682*s_cosx[2]-0.0000000000048966*s_sinx[2];
    s_T[74] = s_sinx[2];
    s_T[76] = 0.0000000000000000000000010127*s_cosx[2]+s_sinx[2];
    s_T[77] = 0.00000000000020682*s_sinx[2]-0.0000000000048966*s_cosx[2];
    s_T[78] = s_cosx[2];
    s_T[108] = s_cosx[3];
    s_T[109] = 0.0000000000048966*s_sinx[3];
    s_T[110] = s_sinx[3];
    s_T[112] = -s_sinx[3];
    s_T[113] = 0.0000000000048966*s_cosx[3];
    s_T[114] = s_cosx[3];
    s_T[144] = 0.00000000000020682*s_sinx[4]-s_cosx[4];
    s_T[145] = 0.0000000000048966*s_sinx[4];
    s_T[146] = 0.00000000000020682*s_cosx[4]+s_sinx[4];
    s_T[148] = 0.00000000000020682*s_cosx[4]+s_sinx[4];
    s_T[149] = 0.0000000000048966*s_cosx[4];
    s_T[150] = -0.00000000000020682*s_sinx[4]+s_cosx[4];
    s_T[180] = s_cosx[5];
    s_T[181] = 0.0000000000048966*s_sinx[5];
    s_T[182] = s_sinx[5];
    s_T[184] = -s_sinx[5];
    s_T[185] = 0.0000000000048966*s_cosx[5];
    s_T[186] = s_cosx[5];
    s_T[216] = 0.00000000000020682*s_sinx[6]-s_cosx[6];
    s_T[217] = 0.0000000000048966*s_sinx[6];
    s_T[218] = 0.00000000000020682*s_cosx[6]+s_sinx[6];
    s_T[220] = 0.00000000000020682*s_cosx[6]+s_sinx[6];
    s_T[221] = 0.0000000000048966*s_cosx[6];
    s_T[222] = -0.00000000000020682*s_sinx[6]+s_cosx[6];
}

template <typename T>
__host__ __device__ 
void loadTdx4(T *s_Tdx, T *s_cosx, T *s_sinx){
    s_Tdx[0] = -s_sinx[0];
    s_Tdx[1] = s_cosx[0];
    s_Tdx[4] = -s_cosx[0];
    s_Tdx[5] = -s_sinx[0];
    s_Tdx[16] = 0.0000000000000000000000010127*s_cosx[1]+s_sinx[1];
    s_Tdx[17] = 0.00000000000020682*s_sinx[1]-0.0000000000048966*s_cosx[1];
    s_Tdx[18] = s_cosx[1];
    s_Tdx[20] = -0.0000000000000000000000010127*s_sinx[1]+s_cosx[1];
    s_Tdx[21] = 0.00000000000020682*s_cosx[1]+0.0000000000048966*s_sinx[1];
    s_Tdx[22] = -s_sinx[1];
    s_Tdx[32] = 0.0000000000000000000000010127*s_cosx[2]+s_sinx[2];
    s_Tdx[33] = 0.00000000000020682*s_sinx[2]-0.0000000000048966*s_cosx[2];
    s_Tdx[34] = s_cosx[2];
    s_Tdx[36] = -0.0000000000000000000000010127*s_sinx[2]+s_cosx[2];
    s_Tdx[37] = 0.00000000000020682*s_cosx[2]+0.0000000000048966*s_sinx[2];
    s_Tdx[38] = -s_sinx[2];
    s_Tdx[48] = -s_sinx[3];
    s_Tdx[49] = 0.0000000000048966*s_cosx[3];
    s_Tdx[50] = s_cosx[3];
    s_Tdx[52] = -s_cosx[3];
    s_Tdx[53] = -0.0000000000048966*s_sinx[3];
    s_Tdx[54] = -s_sinx[3];
    s_Tdx[64] = 0.00000000000020682*s_cosx[4]+s_sinx[4];
    s_Tdx[65] = 0.0000000000048966*s_cosx[4];
    s_Tdx[66] = -0.00000000000020682*s_sinx[4]+s_cosx[4];
    s_Tdx[68] = -0.00000000000020682*s_sinx[4]+s_cosx[4];
    s_Tdx[69] = -0.0000000000048966*s_sinx[4];
    s_Tdx[70] = -0.00000000000020682*s_cosx[4]-s_sinx[4];
    s_Tdx[80] = -s_sinx[5];
    s_Tdx[81] = 0.0000000000048966*s_cosx[5];
    s_Tdx[82] = s_cosx[5];
    s_Tdx[84] = -s_cosx[5];
    s_Tdx[85] = -0.0000000000048966*s_sinx[5];
    s_Tdx[86] = -s_sinx[5];
    s_Tdx[96] = 0.00000000000020682*s_cosx[6]+s_sinx[6];
    s_Tdx[97] = 0.0000000000048966*s_cosx[6];
    s_Tdx[98] = -0.00000000000020682*s_sinx[6]+s_cosx[6];
    s_Tdx[100] = -0.00000000000020682*s_sinx[6]+s_cosx[6];
    s_Tdx[101] = -0.0000000000048966*s_sinx[6];
    s_Tdx[102] = -0.00000000000020682*s_cosx[6]-s_sinx[6];
}
/*** PLANT SPECIFIC RBDYN HELPERS ***/

/*** GENERAL PURPOSE RBDYN HELPERS ***/
template <typename T>
__host__ __device__ 
void loadAdjoint(T *dst, T *src){
   dst[0] = 0.0;
   dst[1] = src[2];
   dst[2] = -src[1];
   dst[3] = -src[2];
   dst[4] = 0.0;
   dst[5] = src[0];
   dst[6] = src[1];
   dst[7] = -src[0];
   dst[8] = 0.0;
}
template <typename T>
__host__ __device__ 
void loadAdjoint(T *dst, T src0, T src1, T src2){
   dst[0] = 0.0;
   dst[1] = src2;
   dst[2] = -src1;
   dst[3] = -src2;
   dst[4] = 0.0;
   dst[5] = src0;
   dst[6] = src1;
   dst[7] = -src0;
   dst[8] = 0.0;
}

template <typename T>
__host__ __device__ 
void crfm(T *dst, T *src, int f_flag){
   // don't think there is a better way to do this.... note make sure to clear first
   dst[1] = src[2];
   dst[2] = -src[1];
   dst[6] = -src[2];
   dst[8] = src[0];
   dst[12] = src[1];
   dst[13] = -src[0];
   dst[22] = src[2];
   dst[23] = -src[1];
   dst[27] = -src[2];
   dst[29] = src[0];
   dst[33] = src[1];
   dst[34] = -src[0];
   if (f_flag){
      dst[19] = src[5];
      dst[20] = -src[4];
      dst[24] = -src[5];
      dst[26] = src[3];
      dst[30] = src[4];
      dst[31] = -src[3];
   }
   else {
      dst[4] = src[5];
      dst[5] = -src[4];
      dst[9] = -src[5];
      dst[11] = src[3];
      dst[15] = src[4];
      dst[16] = -src[3];
   }
}

template <typename T>
__host__ __device__ 
void crfmz(T *dst, T *src, int f_flag){
   // this one is slower as it does all of the zeroing
   crfm(dst, src, f_flag);
   dst[0] = 0;
   dst[3] = 0;
   dst[7] = 0;
   dst[10] = 0;
   dst[14] = 0;
   dst[17] = 0;
   dst[18] = 0;
   dst[21] = 0;
   dst[25] = 0;
   dst[28] = 0;
   dst[32] = 0;
   dst[35] = 0;
   if (!f_flag){
      dst[19] = 0;
      dst[20] = 0;
      dst[24] = 0;
      dst[26] = 0;
      dst[30] = 0;
      dst[31] = 0;
   }
   else {
      dst[4] = 0;
      dst[5] = 0;
      dst[9] = 0;
      dst[11] = 0;
      dst[15] = 0;
      dst[16] = 0;
   }
}
/*** GENERAL PURPOSE RBDYN HELPERS ***/

/*** KINEMATICS AND DYNAMICS HELPERS ***/
template <typename T>
__host__ __device__ __forceinline__
void load_I(T *s_I, T *d_I){
   int start, delta; singleLoopVals(&start,&delta);
   for (int ind = start; ind < 36*NUM_POS; ind += delta){
      s_I[ind] = d_I[ind];
   }
}

template <typename T>
__host__ __device__ __forceinline__
void load_Tb(T *s_x, T *s_Tbody, T *d_Tbody, T *s_sinx, T *s_cosx, T *s_dTbody = nullptr){
   int start, delta; singleLoopVals(&start,&delta);
   // compute sin/cos in parallel as well as Tbase
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
         s_sinx[ind] = std::sin(s_x[ind]);    
         s_cosx[ind] = std::cos(s_x[ind]);
   }
   #pragma unroll
   for (int ind = start; ind < 36*NUM_POS; ind += delta){
      // #ifdef __CUDA_ARCH__
         s_Tbody[ind] = d_Tbody[ind];
      // #endif
      if (s_dTbody != nullptr){s_dTbody[ind] = 0.0;} // if need to load dTbody also need to zero
   }
   hd__syncthreads();
   #ifdef __CUDA_ARCH__
      // load in Tbody specifics in one thread
      if(threadIdx.x == 0 && threadIdx.y == 0){updateT(s_Tbody,s_cosx,s_sinx);}
      // load in dTbody specifics in another warp if needed
      if(s_dTbody != nullptr && threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1){loadTdx4(s_dTbody,s_cosx,s_sinx);}
   #else
      // load Tbody and dTbody serially
      updateT(s_Tbody,s_cosx,s_sinx);
      if(s_dTbody != nullptr){loadTdx4(s_dTbody,s_cosx,s_sinx);}
   #endif
}


template <typename T>
__host__ __device__ __forceinline__
void compute_T_TA_J(T *s_Tbody, T *s_Tworld, T *s_TA = nullptr, T *s_J = nullptr){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   int start, delta; singleLoopVals(&start,&delta);
   int TA_J_Flag = s_TA != nullptr && s_J != nullptr;
   // ky is now going to be the body and kx is the array val
   // compute world Ts in T (T[i] = T[i-1]*Tbody[i])
   T *Tb = s_Tbody; T *Ti = s_Tworld; T *Tim1 = s_Tworld;
   #pragma unroll
   for (int body = 0; body < NUM_POS; body++){
      #pragma unroll
      for (int ky = starty; ky < 4; ky += dy){
         #pragma unroll
         for (int kx = startx; kx < 4; kx += dx){
            // row kx of Tim1 * column ky of Tb unless body 0 then copy
            T val = 0.0;
            if (body == 0){val = Tb[ky*4+kx];}
            else {
               #pragma unroll
               for (int i = 0; i < 4; i++){val += Tim1[kx + 4 * i]*Tb[ky * 4 + i];}
            }
            Ti[kx + 4 * ky] = val;
            // store transpose of TL 3x3 of T in TL and BR 3x3 of s_TA for adjoint comp
            if (TA_J_Flag && kx < 3 && ky < 3){
               s_TA[body*36 + kx * 6 + ky] = val;
               s_TA[body*36 + (kx+3) * 6 + (ky+3)] = val;
            }
         }
      }
      // inc the pointers
      Tim1 = Ti; Ti += 36; Tb += 36;
      hd__syncthreads();
   }
   if (!TA_J_Flag){return;}
   // compute adjoint transform of homogtransInv of T -> temp and of T but only 3rd column -> J (T in temp2 and T' stored in TL and BR of TA already)
   // since 4x4 only takes up the first 16 vals we can compute the phats in the last two 3x3 = 18 vals
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      T *phatTA = &s_Tbody[16 + 36*ind]; T *phatJ = &s_Tbody[25 + 36*ind]; Ti = &s_Tworld[36*ind];
      // for TA need to load in the result of -TA[3x3]*T[3x1_4th column]
      T tempVals0 = -1.0*(Ti[0]*Ti[12] + Ti[1]*Ti[13] + Ti[2]*Ti[14]);
      T tempVals1 = -1.0*(Ti[4]*Ti[12] + Ti[5]*Ti[13] + Ti[6]*Ti[14]); 
      T tempVals2 = -1.0*(Ti[8]*Ti[12] + Ti[9]*Ti[13] + Ti[10]*Ti[14]);
      loadAdjoint(phatTA,tempVals0,tempVals1,tempVals2);        
      // for J it is just the standard phat loading
      loadAdjoint(phatJ,&Ti[12]);
   }
   hd__syncthreads();
   // Finish TA and J by computing phat * T
   Ti = s_Tworld; T *phatTA = &s_Tbody[16]; T *phatJ = &s_Tbody[25];
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){
      #pragma unroll
      for (int kx = startx; kx < 9; kx += dx){
         int row = kx % 3; int column = kx / 3; T val = 0.0;
         // first for s_TA
         #pragma unroll
         for (int i = 0; i < 3; i++){
            val += phatTA[36*ky + row + 3 * i] * s_TA[36*ky + column * 6 + i]; // TL 3x3
         }
         s_TA[36*ky + column * 6 + (row + 3)] = val; // store in BL 3x3
         s_TA[36*ky + (column + 3) * 6 + row] = 0.0; // zero out TR of TA
         // then for s_J (but note only one column to compute which is 3rd column)
         if (column == 2){
            T val = 0.0;
            #pragma unroll
            for (int i = 0; i < 3; i++){
               val += phatJ[36*ky + row + 3 * i] * Ti[36*ky + 8 + i]; // 3rd column of T times pHat row
            }
            s_J[6*ky + row + 3] = val; // store in last 3 of J
            s_J[6*ky + row] = Ti[36*ky + 8 + row]; // load in 3rd column of T into first three
         }
      }
   }
}

// Looped such that we reduce memory from 36*NB*NB to 36*NB + 16*NB for s_dT
template <typename T>
__host__ __device__ __forceinline__
void compute_dT_dTA_dJ(T *s_Tbody, T *s_dTbody, T *s_T, T *s_dT, T *s_dTp, T *s_TA = nullptr, T *s_dTA = nullptr, T *s_dJ = nullptr){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   int start, delta; singleLoopVals(&start,&delta);
   int TA_J_Flag = s_TA != nullptr && s_dTA != nullptr && s_dJ != nullptr;
   T *Tb = s_Tbody; T *Ti = s_T; T *TA = s_TA; T *Tim1 = s_T;
   T *phatTA = &Tb[16]; T *phatJ = &Tb[25]; T *dTb = s_dTbody;
   #pragma unroll
   for (int bodyi = 0; bodyi < NUM_POS; bodyi++){
      // compute world dTs (dT[i,j] = dT[i-1,j]*Tbody[i] + (i == j ? T[i-1]*dTbody[i] : 0))
      #pragma unroll
      for (int bodyj = starty; bodyj < NUM_POS; bodyj += dy){
         T *dTij = &s_dT[36*bodyj]; T *dTim1 = &s_dTp[16*bodyj]; T *dTA = &s_dTA[36*(NUM_POS*bodyi+bodyj)];
         #pragma unroll
         for (int ind = startx; ind < 16; ind += dx){
            int ky = ind / 4; int kx = ind % 4; T val = 0.0;
            if (bodyi == 0){val += bodyi == bodyj ? dTb[ky * 4 + kx] : 0.0;}
            else{
               #pragma unroll
               for (int i = 0; i < 4; i++){
                  val += dTim1[kx + 4 * i]*Tb[ky * 4 + i] + (bodyi == bodyj ? Tim1[kx + 4 * i]*dTb[ky * 4 + i] : 0.0);
               }
            }
            dTij[kx + 4 * ky] = val;
            // store transpose of TL 3x3 of T in TL and BR 3x3 of dTA for adjoint comp
            if (TA_J_Flag && kx < 3 && ky < 3){
               dTA[kx * 6 + ky] = val; // TL
               dTA[(kx+3) * 6 + (ky+3)] = val; // BR
               dTA[(kx+3) * 6 + ky] = 0.0; // zero out TR of TA
            }
         }
      }
      hd__syncthreads();
      if (TA_J_Flag){
         // then also compute the derivatives of the phats note that load adjoint is simply a transformation and thus we only need to compute
         // the product rule with respect to the multiplication on tempVals -- note only need one thread per body
         for (int bodyj = start; bodyj < NUM_POS; bodyj += delta){
            T *dTij = &s_dT[36*bodyj]; T *dphatTA = &dTij[16]; T *dphatJ = &dTij[25];
            // for TA need to load in the result of -TA[3x3]*T[3x1_4th column]
            T tempVals0 = -1.0*(dTij[0]*Ti[12] + dTij[1]*Ti[13] + dTij[2]*Ti[14] + Ti[0]*dTij[12] + Ti[1]*dTij[13] + Ti[2]*dTij[14]);
            T tempVals1 = -1.0*(dTij[4]*Ti[12] + dTij[5]*Ti[13] + dTij[6]*Ti[14] + Ti[4]*dTij[12] + Ti[5]*dTij[13] + Ti[6]*dTij[14]); 
            T tempVals2 = -1.0*(dTij[8]*Ti[12] + dTij[9]*Ti[13] + dTij[10]*Ti[14] + Ti[8]*dTij[12] + Ti[9]*dTij[13] + Ti[10]*dTij[14]);
            loadAdjoint(dphatTA,tempVals0,tempVals1,tempVals2);
            // for J it is just the standard phat loading
            loadAdjoint(dphatJ,&dTij[12]);
         }
         hd__syncthreads();
         // Finish dTA and dJ by computing dphat * T + phat * dT
         #pragma unroll
         for (int bodyj = starty; bodyj < NUM_POS; bodyj += dy){
            T *dTij = &s_dT[36*bodyj]; T *dphatTA = &dTij[16]; T *dphatJ = &dTij[25];
            T *dTA = &s_dTA[36*(NUM_POS*bodyi+bodyj)]; T *dJ = &s_dJ[6*(NUM_POS*bodyi+bodyj)];
            #pragma unroll
            for (int kx = startx; kx < 9; kx += dx){
               int column = kx / 3; int row = kx % 3; T val = 0.0;
               // first for dTA
               #pragma unroll
               for (int i = 0; i < 3; i++){
                  val += phatTA[row + 3 * i] * dTA[column * 6 + i] + dphatTA[row + 3 * i] * TA[column * 6 + i]; // TL 3x3
               }
               dTA[column * 6 + (row + 3)] = val; // store in BL 3x3
               // then for s_J (but note only one column to compute which is 3rd column)
               if (column == 2){
                  T val = 0.0;
                  #pragma unroll
                  for (int i = 0; i < 3; i++){
                     val += dphatJ[row + 3 * i] * Ti[8 + i] + phatJ[row + 3 * i] * dTij[8 + i]; // 3rd column of T times pHat row
                  }
                  dJ[row + 3] = val; // store in last 3 of J
                  dJ[row] = dTij[8 + row]; // load in 3rd column of dT into first three
               }
            }
         }
      }
      hd__syncthreads();
      // save down dTij into dTp for next round
      #pragma unroll
      for (int bodyj = starty; bodyj < NUM_POS; bodyj += dy){
         T *dTij = &s_dT[36*bodyj]; T *dTim1 = &s_dTp[16*bodyj];
         #pragma unroll
         for (int kx = startx; kx < 16; kx += dx){dTim1[kx] = dTij[kx];}
      }
      // inc the pointers (the rest move on bodyi inc)
      Tim1 = Ti; Ti += 36; Tb += 36; dTb += 16;
      if (TA_J_Flag){TA += 36; phatTA += 36; phatJ += 36;}
      hd__syncthreads();
   }
}

// compute Iw, Icrbs, twsits, and dIw if needed uses TA,J,x, and temp space as well as dTA if dIw is needed
template <typename T>
__host__ __device__ __forceinline__
void compute_Iw_Icrbs_twist(T *s_I, T *s_Icrbs, T *s_twist, T *s_TA, T *s_J, T *s_x, T *s_temp, T *s_dTA = nullptr, T *s_temp2 = nullptr){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   int start, delta; singleLoopVals(&start,&delta);
   // I = I*TA to start the inertia comp -- store in temp
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){
      // row L * column R stores in (r,c)
      #pragma unroll
      for (int kx = startx; kx < 36; kx += dx){
         int r = kx % 6;
         int c = kx / 6;
         T val = 0.0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            val += s_I[36*ky + r + 6 * i] * s_TA[36*ky + c * 6 + i];
         }
         s_temp[36*ky + c * 6 + r] = val; 
      }
   }
   hd__syncthreads();
   // compute dIw if needed
   if (s_dTA != nullptr){
      // we are now going to start running into memory issues so we are going to loop NUM_POS times
      // lets use Icrbs and temp as the temp space and load the finals in temp2 and then back into dTA to save space
      // dIW = dTA'*(I*TA) + TA'*(I*dTA) -- note I*TA is already in s_temp
      // 
      #pragma unroll
      for (int bodyi = 0; bodyi < NUM_POS; bodyi++){
         #pragma unroll
         for (int ky = starty; ky < NUM_POS; ky += dy){
            #pragma unroll
            for (int kx = startx; kx < 36; kx += dx){
               int r = kx % 6;
               int c = kx / 6;
               T val = 0.0;
               #pragma unroll
               for (int i = 0; i < 6; i++){
                  val += s_I[36*bodyi + r + 6 * i] * s_dTA[36*(bodyi*NUM_POS+ky) + c * 6 + i];
               }
               s_Icrbs[36*ky + c * 6 + r] = val;   
            }
         }
         hd__syncthreads();
         #pragma unroll
         for (int ky = starty; ky < NUM_POS; ky += dy){
            #pragma unroll
            for (int kx = startx; kx < 36; kx += dx){
               T val = 0.0;
               int r = kx % 6;
               int c = kx / 6;
               #pragma unroll
               for (int i = 0; i < 6; i++){
                  val += s_dTA[36*(bodyi*NUM_POS+ky) + r * 6 + i] * s_temp[36*bodyi + c * 6 + i];
                  val += s_TA[36*bodyi + r * 6 + i] * s_Icrbs[36*ky + c * 6 + i];
               }
               s_temp2[36*ky + c*6+r] = val; // load this bodyi into temp2 
            }
         }
         hd__syncthreads();
         //T *s_dIw = &s_dTA[36*bodyi*NUM_POS];
         #pragma unroll
         for (int ky = starty; ky < NUM_POS; ky += dy){
            #pragma unroll
            for (int kx = startx; kx < 36; kx += dx){
               s_dTA[36*(bodyi*NUM_POS+ky) + kx] = s_temp2[36*ky+kx]; // then copy over into s_dTA
            }
         }
         // no sync needed because can start computing next step from next bodyi without incident
      }
   }
   // IW = TA'*(I*TA) to finish the inertia comp which we can now safely overwrite s_I
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){
      // row L * column R stores in (r,c)
      #pragma unroll
      for (int kx = startx; kx < 36; kx += dx){
         int r = kx % 6;
         int c = kx / 6;
         T val = 0.0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            val += s_TA[36*ky + r * 6 + i] * s_temp[36*ky + c * 6 + i];
         }
         s_I[36*ky + c * 6 + r] = val; 
      }
   }
   hd__syncthreads();
   // compute the recursion for the CRBI which is just a summation of the world Is
   // and finish the twist recursion which is also a summation
   // sum IW to IC IC[i] = Sum_j>i IC[j] matrix 6x6 (avoids syncthreads and only NUM_POS wasted additions is probs faster)
   #pragma unroll
   for (int ind = start; ind < 36; ind += delta){
      T val = 0.0;
      #pragma unroll
      for (int body = NUM_POS-1; body >= 0; body--){
         val += s_I[36*body + ind];
         s_Icrbs[36*body + ind] = val;
         // and clear Temp and TA for later
         s_TA[36*body + ind] = 0.0;
         s_temp[36*body + ind] = 0.0;
      }
   }
   // compute the twist recursion -- twist[i] = SUM_j<=i twist[j] where (twists[j] = J[j]*x[nbodies+j])
   #pragma unroll
   for (int ind = start; ind < 6; ind += delta){
      #pragma unroll
      for (int body = 0; body < NUM_POS; body++){
         s_twist[6*body + ind] = s_J[6*body + ind] * s_x[NUM_POS+body] + (body > 0 ? s_twist[6*(body-1) + ind] : 0.0);
      }
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_JdotV(T *s_JdotV, T *s_twist, T *s_J, T *s_x, T *s_temp){
   int start, delta; singleLoopVals(&start,&delta);
   // compute the CRMs of the twists -- temp needs to have been cleared earlier
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      crfm(&s_temp[36*ind],&s_twist[6*ind],0);
   }
   hd__syncthreads();   
   // now compute JdotV[i] = JdotV[i-1] + crms[i]*J[i]*s_x[NUM_POS + i] and store in s_JdotV
   #pragma unroll
   for (int body = 0; body < NUM_POS; body++){
      #pragma unroll
      for (int ind = start; ind < 6; ind += delta){
         T val = 0.0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            val += s_temp[36*body + ind + i*6]*s_J[6*body + i];
         }
         s_JdotV[6*body + ind] = s_x[NUM_POS + body]*val + (body > 0 ? s_JdotV[6*(body-1) + ind] : 0.0);
      }
      hd__syncthreads();
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_dtwist(T *s_dTwist, T *s_J, T *s_dJ, T *s_x){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   // dtwist[i,j] (NB*NB*6*2 b/c qd) = dJ[i,j]*qd[i] + J[i]*dqd[i,j] + dtwist[i-1,j]
   #pragma unroll
   for (int body = 0; body < NUM_POS; body++){ // main body
      // first do the dqs where we only have dJ[i,j]*qd[i] + dtwist[i-1,j]
      #pragma unroll
      for (int ky = starty; ky < NUM_POS; ky += dy){ // derivative body
         #pragma unroll
         for (int kx = startx; kx < 6; kx += dx){ // index
            T val = s_dJ[6*(body*NUM_POS + ky) + kx]*s_x[NUM_POS + body];
            if (body > 0){
               val += s_dTwist[6*((body-1)*2*NUM_POS + ky) + kx];
            }
            s_dTwist[6*(body*2*NUM_POS + ky) + kx] = val;
         }
      }
      // then the dqds where we only have J[i]*dqd[i,j] + dtwist[i-1,j]
      #pragma unroll
      for (int ky = starty; ky < NUM_POS; ky += dy){ // derivative body
         #pragma unroll
         for (int kx = startx; kx < 6; kx += dx){ // index
            T val = ky == body ? s_J[6*body + kx] : 0.0;
            if (body > 0){
               val += s_dTwist[6*((body-1)*2*NUM_POS + NUM_POS + ky) + kx];
            }
            s_dTwist[6*(body*2*NUM_POS + NUM_POS + ky) + kx] = val;
         }
      }
      hd__syncthreads();
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_dJdotV(T *s_dJdotV, T *s_twist, T *s_dTwist, T *s_J, T *s_dJ, T *s_x, T *s_temp, T *s_temp2){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   int start, delta; singleLoopVals(&start,&delta);
   // dJdotV[i] (NB*NB*6*2 b/c qd) = crm(dtwist)*J*qd + crm(twist)*(dJ*qd + J*dqd) + dJdotV[i-1] 
   // first form the crms in s_temp and s_temp2 (need two passes b/c it assume each is 36*NB and we need 36*NB*2 for each)
   // then mulitply out to get the answer
   // first lets load in the crm(twist) into s_temp2 -- note we need to zero it all first so use the zeroing crfmz b/c the constant syncing
   // to rezero will probably be slower
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      crfmz(&s_temp2[36*ind],&s_twist[6*ind],0);
   }
   // then we loop by body forming the dwtists and the results
   #pragma unroll
   for (int body = 0; body < NUM_POS; body++){ // main body
      // first form the dcrms for the qs
      #pragma unroll
      for (int ind = start; ind < NUM_POS; ind += delta){
         crfmz(&s_temp[36*ind],&s_dTwist[6*(body*2*NUM_POS + ind)],0);
      }
      hd__syncthreads();
      // then multiply for the qs so dqd = 0 thus we need = (crm(dtwist)*J + crm(twist)*dJ)*qd + dJdotV[i-1]
      #pragma unroll
      for (int ky = starty; ky < NUM_POS; ky += dy){ // derivative body
         #pragma unroll
         for (int kx = startx; kx < 6; kx += dx){ // index
            T val = 0.0;
            #pragma unroll
            for (int i = 0; i < 6; i++){
               val += s_temp[36*ky + kx + 6 * i]*s_J[6*body + i] + s_temp2[36*body + kx + 6 * i]*s_dJ[6*(body*NUM_POS + ky) + i];
            }
            val *= s_x[NUM_POS + body];
            if (body > 0){
               val += s_dJdotV[6*((body-1)*2*NUM_POS + ky) + kx];
            }
            s_dJdotV[6*(body*2*NUM_POS + ky) + kx] = val;
         }
      }
      hd__syncthreads();
      // then form the crms for the qds but note that the twists dont change its just the dtwists
      #pragma unroll
      for (int ind = start; ind < NUM_POS; ind += delta){
         crfmz(&s_temp[36*ind],&s_dTwist[6*(body*2*NUM_POS + NUM_POS + ind)],0);
      }
      hd__syncthreads();
      // then multiply for the qds so dJ = 0 thus we need = (crm(dtwist)*qd + crm(twist)*dqd)*J + dJdotV[i-1] 
      #pragma unroll
      for (int ky = starty; ky < NUM_POS; ky += dy){ // derivative body
         #pragma unroll
         for (int kx = startx; kx < 6; kx += dx){ // index
            T val = 0.0;
            #pragma unroll
            for (int i = 0; i < 6; i++){
               val += (s_temp[36*ky + kx + 6 * i]*s_x[NUM_POS + body] + (ky == body ? s_temp2[36*body + kx + 6 * i] : 0.0))*s_J[6*body + i];
            }
            if (body > 0){
               val += s_dJdotV[6*((body-1)*2*NUM_POS + NUM_POS + ky) + kx];
            }
            s_dJdotV[6*(body*2*NUM_POS + NUM_POS + ky) + kx] = val;
         }
      }
      hd__syncthreads();
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_M_Tau(T *s_M, T *s_Tau, T *s_W, T *s_JdotV, T *s_F, T *s_Icrbs, T *s_twist, T *s_J, T *s_I, T *s_x, T *s_u, T *s_temp, T *s_temp2, T *s_TA){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   int start, delta; singleLoopVals(&start,&delta);
   // compute sub parts for wrenches and the force matrix for the mass matrix comp
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){
      // compute wrench subparts
      #pragma unroll
      for (int kx = startx; kx < 6; kx += dx){
         // temp_c1 = I_world * W_twist (holding the twist)
         // temp_c2 = I_world * grav + JdV
         // form the force matrices for the mass matrix comp (F = I_crbs[i]*J[i])
         T val = 0.0;
         T val2 = 0;
         T val3 = 0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            int Iind = 36*ky + kx + i * 6;
            val  += s_I[Iind] * s_twist[6*ky + i];
            val2 += s_I[Iind] * (s_JdotV[6*ky + i] + (i == 5 ? GRAVITY : 0.0)); //for arm gravity vec is [0 0 0 0 0 g]
            val3 += s_Icrbs[Iind] * s_J[6*ky + i];
         }
         s_temp[36*ky + kx] = val;        
         s_temp[36*ky + 6 + kx] = val2;
         s_F[ky*6 + kx] = val3;
      }
      // finally form the crfs for the wrenches (TA cleared earlier)
      int flag = 1; // optimized away in cpu case
      #ifdef __CUDA_ARCH__
         flag = threadIdx.x == blockDim.x - 1; // use last thread b/c less likely to be looping above
      #endif
      if (flag){crfm(&s_TA[36*ky],&s_twist[6*ky],1);}
   }
   hd__syncthreads();
   // W[i] and mass matrix
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){
      // compute W = crf(W_twist) * IC * W_twist + IC*(g + JdV) = <<<TA*temp_c1 + temp_c2 = W>>>
      #pragma unroll
      for (int kx = startx; kx < 6; kx += dx){
         T val = 0.0;
         for (int i = 0; i < 6; i++){
            val += s_TA[36*ky + kx + i * 6]*s_temp[36*ky + i];
         }
         s_W[6*ky + kx] = val + s_temp[36*ky + 6 + kx];
         //printf("Body[%d]_[%d]: W[%f] = IgJdV[%f] + crf*IW[%f]\n",ky,kx,s_W[kx],s_temp[36*ky + 6 + kx],val);
      }
      // and at the same time the Mass matrix which we store back in s_temp2
      #pragma unroll
      for (int kx = startx; kx < NUM_POS; kx += dx){
         // M(i,j<=1) = M(j<=1,i) = J[j]*F[i]
         int jInd, iInd;
         if (kx <= ky){
            jInd = kx;
            iInd = ky;
         }
         else{
            jInd = ky;
            iInd = kx;
         }
         T val = 0.0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            val += s_J[6*jInd + i] * s_F[6*iInd + i];
         }
         s_M[ky*NUM_POS + kx] = val;
         // also load in an identity next to it to prep for inverse
         s_M[(ky+NUM_POS)*NUM_POS + kx] = (kx == ky) ? 1.0 : 0.0;
      }
   }
   hd__syncthreads();
   // net W: sum net wrenches W[i] = Sum_j>=i W[j] vector 6x1
   #pragma unroll
   for (int ind = start; ind < 6; ind += delta){
      T val = 0.0;
      #pragma unroll
      for (int body = NUM_POS - 1; body >= 0; body--){
         val += s_W[6*body + ind];
         s_W[6*body + ind] = val;
      }
   }
   hd__syncthreads();
   // compute the bias force
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      // C(i) = W(i)*J(i) -- store in end of temp2 (also subtract from u for later we are forming tau)
      T val = 0.0;
      for (int i = 0; i < 6; i++){
         val += s_J[6*ind + i] * s_W[6*ind + i];
      }     
      // for our robot damping is all velocity dependent and =0.5v and B = I so tau = u-(c+0.5v)
      s_Tau[ind] = s_u[ind] - (val + 0.5*s_x[NUM_POS+ind]);
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_dWb(T *s_dWb, T *s_JdotV, T *s_dJdotV, T *s_twist, T *s_dTwist, T *s_Iw, T *s_dIw, T *s_temp, T *s_temp2, T *s_temp3){
   int start, delta; singleLoopVals(&start,&delta);
   // dWb (NB*NB*6) = dIw*([0--9.81] + JdotV) + Iw*dJdotV + crf(dtwist)*Iw*twist + crf(twist)*(dIw*twist + Iw*dtwist)
   // again first form the crfs in s_temp and s_temp2 (need two passes b/c it is assumed 36*NB and we need 36*NB*2 for each)
   // then mulitply out to get the answer -- note s_temp3 only needs to be 6*NB in size because storing temp totals
   // first form the crfs
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      crfmz(&s_temp[36*ind],&s_twist[6*ind],1);
   }
   hd__syncthreads();
   #pragma unroll
   for (int body = 0; body < NUM_POS; body++){ // main body
      // fist form the dcrfs for the qs
      #pragma unroll
      for (int ind = start; ind < NUM_POS; ind += delta){
         crfmz(&s_temp2[36*ind],&s_dTwist[6*(body*2*NUM_POS + ind)],1);
      }
      hd__syncthreads();
      // the multiply dIw*([0--9.81] + JdotV) + Iw*dJdotV + crf(dtwist)*Iw*twist + crf(twist)*(dIw*twist + Iw*dtwist)
      // our issue here is that we actually need far more space than we have so we are going to loop this again unfortunately
      #pragma unroll
      for (int dbody = 0; dbody < NUM_POS; dbody++){ // derivative body
         #pragma unroll
         for (int ind = start; ind < 6; ind += delta){ // ind
            T val0 = 0; //dIw*([0--9.81] + JdotV) + Iw*dJdotV
            T val1 = 0; //crf(dtwist)*Iw*twist
            T val2 = 0; //crf(twist)*(dIw*twist + Iw*dtwist)
            #pragma unroll
            for (int i = 0; i < 6; i++){
               T Iw = s_Iw[36*body + ind + 6 * i];
               T dIw = s_dIw[36*(body*NUM_POS + dbody) + ind + 6 * i];
               T tw = s_twist[6*body + i];
               T dtw = s_dTwist[6*(body*2*NUM_POS + dbody) + i];
               T dJdV = s_dJdotV[6*(body*2*NUM_POS + dbody) + i];
               val0 += dIw*(s_JdotV[6*body + i] + (i == 5 ? 9.81 : 0.0)) + Iw*dJdV;
               val1 += Iw*tw;
               val2 += dIw*tw + Iw*dtw;
            }
            // store the temp vals in s_temp3
            s_temp3[3*ind] = val0;
            s_temp3[3*ind + 1] = val1;
            s_temp3[3*ind + 2] = val2;
         }
         hd__syncthreads();
         // now finish it off with val0 + crf(dtwist)*val1 + crf(twist)*val2
         #pragma unroll
         for (int ind = start; ind < 6; ind += delta){ // ind
            T val = s_temp3[3*ind];
            #pragma unroll
            for (int i = 0; i < 6; i++){
               val += s_temp2[36*dbody + ind + 6 * i]*s_temp3[3*i + 1] + s_temp[36*body + ind + 6 * i]*s_temp3[3*i + 2];
            }
            s_dWb[6*(body*2*NUM_POS + dbody) + ind] = val;
         }
         hd__syncthreads();
      }
      // then form the crfs for the qds and again note that only the dtwist changes
      #pragma unroll
      for (int ind = start; ind < NUM_POS; ind += delta){
         crfmz(&s_temp2[36*ind],&s_dTwist[6*(body*2*NUM_POS + NUM_POS + ind)],1);
      }
      hd__syncthreads();
      // then multiply again
      #pragma unroll
      for (int dbody = 0; dbody < NUM_POS; dbody++){ // derivative body
         #pragma unroll
         for (int ind = start; ind < 6; ind += delta){ // ind
            // note now that dIw == 0 so can drop those terms
            T val0 = 0; //Iw*dJdotV
            T val1 = 0; //crf(dtwist)*Iw*twist
            T val2 = 0; //crf(twist)*Iw*dtwist
            #pragma unroll
            for (int i = 0; i < 6; i++){
               T Iw = s_Iw[36*body + ind + 6 * i];
               T tw = s_twist[6*body + i];
               T dtw = s_dTwist[6*(body*2*NUM_POS + NUM_POS + dbody) + i];
               T dJdV = s_dJdotV[6*(body*2*NUM_POS + NUM_POS + dbody) + i];
               val0 += Iw*dJdV;
               val1 += Iw*tw;
               val2 += Iw*dtw;
            }
            // store the temp vals in s_temp3
            s_temp3[3*ind] = val0;
            s_temp3[3*ind + 1] = val1;
            s_temp3[3*ind + 2] = val2;
         }
         hd__syncthreads();
         // now finish it off with val0 + crf(dtwist)*val1 + crf(twist)*val2
         #pragma unroll
         for (int ind = start; ind < 6; ind += delta){ // ind
            T val = s_temp3[3*ind];
            #pragma unroll
            for (int i = 0; i < 6; i++){
               val += s_temp2[36*dbody + ind + 6 * i]*s_temp3[3*i + 1] + s_temp[36*body + ind + 6 * i]*s_temp3[3*i + 2];
            }
            s_dWb[6*(body*2*NUM_POS + NUM_POS + dbody) + ind] = val;
         }
         hd__syncthreads();
      }
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_dTau(T *s_dTau, T *s_dWb, T *s_W, T *s_J, T *s_dJ){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   // dTau = -dC for the arm and dC (NB*NB) = dJ*W + J*SUM(dWb) + 0.5dqd(aka eye) -- note dJ only exists for qs
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){ // main body
      #pragma unroll
      for (int kx = startx; kx < 2*NUM_POS; kx += dx){ // derivative body
         T val = 0.0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            T dW = 0;
            #pragma unroll
            for (int j = ky; j < NUM_POS; j++){
               dW += s_dWb[6*(j*2*NUM_POS + kx) + i];
            }
            val += (kx < NUM_POS ? s_dJ[6*(ky*NUM_POS + kx) + i]*s_W[6*ky + i] : 0.0) + s_J[6*ky + i]*dW;
         }
         s_dTau[kx*NUM_POS + ky] = -1*(val + (kx - NUM_POS == ky ? 0.5 : 0.0));
      }
   }
}

// Looped version of comptueDTau for memory efficiency so we only need current and parent of each body since serial chain
// so that means NB*6*2 for (dtwist, dtwistp, dJdotV, dJdotVp, dWb, dWp, dTau)
// temp = 36*NB, temp2 = 36*NB, temp3 = 36*NB, temp4 = 6*NB;
template <typename T>
__host__ __device__ __forceinline__
void compute_dTau(T *s_dTau, T *s_W, T *s_dWb, T *s_dWp, T *s_JdotV, T *s_dJdotV, T *s_dJdotVp, T *s_twist, T *s_dTwist, T *s_dTwistp, T *s_Iw, T *s_dIw, T *s_J, T *s_dJ, T *s_x, T *s_temp, T *s_temp2, T *s_temp3, T *s_temp4){
   int start, delta; singleLoopVals(&start,&delta);
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   // we loop over bodies for memory efficiency but also pre-compute some terms that don't change
   // first clear the prev variables and temp vars for the crfms
   #pragma unroll
   for (int ind = start; ind < 36*NUM_POS; ind += delta){
      s_temp[ind] = 0.0;
      s_temp2[ind] = 0.0;
      s_temp3[ind] = 0.0;
      if (ind < 12*NUM_POS){
         s_dWp[ind] = 0.0;
         s_dJdotVp[ind] = 0.0;
         s_dTwistp[ind] = 0.0;
      }
   }
   hd__syncthreads();
   // first up crm(twist) into s_temp
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      crfm(&s_temp[36*ind],&s_twist[6*ind],0);
   }
   // then start the loops
   #pragma unroll
   for (int body = 0; body < NUM_POS; body++){ // main body
      // dTwist[i,j] (NB*6*2 b/c qd) = dJ[i,j]*qd[i] + J[i]*dqd[i,j] + dtwist[i-1,j]
      // first do the dqs where we only have dJ[i,j]*qd[i] + dtwist[i-1,j]
      // then the dqds where we only have J[i]*dqd[i,j] + dtwist[i-1,j]
      #pragma unroll
      for (int ky = starty; ky < 2*NUM_POS; ky += dy){ // derivative body
         #pragma unroll
         for (int kx = startx; kx < 6; kx += dx){ // index
            T val;
            if (body < NUM_POS){val = s_dJ[6*(body*NUM_POS + ky) + kx]*s_x[NUM_POS + body];}
            else{val = ky == body ? s_J[6*body + kx] : 0.0;}
            if (body > 0){val += s_dTwistp[6*ky + kx];
            }
            s_dTwist[6*ky + kx] = val;
         }
      }
      hd__syncthreads();
      // dJdotV[i] (NB*6*2 b/c qd) = crm(dtwist)*J*qd + crm(twist)*(dJ*qd + J*dqd) + dJdotV[i-1] 
      // first form the dcrms for the qs and qds
      #pragma unroll
      for (int ind = start; ind < 2*NUM_POS; ind += delta){
         if (ind < NUM_POS){
            crfm(&s_temp2[36*ind],&s_dTwist[6*ind],0);   
         }
         else{
            crfm(&s_temp3[36*(ind-NUM_POS)],&s_dTwist[6*ind],0);
         }
      }
      hd__syncthreads();
      // then multiply for the qs so dqd = 0 thus we need = (crm(dtwist)*J + crm(twist)*dJ)*qd + dJdotV[i-1]
      // then multiply for the qds so dJ = 0 thus we need = (crm(dtwist)*qd + crm(twist)*dqd)*J + dJdotV[i-1] 
      #pragma unroll
      for (int ky = starty; ky < 2*NUM_POS; ky += dy){ // derivative body
         #pragma unroll
         for (int kx = startx; kx < 6; kx += dx){ // index
            T val = 0.0;
            #pragma unroll
            for (int i = 0; i < 6; i++){
               if (ky < NUM_POS){
                  val += (s_temp2[36*ky + kx + 6 * i]*s_J[6*body + i] + s_temp[36*body + kx + 6 * i]*s_dJ[6*(body*NUM_POS + ky) + i])*s_x[NUM_POS + body];
               }
               else{
                  val += (s_temp3[36*ky + kx + 6 * i]*s_x[NUM_POS + body] + (ky == body ? s_temp[36*body + kx + 6 * i] : 0.0))*s_J[6*body + i];
               }
            }
            if (body > 0){val += s_dJdotVp[6*ky + kx];}
            s_dJdotV[6*ky + kx] = val;
         }
      }
      hd__syncthreads();
      // dWb (NB*NB*6) = dIw*([0--9.81] + JdotV) + Iw*dJdotV + crf(dtwist)*Iw*twist + crf(twist)*(dIw*twist + Iw*dtwist)
       // first form the dcrms for the qs and qds
      #pragma unroll
      for (int ind = start; ind < 2*NUM_POS; ind += delta){
         if (ind < NUM_POS){
            crfm(&s_temp2[36*ind],&s_dTwist[6*ind],1);   
         }
         else{
            crfm(&s_temp3[36*(ind-NUM_POS)],&s_dTwist[6*ind],1);
         }
      }
      hd__syncthreads();
      // the multiply dIw*([0--9.81] + JdotV) + Iw*dJdotV + crf(dtwist)*Iw*twist + crf(twist)*(dIw*twist + Iw*dtwist)
      // our issue here is that we actually need far more space than we have so we are going to loop this again unfortunately
      #pragma unroll
      for (int dbody = 0; dbody < 2*NUM_POS; dbody++){ // derivative body
         #pragma unroll
         for (int ind = start; ind < 6; ind += delta){ // ind
            // note: for dbody > NUM_POS then dIw = 0 so drop those terms
            T val0 = 0; //dIw*([0--9.81] + JdotV) + Iw*dJdotV
            T val1 = 0; //crf(dtwist)*Iw*twist
            T val2 = 0; //crf(twist)*(dIw*twist + Iw*dtwist)
            #pragma unroll
            for (int i = 0; i < 6; i++){
               T Iw = s_Iw[36*body + ind + 6 * i];
               
               T tw = s_twist[6*body + i];
               T dtw = s_dTwist[6*(body*2*NUM_POS + dbody) + i];
               T dJdV = s_dJdotV[6*(body*2*NUM_POS + dbody) + i];
               if (dbody < NUM_POS){
                  T dIw = s_dIw[36*(body*NUM_POS + dbody) + ind + 6 * i];   
                  val0 += dIw*(s_JdotV[6*body + i] + (i == 5 ? 9.81 : 0.0));
                  val2 += dIw*tw;
               }
               val0 += Iw*dJdV;
               val1 += Iw*tw;
               val2 += Iw*dtw;
            }
            // store the temp vals in s_temp4
            s_temp4[3*ind] = val0;
            s_temp4[3*ind + 1] = val1;
            s_temp4[3*ind + 2] = val2;
         }
         hd__syncthreads();
         // now finish it off with val0 + crf(dtwist)*val1 + crf(twist)*val2
         #pragma unroll
         for (int ind = start; ind < 6; ind += delta){ // ind
            T val = s_temp4[3*ind];
            #pragma unroll
            for (int i = 0; i < 6; i++){
               if (ind < NUM_POS){
                  val += s_temp2[36*dbody + ind + 6 * i]*s_temp4[3*i + 1] + s_temp[36*body + ind + 6 * i]*s_temp4[3*i + 2];   
               }
               else{
                  val += s_temp3[36*(dbody-NUM_POS) + ind + 6 * i]*s_temp4[3*i + 1] + s_temp[36*body + ind + 6 * i]*s_temp4[3*i + 2];
               }
            }
            s_dWb[6*dbody + ind] = val;
         }
         hd__syncthreads();
      }
      // dTau = -dC for the arm and dC (NB*NB) = dJ*W + J*SUM(dWb) + 0.5dqd(aka eye) -- note dJ only exists for qs
      #pragma unroll
      for (int ind = start; ind < 2*NUM_POS; ind += delta){
         T val = 0.0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            T dW = s_dWb[6*ind + i] + s_dWp[6*ind + i]; // sum for W
            val += (ind < NUM_POS ? s_dJ[6*(body*NUM_POS + ind) + i]*s_W[6*body + i] : 0.0) + s_J[6*body + i]*dW;
         }
         s_dTau[ind*NUM_POS + body] = -1*(val + (ind - NUM_POS == body ? 0.5 : 0.0));
      }
      hd__syncthreads();
      // now save current into prev for next round
      #pragma unroll
      for (int ind = start; ind < 2*NUM_POS*6; ind += delta){
         s_dWp[ind] = s_dWb[ind];
         s_dJdotVp[ind] = s_JdotV[ind];
         s_dTwistp[ind] = s_dTwist[ind];
      }
      hd__syncthreads();
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_qdd(T *s_qdd, T *s_Minv, T *s_Tau){
   int start, delta; singleLoopVals(&start,&delta);
   // for the arm B = I so qdd = Hinv*tau
   #pragma unroll
   for (int ind = start; ind < NUM_POS; ind += delta){
      T val = 0.0;
      for (int i = 0; i < NUM_POS; i++){
         val += s_Minv[ind + NUM_POS*i] * s_Tau[i];
      }     
      s_qdd[ind] = val;
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_dM(T *s_dM, T *s_Icrbs, T *s_dIw, T *s_J, T *s_dJ, T *s_F, T *s_temp){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   // dM[i,j]/dk = dJ[j,k]*Icrbs[i]*J[i] + J[j]*(dIcrbs[i,k]*J[i] + Icrbs[i]*dJ[i,k])
   // so we need s_F[i] = s_Icrbs[i]*s_J[i]
   //            s_temp[i,k] = dIcrbs[i,k]*J[i] + Icrbs[i]*dJ[i,k]
   // note that we only have dIw and dIcrbs = SUM_j>i dIw[i] so need to dynamically sum
   #pragma unroll
   for (int bodyi = starty; bodyi < NUM_POS; bodyi += dy){
      for (int kx = startx; kx < NUM_POS*6; kx += dx){
         int bodyk = kx / 6;
         int r = kx % 6;
         T val = 0.0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            // need to sum up the dIw for the dIcrbs
            T dIcrbs = 0;
            #pragma unroll
            for (int j = bodyi; j < NUM_POS; j++){
               dIcrbs += s_dIw[36*(j*NUM_POS + bodyk) + r + 6 * i];
            }
            //dIcrbs[i,k]*J[i] + Icrbs[i]*dJ[i,k]
            val += dIcrbs*s_J[6*bodyi + i] + s_Icrbs[36*bodyi + r + 6 * i]*s_dJ[6*(bodyi*NUM_POS + bodyk) + i];
         }
         s_temp[6*(bodyi*NUM_POS + bodyk) + r] = val;
      }
      int reps = 0;
      #ifdef __CUDA_ARCH__
         if(threadIdx.x >= blockDim.x - 6){reps = 1;} // possibly in separate warp
      #else
         reps = 6;
      #endif
      #pragma unroll
      for(int rep = 0; rep < reps; rep++){
         #ifdef __CUDA_ARCH__
            int r = threadIdx.x % 6;
         #else
            int r = rep;
         #endif
         T val = 0.0;
         #pragma unroll
         for (int i = 0; i < 6; i++){val += s_Icrbs[36*bodyi + r + 6 * i]*s_J[6*bodyi + i];}
         s_F[bodyi*6 + r] = val;
      }
   }
   hd__syncthreads();
   // now dM[i,j]/dk = dJ[j,k]*F[i] + J[j]*temp[i][k] --> store in s_dM for now
   #pragma unroll
   for (int bodyk = starty; bodyk < NUM_POS; bodyk += dy){
      #pragma unroll
      for (int kx = startx; kx < NUM_POS*NUM_POS; kx += dx){
         int r = kx % NUM_POS;
         int c = kx / NUM_POS;
         int jInd, iInd;
         if (r <= c){
            jInd = r;
            iInd = c;
         }
         else{
            jInd = c;
            iInd = r;
         }
         T val = 0.0;
         #pragma unroll
         for (int i = 0; i < 6; i++){
            val += s_dJ[6*(jInd*NUM_POS + bodyk) + i] * s_F[6*iInd + i] + s_J[6*jInd + i] * s_temp[6*(iInd*NUM_POS + bodyk) + i];
         }
         s_dM[NUM_POS*NUM_POS*bodyk + c * NUM_POS + r] = val;
      }
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_dqdd_dM(T *s_dqdd, T *s_dM, T *s_Minv, T *s_qdd, T *s_temp){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   // then compute first half of the compute for dqdd (-Minv^T*dM*Minv*tau = -Minv^T*dM*qdd)
   // note dqdd_dx will be 2*NUM_POS*NUM_POS but dM part is only half of the first NUM_POS*NUM_POS part
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){ // pick the dx
      #pragma unroll
      for (int kx = startx; kx < NUM_POS; kx += dx){ // pick the row output
         T *dM = &s_dM[NUM_POS*NUM_POS*ky + kx];
         // then dM(stride by k)*qdd in tspace
         T val = 0.0;
         #pragma unroll
         for (int i = 0; i < NUM_POS; i++){
            val += dM[i*NUM_POS]*s_qdd[i];
         }
         s_temp[ky*NUM_POS + kx] = val;
      }
   }
   hd__syncthreads();
   // now we need -Minv^T*tspace -> dqdd
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){ // pick the dx
      #pragma unroll
      for (int kx = startx; kx < NUM_POS; kx += dx){ // pick the row output
         T val = 0.0;
         #pragma unroll
         for (int i = 0; i < NUM_POS; i++){
            val += s_Minv[kx*NUM_POS + i]*s_temp[ky*NUM_POS + i];
         }
         s_dqdd[ky*NUM_POS + kx] = -1.0*val;
         s_dqdd[(ky+NUM_POS)*NUM_POS + kx] = 0.0; // set dqd part to 0 for now
      }
   }
}

template <typename T>
__host__ __device__ __forceinline__
void finish_dqdd(T *s_dqdd, T *s_dTau, T *s_Minv){
   int starty, dy, startx, dx; doubleLoopVals(&starty,&dy,&startx,&dx);
   T *s_dqdd_du = &s_dqdd[2*NUM_POS*NUM_POS];
   #pragma unroll
   for (int ky = starty; ky < NUM_POS; ky += dy){ // row of invH
      #pragma unroll
      for (int kx = startx; kx < 2*NUM_POS; kx += dx){ // col of dTau
         T val = 0.0;
         #pragma unroll
         for (int i = 0; i < NUM_POS; i++){
            val += s_Minv[ky + NUM_POS * i] * s_dTau[kx*NUM_POS + i];
         }
         s_dqdd[kx*NUM_POS + ky] += val;
         // also load in dqdd_du which for arm is just Hinv
         if (kx < NUM_POS){s_dqdd_du[kx*NUM_POS + ky] = s_Minv[kx*NUM_POS + ky];}
      }
   }
}

template <typename T>
__host__ __device__ __forceinline__
void compute_eePos(T *s_T, T *s_eePos, T *s_dT = nullptr, T *s_deePos = nullptr, T *s_temp = nullptr){
   int start, delta; singleLoopVals(&start,&delta);
   #ifdef __CUDA_ARCH__
      bool thread0Flag = threadIdx.x == 0 && threadIdx.y == 0;
   #else
      bool thread0Flag = 1;
   #endif
   T *Tee = &s_T[(NUM_POS-1)*36];
   // get hand pos and factors for multiplication in one thread
   if (thread0Flag){
      s_eePos[0] = Tee[12];
      s_eePos[1] = Tee[13];
      s_eePos[2] = Tee[14];
      s_eePos[3] = (T) atan2(Tee[6],Tee[10]);
      s_eePos[4] = (T) atan2(-Tee[2],sqrt(Tee[6]*Tee[6]+Tee[10]*Tee[10]));
      s_eePos[5] = (T) atan2(Tee[1],Tee[0]);
   }
   // if computing derivatives first compute factors in temp memory
   bool dFlag = (s_dT != nullptr) && (s_deePos != nullptr) && (s_temp != nullptr);
   if (dFlag){
      if (thread0Flag){
         T factor3 = Tee[6]*Tee[6] + Tee[10]*Tee[10];
         T factor4 = 1/(Tee[2]*Tee[2] + factor3);
         T factor5 = 1/(Tee[1]*Tee[1] + Tee[0]*Tee[0]);
         T sqrtfactor3 = (T) sqrt(factor3);
         s_temp[0] = -Tee[6]/factor3;
         s_temp[1] = Tee[10]/factor3;
         s_temp[2] = Tee[2]*Tee[6]*factor4/sqrtfactor3;
         s_temp[3] = Tee[2]*Tee[10]*factor4/sqrtfactor3;
         s_temp[4] = -sqrtfactor3*factor4;
         s_temp[5] = -Tee[1]*factor5;
         s_temp[6] = Tee[0]*factor5;
      }
      hd__syncthreads();
      // then compute all dk in parallel (note dqd is 0 so only dq needed)
      #pragma unroll
      for (int k = start; k < NUM_POS; k += delta){
         //T *dT = &s_dT[36*(NUM_POS*(NUM_POS-1) + k)];
         T *dT = &s_dT[36*k]; // looped variant only saves final dT[i]/dx but thats all we need
         s_deePos[k*6] = dT[12];
         s_deePos[k*6 + 1] = dT[13];
         s_deePos[k*6 + 2] = dT[14];
         s_deePos[k*6 + 3] = s_temp[0]*dT[10] + s_temp[1]*dT[6];
         s_deePos[k*6 + 4] = s_temp[2]*dT[6]  + s_temp[3]*dT[10] + s_temp[4]*dT[2];
         s_deePos[k*6 + 5] = s_temp[5]*dT[0]  + s_temp[6]*dT[1];
      }
   }
}
template <typename T>
__host__ __device__ __forceinline__
void compute_eePos(T *s_eePos, T *s_T, T *s_Tb, T *s_sinq, T *s_cosq, T *s_x, T *d_Tbody){
   load_Tb(s_x,s_Tb,d_Tbody,s_cosq,s_sinq);
   hd__syncthreads();
   // then compute Tbody -> T
   compute_T_TA_J(s_Tb,s_T);
   hd__syncthreads();
   //compute the hand position
   compute_eePos(s_T,s_eePos);
}
template <typename T>
__host__ __device__ __forceinline__
void compute_eePos(T *s_T, T *s_eePos, T *s_dT, T *s_deePos, T *s_sinq, 
                   T *s_Tb, T *s_dTb, T *s_x, T *s_cosq, T *d_Tbody){
   load_Tb(s_x,s_Tb,d_Tbody,s_cosq,s_sinq,s_dTb);
   hd__syncthreads();
   // then compute Tbody -> T
   compute_T_TA_J(s_Tb,s_T);
   hd__syncthreads();
   // then computde T, dTbody -> dT
   T *s_dTp = &s_dTb[16*NUM_POS]; // 16*NUM_POS so 32*NUM_POS b/c using compressed dTb
   compute_dT_dTA_dJ(s_Tb,s_dTb,s_T,s_dT,s_dTp);
   hd__syncthreads();
   //compute the hand position and derivative use sinq as temp space
   compute_eePos(s_T,s_eePos,s_dT,s_deePos,s_sinq);
}
template <typename T>
__host__ __forceinline__
void compute_eePos_scratch(T *x, T *eePos){
   T s_cosq[NUM_POS];         T s_sinq[NUM_POS];      T s_Tb[36*NUM_POS];  
   T s_T[36*NUM_POS];         T Tbody[36*NUM_POS];
   initT<T>(Tbody);           load_Tb(x,s_Tb,Tbody,s_cosq,s_sinq);   
   compute_T_TA_J(s_Tb,s_T);  compute_eePos(s_T,eePos);
}

template <typename T>
__host__ __device__ __forceinline__
void dynamics(T *s_qdd, T *s_x, T *s_u, T *d_I, T *d_Tbody, T *s_eePos = nullptr, int reps = 1){
   #ifdef __CUDA_ARCH__
      __shared__ T s_I[36*NUM_POS]; // standard inertias -->  world inertias
      __shared__ T s_Icrbs[36*NUM_POS]; // Icrbs inertias
      __shared__ T s_J[6*NUM_POS]; // kinsol.J transformation matricies
      __shared__ T s_temp[36*NUM_POS]; // temp work space (load Tbody mats into here) --> and JdotV
      __shared__ T s_temp2[36*NUM_POS]; // temp work space (compute T mats into here) --> and crm(twist) --> H and C (note if NUM_POS > 17 then need 2*NUM_POS*NUM_POS + NUM_POS)
      __shared__ T s_TA[36*NUM_POS]; // adjoint transpose --> crf(twist)
      __shared__ T s_W[6*NUM_POS]; // to store net wrenches  
      __shared__ T s_F[6*NUM_POS]; // to store forces in joint axis
      __shared__ T s_JdotV[6*NUM_POS]; // JdotV vectors
   #else
      T s_I[36*NUM_POS]; // standard inertias -->  world inertias
      T s_Icrbs[36*NUM_POS]; // Icrbs inertias
      T s_J[6*NUM_POS]; // kinsol.J transformation matricies
      T s_temp[36*NUM_POS]; // temp work space (load Tbody mats into here) --> and JdotV
      T s_temp2[36*NUM_POS]; // temp work space (compute T mats into here) --> and crm(twist) --> H and C (note if NUM_POS > 17 then need 2*NUM_POS*NUM_POS + NUM_POS)
      T s_TA[36*NUM_POS]; // adjoint transpose --> crf(twist)
      T s_W[6*NUM_POS]; // to store net wrenches  
      T s_F[6*NUM_POS]; // to store forces in joint axis
      T s_JdotV[6*NUM_POS]; // JdotV vectors
   #endif
   for(int iter = 0; iter < reps; iter++){
      T *s_xk = &s_x[STATE_SIZE*iter];
      T *s_uk = &s_u[NUM_POS*iter];
      T *s_qddk = &s_qdd[NUM_POS*iter];
      // load in I and Tbody (use W and F as temp mem)
      load_Tb(s_xk,s_temp,d_Tbody,s_F,s_W);
      load_I(s_I,d_I);
      hd__syncthreads();
      // then compute Tbody -> T -> TA & J (T and Tbody in scratch mem) again F for temp mem
      compute_T_TA_J(s_temp,s_temp2,s_TA,s_J);
      hd__syncthreads();
      // if we are asked for eePos then compute
      if (s_eePos != nullptr){compute_eePos(s_temp2,s_eePos); hd__syncthreads();}
      // then compute Iworld, Icrbs, twists (in W) and clear temp and TA for later
      compute_Iw_Icrbs_twist(s_I,s_Icrbs,s_W,s_TA,s_J,s_xk,s_temp);
      hd__syncthreads();
      // then JdotV (twists in W)
      compute_JdotV(s_JdotV,s_W,s_J,s_x,s_temp);
      hd__syncthreads();
      // finally compute F > biasForce(Tau) & [massMatrix|I] from twists in W and JdotV and Icrbs etc.
      T *s_M = s_temp2; // reuse scratch mem and note that s_TA cleared in inertia comp so use as scratch mem as well
      T *s_Tau = &s_temp2[2*NUM_POS*NUM_POS];
      compute_M_Tau(s_M, s_Tau, s_W, s_JdotV, s_F, s_Icrbs, s_W, s_J, s_I, s_xk, s_uk, s_temp, s_temp2, s_TA);
      hd__syncthreads();
      // invert Mass matrix -- assumes more threads than NUM_POS +1 by NUM_POS -- writes out [I|M^{-1}]
      #ifdef __CUDA_ARCH__
         int err = invertMatrix<T,NUM_POS,1>(s_M,s_F);
      #else
         int err = invertMatrix<T,NUM_POS>(s_M,s_F);
      #endif
      // TBD: DO SOMETHING WITH THE ERROR
      T *s_Minv = &s_temp2[NUM_POS*NUM_POS];
      hd__syncthreads();
      // finally compute qdd
      compute_qdd(s_qddk,s_Minv,s_Tau);
   }
}

template <typename T>
__host__ __device__ __forceinline__
void dynamicsGradient(T *s_dqdd, T *s_qdd, T *s_x, T *s_u, T *d_I, T *d_Tbody){
   #ifdef __CUDA_ARCH__
      __shared__ T s_I[36*NUM_POS]; // standard inertis -->  world inertias
      __shared__ T s_Icrbs[36*NUM_POS]; // crbs inertias (load dTbody here to start)
      __shared__ T s_TA[42*NUM_POS]; // adjoint transpose --> crf(twist)
      __shared__ T s_dTA[36*NUM_POS*NUM_POS]; // derive adjoint transpose --> dM --> dTwist, dJdotV, dW
      __shared__ T s_J[6*NUM_POS]; // kinsol.J transformation matricies
      __shared__ T s_dJ[6*NUM_POS*NUM_POS]; // derivative of J
      __shared__ T s_JdotV[6*NUM_POS]; // JdotV vectors
      __shared__ T s_twist[6*NUM_POS]; // twist vectors
      __shared__ T s_W[6*NUM_POS]; // to store net wrenches 
      __shared__ T s_F[6*NUM_POS]; // to store forces in joint axis
      __shared__ T s_temp[36*NUM_POS]; // temp work space (load Tbody into here)
      __shared__ T s_temp2[36*NUM_POS]; // temp work space (compute T mats into here) --> H and C (note if NUM_POS > 17 then need 2*NUM_POS*NUM_POS + NUM_POS)
      __shared__ T s_temp3[36*NUM_POS*NUM_POS]; // compute dT here --> then compute dIw here
   #else
      T s_I[36*NUM_POS]; // standard inertis -->  world inertias
      T s_Icrbs[36*NUM_POS]; // crbs inertias (load dTbody here to start)
      T s_TA[42*NUM_POS]; // adjoint transpose --> crf(twist)
      T s_dTA[36*NUM_POS*NUM_POS]; // derive adjoint transpose --> dM --> dTwist, dJdotV, dW
      T s_J[6*NUM_POS]; // kinsol.J transformation matricies
      T s_dJ[6*NUM_POS*NUM_POS]; // derivative of J
      T s_JdotV[6*NUM_POS]; // JdotV vectors
      T s_twist[6*NUM_POS]; // twist vectors
      T s_W[6*NUM_POS]; // to store net wrenches 
      T s_F[6*NUM_POS]; // to store forces in joint axis
      T s_temp[36*NUM_POS]; // temp work space (load Tbody into here)
      T s_temp2[36*NUM_POS]; // temp work space (compute T mats into here) --> H and C (note if NUM_POS > 17 then need 2*NUM_POS*NUM_POS + NUM_POS)
      T s_temp3[36*NUM_POS*NUM_POS]; // compute dT here --> then compute dIw here
   #endif
   // compute Tbody and dTbody (in temp and Icrbs) and use W and F as temp memory
   T *s_Tb = s_temp; // 36*NUM_POS
   T *s_dTb = s_temp2; // 16*NUM_POS
   load_Tb(s_x,s_Tb,d_Tbody,s_W,s_F,s_dTb);
   load_I(s_I,d_I);
   hd__syncthreads();
   T *s_T = s_Icrbs; // 16*NUM_POS
   // then compute Tbody -> T -> TA & J (T and Tbody in scratch mem)
   compute_T_TA_J(s_Tb,s_T,s_TA,s_J);
   hd__syncthreads();
   // then computde dTbody,T, TA -> dT -> dTA & dJ
   T *s_dT = s_temp3; // 36*NUM_POS
   T *s_dTp = &s_temp2[16*NUM_POS]; // 16*NUM_POS so 32*NUM_POS
   compute_dT_dTA_dJ(s_Tb,s_dTb,s_T,s_dT,s_dTp,s_TA,s_dTA,s_dJ);
   hd__syncthreads();
   // compute Iworld, Icrbs, twists, and dIw (in dTA b/c now done with that) and clear temp and TA for later
   compute_Iw_Icrbs_twist(s_I,s_Icrbs,s_twist,s_TA,s_J,s_x,s_temp,s_dTA,s_temp2);
   T *s_dIw = s_dTA;
   hd__syncthreads();
   // now finish normal comp before doing rest of dervatives so compute JdotV
   compute_JdotV(s_JdotV,s_twist,s_J,s_x,s_temp);
   hd__syncthreads();
   // then compute F > biasForce(Tau) & [massMatrix|I]
   T *s_M = s_temp2; // reuse scratch mem and note that s_TA cleared in inertia comp so use as scratch mem as well
   T *s_Tau = &s_temp2[2*NUM_POS*NUM_POS];
   compute_M_Tau(s_M, s_Tau, s_W, s_JdotV, s_F, s_Icrbs, s_twist, s_J, s_I, s_x, s_u, s_temp, s_temp2, s_TA);
   hd__syncthreads();
   // invert Mass matrix -- assumes more threads than NUM_POS +1 by NUM_POS -- writes out [I|M^{-1}]
   #ifdef __CUDA_ARCH__
      int err = invertMatrix<T,NUM_POS,1>(s_M,s_F);
   #else
      int err = invertMatrix<T,NUM_POS>(s_M,s_F);
   #endif
   // TBD: DO SOMETHING WITH THE ERROR
   T *s_Minv = &s_temp2[NUM_POS*NUM_POS];
   hd__syncthreads();
   // finally compute qdd
   compute_qdd(s_qdd,s_Minv,s_Tau);
   
   // ---------------------------------------------
   // note we now have:
   //    -J, dJ in s_J, s_dJ
   //    -Iw, dIw in s_I, s_dTA
   //    -Icrbs in s_Icrbs
   //    -invM in s_temp2
   //    -C    in s_temp2
   //    -twists in s_twist
   //    -netW   in s_W
   //    -JdotV in s_JdotV
   //    -u in s_u
   //    -qdd,qd in s_x
   // we should also be about maxed out in shared memory so we need
   // to loop to finish this up and note that we still have free:
   // s_temp[36*NB],s_F[6*NB],s_TA[42*NB = 6*NB*NB],s_temp3[36*NB*NB]
   // ---------------------------------------------

   // IF WE CAN LOOP TO REDUCE THE USAGE OF s_temp3 down to 36*NB we can reduce it in size b/c not using big before here

   // dM is going to be size NB*NB*NB so can store in s_temp3 as long as NB < 36
   T *s_dM = s_temp3;
   compute_dM(s_dM,s_Icrbs,s_dIw,s_J,s_dJ,s_F,s_TA);
   hd__syncthreads();
   // the comptue first half of dqdd with relation to dM (dqdd_du is just Minv which we already have for arm)
   compute_dqdd_dM(s_dqdd,s_dM,s_Minv,s_qdd,s_temp);
   hd__syncthreads();
   // now we need to form dTau to compute the other half of dqdd_dx
   // unfortunately we need to loop to save memory
   // note we can reuse the temp memory space as we are done with dM
   // T *s_dTwist = s_temp3;
   // T *s_dTwistp = &s_temp3[12*NUM_POS];
   // T *s_dJdotV = &s_temp3[2*12*NUM_POS];
   // T *s_dJdotVp = &s_temp3[3*12*NUM_POS];
   // T *s_dWb = &s_temp3[4*12*NUM_POS];
   // T *s_dWp = &s_temp3[5*12*NUM_POS];
   // T *s_dTau = &s_temp3[6*12*NUM_POS];
   // compute_dTau(s_dTau,s_W,s_dWb,s_dWp,s_JdotV,s_dJdotV,s_dJdotVp,s_twist,s_dTwist,s_dTwistp,s_I,s_dIw,s_J,s_dJ,s_x,s_temp,s_temp2,s_Icrbs,s_F);
   T *s_dTwist = s_temp3;
   T *s_dJdotV = &s_temp3[6*(2*NUM_POS)*NUM_POS];
   T *s_dWb = &s_temp3[6*(4*NUM_POS)*NUM_POS];
   // so compute dTwist then dJdotV then dWb
   compute_dtwist(s_dTwist,s_J,s_dJ,s_x);
   hd__syncthreads();
   compute_dJdotV(s_dJdotV,s_twist,s_dTwist,s_J,s_dJ,s_x,s_temp,s_TA);
   hd__syncthreads();
   compute_dWb(s_dWb,s_JdotV,s_dJdotV,s_twist,s_dTwist,s_I,s_dIw,s_temp,s_TA,s_F);
   hd__syncthreads();
   // use those to comptue dTau
   T *s_dTau = s_temp;
   compute_dTau(s_dTau,s_dWb,s_W,s_J,s_dJ);
   hd__syncthreads();
   // finally dqdd += invH*dTau to compute the second part of dqdd_dx and load dqdd_du
   finish_dqdd(s_dqdd,s_dTau,s_Minv);
}
/*** KINEMATICS AND DYNAMICS HELPERS ***/