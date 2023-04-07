/*
Reproduce minimal GPU-accelerrated iLQR implementation for the pendulum plant
*/

// DDPHelpers.cu also includes the config.h where the plant is confgured as a pendulum (PLANT = 1)
#include "DDPHelpers.cuh"

template <typename T>
__host__ //runs on host
void testGPU(){
    //Define project variables
    

    //Allocate memory for these variables


    //Run the iLQR GPU function





    // Free variables
}





// *************************************************************************************************


int main()
{   
    // call the GPU iLQR implementation to solve the inverted pendulum system
    // config.h Line 61: typedef float algType;
	testGPU<algType>();
    
    return 0;
}
