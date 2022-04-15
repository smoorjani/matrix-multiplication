#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#include "cuda.h"

#define NUMBLOCKS 1
#define BLOCKSIZE 1024
#define REGISTER 1
#define BUFFSIZE 96
#define WARPSIZE 32

#define T float

#define PRINT 0 
enum TiledSpMM_order_t {TiledSpMM_COLMAJOR, TiledSpMM_ROWMAJOR};
#define ORDER TiledSpMM_COLMAJOR

//TiledSpMM HANDLE STRUCTURE
struct TiledSpMM_handle {
  int M;
  int N;
  int K;

  int numblock;
  int tilesize;
  int *tiledispl;
  int *mapdispl;
  int *mapindex;
  int *elldispl;
  unsigned short *ellindex;
  T *ellvalue;

  dim3 grid;
  dim3 block;
};


 //TiledSpMM KERNEL
__global__ void __launch_bounds__(BLOCKSIZE,NUMBLOCKS) kernel_TiledELL(T *B, T *C,
                                                                       int M,
								       int N,
								       int K,
                                                                       int tilesize,
                                                                       int *tiledispl,
                                                                       int *mapdispl,
                                                                       int *mapindex,
                                                                       int *elldispl,
                                                                       unsigned short *ellindex,
                                                                       T *ellvalue){
  extern __shared__ T shared[];
  T acc[REGISTER] = {0.0};
  ellindex += threadIdx.x%WARPSIZE;
  ellvalue += threadIdx.x%WARPSIZE;
  int coloffset = blockIdx.y*REGISTER;
  #pragma unroll(1)
  for(int tile = tiledispl[blockIdx.x]; tile < tiledispl[blockIdx.x+1]; tile++){
    int mapoffset = mapdispl[tile];

    //COLUMN-MAJOR FORMAT
    if(ORDER == TiledSpMM_COLMAJOR)
    #pragma unroll(1)
    for(int n = threadIdx.x; n < mapdispl[tile+1]-mapoffset; n += blockDim.x){
      int index = mapindex[mapoffset+n];
      #pragma unroll
      for(int k = 0; k < REGISTER; k++)
        if(coloffset + k < K)
            shared[k*tilesize+n] = B[(coloffset+k)*N+index];
    }
    //ROW-MAJOR FORMAT
    if(ORDER == TiledSpMM_ROWMAJOR)
    #pragma unroll
    for(int k = 0; k < REGISTER; k++)
      #pragma unroll(1)
      for(int n = threadIdx.x; n < mapdispl[tile+1]-mapoffset; n += blockDim.x)
        shared[k*tilesize+n] = B[mapindex[mapoffset+n]*K+coloffset+k];

    __syncthreads();
    int warp = tile*(blockDim.x/WARPSIZE)+threadIdx.x/WARPSIZE;
    #pragma unroll(8)
    for(int m = elldispl[warp]; m < elldispl[warp+1]; m++){
      int index = ellindex[m*(long)WARPSIZE];
      T value = ellvalue[m*(long)WARPSIZE];
      #pragma unroll
      for(int k = 0; k < REGISTER; k++)
        acc[k] += shared[k*tilesize+index]*value;
    }
    __syncthreads();
    int m = blockIdx.x*blockDim.x+threadIdx.x;
    if(m < M)
      #pragma unroll
      for(int k = 0; k < REGISTER; k++)
        if(coloffset + k < K){
          if(ORDER == TiledSpMM_COLMAJOR)
            C[(coloffset+k)*M+m] = acc[k];
          if(ORDER == TiledSpMM_ROWMAJOR)
            C[m*K+coloffset+k] = acc[k];
	}
  }
};

void print_arr(T* arr, int size) {
  for (int i = 0; i < size; i++) {
    printf("%f ", T(arr[i])); 
  }
  printf("\n");
}
  

//CONVERTS COO->CSR
void TiledSpMM_coo2csr(int M, int N, long nnz, int *coo_rowidx, int *coo_colidx, T *coo_value, long **csr_displ, long **csr_index, T **csr_value) {

  long *csrindex = new long[nnz]; //csr_col
  long *csrdispl = new long[M + 1]; //csr_row
  T *csrvalue = new T[nnz];//csr_val

  for (int i = 0; i < M + 1; i++) {
    csrdispl[i] = 0;
  }
    
  for (int i = 0; i < nnz; i++) {
    csrvalue[i] = coo_value[i];
    csrindex[i] = coo_colidx[i];
    csrdispl[coo_rowidx[i] + 1]++;
  }

  for (int i = 0; i < M; i++) {
    csrdispl[i + 1] += csrdispl[i];
  }

  *csr_displ = csrdispl;
  *csr_index = csrindex;
  *csr_value = csrvalue;

}


void TiledSpMM_inspect(int M, int N, int K, long *csr_displ, long *csr_index, T *csr_value, TiledSpMM_handle *handle) {

  //PRINT MATRIX DIMENSIONS
  if (PRINT == 1) {
    printf("\nA: %d NZ (%.2f GB) B: %d x %d (%.2f GB) C: %d x %d (%.2f GB)\n",csr_displ[M],csr_displ[M]*(sizeof(int)+sizeof(T))/1.0e9,N,K,M*sizeof(T)*K/1.0e9,M,K,M*sizeof(T)*K/1.0e9);
    double density = csr_displ[M]/(M*(double)N);
    printf("DENSITY: %e SPARSITY: %e\n", density, 1-density);
    int maxnz = 0;
    int minnz = N;
    for(int m = 0; m < M; m++) {
      int nz = csr_displ[m+1] - csr_displ[m];
      if(nz > maxnz) maxnz = nz;
      if(nz < minnz) minnz = nz;
    }
    printf("ROW STATS: MAXNZ: %d MINNZ: %d AVNZ: %.2f\n",maxnz,minnz,N*density);
    printf("\n");
  }

  int numblock = (M+BLOCKSIZE-1)/BLOCKSIZE;
  int tilesize = ((BUFFSIZE*1024)/sizeof(T))/REGISTER;
  if (PRINT == 1) {
    printf("BUFFSIZE %d KB (%d ELEMENTS)\n",BUFFSIZE,tilesize);
    printf("NUMBLOCK %d BLOCKSIZE %d NUMWARP %d\n",numblock,BLOCKSIZE,(numblock*BLOCKSIZE)/WARPSIZE);
    printf("PREPROC BUFFER: %lu (%.2f GB)\n",numblock*(long)N,sizeof(int)*numblock*(T)N/1.0e9);
  }

  //FIRST LOOP
  int *tagbuff = new int[numblock*(long)N];
  int *numtile = new int[numblock];

  #pragma parallel omp for
  for(int block = 0; block < numblock; block++){
    int *tag = tagbuff + block*(long)N;
    for(int n = 0; n < N; n++)
      tag[n] = -1;
    for(int m = block*BLOCKSIZE; m < (block+1)*BLOCKSIZE; m++)
      if(m < M)
        for(long l = csr_displ[m]; l < csr_displ[m+1]; l++)
          tag[csr_index[l]] = 0;
    int footprint = 0;
    for(int n = 0; n < N; n++)
      if(tag[n] > -1){
        tag[n] = footprint;
        footprint++;
      }
    numtile[block] = (footprint+tilesize-1)/tilesize;
  }
  int *tiledispl = new int[numblock+1];
  tiledispl[0] = 0;
  for(int m = 1; m < numblock+1; m++)
    tiledispl[m] = tiledispl[m-1] + numtile[m-1];
  
  if (PRINT == 1) {
    printf("NUMBER OF TILES: %d (%f PER BLOCK)\n",tiledispl[numblock],tiledispl[numblock]/(T)numblock);
  }
  
  //SECOND LOOP
  int *mapnz = new int[tiledispl[numblock]];
  #pragma omp parallel for
  for(int tile = 0; tile < tiledispl[numblock]; tile++)
    mapnz[tile] = 0;
  int *ellnz = new int[tiledispl[numblock]*(BLOCKSIZE/WARPSIZE)];

  #pragma omp parallel for
  for(int block = 0; block < numblock; block++){
    int *tag = tagbuff + block*(long)N;
    for(int n = 0; n < N; n++)
      if(tag[n] > -1)
        mapnz[tiledispl[block]+tag[n]/tilesize]++;
    int threadnz[numtile[block]*BLOCKSIZE];
    for(int n = 0; n < numtile[block]*BLOCKSIZE; n++)
      threadnz[n] = 0;
    for(int thread = 0; thread < BLOCKSIZE; thread++){
      int m = block*BLOCKSIZE+thread;
      if(m < M)
        for(long l = csr_displ[m]; l < csr_displ[m+1]; l++)
          threadnz[(tag[csr_index[l]]/tilesize)*BLOCKSIZE+thread]++;
    }
    for(int tile = 0; tile < numtile[block]; tile++)
      for(int warp = 0; warp < BLOCKSIZE/WARPSIZE; warp++){
        int max = 0;
        for(int thread = 0; thread < WARPSIZE; thread++)
          if(threadnz[tile*BLOCKSIZE+warp*WARPSIZE+thread] > max)
            max = threadnz[tile*BLOCKSIZE+warp*WARPSIZE+thread];
        ellnz[(tiledispl[block]+tile)*(BLOCKSIZE/WARPSIZE)+warp] = max;
      }
  }


  int *elldispl = new int[tiledispl[numblock]*(BLOCKSIZE/WARPSIZE)+1];
  elldispl[0] = 0;
  for(int n = 1; n < tiledispl[numblock]*(BLOCKSIZE/WARPSIZE)+1; n++)
    elldispl[n] = elldispl[n-1] + ellnz[n-1];
  
  long elltotal = elldispl[tiledispl[numblock]*(BLOCKSIZE/WARPSIZE)]*(long)WARPSIZE;

  if (PRINT == 1) {
    printf("SLICED-ELL DIMENSION: %d\n", elldispl[tiledispl[numblock]*(BLOCKSIZE/WARPSIZE)]);
    printf("ELL TOTAL %ld (%f GB) ZERO-PADDING OVERHEAD: %f%%\n", elltotal, elltotal*(sizeof(T)+sizeof(unsigned short))/1.0e9, ((elltotal-csr_displ[M])/(T)csr_displ[M])*100);
  }
  
  //THIRD LOOP
  unsigned short *ellindex = new unsigned short[elltotal];
  T *ellvalue = new T[elltotal];
  #pragma omp parallel for
  for(long n = 0; n < elltotal; n++){
    ellindex[n] = 0;
    ellvalue[n] = 0.0;
  }

  int *mapdispl = new int[tiledispl[numblock]+1];
  mapdispl[0] = 0;
  for(int tile = 1; tile < tiledispl[numblock]+1; tile++)
    mapdispl[tile] = mapdispl[tile-1] + mapnz[tile-1];

  int *mapindex = new int[mapdispl[tiledispl[numblock]]];

  #pragma omp parallel for
  for(int block = 0; block < numblock; block++){
    int *tag = tagbuff + block*(long)N;
    for(int n = 0; n < N; n++)
      if(tag[n] > -1)
        mapindex[mapdispl[tiledispl[block]+tag[n]/tilesize]+tag[n]%tilesize] = n;
    int threadnz[numtile[block]*BLOCKSIZE];
    for(int n = 0; n < numtile[block]*BLOCKSIZE; n++)
      threadnz[n] = 0;
    for(int thread = 0; thread < BLOCKSIZE; thread++){
      int m = block*BLOCKSIZE+thread;
      if(m < M)
        for(long l = csr_displ[m]; l < csr_displ[m+1]; l++){
          int foot = tag[csr_index[l]];
          int tile = tiledispl[block] + foot/tilesize;
          int colind = thread%WARPSIZE;
          int rowind = elldispl[tile*(BLOCKSIZE/WARPSIZE)+thread/WARPSIZE]+threadnz[(foot/tilesize)*BLOCKSIZE+thread];
          ellindex[rowind*(long)WARPSIZE+colind] = foot%tilesize;
          ellvalue[rowind*(long)WARPSIZE+colind] = csr_value[l];
          threadnz[(foot/tilesize)*BLOCKSIZE+thread]++;
        }
    }
  }

  //delete[] csr_displ;
  //delete[] csr_index;
  //delete[] csr_value;
  delete[] tagbuff;
  delete[] numtile;
  delete[] ellnz;
  delete[] mapnz;
  

  handle->M = M;
  handle->N = N;
  handle->K = K;
  handle->numblock = numblock;
  handle->tilesize = tilesize;

  cudaFuncSetAttribute(kernel_TiledELL, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024);

  cudaSetDevice(0);
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
    
  int numBlocks;        // Occupancy in terms of active blocks
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,kernel_TiledELL,BLOCKSIZE,tilesize*sizeof(T));
  int activeWarps = numBlocks * BLOCKSIZE / deviceProp.warpSize;
  int maxWarps = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
  
  //REPORT FURTHER STATISTICS
  if (PRINT == 1) {
    
    printf("TILING MAP SIZE: %d (%f GB)\n", mapdispl[tiledispl[numblock]], sizeof(int)*(double)mapdispl[tiledispl[numblock]]/1.0e9);
    printf("SHARED-MEMORY DATA REUSE: %f DUPLICATE MEMORY ACCESS: %f\n", csr_displ[M]/(double)mapdispl[tiledispl[numblock]], mapdispl[tiledispl[numblock]]/(double)N);

    system("nvidia-smi");
    printf("\n");
    printf("Device Count: %d\n",deviceCount);
    printf("Device %d name: %s\n",dev,deviceProp.name);
    printf("Computational Capabilities: %d, %d\n",deviceProp.major,deviceProp.minor);
    printf("Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
    printf("Maximum constant memory size: %lu\n",deviceProp.totalConstMem);
    printf("Maximum shared memory size per block: %lu\n",deviceProp.sharedMemPerBlock);
    printf("Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf("Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    printf("Warp size: %d\n",deviceProp.warpSize);
    printf("\n");
    printf("Occupancy: %f%%\n",(T)activeWarps/maxWarps*100);
  }

  int *tiledispl_d;
  int *mapdispl_d;
  int *mapindex_d;
  int *elldispl_d;
  unsigned short *ellindex_d;
  T* ellvalue_d;

  cudaMalloc(&tiledispl_d, (numblock+1)*sizeof(int));
  cudaMalloc(&mapdispl_d,(tiledispl[numblock]+1)*sizeof(int));
  cudaMalloc(&mapindex_d, mapdispl[tiledispl[numblock]]*sizeof(int));
  cudaMalloc(&elldispl_d, (tiledispl[numblock]*(BLOCKSIZE/WARPSIZE)+1)*sizeof(int));
  cudaMalloc(&ellindex_d, elldispl[tiledispl[numblock]*(BLOCKSIZE/WARPSIZE)]*WARPSIZE*sizeof(unsigned short));
  cudaMalloc(&ellvalue_d, elldispl[tiledispl[numblock]*(BLOCKSIZE/WARPSIZE)]*WARPSIZE*sizeof(T));

  handle->tiledispl = tiledispl_d;
  handle->mapdispl = mapdispl_d;
  handle->mapindex = mapindex_d;
  handle->elldispl = elldispl_d;
  handle->ellindex = ellindex_d;
  handle->ellvalue = ellvalue_d;

  cudaMemcpy(tiledispl_d, tiledispl, (numblock+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(mapdispl_d, mapdispl, (tiledispl[numblock]+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(mapindex_d, mapindex, mapdispl[tiledispl[numblock]]*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(elldispl_d, elldispl, (tiledispl[numblock]*(BLOCKSIZE/WARPSIZE)+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(ellindex_d, ellindex, elldispl[tiledispl[numblock]*(BLOCKSIZE/WARPSIZE)]*WARPSIZE*sizeof(unsigned short), cudaMemcpyHostToDevice);
  cudaMemcpy(ellvalue_d, ellvalue, elldispl[tiledispl[numblock]*(BLOCKSIZE/WARPSIZE)]*WARPSIZE*sizeof(T), cudaMemcpyHostToDevice);

  dim3 grid(numblock, (K+REGISTER-1)/REGISTER);
  dim3 block(BLOCKSIZE);

  if (PRINT == 1) {
    printf("grid: %d x %d x %d\n", grid.x, grid.y, grid.z);
    printf("block: %d x %d x %d\n", block.x, block.y, block.z);
  }
  
  handle->grid = grid;
  handle->block = block;
};


void TiledSpMM_multiply(T *B, T *C, TiledSpMM_handle handle) {
  kernel_TiledELL<<<handle.grid,
                             handle.block,
                             handle.tilesize*sizeof(T)>>>(B, C,
                                                          handle.M,
                                                          handle.N,
                                                          handle.K,
                                                          handle.tilesize,
                                                          handle.tiledispl,
                                                          handle.mapdispl,
                                                          handle.mapindex,
                                                          handle.elldispl,
                                                          handle.ellindex,
                                                          handle.ellvalue);
};


void TiledSpMM_free(TiledSpMM_handle handle) {
  cudaFree(handle.tiledispl);
  cudaFree(handle.mapdispl);
  cudaFree(handle.mapindex);
  cudaFree(handle.elldispl);
  cudaFree(handle.ellindex);
  cudaFree(handle.ellvalue);
};
