/*
 kmeans3.cu:  	Cuda implementation of the K-means algorithm.

 input:		 N 	number of data
		 DIMS	number of dimensions
		 K	number of clusters
		 threads number of threads per block
		 D	debug 1, no debug 0
		 F	output filename

 output:	 means at the beggining
		 means at the end
		 number of iterations, execution time, throughput
		 labeled data with cluster in the output filename [optional]
                 source data in filename.src
		 lableled data in filename.dat

 procedure:	 Generate random means.
		 Generate random data with a normal distribution for each mean using a
		 standard deviation
		 Copy data to the GPU
		 Apply k-means algorith iteratively in the GPU controled by the CPU
		 Copy results to CPU
		 Save results to the output file (optional).

 author:         Arturo Diaz-Perez
 date:           February 18, 2020.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>

#define MAX_VALUE  1000
#define MAXITER    1000
#define RNDM      ((float )rand()/(float )(RAND_MAX))


__global__ void initializingClusterCounters(int *d_count, int no_clusters)
{
    int i = threadIdx.x;

    // if( i < no_clusters )
        d_count[i] = 0;
}

__global__ void computingMinCluster( int *d_data, int *d_cluster, int *d_mean, int *d_count,
           int no_data, int no_dims, int no_clusters, int *no_changes ) {
    int i = blockIdx.x;  // a block per data
    int k = threadIdx.x; // a thread per cluster
    int error, *scluster;

    extern __shared__ int sdata[];

    scluster = &(sdata[blockDim.x]);
    __syncthreads();

    sdata[k] = 0;
    scluster[k] = k;

    for( int m = 0; m < no_dims; m++ ) {
        error = d_data[i*no_dims+m] - d_mean[k*no_dims+m];
        sdata[k] += error*error;
    }
    __syncthreads();

    // Reduction process to compute the minimum.
    // This version requires the number of cluster a power of 2
    // Modifications needed if the number of clusters is different
    for( unsigned int s=blockDim.x/2; s > 0; s >>= 1 ) {
        if( k < s ) {
            if( sdata[k] > sdata[k+s] ) {
                sdata[k] = sdata[k + s];
                scluster[k] = scluster[k+s];
            }
        }
        __syncthreads();
    }

    // write result for this block to global mem cluster[i] <- min

    if((scluster[0] != d_cluster[i]) && k == 0 ) {
        no_changes[i] = 1;
        d_cluster[i] = scluster[0];
    }
    else{
      no_changes[i] = 0;
    }
    if( k == 0 ) atomicAdd( &(d_count[scluster[0]]), 1);
}

__global__ void prescan(int *g_odata, int *g_idata, int n) {

    extern __shared__ int temp[];

    int blid = blockIdx.x;
    int thid = threadIdx.x;

    int offset = 1;

    temp[2*blid*thid] = g_idata[2*blid*thid];
    temp[2*blid*thid+1] = g_idata[2*blid*thid+1];

    for (int d = n>>1; d > 0; d >>= 1)
    {
      __syncthreads();

      if (blid*thid < d)
      {
        int ai = offset*(2*blid*thid+1)-1;
        int bi = offset*(2*blid*thid+2)-1;

        temp[bi] += temp[ai];
      }
      offset *= 2;
    }

    if (blid*thid == 0)
    {
      temp[n - 1] = 0;
    }

    for (int d = 1; d < n; d *= 2)
    {
      offset >>= 1;
      __syncthreads();

        if (blid*thid < d)
        {
          int ai = offset*(2*blid*thid+1)-1;
          int bi = offset*(2*blid*thid+2)-1;

          int t = temp[ai];
          temp[ai]  = temp[bi];
          temp[bi] += t;
        }
    }

    __syncthreads();

    g_odata[2*blid*thid] = temp[2*blid*thid];
    g_odata[2*blid*thid+1] = temp[2*blid*thid+1];
}

__global__ void initializeMeans( int *d_mean, int no_dims, int no_clusters )
{
    int i = blockIdx.x;
    int m = threadIdx.x;

    // if( i < no_clusters && m < no_dims )
        d_mean[i*no_dims+m] = 0;
}

// This is to update means once each data is assigned to a cluster
__global__ void addDataToCluster( int *d_data, int *d_cluster, int *d_mean,
           int no_data, int no_dims, int no_clusters )
{
    int i = blockIdx.x;
    int m = threadIdx.x;

    // if( i < no_clusters && m < no_dims ) {
        int k = d_cluster[i];
        // d_mean[k*no_dims+m] += d_data[i*no_dims+m];
        // this is not efficient since all threads perform atomic adds
        atomicAdd( &(d_mean[k*no_dims+m]), d_data[i*no_dims+m] );
    // }
}

// Performs the final division to update the means
__global__ void divideMeans( int *d_mean, int no_dims, int no_clusters, int *d_count )
{
    int i = blockIdx.x;
    int m = threadIdx.x;

    // if( i < no_clusters && m < no_dims )
        d_mean[i*no_dims+m] /= d_count[i];
}

// Outputs mean values
void printMeans( int *mean, int *count, int no_dims, int no_clusters)
{
    printf( "Means of each cluster\n" );
    int sum_clusters = 0;
    for( int k = 0; k < no_clusters; k++ ) {
        sum_clusters += count[k];
        printf( "Mean[%2d] = [ ", k );
        for( int m = 0; m < no_dims; m++ )
            printf( "%3d, ", mean[k*no_dims+m] );
        printf( "]. %6d data\n", count[k] );
    }
    printf( "Total data in clusters %7d\n", sum_clusters );
}

// Output cluster counters
void printCounters( int *count, int no_clusters )
{
    printf( "Counters of each cluster\n" );
    int ndata = 0;
    for( int i = 0; i < no_clusters; i++) {
        printf("Count[%d] =  %d\n", i, count[i] );
        ndata += count[i];
    }
    printf( "Total data in clusters %d\n", ndata );
}

// Global variables for problem parameters
int no_data, no_dims, no_clusters;
int numberOfBlocks,threadsPerBlock = 128;
int totalThreadsPerBlock;
int Debug = 0;
char fname[50] = "";
char src_file[50] = "";
char out_file[50] = "";

// Check arguments in the command line
void checkArgs( int argc, char **argv )
{
    if( argc < 6 ) {
       printf( "Usage %s N DIMS CLUSTERS ThreadsPerBlock Debug [file output]\n", argv[0] );
       exit( 0 );
    }

    no_data = atoi( argv[1] ) * 1024;
    if( no_data > (1<<25) ) {
       printf( "Size <= 32M. Size set to 1M\n" );
       no_data = 1024*1024;
    }

    no_dims = atoi( argv[2] );
    if( no_dims < 0  || no_dims > 32 ) {
       no_dims = 8;
       printf( "1 <= DIMS <= 32. DIMS set to %d\n", no_dims );
    }

    no_clusters = atoi( argv[3] );
    if( no_clusters < 0  || no_clusters > 32 ) {
       no_clusters = 8;
       printf( "1 <= K <= 32. K set to %d\n", no_clusters );
    }

    threadsPerBlock = atoi( argv[4] );
    totalThreadsPerBlock = threadsPerBlock * no_clusters;
    if( (totalThreadsPerBlock <  32) || (totalThreadsPerBlock > 1024) ) {
       threadsPerBlock = 1024 / no_clusters;
       printf( "Threads per block set to %d\n", threadsPerBlock );
    }

    Debug = atoi( argv[5] );
    if( Debug != 0 ) Debug = 1;

    if( argc > 6 ) {
        strcpy( fname, argv[6] );
        strcpy( src_file, fname );
        strcat( src_file, ".src" );
        strcpy( out_file, fname );
        strcat( out_file, ".dat" );
    }

}

// Save all labeled data
void fileSaveData( char *fn, int *data, int *cluster, int no_data, int no_dims )
{
    FILE *fp;
    int  i, k;

    if( !(fp = fopen( fn, "w" )) ) {
        printf( "Error: not possible open output file\n" );
        return ;
    }
    for( i = 0; i < no_data; i++ ) {
        for( k = 0; k < no_dims; k++ )
            fprintf(fp, "\t%5d", data[i*no_dims + k] );
        fprintf( fp, "\t%5d\n", cluster[i] );
    }
    fclose( fp );
}

// Generate data uniform distributed in (0,0) - (MAX_VALUE,MAXVALUE) rectangle
// Initialize means choosing random point inside the region
//
void generateUniformData( int *data, int *cluster, int *mean, int *count,
                          int no_data, int no_dims, int no_cluster, int *no_changes, int *no_changes_sum)
{
   int i, k, m;

   srand( time(0) );
    for( k = 0; k < no_clusters; k++ ) {
        count[k] = 0;
        for( m = 0; m < no_dims; m++ )
            mean[k*no_dims+m] = (float )(((float )rand()/(float ) RAND_MAX)*MAX_VALUE);
    }

    for( i = 0; i < no_data; i++ ) {
        no_changes[i] = 0;
        no_changes_sum[i] = 0;
        cluster[i] = (float )(((float )rand()/(float ) RAND_MAX)*no_clusters);
        count[cluster[i]]++;
        for( m = 0; m < no_dims; m++ )
            data[i*no_dims+m] = (float )(((float )rand()/(float ) RAND_MAX)*MAX_VALUE);
    }
}


double rand_gen() {
   // return a uniformly distributed random value
   return ( (double)(rand()) + 1. )/( (double)(RAND_MAX) + 1. );
}

double normalRandom() {
   // return a normally distributed random value
   double v1=rand_gen();
   double v2=rand_gen();
   return cos(2*3.14*v2)*sqrt(-2.*log(v1));
}


// Generate data normal distributed around each mean with random standard deviation
// Initial data in (0,0) - (MAX_VALUE,MAXVALUE) rectangle
// Initialize means choosing random point inside the region
//
void generateNormalData( int *data, int *cluster, int *mean, int *count, float *ratio,
                          int no_data, int no_dims, int no_cluster )
{
   int i, k, m, centroid;

   srand( time(0) );
   for( k = 0; k < no_clusters; k++ ) {
       count[k] = 0;
       for( m = 0; m < no_dims; m++ ) {
           mean[k*no_dims+m] = (float )(RNDM*MAX_VALUE);
           if( rand() > RAND_MAX/2 ) mean[k*no_dims+m] = -mean[k*no_dims+m];
       }
       ratio[k] = (float )(MAX_VALUE)*0.2*RNDM;
   }

   for( i = 0; i < no_data; i++ ) {
       centroid = (float )(no_clusters*RNDM);
       cluster[i] = (float )(no_clusters*RNDM);
       count[cluster[i]]++;
       for( m = 0; m < no_dims; m++ ) {
           float xm = normalRandom()*ratio[centroid];
           data[i*no_dims+m] = xm + mean[centroid*no_dims+m];
       }
   }
}


int main(int argc, char **argv)
{

    int iter;
    int *no_changes;
    int *no_changes_sum;
    int *data, *cluster, *mean, *count;
    float *ratio;
    cudaEvent_t     start, stop;
    float           elapsedTime, bw;

    checkArgs( argc, argv );

    // generate the input array on the host
    data = (int *) malloc( no_data*no_dims*sizeof(int) );
    cluster = (int *) malloc( no_data*sizeof( int ) );
    mean = (int *) malloc( no_clusters*no_dims*sizeof( int ) );
    count = (int *) malloc( no_clusters*sizeof( int ) );
    ratio = (float *) malloc( no_clusters*sizeof( float ) );
    no_changes = (int *) malloc( no_data*sizeof( int ) );
    no_changes_sum = (int *) malloc( no_data*sizeof( int ) );

    if( data == NULL || cluster == NULL || mean == NULL || count == NULL ||
        ratio == NULL || no_changes==NULL || no_changes_sum==NULL) {
       printf( "Error allocating memory in CPU\n" );
       exit( 0 );
    }

    // Get information about GPU device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %ld MB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, devProps.totalGlobalMem/(1024*1024),
               (int)devProps.major, (int)devProps.minor,
               (int)devProps.clockRate);
    }

    // declare GPU memory pointers
    int *d_data;	// array for all data
    int *d_cluster;	// array for labeling data with cluster
    int *d_mean;	// array for means
    int *d_count;	// array for each cluster's counter
    int *d_no_changes;	// variable to mark if there are changes in the labeling
    int *d_no_changes_sum;
    // allocate GPU memory
    int r1 = cudaMalloc((void **) &d_data, no_data*no_dims*sizeof( int ) );
    int r2 = cudaMalloc((void **) &d_cluster, no_data*sizeof( int ));
    int r3 = cudaMalloc((void **) &d_mean, no_clusters*no_dims*sizeof( int ));
    int r4 = cudaMalloc((void **) &d_count, no_clusters*sizeof( int ));
    int r5 = cudaMalloc((void **) &d_no_changes, no_data*sizeof( int ));
    int r6 = cudaMalloc((void **) &d_no_changes_sum, no_data*sizeof( int ));
    if( r1 || r2 || r3 || r4 || r6 || r5) {
       printf( "Error allocating memory in GPU\n" );
       exit( 0 );
    }

    generateUniformData( data, cluster, mean, count, no_data, no_dims, no_clusters, no_changes, no_changes_sum);
    // generateNormalData( data, cluster, mean, count, ratio, no_data, no_dims, no_clusters );
    if( strlen( fname ) > 1 ) fileSaveData( src_file, data, cluster, no_data, no_dims );

    printMeans( mean, count, no_dims, no_clusters );
    // printCounters( count, no_clusters );


    // transfer the arrays to the GPU
    cudaMemcpy(d_data, data, no_data*no_dims*sizeof( int ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cluster, cluster, no_data*sizeof( int ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, mean, no_clusters*no_dims*sizeof( int ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, count, no_clusters*sizeof( int ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_no_changes, no_changes, no_data*sizeof( int ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_no_changes_sum, no_changes_sum, no_data*sizeof( int ), cudaMemcpyHostToDevice);

    numberOfBlocks = no_data / threadsPerBlock;
    if( no_data > numberOfBlocks*threadsPerBlock ) numberOfBlocks++;


    // Creating events to estimate execution time
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    cudaEventRecord( start, 0 ); // Starting clock

    dim3 Grid(no_data);
    dim3 Blocks(no_clusters);
    //no_changes_sum[0] = 1;
    for( iter = 0; (iter < MAXITER); iter++ ) {
      // launch the kernel
      if( Debug || (iter % 5 == 0) )
          printf(" ************************ ITERATION %d\n", iter );

      if( Debug ) printf("Running kernel to initializa counter\n");
      initializingClusterCounters<<<1,no_clusters>>>(d_count, no_clusters );

      // launch kernel to compute minimum distance cluster
      int size_sharedmem = 2*no_clusters*sizeof(int);
      int size_sharedmem_sum = 2*no_data*sizeof(int);

      if( Debug ) printf("Running kernel to update close clusters\n");
      computingMinCluster<<<no_data, no_clusters, size_sharedmem>>>(
         d_data, d_cluster, d_mean, d_count,
         no_data, no_dims, no_clusters, d_no_changes );
       if(Debug) printf("prescan gfhghdfghdfghdfgdfgh");
      prescan<<< numberOfBlocks, threadsPerBlock , size_sharedmem_sum>>>(d_no_changes_sum, d_no_changes, no_data);
      // copy back the sum from GPU
      if( Debug )
      cudaMemcpy(count, d_count, no_clusters*sizeof( int ), cudaMemcpyDeviceToHost);
      cudaMemcpy(&no_changes, d_no_changes, no_data*sizeof( int ), cudaMemcpyDeviceToHost);
      cudaMemcpy(&no_changes_sum, d_no_changes_sum, no_data*sizeof( int ), cudaMemcpyDeviceToHost);
      if( Debug ) printf( "Value of no_changes: %d\n", no_changes_sum[0] );
      if( !(no_changes_sum[0] > 0) ) break;

      if( Debug ) {
          printf( "Counters after GPU invocation\n" );
          printCounters( count, no_clusters );
      }

      if( Debug ) printf("Running kernel to set means to zero\n");
      initializeMeans<<<no_clusters,no_dims>>>(
           d_mean, no_dims, no_clusters );

      // Copy back means from GPU
      if( Debug ) {
          cudaMemcpy(mean, d_mean, no_clusters*no_dims*sizeof( int ), cudaMemcpyDeviceToHost);
          printf( "Means after GPU initializeMeans\n" );
          printMeans( mean, count, no_dims, no_clusters );
      }

      addDataToCluster<<<no_data,no_dims>>>( d_data, d_cluster, d_mean,
           no_data, no_dims, no_clusters );

      // Copy back means from GPU
      if( Debug ) {
          cudaMemcpy(mean, d_mean, no_clusters*no_dims*sizeof( int ), cudaMemcpyDeviceToHost);
          printf( "Means after GPU addDataToCluster\n" );
          printMeans( mean, count, no_dims, no_clusters );
      }

      divideMeans<<<no_clusters,no_dims>>>( d_mean, no_dims, no_clusters, d_count );

      // copy back means from GPU
      if( Debug ) {
          cudaMemcpy(mean, d_mean, no_clusters*no_dims*sizeof( int ), cudaMemcpyDeviceToHost);
          printf( "Means after GPU divideMeans\n" );
          printMeans( mean, count, no_dims, no_clusters );
      }

    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );

    cudaMemcpy(cluster, d_cluster, no_data*sizeof( int ), cudaMemcpyDeviceToHost);
    cudaMemcpy(count, d_count, no_clusters*sizeof( int ), cudaMemcpyDeviceToHost);
    cudaMemcpy(mean, d_mean, no_clusters*no_dims*sizeof( int ), cudaMemcpyDeviceToHost);

    printf( "Means at the end of %d iterations\n", iter );
    printf( "Means after GPU divideMeans\n" );
    printMeans( mean, count, no_dims, no_clusters );

    bw = (float )no_data*(float )no_dims*(float )no_clusters*(float )iter;
    bw /= elapsedTime*1000000.0;
    printf( "Kmeans GPU execution time: %7.3f ms, Throughput %6.2f GFLOPS\n", elapsedTime, bw );
    if( strlen( fname ) > 1 ) fileSaveData( out_file, data, cluster, no_data, no_dims );

    // Destroying event variables
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    // free GPU memory allocation
    cudaFree(d_data);
    cudaFree(d_cluster);
    cudaFree(d_mean );
    cudaFree(d_count );

    free( data );
    free( cluster );
    free( mean );
    free( count );

    return 0;
}
