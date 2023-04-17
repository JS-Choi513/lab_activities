#include <iostream>
#include <cstdlib>

int main(int argc, char** argv)
{
  cudaError_t error;

  int number_of_devices;
  error = cudaGetDeviceCount(&number_of_devices);
  if (error)
  {
    std::cout << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  for (int dev_idx = 0; dev_idx < number_of_devices; ++dev_idx)
  {
    cudaDeviceProp props;
    error = cudaGetDeviceProperties(&props, dev_idx);
    if (error)
    {
      std::cout << cudaGetErrorString(error) << std::endl;
      exit(-1);
    }

    printf("Device Index: %d\n", dev_idx);
    printf(
        "  - name                       : %s\n"
        "  - totalGlobalMem             : %zu bytes\n"
        "  - sharedMemPerBlock          : %zu bytes\n"
        "  - regsPerBlock               : %d\n"
        "  - warpSize                   : %d\n"
        "  - memPitch                   : %zu bytes\n"
        "  - maxThreadsPerBlock         : %d\n"
        "  - maxThreadsDim              : %d x %d x %d\n"
        "  - maxGridSize                : %d x %d x %d\n"
        "  - clockRate                  : %d KHz\n"
        "  - totalConstMem              : %zu bytes\n"
        "  - major                      : %d\n"
        "  - minor                      : %d\n"
        "  - textureAlignment           : %zu\n"
        "  - deviceOverlap              : %d(1 for True, 0 for False)\n"
        "  - multiProcessorCount        : %d\n"
        "  - kernelExecTimeoutEnabled   : %d(1 for True, 0 for False)\n"
        "  - integrated                 : %d(1 for True, 0 for False)\n"
        "  - canMapHostMemory           : %d(1 for True, 0 for False)\n"
        "  - computeMode                : %d(0 for Default, 1 for Exclusive, 2 for Prohibited, 3 for ExclusiveProcess)\n"
        "  - maxTexture1D               : %d\n"
        "  - maxTexture2D               : %d x %d\n"
        "  - maxTexture3D               : %d x %d x %d\n"
        "  - maxTexture1DLayered        : %d x %d\n"
        /*"  - maxTexture2DLayered        : %d x %d x %d\n"*/
        "  - surfaceAlignment           : %zu\n"
        "  - concurrentKernels          : %d(1 for True, 0 for False)\n"
        "  - ECCEnabled                 : %d(1 for True, 0 for False)\n"
        "  - pciBusID                   : %08X\n"
        "  - pciDeviceID                : %08X\n"
        "  - pciDomainID                : %08X\n"
        "  - tccDriver                  : %d(1 for True, 0 for False)\n"
        "  - asyncEngineCount           : %d\n"
        "  - unifiedAddressing          : %d(1 for True, 0 for False)\n"
        "  - memoryClockRate            : %d KHz\n"
        "  - memoryBusWidth             : %d bits\n"
        "  - l2CacheSize                : %d bytes\n"
        "  - maxThreadsPerMultiProcessor: %d\n",
        props.name,
        props.totalGlobalMem,
        props.sharedMemPerBlock,
        props.regsPerBlock,
        props.warpSize,
        props.memPitch,
        props.maxThreadsPerBlock,
        props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2],
        props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2],
        props.clockRate,
        props.totalConstMem,
        props.major,
        props.minor,
        props.textureAlignment,
        props.deviceOverlap,
        props.multiProcessorCount,
        props.kernelExecTimeoutEnabled,
        props.integrated,
        props.canMapHostMemory,
        props.computeMode,
        props.maxTexture1D,
        props.maxTexture2D[0], props.maxTexture2D[1],
        props.maxTexture3D[0], props.maxTexture3D[1], props.maxTexture3D[2],
        props.maxTexture1DLayered[0], props.maxTexture1DLayered[1],
        /*props.maxTexture2DLayered[0], props.maxTexture2DLayered[1], maxTexture2DLayered[2],*/
        props.surfaceAlignment,
        props.concurrentKernels,
        props.ECCEnabled,
        props.pciBusID,
        props.pciDeviceID,
        props.pciDomainID,
        props.tccDriver,
        props.asyncEngineCount,
        props.unifiedAddressing,
        props.memoryClockRate,
        props.memoryBusWidth,
        props.l2CacheSize,
        props.maxThreadsPerMultiProcessor);
  }

  return 0;
}
