/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkOclBaseData_h
#define mitkOclBaseData_h

#include "MitkOpenCLExports.h"
#include "mitkOclUtils.h"

#include "mitkOpenCL.h"

#include <mitkCommon.h>
#include <itkObjectFactory.h>
#include <itkObject.h>

namespace mitk
{

  class MITKOPENCL_EXPORT OclBaseData : public itk::Object
  {
  public:
    mitkClassMacroItkParent(OclBaseData, itk::Object);

    /*!  \brief Copies the RAM-stored data to GPU-MEM.
     * This method has to be implemented for each data type.
     */
    virtual int TransferDataToGPU(cl_command_queue) = 0;

    /*! \brief Copies the in GPU-MEM stored data to RAM
     * This method has to be implemented for each data type.
     */
    virtual void* TransferDataToCPU(cl_command_queue) = 0;

    /** \brief Set the modified flag for one of the GPU buffer */
    void GPUModified();

    /** \brief Set the modified flag for one of the CPU buffer */
    void CPUModified();


    /** \brief Get the pointer to the buffer on the device (GPU) */
    virtual cl_mem GetGPUBuffer();

  protected:
    /** \brief Default constructor. */
    OclBaseData();

    /** \brief Default destructor. */
    virtual ~OclBaseData();

  private:

    /** \brief Size of the data buffer. CPU size equals GPU size. */
    size_t m_BufferSize;

    /** \brief Pointer to the buffer on the device (GPU) */
    cl_mem m_GPUBuffer;

    /** \brief Pointer to the buffer on the host (CPU) */
    void* m_CPUBuffer;
  };

}

#endif
