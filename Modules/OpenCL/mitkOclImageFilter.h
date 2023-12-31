/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkOclImageFilter_h
#define mitkOclImageFilter_h

#include "mitkOclFilter.h"
#include "mitkOclImage.h"

#define FILTER_UCHAR 0
#define FILTER_SHORT 1

namespace mitk
{
class OclFilter;
class OclImageFilter;

   /**
    * \brief The OclImageFilter is the topmost class for all filter which take images as input.
    *
    * The input image can be intialized via an oclImage or an mitk::Image.
    * This makes it possible to create a filter pipeline of GPU-based filters
    * and to bind this part into the CPU (ITK) filter pipeline.
    */
class MITKOPENCL_EXPORT OclImageFilter: public OclFilter
{
public:
  /**
   * @brief SetInput SetInput Set the input image (as mitk::OclImage).
   * @param image The image in mitk::OclImage.
   */
  void SetInput(mitk::OclImage::Pointer image);

  /**
   * @brief SetInput Set the input image (as mitk::Image).
   * @param image The image in mitk::Image.
   */
  void SetInput(mitk::Image::Pointer image);

protected:
  OclImageFilter();

  virtual ~OclImageFilter();

  /** The input image */
  mitk::OclImage::Pointer m_Input;

  /*! Current (last used) image pixel type */
  short m_CurrentType;
};
}
#endif
