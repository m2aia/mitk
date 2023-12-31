/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkVolumeCalculator_h
#define mitkVolumeCalculator_h

#include "itkImage.h"
#include "itkObject.h"
#include "itkObjectFactory.h"

#include "mitkImage.h"
#include "mitkImageTimeSelector.h"
#include <MitkCoreExports.h>

namespace mitk
{
  /**
   * @brief Calculates the volume of a mitk::Image.
   * The given volume is in milliliters or as a voxel count.
   * Voxels are counted if their gray value is above a threshold (see SetThreshold), the default threshold is 0.
   *
   * The filter works for 2D, 3D and 3D+t. In the 3D+t case a vector of volumes is provided (see GetVolumes()).
   */
  class MITKCORE_EXPORT VolumeCalculator : public itk::Object
  {
  public:
    mitkClassMacroItkParent(VolumeCalculator, itk::Object);

    itkFactorylessNewMacro(Self);

    itkCloneMacro(Self);

    itkSetObjectMacro(Image, Image);
    /**
    * Sets threshold, all voxels that are equal or greater are accepted.
    */
    itkSetMacro(Threshold, int);
    /**
    * In case of a 2D image this returns the calculated volume in square cm, in case of 3D images the result is given
    * in cubic cm. For 4D data refer to the GetVolumes() function.
    * @see GetVolumes()
    */
    itkGetMacro(Volume, float);
    /**
    * Returns the number of accepted voxels.
    */
    itkGetMacro(VoxelCount, unsigned long int);
    /**
    * Returns the volume for 4D images, as a vector where each volume is represented by an item in the vector (in cubic
    cm).
    \note Also for 2D+t images the result will be returned in cubic cm.
    */
    std::vector<float> GetVolumes();
    void ComputeVolume();

    void ComputeVolumeFromImageStatistics();
    static float ComputeVolume(Vector3D spacing, unsigned int voxelCount);

  protected:
    VolumeCalculator();
    ~VolumeCalculator() override;

    template <typename TPixel, unsigned int VImageDimension>
    void InternalCompute(const itk::Image<TPixel, VImageDimension> *itkImage);

    Image::ConstPointer m_Image;
    int m_Threshold;
    float m_Volume;
    unsigned long int m_VoxelCount;
    std::vector<float> m_Volumes;
    ImageTimeSelector::Pointer m_TimeSelector;
  };

} // namespace mitk

#endif
