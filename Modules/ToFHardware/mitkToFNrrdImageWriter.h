/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef mitkToFNrrdImageWriter_h
#define mitkToFNrrdImageWriter_h

#include <MitkToFHardwareExports.h>
#include "mitkToFImageWriter.h"

#include <fstream>

namespace mitk
{
  /**
  * @brief Writer class for ToF nrrd images
  *
  * This writer class allows streaming of ToF data into a nrrd file. This class uses the itkNrrdImageIO class
  * Writer can simultaneously save "distance", "intensity" and "amplitude" image.
  * Images can be written as 3D volume (ToFImageType::ToFImageType3D) or temporal image stack (ToFImageType::ToFImageType2DPlusT)
  *
  * @ingroup ToFHardware
  */
  class MITKTOFHARDWARE_EXPORT ToFNrrdImageWriter : public ToFImageWriter
  {
  public:
    mitkClassMacro( ToFNrrdImageWriter , ToFImageWriter );
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

    /*!
    \brief Open file(s) for writing
    */
    void Open() override;
    /*!
    \brief Close file(s) add header and write
    */
    void Close() override;
    /*!
    \brief Add new data to file.
    */
    void Add(float* distanceFloatData, float* amplitudeFloatData, float* intensityFloatData, unsigned char* rgbData=nullptr) override;

  protected:

    std::ofstream m_DistanceOutfile; ///< file for distance image
    std::ofstream m_AmplitudeOutfile; ///< file for amplitude image
    std::ofstream m_IntensityOutfile; ///< file for intensity image
    std::ofstream m_RGBOutfile; ///< file for intensity image

  private:

    ToFNrrdImageWriter();
    ~ToFNrrdImageWriter() override;

    /*!
    \brief Open file by filename to gain write access to it.
    */
    void OpenStreamFile(std::ofstream &outfile, std::string outfileName);
    /*!
    \brief Close file after work on it is finished.
    */
    void CloseStreamFile(std::ofstream &outfile, std::string fileName);
    /*!
    \brief Write image information to the NrrdFile.
    */
    void ConvertStreamToNrrdFormat( std::string fileName );
  };
} //END mitk namespace
#endif
