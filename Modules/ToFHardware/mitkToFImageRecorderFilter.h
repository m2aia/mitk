/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef mitkToFImageRecorderFilter_h
#define mitkToFImageRecorderFilter_h

#include <mitkImageToImageFilter.h>
#include <MitkToFHardwareExports.h>

#include <mitkToFImageWriter.h>

namespace mitk
{
  /**
  * @brief Filter that allows recording of processed ToF images
  * Internally the ToFImageWriter is used for writing.
  *
  * @ingroup ToFProcessing
  */
  class MITKTOFHARDWARE_EXPORT ToFImageRecorderFilter : public ImageToImageFilter
  {
  public:
    mitkClassMacro(ToFImageRecorderFilter, ImageToImageFilter);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

    /*!
    \brief Set file name for writing image files
    This filename will be appended by "_DistanceImage", "_AmplitudeImage", or "_IntensityImage" for the single images
    \param fileName base file name to save image filtes
    */
    void SetFileName(std::string fileName);
    /*!
    \brief Set image type for recording
    \param tofImageType either 3D Volume (ToFImageType3D) or temporal image stack (ToFImageType2DPlusT)
    */
    void SetImageType(ToFImageWriter::ToFImageType tofImageType);
    /*!
    \brief Returns a pointer to the ToFImageWriter internally used
    \return ToFImageWriter
    */
    ToFImageWriter::Pointer GetToFImageWriter();
    /*!
    \brief Sets a pointer to the ToFImageWriter internally used
    \param tofImageWriter ToFImageWriter
    */
    void SetToFImageWriter(ToFImageWriter::Pointer tofImageWriter);
    /*!
    \brief start recording of data
    */
    void StartRecording();
    /*!
    \brief stop recording of data
    */
    void StopRecording();

    using itk::ProcessObject::SetInput;

    /*!
    \brief sets the input of this filter
    \param input is the distance image of e.g. a ToF camera
    */
    void SetInput( const InputImageType* input) override;
    /*!
    \brief sets the input of this filter at idx
    \param idx number of the current input
    \param input is the distance image of e.g. a ToF camera
    */
    void SetInput(unsigned int idx,  const InputImageType* input) override;
    /*!
    \brief returns the input of this filter
    */
     Image* GetInput();
    /*!
    \brief returns the input with id idx of this filter
    */
     Image* GetInput(unsigned int idx);

  protected:
    /*!
    \brief standard constructor
    */
    ToFImageRecorderFilter();
    /*!
    \brief standard destructor
    */
    ~ToFImageRecorderFilter() override;
    /*!
    \brief method generating the output of this filter. Called in the updated process of the pipeline.
    This method generates the output of the ToFSurfaceSource: The generated surface of the 3d points
    */
    void GenerateData() override;
    /**
    * \brief Create an output for each input
    *
    * This Method sets the number of outputs to the number of inputs
    * and creates missing outputs objects.
    * \warning any additional outputs that exist before the method is called are deleted
    */
    void CreateOutputsForAllInputs();

    bool m_RecordingStarted; ///< flag indicating if filter is currently recording
    ToFImageWriter::Pointer m_ToFImageWriter; ///< image writer used for streaming input data to file
    std::string m_FileExtension;
  };
} //END mitk namespace
#endif
