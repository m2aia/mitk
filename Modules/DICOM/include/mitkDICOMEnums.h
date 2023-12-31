/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkDICOMEnums_h
#define mitkDICOMEnums_h

#include <string>
#include <vector>
#include <utility>

namespace mitk
{
  typedef std::vector<std::string> StringList;
  typedef std::vector<bool> BoolList;

  /**
   \ingroup DICOMModule
   \brief How the mitk::Image spacing should be interpreted (see mitk::DICOMFileReader).

   Compare DICOM PS 3.3 10.7 (Basic Pixel Spacing Calibration Macro).
 */
 typedef enum
  {
    SpacingInPatient,  ///< distances are mm within a patient
    SpacingAtDetector, ///< distances are mm at detector surface
    SpacingUnknown     ///< NO spacing information is present, we use (1,1) as default
  } PixelSpacingInterpretation;

  /**
   \ingroup DICOMModule
    \brief Describes how well the reader is tested for a certain file type (see mitk::DICOMFileReader).

    Applications should not rely on the outcome for images which are reported
    Implemented or Unsupported.

    Errors to load images which are reported as Supported
    are considered bugs. For PartlySupported please check the reader documentation on specifics.
  */
  typedef enum
  {
    SOPClassSupported,       ///< loader code and tests are established
    SOPClassPartlySupported, ///< loader code and tests are established for specific parts of a SOP Class
    SOPClassImplemented,     ///< loader code is implemented but not accompanied by tests
    SOPClassUnsupported,     ///< loader code is not known to work with this SOP Class
    SOPClassUnknown,         ///< loader did not yet inspect any images, unknown fitness
  } ReaderImplementationLevel;


  /// Convert mitk::PixelSpacingInterpretation to a human readable string.
  std::string PixelSpacingInterpretationToString(const PixelSpacingInterpretation& value);
  /// Convert mitk::ReaderImplementationLevel to a human readable string.
  std::string ReaderImplementationLevelToString( const ReaderImplementationLevel& enumValue );
}

#endif
