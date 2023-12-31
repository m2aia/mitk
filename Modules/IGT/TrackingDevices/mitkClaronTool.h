/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkClaronTool_h
#define mitkClaronTool_h

#include <mitkClaronInterface.h>
#include <mitkTrackingTool.h>

namespace mitk
{
  class ClaronTrackingDevice;
  /** Documentation:
  *   \brief  An object of this class represents a MicronTracker 2 tool.
  *           A tool has to be added to a tracking device which will then
  *           continuously update the tool coordinates.
  *   \ingroup IGT
  */
  class MITKIGT_EXPORT ClaronTool : public TrackingTool
  {
  public:
    friend class ClaronTrackingDevice;
    mitkClassMacro(ClaronTool, TrackingTool);


    /**
    * \brief Loads a tool calibration file. Without this file the tool can not be tracked!
    */
    bool LoadFile(const char* filename);
    /**
    * \brief Loads a tool calibration file. Without this file the tool can not be tracked!
    */
    bool LoadFile(std::string filename);

    std::string GetFile();

    /**
    * \brief Sets the handle of the tool.
    * \param handle The new handle of the tool.
    */
    void SetToolHandle (claronToolHandle handle);

    /**
    * \return Returns the calibration name which is used to identify the tool.
    */
    std::string GetCalibrationName();

    /**
    * \brief Sets the calibration name of the tool. Be careful, only use this method if you know what you are doing.
    * If you want to change the tool name use the method setToolName instead!
    */
    void SetCalibrationName(std::string name);

    /**
    * @return Returns the tool handle of the tool.
    */
    claronToolHandle GetToolHandle();

  protected:
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self)
    ClaronTool();
    ~ClaronTool() override;
    /** \brief Tool handle variable from tracking device */
    claronToolHandle m_ToolHandle;
    /** \brief  Variable which holds the Tool's calibration name */
    std::string m_CalibrationName;
    /** \brief Variable to check filename's format and to get back complete filename */
    std::string m_Filename;
  };
}//mitk
#endif
