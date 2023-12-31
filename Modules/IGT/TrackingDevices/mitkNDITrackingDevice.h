/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef mitkNDITrackingDevice_h
#define mitkNDITrackingDevice_h

#include "mitkTrackingDevice.h"

#include <MitkIGTExports.h>
#include <thread>
#include <mutex>
#include <vector>

#include "mitkNDIProtocol.h"
#include "mitkNDIPassiveTool.h"
#include "mitkSerialCommunication.h"


namespace mitk
{
  class NDIProtocol;

  /** Documentation
  * \brief superclass for specific NDI tracking Devices that use serial communication.
  *
  * implements the TrackingDevice interface for NDI tracking devices (POLARIS, AURORA)
  *
  * \ingroup IGT
  */
  class MITKIGT_EXPORT NDITrackingDevice : public TrackingDevice
  {
    friend class NDIProtocol;

  public:
    typedef std::vector<NDIPassiveTool::Pointer> Tool6DContainerType;  ///< List of 6D tools of the correct type for this tracking device

    typedef mitk::TrackingDeviceType NDITrackingDeviceType;  ///< This enumeration includes the two types of NDI tracking devices (Polaris, Aurora).
    typedef mitk::SerialCommunication::PortNumber PortNumber; ///< Port number of the serial connection
    typedef mitk::SerialCommunication::BaudRate BaudRate;     ///< Baud rate of the serial connection
    typedef mitk::SerialCommunication::DataBits DataBits;     ///< Number of data bits used in the serial connection
    typedef mitk::SerialCommunication::Parity Parity;         ///< Parity mode used in the serial connection
    typedef mitk::SerialCommunication::StopBits StopBits;     ///< Number of stop bits used in the serial connection
    typedef mitk::SerialCommunication::HardwareHandshake HardwareHandshake; ///< Hardware handshake mode of the serial connection
    typedef mitk::NDIPassiveTool::TrackingPriority TrackingPriority; ///< Tracking priority used for tracking a tool

    mitkClassMacro(NDITrackingDevice, TrackingDevice);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

    /**
    * \brief Set the type of the NDI Tracking Device because it can not jet handle this itself
    */
    //itkSetMacro(Type, TrackingDeviceType);


    /**
    * \brief initialize the connection to the tracking device
    *
    * OpenConnection() establishes the connection to the tracking device by:
    * - initializing the serial port with the given parameters (port number, baud rate, ...)
    * - connection to the tracking device
    * - initializing the device
    * - initializing all manually added passive tools (user supplied srom file)
    * - initializing active tools that are connected to the tracking device
    * @throw mitk::IGTHardwareException Throws an exception if there are errors while connecting to the device.
    * @throw mitk::IGTException Throws a normal IGT exception if an error occures which is not related to the hardware.
    */
    bool OpenConnection() override;

    /**
    * \brief Closes the connection
    *
    * CloseConnection() resets the tracking device, invalidates all tools and then closes the serial port.
    */
    bool CloseConnection() override;

    /** @throw mitk::IGTHardwareException Throws an exception if there are errors while connecting to the device. */
    bool InitializeWiredTools();

    /** Sets the rotation mode of this class. See documentation of enum RotationMode for details
     *  on the different modes.
     */
    void SetRotationMode(RotationMode r) override;

    /**
    * \brief TestConnection() tries to connect to a NDI tracking device on the current port/device and returns which device it has found
    *
    * TestConnection() tries to connect to a NDI tracking device on the current port/device.
    * \return It returns the type of the device that answers at the port/device. Throws an exception if no device is available on that port.
    * @throw mitk::IGTHardwareException Throws an exception if there are errors while connecting to the device.
    */
    virtual mitk::TrackingDeviceType TestConnection();

    /**
    * \brief retrieves all wired tools from the tracking device
    *
    * This method queries the tracking device for all wired tools, initializes them and creates TrackingTool representation objects
    * for them
    * \return True if the method was executed successful.
    * @throw mitk::IGTHardwareException Throws an exception if there are errors while connecting to the device.
    * @throw mitk::IGTException Throws a normal IGT exception if an error occures which is not related to the hardware.
    */
    bool DiscoverWiredTools();

    /**
    * \brief Start the tracking.
    *
    * A new thread is created, which continuously reads the position and orientation information of each tool and stores them inside the tools.
    * Depending on the current operation mode (see SetOperationMode()), either the 6D tools (ToolTracking6D), 5D tools (ToolTracking5D),
    * 3D marker positions (MarkerTracking3D) or both 6D tools and 3D markers (HybridTracking) are updated.
    * Call StopTracking() to stop the tracking thread.
    */
    bool StartTracking() override;

    /**
    * \brief return the tool with index toolNumber
    */
    TrackingTool* GetTool(unsigned int toolNumber) const override;

    mitk::TrackingTool* GetToolByName(std::string name) const override;
    /**
    * \brief return current number of tools
    */
    unsigned int GetToolCount() const override;

    /**
    * \brief Create a passive 6D tool with toolName and fileName and add it to the list of tools
    *
    * This method will create a new NDIPassiveTool object, load the SROM file fileName,
    * set the tool name toolName and the tracking priority p and then add
    * it to the list of tools. It returns a pointer of type mitk::TrackingTool to the tool
    * that can be used to read tracking data from it.
    * This is the only way to add tools to NDITrackingDevice.
    * @throw mitk::IGTHardwareException Throws an exception if there are errors while adding the tool.
    *
    * \warning adding tools is not possible in tracking mode, only in setup and ready.
    */
    mitk::TrackingTool* AddTool(const char* toolName, const char* fileName, TrackingPriority p = NDIPassiveTool::Dynamic);

    /**
    * \brief Remove a passive 6D tool from the list of tracked tools.
    *
    * \warning removing tools is not possible in tracking mode, only in setup and ready modes.
    */
    virtual bool RemoveTool(TrackingTool* tool);

    /**
    * \brief reloads the srom file and reinitializes the tool
    */
    virtual bool UpdateTool(mitk::TrackingTool* tool);

    virtual void SetPortNumber(const PortNumber _arg); ///< set port number for serial communication
    itkGetConstMacro(PortNumber, PortNumber);          ///< returns the port number for serial communication
    virtual void SetDeviceName(std::string _arg);      ///< set device name (e.g. COM1, /dev/ttyUSB0). If this is set, PortNumber will be ignored
    itkGetStringMacro(DeviceName);                     ///< returns the device name for serial communication
    virtual void SetBaudRate(const BaudRate _arg);     ///< set baud rate for serial communication
    itkGetConstMacro(BaudRate, BaudRate);              ///< returns the baud rate for serial communication
    virtual void SetDataBits(const DataBits _arg);     ///< set number of data bits
    itkGetConstMacro(DataBits, DataBits);              ///< returns the data bits for serial communication
    virtual void SetParity(const Parity _arg);         ///< set parity mode
    itkGetConstMacro(Parity, Parity);                  ///< returns the parity mode
    virtual void SetStopBits(const StopBits _arg);     ///< set number of stop bits
    itkGetConstMacro(StopBits, StopBits);              ///< returns the number of stop bits
    virtual void SetHardwareHandshake(const HardwareHandshake _arg);  ///< set use hardware handshake for serial communication
    itkGetConstMacro(HardwareHandshake, HardwareHandshake);              ///< returns the hardware handshake setting
    virtual void SetIlluminationActivationRate(const IlluminationActivationRate _arg); ///< set activation rate of IR illumator for polaris
    itkGetConstMacro(IlluminationActivationRate, IlluminationActivationRate);          ///< returns the activation rate of IR illumator for polaris
    virtual void SetDataTransferMode(const DataTransferMode _arg);    ///< set data transfer mode to text (TX) or binary (BX). \warning: only TX is supportet at the moment
    itkGetConstMacro(DataTransferMode, DataTransferMode);              ///< returns the data transfer mode
    virtual bool Beep(unsigned char count);   ///< Beep the tracking device 1 to 9 times

    NDIErrorCode GetErrorCode(const std::string* input);  ///< returns the error code for a string that contains an error code in hexadecimal format

    virtual bool SetOperationMode(OperationMode mode);  ///< set operation mode to 6D tool tracking, 3D marker tracking or 6D&3D hybrid tracking (see OperationMode)
    virtual OperationMode GetOperationMode();           ///< get current operation mode

    /**
    * \brief Get 3D marker positions (operation mode must be set to MarkerTracking3D or HybridTracking)
    */
    virtual bool GetMarkerPositions(MarkerPointContainerType* markerpositions);

    /**
    * \brief Get major revision number from tracking device
    * should not be called directly after starting to track
    **/
    virtual int GetMajorFirmwareRevisionNumber();

    /**
    * \brief Get revision number from tracking device as string
    * should not be called directly after starting to track
    **/
    virtual const char* GetFirmwareRevisionNumber();

    /** @return Returns true if this device can autodetects its tools. */
    bool AutoDetectToolsAvailable() override;

    /** @return Returns true if it is possible to add a single tool. True for Polaris, false for Aurora.*/
    bool AddSingleToolIsAvailable() override;

    /** Autodetects tools from this device and returns them as a navigation tool storage.
    *  @return Returns the detected tools. Returns an empty storage if no tools are present
    *          or if detection is not possible
    */
    mitk::NavigationToolStorage::Pointer AutoDetectTools() override;



  protected:

    typedef std::vector<std::string> NDITrackingVolumeContainerType;  ///< vector of tracking volumes
    typedef std::vector<int> TrackingVolumeDimensionType;          ///< List of the supported tracking volume dimensions.

        /**
    * \brief Get number of supported tracking volumes, a vector containing the supported volumes and
    * a vector containing the signed dimensions in mm. For each volume 10 boundaries are stored in the order of
    * the supported volumes (see AURORA API GUIDE: SFLIST p.54).
    **/
    virtual bool GetSupportedVolumes(unsigned int* numberOfVolumes, NDITrackingVolumeContainerType* volumes, TrackingVolumeDimensionType* volumesDimensions);

       /**
    * \brief Sets the desired tracking volume. Returns true if the volume type could be set. It is set in the OpenConnection() Method and sets the tracking volume out of m_Data.
    * @throw mitk::IGTHardwareException Throws an IGT hardware exception if the volume could not be set.
    **/
    virtual bool SetVolume(mitk::TrackingDeviceData volume);

    /**
    * \brief Add a passive 6D tool to the list of tracked tools. This method is used by AddTool
    * @throw mitk::IGTHardwareException Throws an exception if there are errors while adding the tool.
    * \warning adding tools is not possible in tracking mode, only in setup and ready.
    */
    virtual bool InternalAddTool(NDIPassiveTool* tool);

    /* Methods for NDIProtocol friend class */
    virtual void InvalidateAll();             ///< invalidate all tools
    NDIPassiveTool* GetInternalTool(std::string portHandle); ///< returns the tool object that has been assigned the port handle or nullptr if no tool can be found

    /**
    * \brief free all port handles that need to be freed
    *
    * This method retrieves a list of all port handles that need to be freed (e.g. tool got disconnected)
    * and frees the handles at the tracking device and it removes the tools from the internal tool list
    * \warning This method can remove TrackingTools from the tool list! After calling this method, GetTool(i) could return
    *          a different tool, because tool indices could have changed.
    * @throw mitk::IGTHardwareException Throws an exception if there are errors while communicating with the device.
    * \return returns NDIOKAY if everything was sucessfull, returns an error code otherwise
    */
    NDIErrorCode FreePortHandles();

    NDIErrorCode Send(const std::string* message, bool addCRC = true);      ///< Send message to tracking device
    NDIErrorCode Receive(std::string* answer, unsigned int numberOfBytes);  ///< receive numberOfBytes bytes from tracking device
    NDIErrorCode ReceiveByte(char* answer);   ///< lightweight receive function, that reads just one byte
    NDIErrorCode ReceiveLine(std::string* answer); ///< receive characters until the first LF (The LF is included in the answer string)
    void ClearSendBuffer();                   ///< empty send buffer of serial communication interface
    void ClearReceiveBuffer();                ///< empty receive buffer of serial communication interface
    const std::string CalcCRC(const std::string* input);  ///< returns the CRC16 for input as a std::string

public:

    /**
    * \brief TrackTools() continuously polls serial interface for new 6d tool positions until StopTracking is called.
    *
    * Continuously tracks the 6D position of all tools until StopTracking() is called.
    * This function is executed by the tracking thread (through StartTracking() and ThreadStartTracking()).
    * It should not be called directly.
    * @throw mitk::IGTHardwareException Throws an exception if there are errors while tracking the tools.
    */
    virtual void TrackTools();

    /**
    * \brief continuously polls serial interface for new 3D marker positions until StopTracking is called.
    *
    * Continuously tracks the 3D position of all markers until StopTracking() is called.
    * This function is executed by the tracking thread (through StartTracking() and ThreadStartTracking()).
    * It should not be called directly.
    */
    virtual void TrackMarkerPositions();

    /**
    * \brief continuously polls serial interface for new 3D marker positions and 6D tool positions until StopTracking is called.
    *
    * Continuously tracks the 3D position of all markers and the 6D position of all tools until StopTracking() is called.
    * This function is executed by the tracking thread (through StartTracking() and ThreadStartTracking()).
    * It should not be called directly.
    */
    virtual void TrackToolsAndMarkers();

    /**
    * \brief start method for the tracking thread.
    */
    void ThreadStartTracking();

  protected:
    NDITrackingDevice();          ///< Constructor
    ~NDITrackingDevice() override; ///< Destructor

    std::string m_DeviceName;///< Device Name
    PortNumber m_PortNumber; ///< COM Port Number
    BaudRate m_BaudRate;     ///< COM Port Baud Rate
    DataBits m_DataBits;     ///< Number of Data Bits per token
    Parity m_Parity;         ///< Parity mode for communication
    StopBits m_StopBits;     ///< number of stop bits per token
    HardwareHandshake m_HardwareHandshake; ///< use hardware handshake for serial port connection
    ///< which tracking volume is currently used (if device supports multiple volumes) (\warning This parameter is not used yet)
    IlluminationActivationRate m_IlluminationActivationRate; ///< update rate of IR illuminator for Polaris
    DataTransferMode m_DataTransferMode;  ///< use TX (text) or BX (binary) (\warning currently, only TX mode is supported)
    Tool6DContainerType m_6DTools;        ///< list of 6D tools

    mutable std::mutex m_ToolsMutex; ///< mutex for coordinated access of tool container
    mitk::SerialCommunication::Pointer m_SerialCommunication;    ///< serial communication interface
    std::mutex m_SerialCommunicationMutex; ///< mutex for coordinated access of serial communication interface
    NDIProtocol::Pointer m_DeviceProtocol;    ///< create and parse NDI protocol strings

    std::thread m_Thread;                 ///< ID of tracking thread
    OperationMode m_OperationMode;  ///< tracking mode (6D tool tracking, 3D marker tracking,...)
    std::mutex m_MarkerPointsMutex;  ///< mutex for marker point data container
    MarkerPointContainerType m_MarkerPoints;          ///< container for markers (3D point tracking mode)
  };
} // namespace mitk
#endif
