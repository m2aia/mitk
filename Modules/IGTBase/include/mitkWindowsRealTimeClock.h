/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkWindowsRealTimeClock_h
#define mitkWindowsRealTimeClock_h

#include "MitkIGTBaseExports.h"
#include "mitkRealTimeClock.h"


namespace mitk {

  /**
  * \brief realtimeclock implementation for windows-systems
  *
  * This class provides a RealTimeClock for windows-systems.
  * Internally, it uses the QueryPerformanceCounter and the QueryPerformaceFrequency.
  * It polls the current tick-counter, that counts from bootup.
  * is supposed to be the most accurate time you can get on a windows-system.
  *
  * \ingroup Navigation
  */

  class MITKIGTBASE_EXPORT WindowsRealTimeClock : public RealTimeClock
  {
  public:
    mitkClassMacro(WindowsRealTimeClock, mitk::RealTimeClock);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

    /**
    * \brief basic contructor
    */
    WindowsRealTimeClock();

    /**
    * \brief basic destructor
    */
    virtual ~WindowsRealTimeClock();

    /**
    * \brief returns the current time in milliseconds as a double
    *
    * Uses the QueryPerformanceCounter to acquire an accurate time.
    * (Ticks counted / Frequency) = time in ms
    */
    virtual double GetCurrentStamp();

    /**
    * \brief returns the QueryPerformanceFrequency
    */
    virtual LARGE_INTEGER GetFrequency();

  protected:

    void SetFrequency();

    /**
    * \brief Frequency needed to calculate time from tick-counter
    */
    LARGE_INTEGER m_Frequency;

  }; //namespace
}






#endif
