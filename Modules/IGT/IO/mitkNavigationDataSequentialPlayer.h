/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkNavigationDataSequentialPlayer_h
#define mitkNavigationDataSequentialPlayer_h

#include <mitkNavigationDataPlayerBase.h>

namespace mitk
{
  /**Documentation
  * \brief This class is a slightly changed reimplementation of the
  * NavigationDataPlayer which does not care about timestamps and just
  * outputs the navigationdatas in their sequential order
  *
  * \ingroup IGT
  */
  class MITKIGT_EXPORT NavigationDataSequentialPlayer
    : public NavigationDataPlayerBase
  {
  public:
    mitkClassMacro(NavigationDataSequentialPlayer, NavigationDataPlayerBase);
    itkNewMacro(Self);

    /**
    * \brief Advance the output to the i-th snapshot of mitk::NavigationData.
    * E.g. if you want to have the NavData of snapshot
    * 18 then you can call GoToSnapshot(17). Index begins at 0.
    * You can only go back if m_Repeat is set true.
    * This method internally calls GenerateData, so outputs are refreshed automatically
    *
    * Filter output is updated inside the function.
    *
    * @throw mitk::IGTException Throws an exception if cannot go back to particular snapshot.
    */
    void GoToSnapshot(unsigned int i);

    /**
    * \brief Advance the output to the next snapshot of mitk::NavigationData.
    * Filter output is updated inside the function.
    *
    * \return false if no next snapshot is available (happens only if m_Repeat is set to false).
    * @throw mitk::IGTException Throws an exception if an output is null.
    */
    bool GoToNextSnapshot();

    /**
    * \brief Used for pipeline update just to tell the pipeline
    * that we always have to update
    */
    void UpdateOutputInformation() override;

  protected:
    NavigationDataSequentialPlayer();
    ~NavigationDataSequentialPlayer() override;

    /**
    * \brief Does nothing.
    * mitk::NavigationDataSequentialPlayer::GoToNextSnapshot() should be called
    * for generating next data.
    */
    void GenerateData() override;
  };
} // namespace mitk

#endif
