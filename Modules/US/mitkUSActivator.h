/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkUSActivator_h
#define mitkUSActivator_h

#include "mitkUSVideoDevice.h"

// Microservices
#include <usModuleContext.h>
#include <usModuleActivator.h>

namespace mitk
{
/**
  * \brief Module activator for the US module.
  * Loads mitk::USVideoDevice objects from hard disk on module load and write
  * them to hard disk on module unload.
  *
  * Pointers to mitk::USDevice objects are held to make sure that they
  * will not be deleted while the module is loaded. A service event listener is
  * registered, so that pointers to devices which are registered into micro
  * service from a plugin for example can be held here, too.
  */
class USActivator : public us::ModuleActivator {
public:

  USActivator();
  ~USActivator() override;

  /**
    * \brief The mitk::USVideoDevice obejcts are loaded from hard disk and registered into micro service.
    */
  void Load(us::ModuleContext* context) override;

  /**
    * \brief Registered mitk::USVideoDevice objects are stored to hard disk an deregistered from micro service.
    */
  void Unload(us::ModuleContext* context) override;

protected:
  /**
    *\brief Listens to ServiceRegistry changes and updates the list of mitk::USDevice object accordingly.
    */
  void OnServiceEvent(const us::ServiceEvent event);

  us::ModuleContext*                    m_Context;
  std::vector<USDevice::Pointer>        m_Devices;
};
} // namespace mitk

US_EXPORT_MODULE_ACTIVATOR(mitk::USActivator)

#endif
