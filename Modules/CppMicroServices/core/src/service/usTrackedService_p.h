/*============================================================================

  Library: CppMicroServices

  Copyright (c) German Cancer Research Center (DKFZ)
  All rights reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

============================================================================*/


#ifndef USTRACKEDSERVICE_H
#define USTRACKEDSERVICE_H

#include "usTrackedServiceListener_p.h"
#include "usModuleAbstractTracked_p.h"
#include "usServiceEvent.h"

US_BEGIN_NAMESPACE

/**
 * This class is not intended to be used directly. It is exported to support
 * the CppMicroServices module system.
 */
template<class S, class TTT>
class TrackedService : public TrackedServiceListener,
    public ModuleAbstractTracked<ServiceReference<S>, TTT, ServiceEvent>
{

public:

  typedef typename TTT::TrackedType T;

  TrackedService(ServiceTracker<S,TTT>* serviceTracker,
                 ServiceTrackerCustomizer<S,T>* customizer);

  /**
   * Method connected to service events for the
   * <code>ServiceTracker</code> class. This method must NOT be
   * synchronized to avoid deadlock potential.
   *
   * @param event <code>ServiceEvent</code> object from the framework.
   */
  void ServiceChanged(const ServiceEvent event) override;

private:

  typedef ModuleAbstractTracked<ServiceReference<S>, TTT, ServiceEvent> Superclass;

  ServiceTracker<S,TTT>* serviceTracker;
  ServiceTrackerCustomizer<S,T>* customizer;

  /**
   * Increment the tracking count and tell the tracker there was a
   * modification.
   *
   * @GuardedBy this
   */
  void Modified() override;

  /**
   * Call the specific customizer adding method. This method must not be
   * called while synchronized on this object.
   *
   * @param item Item to be tracked.
   * @param related Action related object.
   * @return Customized object for the tracked item or <code>null</code>
   *         if the item is not to be tracked.
   */
  T CustomizerAdding(ServiceReference<S> item, const ServiceEvent& related) override;

  /**
   * Call the specific customizer modified method. This method must not be
   * called while synchronized on this object.
   *
   * @param item Tracked item.
   * @param related Action related object.
   * @param object Customized object for the tracked item.
   */
  void CustomizerModified(ServiceReference<S> item,
                          const ServiceEvent& related, T object) override ;

  /**
   * Call the specific customizer removed method. This method must not be
   * called while synchronized on this object.
   *
   * @param item Tracked item.
   * @param related Action related object.
   * @param object Customized object for the tracked item.
   */
  void CustomizerRemoved(ServiceReference<S> item,
                         const ServiceEvent& related, T object) override ;
};

US_END_NAMESPACE

#include "usTrackedService.tpp"

#endif // USTRACKEDSERVICE_H
