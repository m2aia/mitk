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


#include <usModuleActivator.h>
#include <usModulePropsInterface.h>

#include <usServiceTracker.h>
#include <usServiceTrackerCustomizer.h>

#include "usFooService.h"

US_BEGIN_NAMESPACE

class ActivatorSL1 :
    public ModuleActivator, public ModulePropsInterface,
    public ServiceTrackerCustomizer<FooService>
{

public:

  ActivatorSL1()
    : tracker(nullptr), context(nullptr)
  {

  }

  ~ActivatorSL1() override
  {
    delete tracker;
  }

  void Load(ModuleContext* context) override
  {
    this->context = context;

    InterfaceMap im = MakeInterfaceMap<ModulePropsInterface>(this);
    im.insert(std::make_pair(std::string("ActivatorSL1"), this));
    sr = context->RegisterService(im);

    delete tracker;
    tracker = new FooTracker(context, this);
    tracker->Open();
  }

  void Unload(ModuleContext* /*context*/) override
  {
    tracker->Close();
  }

  const Properties& GetProperties() const override
  {
    return props;
  }

  FooService* AddingService(const ServiceReferenceType& reference) override
  {
    props["serviceAdded"] = true;

    FooService* fooService = context->GetService<FooService>(reference);
    fooService->foo();
    return fooService;
  }

  void ModifiedService(const ServiceReferenceType& /*reference*/, FooService* /*service*/) override
  {}

  void RemovedService(const ServiceReferenceType& /*reference*/, FooService* /*service*/) override
  {
    props["serviceRemoved"] = true;
  }

private:

  ModulePropsInterface::Properties props;

  ServiceRegistrationU sr;

  typedef ServiceTracker<FooService> FooTracker;

  FooTracker* tracker;
  ModuleContext* context;

}; // ActivatorSL1

US_END_NAMESPACE

US_EXPORT_MODULE_ACTIVATOR(US_PREPEND_NAMESPACE(ActivatorSL1))
