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

#include "usTestModuleBService.h"

#include <usModuleActivator.h>
#include <usModuleContext.h>


US_BEGIN_NAMESPACE

struct TestModuleImportedByB : public TestModuleBService
{

  TestModuleImportedByB(ModuleContext* mc)
  {
    US_INFO << "Registering TestModuleImportedByB";
    mc->RegisterService<TestModuleBService>(this);
  }

};

class TestModuleImportedByBActivator : public ModuleActivator
{
public:

  TestModuleImportedByBActivator() : s(nullptr) {}
  ~TestModuleImportedByBActivator() override { delete s; }

  void Load(ModuleContext* context) override
  {
    s = new TestModuleImportedByB(context);
  }

  void Unload(ModuleContext*) override
  {
  }

private:

  TestModuleImportedByB* s;
};

US_END_NAMESPACE

US_EXPORT_MODULE_ACTIVATOR(US_PREPEND_NAMESPACE(TestModuleImportedByBActivator))
