/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkRenderingManagerFactory_h
#define mitkRenderingManagerFactory_h

#include "mitkRenderingManager.h"

namespace mitk
{
  /**
   * \brief Factory interface for facilitating the usage of a platform-specific
   * mitk::RenderingManager instance.
   *
   * This class provides an interface for creating the required singleton
   * instance of RenderingManager. Concrete platform-specific subclasses should
   * be instantiated once during startup of the framework (e.g. as a static
   * instance). Their constructor then merely has to call
   * mitk::RenderingManager::SetFactory().
   *
   * \note Instead of using an external class for the manager
   * instantiation, the factory mechanism could be integrated into the
   * RenderingManager (and its subclasses) itself. However, this would make
   * the framework specific instantiation more complicated. Simply creating a
   * static instance somewhere would not work since RenderingManager derives from
   * itk::Object, which itself depends on the initialization of static members
   * (which is problematic since the order of static member initializations
   * cannot easily be controlled).
   * \ingroup Renderer
   */
  class MITKCORE_EXPORT RenderingManagerFactory
  {
  public:
    virtual ~RenderingManagerFactory(){};

    /** \brief Factory method to create platform specific instances of
     * RenderingManager. */
    virtual RenderingManager::Pointer CreateRenderingManager() const = 0;

  protected:
    RenderingManagerFactory(){};

  private:
  };

  /**
   * Factory for the TestingRenderingManager
   */
  class MITKCORE_EXPORT TestingRenderingManagerFactory : public RenderingManagerFactory
  {
  public:
    TestingRenderingManagerFactory()
    {
      if (!mitk::RenderingManager::HasFactory())
      {
        mitk::RenderingManager::SetFactory(this);
      }
    };

    ~TestingRenderingManagerFactory() override{};

    mitk::RenderingManager::Pointer CreateRenderingManager() const override
    {
      TestingRenderingManager::Pointer specificSmartPtr = TestingRenderingManager::New();
      RenderingManager::Pointer smartPtr = specificSmartPtr.GetPointer();
      return smartPtr;
    };
  };

} // namespace mitk

#endif
