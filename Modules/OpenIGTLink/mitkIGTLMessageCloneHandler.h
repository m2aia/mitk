/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkIGTLMessageCloneHandler_h
#define mitkIGTLMessageCloneHandler_h

#include "itkObject.h"
#include "mitkCommon.h"

#include "igtlObject.h"
#include "igtlMacro.h"
#include "igtlSocket.h"
#include "igtlMessageBase.h"

#include "MitkOpenIGTLinkExports.h"

namespace mitk
{
  /**Documentation
  * \brief Base class for clone handlers for igtl::MessageBase derived message
  * types
  *
  * To enable new message types a clone function must be defined and added to the
  * message factory.
  *
  */
  class MITKOPENIGTLINK_EXPORT IGTLMessageCloneHandler: public itk::Object
  {
  public:
    mitkClassMacroItkParent(IGTLMessageCloneHandler, itk::Object);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

  public:
    virtual igtl::MessageBase::Pointer Clone(igtl::MessageBase*) { return nullptr; }

  protected:
    IGTLMessageCloneHandler() {}
    ~IGTLMessageCloneHandler() override {}
};

/**
 * Description:
 * The mitkIGTMessageCloneClassMacro() macro is to help developers to
 * define message clone handler classes. It generates a chlid class of
 * mitk::IGTLMessageCloneHandler.
 * The developer only needs to implement Clone() after calling this macro.
 * The following code shows how to define a handler that processes IMAGE message:
 *
 * mitkIGTMessageCloneClassMacro(igtl::ImageMessage, TestImageMessageHandler);
 * igtl::MessageBase::Pointer
 * TestImageMessageHandler::Clone(igtl::MessageBase * message)
 * {
 *   // do something
 * }
 **/
#define mitkIGTMessageCloneClassMacro(messagetype, classname)     \
  class classname : public ::mitk::IGTLMessageCloneHandler                 \
  {                                                               \
  public:                                                         \
    mitkClassMacro(classname, mitk::IGTLMessageCloneHandler);                \
    itkFactorylessNewMacro(Self);                                 \
    itkCloneMacro(Self);                                          \
  public:                                                         \
    virtual igtl::MessageBase::Pointer Clone(igtl::MessageBase*);       \
  protected:                                                      \
    classname(){}                                                 \
    ~classname() {}                                               \
  };


} // namespace mitk

#endif
