/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkMouseWheelEvent_h
#define mitkMouseWheelEvent_h

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "mitkBaseRenderer.h"
#include "mitkCommon.h"
#include "mitkInteractionEvent.h"
#include "mitkInteractionEventConst.h"
#include "mitkInteractionPositionEvent.h"

#include <MitkCoreExports.h>

/**
 * Note: A Click with the MiddleButton is to be handled with MousePressEvents
 */

namespace mitk
{
  class MITKCORE_EXPORT MouseWheelEvent : public InteractionPositionEvent
  {
  public:
    mitkClassMacro(MouseWheelEvent, InteractionPositionEvent)
      mitkNewMacro5Param(Self, BaseRenderer *, const Point2D &, MouseButtons, ModifierKeys, int)

        ModifierKeys GetModifiers() const;
    MouseButtons GetButtonStates() const;
    void SetModifiers(ModifierKeys modifiers);
    void SetButtonStates(MouseButtons buttons);
    int GetWheelDelta() const;
    void SetWheelDelta(int delta);

    bool IsSuperClassOf(const InteractionEvent::Pointer &baseClass) const override;

  protected:
    MouseWheelEvent(BaseRenderer * = nullptr,
                    const Point2D &mousePosition = Point2D(),
                    MouseButtons buttonStates = NoButton,
                    ModifierKeys modifiers = NoKey,
                    int wheelDelta = 0);
    ~MouseWheelEvent() override;

    bool IsEqual(const InteractionEvent &) const override;

  private:
    int m_WheelDelta;
    MouseButtons m_ButtonStates;
    ModifierKeys m_Modifiers;
  };
} /* namespace mitk */

#endif
