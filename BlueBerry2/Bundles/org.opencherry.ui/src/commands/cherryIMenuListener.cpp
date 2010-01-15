/*=========================================================================

 Program:   openCherry Platform
 Language:  C++
 Date:      $Date$
 Version:   $Revision$

 Copyright (c) German Cancer Research Center, Division of Medical and
 Biological Informatics. All rights reserved.
 See MITKCopyright.txt or http://www.mitk.org/copyright.html for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 =========================================================================*/

#include "cherryIMenuListener.h"
#include "cherryIMenuManager.h"

namespace cherry {

void
IMenuListener::Events
::AddListener(IMenuListener::Pointer l)
{
  if (l.IsNull()) return;

  menuAboutToShow += Delegate(l.GetPointer(), &IMenuListener::MenuAboutToShow);
  menuAboutToHide += Delegate(l.GetPointer(), &IMenuListener::MenuAboutToHide);
}

void
IMenuListener::Events
::RemoveListener(IMenuListener::Pointer l)
{
  if (l.IsNull()) return;

  menuAboutToShow -= Delegate(l.GetPointer(), &IMenuListener::MenuAboutToShow);
  menuAboutToHide -= Delegate(l.GetPointer(), &IMenuListener::MenuAboutToHide);
}

}

