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

#include "cherryAbstractSelectionService.h"

#include "../cherryIPostSelectionProvider.h"

namespace cherry
{

//ISelectionService::SelectionEvents& AbstractSelectionService::GetSelectionEvents(const std::string& partId)
//{
//  if (partId.empty())
//  {
//    return selectionEvents;
//  }
//
//  return this->GetPerPartTracker(partId)->GetSelectionEvents();
//}

AbstractSelectionService::AbstractSelectionService()
{
  selListener = new SelectionListener(this);
  postSelListener = new PostSelectionListener(this);
}

AbstractSelectionService::SelectionListener::SelectionListener(AbstractSelectionService* service)
 : m_SelectionService(service)
{ }

void AbstractSelectionService::SelectionListener::SelectionChanged(SelectionChangedEvent::Pointer event)
{
  m_SelectionService->FireSelection(m_SelectionService->activePart, event->GetSelection());
}

AbstractSelectionService::PostSelectionListener::PostSelectionListener(AbstractSelectionService* service)
 : m_SelectionService(service)
{ }

void AbstractSelectionService::PostSelectionListener::SelectionChanged(
    SelectionChangedEvent::Pointer event)
{
  m_SelectionService->FirePostSelection(m_SelectionService->activePart, event->GetSelection());
}

void AbstractSelectionService::AddSelectionListener(ISelectionListener::Pointer l)
{
  selectionEvents.selectionChanged +=
    ISelectionService::SelectionEvents::Delegate(l.GetPointer(),
      &ISelectionListener::SelectionChanged);
}

void AbstractSelectionService::AddSelectionListener(const std::string& partId,
    ISelectionListener::Pointer listener)
{
  this->GetPerPartTracker(partId)->AddSelectionListener(listener);
}

void AbstractSelectionService::AddPostSelectionListener(
    ISelectionListener::Pointer l)
{
  selectionEvents.postSelectionChanged +=
    ISelectionService::SelectionEvents::Delegate(l.GetPointer(),
      &ISelectionListener::SelectionChanged);
}

void AbstractSelectionService::AddPostSelectionListener(const std::string& partId,
    ISelectionListener::Pointer listener)
{
  this->GetPerPartTracker(partId)->AddPostSelectionListener(listener);
}

void AbstractSelectionService::RemoveSelectionListener(ISelectionListener::Pointer l)
{
  selectionEvents.selectionChanged -=
    ISelectionService::SelectionEvents::Delegate(l.GetPointer(),
      &ISelectionListener::SelectionChanged);
}

void AbstractSelectionService::RemovePostSelectionListener(
    const std::string& partId, ISelectionListener::Pointer listener)
{
  this->GetPerPartTracker(partId)->RemovePostSelectionListener(listener);
}

void AbstractSelectionService::RemovePostSelectionListener(
    ISelectionListener::Pointer l)
{
  selectionEvents.postSelectionChanged -=
    ISelectionService::SelectionEvents::Delegate(l.GetPointer(),
      &ISelectionListener::SelectionChanged);
}

void AbstractSelectionService::RemoveSelectionListener(const std::string& partId,
    ISelectionListener::Pointer listener)
{
  this->GetPerPartTracker(partId)->RemoveSelectionListener(listener);
}

void AbstractSelectionService::FireSelection(IWorkbenchPart::Pointer part,
    ISelection::ConstPointer sel)
{
   selectionEvents.selectionChanged(part, sel);
}

void AbstractSelectionService::FirePostSelection(IWorkbenchPart::Pointer part,
    ISelection::ConstPointer sel)
{
  selectionEvents.postSelectionChanged(part, sel);
}

AbstractPartSelectionTracker::Pointer AbstractSelectionService::GetPerPartTracker(
    const std::string& partId)
{
  AbstractPartSelectionTracker::Pointer tracker;
  std::map<std::string, AbstractPartSelectionTracker::Pointer>::const_iterator res = perPartTrackers.find(partId);
  if (res == perPartTrackers.end())
  {
    tracker = this->CreatePartTracker(partId);
    perPartTrackers[partId] = tracker;
  }
  else
  {
    tracker = res->second;
  }

  return tracker;
}

ISelection::ConstPointer AbstractSelectionService::GetSelection() const
{
  if (activeProvider.IsNotNull())
  {
    return activeProvider->GetSelection();
  }
  else
  {
    return ISelection::ConstPointer(0);
  }
}

ISelection::ConstPointer AbstractSelectionService::GetSelection(const std::string& partId)
{
  return this->GetPerPartTracker(partId)->GetSelection();
}

void AbstractSelectionService::SetActivePart(IWorkbenchPart::Pointer newPart)
{
  // Optimize.
  if (newPart == activePart)
  {
    return;
  }

  ISelectionProvider::Pointer selectionProvider;

  if (newPart.IsNotNull())
  {
    selectionProvider = newPart->GetSite()->GetSelectionProvider();

    if (selectionProvider.IsNull())
    {
      newPart = 0;
    }
  }

  if (newPart == activePart)
  {
    return;
  }

  if (activePart.IsNotNull())
  {
    if (activeProvider.IsNotNull())
    {
      activeProvider->RemoveSelectionChangedListener(selListener);
      if (activeProvider.Cast<IPostSelectionProvider>().IsNotNull())
      {
        activeProvider.Cast<IPostSelectionProvider>()
        ->RemovePostSelectionChangedListener(postSelListener);
      }
      else
      {
        activeProvider
        ->RemoveSelectionChangedListener(postSelListener);
      }
      activeProvider = 0;
    }
    activePart = 0;
  }

  activePart = newPart;

  if (newPart.IsNotNull())
  {
    activeProvider = selectionProvider;
    // Fire an event if there's an active provider
    activeProvider->AddSelectionChangedListener(selListener);
    ISelection::ConstPointer sel = activeProvider->GetSelection();
    this->FireSelection(newPart, sel);
    if (activeProvider.Cast<IPostSelectionProvider>().IsNotNull())
    {
      activeProvider.Cast<IPostSelectionProvider>()
      ->AddPostSelectionChangedListener(postSelListener);
    }
    else
    {
      activeProvider->AddSelectionChangedListener(postSelListener);
    }
    this->FirePostSelection(newPart, sel);
  }
  else
  {
    this->FireSelection(IWorkbenchPart::Pointer(0), ISelection::ConstPointer(0));
    this->FirePostSelection(IWorkbenchPart::Pointer(0), ISelection::ConstPointer(0));
  }
}

}
