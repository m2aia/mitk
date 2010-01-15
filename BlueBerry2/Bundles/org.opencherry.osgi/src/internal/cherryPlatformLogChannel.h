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

#ifndef CHERRYPLATFORMLOGCHANNEL_H_
#define CHERRYPLATFORMLOGCHANNEL_H_

#include "Poco/Message.h"
#include "Poco/SimpleFileChannel.h"

namespace cherry {

class PlatformLogChannel : public Poco::SimpleFileChannel
{
public:
  PlatformLogChannel(const std::string& path);
  
  void log(const Poco::Message& msg);
  
};

}

#endif /*CHERRYPLATFORMLOGCHANNEL_H_*/
