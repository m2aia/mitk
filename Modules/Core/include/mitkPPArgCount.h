/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkPPArgCount_h
#define mitkPPArgCount_h

#define MITK_PP_ARG_COUNT(...) MITK_PP_ARG_COUNT_((__VA_ARGS__, MITK_PP_RSEQ_N()))

#define MITK_PP_ARG_COUNT_(tuple) MITK_PP_ARG_N tuple

#define MITK_PP_ARG_N(_1,                                                                                              \
                      _2,                                                                                              \
                      _3,                                                                                              \
                      _4,                                                                                              \
                      _5,                                                                                              \
                      _6,                                                                                              \
                      _7,                                                                                              \
                      _8,                                                                                              \
                      _9,                                                                                              \
                      _10,                                                                                             \
                      _11,                                                                                             \
                      _12,                                                                                             \
                      _13,                                                                                             \
                      _14,                                                                                             \
                      _15,                                                                                             \
                      _16,                                                                                             \
                      _17,                                                                                             \
                      _18,                                                                                             \
                      _19,                                                                                             \
                      _20,                                                                                             \
                      _21,                                                                                             \
                      _22,                                                                                             \
                      _23,                                                                                             \
                      _24,                                                                                             \
                      _25,                                                                                             \
                      N,                                                                                               \
                      ...)                                                                                             \
  N

#define MITK_PP_RSEQ_N() 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#endif
