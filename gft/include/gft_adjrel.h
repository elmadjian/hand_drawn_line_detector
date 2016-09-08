#ifndef _GFT_ADJREL_H_
#define _GFT_ADJREL_H_

#include "gft_common.h"

namespace gft{
  namespace AdjRel{

    typedef struct _adjrel {
      int *dx;
      int *dy;
      int n;
    } AdjRel;

    AdjRel *Create(int n);
    void    Destroy(AdjRel **A);
    AdjRel *Clone(AdjRel *A);
    
    AdjRel *Neighborhood_4(); /* 4-neighborhood */
    AdjRel *Neighborhood_8(); /* 8-neighborhood */
    AdjRel *Neighborhood_8_counterclockwise();
    AdjRel *Neighborhood_8_clockwise();
   
    AdjRel *Circular(float r);
    AdjRel *Box(int ncols, int nrows);

    //-----------------------------------
    int GetFrameSize(AdjRel *A);
    
  } /*end AdjRel namespace*/
} /*end gft namespace*/

#endif

