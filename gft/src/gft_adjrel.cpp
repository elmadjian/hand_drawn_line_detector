
#include "gft_adjrel.h"

namespace gft{
  namespace AdjRel{

    AdjRel *Create(int n){
      AdjRel *A=NULL;
      
      A = (AdjRel *) calloc(1,sizeof(AdjRel));
      if (A != NULL){
	A->dx = gft::AllocIntArray(n);
	A->dy = gft::AllocIntArray(n);
	A->n  = n;
      } else {
	gft::Error((char *)MSG1,(char *)"AdjRel::Create");
      }

      return(A);
    }


    void Destroy(AdjRel **A){
      AdjRel *aux;

      aux = *A;
      if (aux != NULL){
	if (aux->dx != NULL) free(aux->dx);
	if (aux->dy != NULL) free(aux->dy);
	free(aux);
	*A = NULL;
      }   
    }
    

    AdjRel *Clone(AdjRel *A){
      AdjRel *C;
      int i;
      
      C = Create(A->n);
      for(i=0; i < A->n; i++){
	C->dx[i] = A->dx[i];
	C->dy[i] = A->dy[i];
      }
      return C;
    }
    
    
    AdjRel *Neighborhood_4(){ /* 4-neighborhood */
      AdjRel *A=NULL;
      A = Create(4+1);
      /* place central pixel at first */
      A->dx[0] = 0;  A->dy[0] = 0;  
      A->dx[1] = 1;  A->dy[1] = 0;  /* right */
      A->dx[2] = 0;  A->dy[2] = -1; /* top */
      A->dx[3] = -1; A->dy[3] = 0;  /* left */
      A->dx[4] = 0;  A->dy[4] = 1;  /* bottom */
      return A;
    }
    

    AdjRel *Neighborhood_8(){ /* 8-neighborhood */
      AdjRel *A=NULL;
      int i,dx,dy;
      A = Create(8+1);
      /* place central pixel at first */
      A->dx[0] = 0;
      A->dy[0] = 0;
      i = 1;
      for(dy = -1; dy <= 1; dy++){
	for(dx = -1; dx <= 1; dx++){
	  if ((dx != 0)||(dy != 0)){
	    A->dx[i] = dx;
	    A->dy[i] = dy;
	    i++;
	  }
	}
      }
      return A;
    }


    AdjRel *Neighborhood_8_counterclockwise(){ /* 8-neighborhood */
      AdjRel *A=NULL;
      A = Create(8+1);
      /* place central pixel at first */
      A->dx[0] =  0; A->dy[0] =  0;
      A->dx[1] = -1; A->dy[1] =  0;
      A->dx[2] = -1; A->dy[2] =  1;
      A->dx[3] =  0; A->dy[3] =  1;
      A->dx[4] =  1; A->dy[4] =  1;      
      A->dx[5] =  1; A->dy[5] =  0;
      A->dx[6] =  1; A->dy[6] = -1;      
      A->dx[7] =  0; A->dy[7] = -1;
      A->dx[8] = -1; A->dy[8] = -1;
      return A;
    }


    AdjRel *Neighborhood_8_clockwise(){ /* 8-neighborhood */
      AdjRel *A=NULL;
      A = Create(8+1);
      /* place central pixel at first */
      A->dx[0] =  0; A->dy[0] =  0;
      A->dx[1] = -1; A->dy[1] =  0;
      A->dx[2] = -1; A->dy[2] = -1;
      A->dx[3] =  0; A->dy[3] = -1;
      A->dx[4] =  1; A->dy[4] = -1;
      A->dx[5] =  1; A->dy[5] =  0;
      A->dx[6] =  1; A->dy[6] =  1;
      A->dx[7] =  0; A->dy[7] =  1;
      A->dx[8] = -1; A->dy[8] =  1;
      return A;
    }
    

    AdjRel *Circular(float r){
      AdjRel *A=NULL;
      int i,n,dx,dy,r0,r2;
      
      n=0;
      r0 = (int)r;
      r2  = (int)(r*r + 0.5);
      for(dy=-r0;dy<=r0;dy++)
	for(dx=-r0;dx<=r0;dx++)
	  if(((dx*dx)+(dy*dy)) <= r2)
	    n++;
      
      A = Create(n);
      i = 1;
      for(dy=-r0;dy<=r0;dy++)
	for(dx=-r0;dx<=r0;dx++)
	  if(((dx*dx)+(dy*dy)) <= r2){
	    if ((dx != 0)||(dy != 0)){
	      A->dx[i] = dx;
	      A->dy[i] = dy;
	      i++;
	    }
	  }
      
      /* place central pixel at first */
      A->dx[0] = 0;
      A->dy[0] = 0;
      
      return(A);
    }
    
    
    AdjRel *Box(int ncols, int nrows){
      AdjRel *A=NULL;
      int i,dx,dy;
      
      if (ncols%2 == 0) ncols++;
      if (nrows%2 == 0) nrows++;
      
      A = Create(ncols*nrows);
      i=1;
      for(dy=-nrows/2;dy<=nrows/2;dy++){
	for(dx=-ncols/2;dx<=ncols/2;dx++){
	  if ((dx != 0)||(dy != 0)){
	    A->dx[i] = dx;
	    A->dy[i] = dy;
	    i++;
	  }
	}
      }
      /* place the central pixel at first */
      A->dx[0] = 0;
      A->dy[0] = 0;
      
      return(A);
    }


    
    int GetFrameSize(AdjRel *A){
      int sz=INT_MIN,i=0;
      
      for (i=0; i < A->n; i++){
	if (abs(A->dx[i]) > sz) 
	  sz = abs(A->dx[i]);
	if (abs(A->dy[i]) > sz) 
	  sz = abs(A->dy[i]);
      }
      return(sz);
    }

    
  } /*end AdjRel namespace*/
} /*end gft namespace*/

