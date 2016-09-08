
#include "gft_filtering.h"

namespace gft{

  namespace Kernel{

    Kernel *Make(char *coefs){
      Kernel *K;
      AdjRel::AdjRel *A;
      int xsize,ysize,i;
      float val;

      sscanf(coefs,"%d",&xsize);
      coefs=strchr(coefs,',')+1;
      sscanf(coefs,"%d",&ysize);
      coefs=strchr(coefs,',')+1;
      
      A = AdjRel::Box(xsize, ysize);
      K = Create(A);
      for(i = 0; i < A->n; i++){
	sscanf(coefs, "%f", &K->val[i]);
	coefs = strchr(coefs,',')+1;
      }

      /* Put the middle value (corresponding to the origin) at the first
	 place to match the vector of coefficients with the vector of
	 displacements (adjacency relation) */
      
      for(i=A->n/2; i > 0; i--){
	val = K->val[i];
	K->val[i] = K->val[i-1];
	K->val[i-1] = val;
      }
      
      AdjRel::Destroy(&A);
      return(K);
    }

    
    Kernel *Create(AdjRel::AdjRel *A){
      Kernel *K=NULL;
      int i;
      
      K = (Kernel *) calloc(1,sizeof(Kernel));
      if(K == NULL){
	gft::Error(MSG1, "Kernel::Create");
      }
      K->val = gft::AllocFloatArray(A->n);
      K->adj = AdjRel::Clone(A);
      return(K);
    }


    Kernel *Clone(Kernel *K){
      Kernel *C;
      
      C = (Kernel *) calloc(1,sizeof(Kernel));
      if(C == NULL)
	gft::Error(MSG1, "Kernel::Clone");
      
      C->val = gft::AllocFloatArray(K->adj->n);
      memcpy(C->val, K->val,
	     sizeof(float)*K->adj->n);
      C->adj = AdjRel::Clone(K->adj);
      return C;
    }


    Kernel *Normalize(Kernel *K){
      Kernel *C = Clone(K);
      float wt=0.0;
      int i;
      for(i = 0; i < K->adj->n; i++)
	wt += K->val[i];
      for(i = 0; i < C->adj->n; i++)
	C->val[i] /= wt;
      
      return C;
    }


    void    Destroy(Kernel **K){
      Kernel *aux;
      aux = *K;
      if(aux != NULL){
	if (aux->val != NULL)   free(aux->val);
	AdjRel::Destroy(&(aux->adj));
	free(aux);
	*K = NULL;
      }
    }


  } //end Kernel namespace


  namespace Image32{

    /*
    void ModeFilterLabel(Image32 *label, float r){
    }
    */

    Image32 *SobelFilter(Image32 *img){
      Image32 *gradx=NULL,*grady=NULL,*grad=NULL;
      Kernel::Kernel *Kx,*Ky;

      Ky = Kernel::Make("3,3,-1.0,-2.0,-1.0,0.0,0.0,0.0,1.0,2.0,1.0");
      Kx = Kernel::Make("3,3,-1.0,0.0,1.0,-2.0,0.0,2.0,-1.0,0.0,1.0");
      gradx = LinearFilter(img, Kx);
      grady = LinearFilter(img, Ky);
      grad  = ImageMagnitude(gradx, grady);
      Destroy(&gradx);
      Destroy(&grady);
      Kernel::Destroy(&Kx);
      Kernel::Destroy(&Ky);
      return(grad);
    }
    

    Image32 *LinearFilter(Image32 *img, Kernel::Kernel *K){
      Image32 *cimg;
      int u_x,u_y;
      float conv;
      unsigned int p, i, x, y;
      
      cimg = Create(img->ncols, img->nrows);
      
      for(u_y = 0; u_y < img->nrows; ++u_y){
	for(u_x = 0; u_x < img->ncols; ++u_x){
	  conv = 0.0;
	  for(i = 0; i < K->adj->n; ++i){
	    x = u_x + K->adj->dx[i];
	    y = u_y + K->adj->dy[i];
	    if ((x >= 0) && (x < img->ncols) && (y >= 0) && (y < img->nrows))
	      conv += (float)img->array[y][x] * K->val[i];
	  }
	  cimg->array[u_y][u_x] = (int)conv;
	}
      }
      return(cimg);
    }


    Image32  *ImageMagnitude(Image32 *imgx, Image32 *imgy){
      Image32 *mag = Create(imgx->ncols, imgx->nrows);
      int p, n = imgx->ncols*imgx->nrows;
      
      for(p = 0; p < n; p++)
        mag->data[p] = ROUND(sqrt(imgx->data[p]*imgx->data[p] + imgy->data[p]*imgy->data[p]));
 
      return(mag);
    }



  } //end Image32 namespace


} //end gft namespace

