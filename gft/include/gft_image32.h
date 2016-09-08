#ifndef _GFT_IMAGE32_H_
#define _GFT_IMAGE32_H_

#include "gft_common.h"

namespace gft{
  namespace Image32{

    /**
     * It supports both linear and two-dimensional access 
     * (i.e., img->data[p] or img->array[y][x] for a pixel
     * (x,y) at address p=x+y*xsize).
     */
    typedef struct _image32 {
      int *data;
      int **array;
      int nrows; /* numero de linhas (altura) */
      int ncols; /* numero de colunas (largura) */
      int n;     /* numero de pixels */
      float dx;
      float dy;
    } Image32;

    /**
     * \brief A constructor.
     */
    Image32 *Create(int ncols,int nrows);

    /**
     * \brief A destructor.
     */
    void    Destroy(Image32 **img);

    /**
     * \brief A copy constructor.
     */
    Image32 *Clone(Image32 *img);


    Image32 *Read(char *filename);
    void     Write(Image32 *img, char *filename);

    Image32 *ConvertToNbits(Image32 *img, int N);

    int     GetMinimumValue(Image32 *img);
    int     GetMaximumValue(Image32 *img);

    void    Set(Image32 *img, int value);

    bool    IsValidPixel(Image32 *img, int x, int y);

    Image32 *Threshold(Image32 *img, int L, int H);
    
    void    DrawRectangle(Image32 *img, 
			  int x1, int y1, 
			  int x2, int y2, int val);
    void    DrawLineDDA(Image32 *img, 
			int x1, int y1, 
			int xn, int yn, int val);

    void    DrawCircle(Image32 *img,
		       int x1, int y1,
		       float r,
		       int val);

    //------------------------------------

    Image32 *AddFrame(Image32 *img, int sz, int value);
    Image32 *RemFrame(Image32 *fimg, int sz);
    
  } //end Image32 namespace
} //end gft namespace

#endif

