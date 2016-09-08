
#include "gft_cimage.h"

namespace gft{
  namespace CImage{


    CImage *Create(int ncols, int nrows){
      CImage *cimg=NULL;
      int i;
      
      cimg = (CImage *) calloc(1, sizeof(CImage));
      for (i=0; i < 3; i++)
	cimg->C[i] = Image32::Create(ncols,nrows);
      return(cimg);
    }


    CImage *Create(CImage *cimg){
      return Create(cimg->C[0]->ncols, cimg->C[0]->nrows);
    }


    CImage *Create(Image32::Image32 *img){
      return Create(img->ncols, img->nrows);
    }


    void    Destroy(CImage **cimg){
      CImage *tmp;
      int i;
      
      tmp = *cimg;
      if (tmp != NULL) {
	for (i=0; i < 3; i++)
	  Image32::Destroy(&(tmp->C[i]));
	free(tmp);
	*cimg = NULL;
      }
    }


    CImage *Read(char *filename){
      CImage *cimg=NULL;
      FILE *fp=NULL;
      char type[10];
      int  t,i,ncols,nrows,n;
      char z[256];
      int l;
      gft::Image32::Image32 *img;

      l = strlen(filename);
      if(strcasecmp(filename+l-3, "pgm") == 0){
	img = gft::Image32::Read(filename);
	cimg = Create(img->ncols, img->nrows);
	for(i = 0; i < img->n; i++){
	  cimg->C[0]->data[i] = img->data[i];
	  cimg->C[1]->data[i] = img->data[i];
	  cimg->C[2]->data[i] = img->data[i];
	}
	gft::Image32::Destroy(&img);
	return cimg;
      }
      
      fp = fopen(filename,"rb");
      if (fp == NULL){
	fprintf(stderr,"Cannot open %s\n",filename);
	exit(-1);
      }
      fscanf(fp,"%s\n",type);
      if((strcmp(type,"P6")==0)){
	gft::NCFgets(z,255,fp);
	t = sscanf(z,"%d %d\n",&ncols,&nrows);
	if(t == EOF || t < 2){
	  gft::NCFgets(z,255,fp);
	  sscanf(z,"%d %d\n",&ncols,&nrows);
	}
	n = ncols*nrows;
	gft::NCFgets(z,255,fp);
	sscanf(z,"%d\n",&i);
	cimg = Create(ncols,nrows);
	for (i=0; i < n; i++){
	  cimg->C[0]->data[i] = fgetc(fp);
	  cimg->C[1]->data[i] = fgetc(fp);
	  cimg->C[2]->data[i] = fgetc(fp);
	}
	fclose(fp);
      }else{
	fprintf(stderr,"Input image must be P6\n");
	exit(-1);
      }
      
      return(cimg);
    }
    

    void    Write(CImage *cimg, char *filename){
      FILE *fp;
      int i,n;
      
      fp = fopen(filename,"w");
      fprintf(fp,"P6\n");
      fprintf(fp,"%d %d\n",cimg->C[0]->ncols,cimg->C[0]->nrows);
      fprintf(fp,"255\n");
      n = cimg->C[0]->ncols*cimg->C[0]->nrows;
      for (i=0; i < n; i++) {
	fputc(cimg->C[0]->data[i],fp);
	fputc(cimg->C[1]->data[i],fp);
	fputc(cimg->C[2]->data[i],fp);
      }
      fclose(fp);
    }
    
    
    CImage *Clone(CImage *cimg){
      CImage *imgc;
      int i;
      
      imgc = (CImage *) calloc(1,sizeof(CImage));
      if (imgc == NULL){
	gft::Error((char *)MSG1,(char *)"CImage::Clone");
      }
      for (i=0; i<3; i++)
	imgc->C[i] = Image32::Clone(cimg->C[i]);
      return imgc;
    }


    CImage *Clone(Image32::Image32 *img){
      CImage *imgc;
      Image32::Image32 *img8;
      int i;
      if( GetMinimumValue(img) < 0 ||
	  GetMaximumValue(img) > 255 )
	img8 = Image32::ConvertToNbits(img, 8);
      else
	img8 = Image32::Clone(img);

      imgc = (CImage *) calloc(1,sizeof(CImage));
      if (imgc == NULL){
	gft::Error((char *)MSG1,(char *)"CImage::Clone");
      }
      imgc->C[0] = img8;
      for (i=1; i<3; i++)
	imgc->C[i] = Image32::Clone(img8);
      return imgc;
    }

    
    void    Set(CImage *cimg, int r, int g, int b){
      Image32::Set(cimg->C[0], r);
      Image32::Set(cimg->C[1], g);
      Image32::Set(cimg->C[2], b);
    }


    void    Set(CImage *cimg, int color){
      Image32::Set(cimg->C[0], gft::Color::Channel0(color));
      Image32::Set(cimg->C[1], gft::Color::Channel1(color));
      Image32::Set(cimg->C[2], gft::Color::Channel2(color));
    }
    
    
    CImage *ColorizeLabel(Image32::Image32 *label){
      CImage *cimg;
      int n,p,l,Lmax = Image32::GetMaximumValue(label);
      int *R,*G,*B;
      
      cimg = Create(label->ncols, label->nrows);
      R = gft::AllocIntArray(Lmax+1);
      G = gft::AllocIntArray(Lmax+1);
      B = gft::AllocIntArray(Lmax+1);
      gft::RandomSeed();
      for(l = 0; l <= Lmax; l++){
	R[l] = gft::RandomInteger(0, 255);
	G[l] = gft::RandomInteger(0, 255);
	B[l] = gft::RandomInteger(0, 255);
      }
      n = label->ncols*label->nrows;
      for(p = 0; p < n; p++){
	l = label->data[p];
	cimg->C[0]->data[p] = R[l];
	cimg->C[1]->data[p] = G[l];
	cimg->C[2]->data[p] = B[l];
      }
      free(R);
      free(G);
      free(B);
      return cimg;
    }


    CImage *RGB2Lab(CImage *cimg){
      CImage *cimg_lab;
      gft::Color::ColorRGB rgb;
      gft::Color::ColorLab lab;
      int p,n;
      n = cimg->C[0]->n;
      cimg_lab = Create(cimg->C[0]->ncols, cimg->C[0]->nrows);
      for(p = 0; p < n; p++){
	rgb.r = cimg->C[0]->data[p];
	rgb.g = cimg->C[1]->data[p];
	rgb.b = cimg->C[2]->data[p];
	lab = gft::Color::RGB2Lab_2(rgb);
	cimg_lab->C[0]->data[p] = ROUND(lab.l);
	cimg_lab->C[1]->data[p] = ROUND(lab.a);
	cimg_lab->C[2]->data[p] = ROUND(lab.b);
      }
      return cimg_lab;
    }


    Image32::Image32 *Lightness(CImage *cimg){
      Image32::Image32 *img;
      int p,n,max,min;
      n = cimg->C[0]->n;
      img = Image32::Create(cimg->C[0]->ncols, cimg->C[0]->nrows);
      for(p = 0; p < n; p++){
	max = MAX(cimg->C[0]->data[p], 
		  MAX(cimg->C[1]->data[p], cimg->C[2]->data[p]));
	min = MIN(cimg->C[0]->data[p], 
		  MIN(cimg->C[1]->data[p], cimg->C[2]->data[p]));
	img->data[p] = ROUND((max+min)/2.0);
      }
      return img;
    }


    /*The luminosity method works best overall and is the 
      default method used if you ask GIMP to change an image 
      from RGB to grayscale */
    /*It averages the values, but it forms a weighted average 
      to account for human perception.
      We’re more sensitive to green than other colors, so green 
      is weighted most heavily. */
    Image32::Image32 *Luminosity(CImage *cimg){
      Image32::Image32 *img;
      int p,n;
      n = cimg->C[0]->n;
      img = Image32::Create(cimg->C[0]->ncols, cimg->C[0]->nrows);
      for(p = 0; p < n; p++){
	img->data[p] = ROUND(0.21 * cimg->C[0]->data[p] + 
			     0.72 * cimg->C[1]->data[p] + 
			     0.07 * cimg->C[2]->data[p]);
      }
      return img;
    }


  } /*end CImage namespace*/
} /*end gft namespace*/
