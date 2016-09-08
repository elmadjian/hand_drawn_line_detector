
#include "gft.h"


/*Funcao auxiliar usada para visualizar os caminhos*/
gft::CImage::CImage *ViewPaths(gft::Image32::Image32 *img,
			       gft::Image32::Image32 *pred,
			       int S[]){
  gft::CImage::CImage *paths;
  int i,p;
  paths = gft::CImage::Clone(img);

  for(i = 1; i <= S[0]; i++){
    p = S[i];
    while(p != NIL){
      (paths->C[0])->data[p] = 255;
      (paths->C[1])->data[p] = 255;
      (paths->C[2])->data[p] = 0;
      p = pred->data[p];
    }
  }
  return paths;
}

  
int main(int argc, char **argv){
  gft::SparseGraph::SparseGraph *sg;
  gft::Image32::Image32 *img, *pred, *label;
  gft::CImage::CImage *paths;
  char file[512];
  float power = 1.0;

  /*Nos vetores abaixo, por convencao, a primeira posicao guarda o numero
    de elementos inseridos no vetor:*/

  /*Pontos de origem:*/
  int S1[] = {7, 148+99*600, 139+165*600, 129+205*600, 123+246*600, 111+315*600, 103+379*600, 90+450*600};
  /*Pontos de destino:*/
  int S2[] = {7, 483+88*600, 466+160*600, 441+228*600, 427+291*600, 408+352*600, 390+403*600, 372+462*600};
  int S[2];
  int i;
  
  if(argc < 2){
    fprintf(stdout,"usage:\n");
    fprintf(stdout,"proj1 <image1>\n");
    exit(0);
  }

  /*Leitura da imagem de entrada*/
  img = gft::Image32::Read(argv[1]);
  /*Calcula grafo da imagem com vizinhanca 8*/
  sg = gft::SparseGraph::ByWeightImage(img, 1.5);
  /*Cria imagem de rotulos*/
  label = gft::Image32::Create(img->ncols, img->nrows);

  /*Para cada ponto de origem:*/
  for(i = 1; i <= S1[0]; i++){
    /*Armazena o ponto de origem atual no vetor S, onde 
      a primeira posicao guarda o numero de elementos 
      armazenados no vetor (1 nesse caso).*/
    S[0] = 1;
    S[1] = S1[i];

    /*Coloca NIL (-1) em todas posicoes da matriz de rotulos*/
    gft::Image32::Set(label, NIL);
    /*O ponto de origem inicia com rotulo 1*/
    label->data[S[1]] = 1;

    /*Executa a IFT com funcao aditiva de custo*/
    pred = gft::ift::pred_IFTSUM(sg, S, label, power);

    /*Gera imagem colorida mostrando os caminhos calculados 
      ate os pontos de destino, que estao armazenados no 
      mapa de predecessores*/
    paths = ViewPaths(img, pred, S2);

    /*Grava imagem colorida com visualizacao dos caminhos no disco*/
    sprintf(file, "paths%02d.ppm", i);
    gft::CImage::Write(paths, file);

    /*Libera memoria do mapa de predecessores e da imagem colorida*/
    gft::CImage::Destroy(&paths);
    gft::Image32::Destroy(&pred);
  }

  /*Libera memoria do grafo*/
  gft::SparseGraph::Destroy(&sg);
  /*Libera memoria da imagem de rotulos*/
  gft::Image32::Destroy(&label);
  /*Libera memoria da imagem de entrada*/
  gft::Image32::Destroy(&img);

  return 0;
}
