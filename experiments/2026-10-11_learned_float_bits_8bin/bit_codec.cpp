#include <cstdint>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <algorithm>
extern "C" {
void encode(const float* x,int n,const int* pos,int k,const uint8_t* lut,uint8_t* out){
 for(int i=0;i<n;i++) for(int d=0;d<64;d++){
  uint32_t u; std::memcpy(&u,x+i*64+d,4); unsigned c=0;
  for(int r=0;r<k;r++) c|=((u>>pos[r])&1u)<<r;
  out[i*64+d]=lut[d*(1<<k)+c];
 }
}
void pack3(const uint8_t* x,int n,uint8_t* out){
 std::memset(out,0,n*24);
 for(int i=0;i<n;i++) for(int d=0;d<64;d++){
  unsigned b=d*3, v=x[i*64+d]; out[i*24+b/8]|=v<<(b%8);
  if(b%8>5) out[i*24+b/8+1]|=v>>(8-b%8);
 }
}
void unpack3(const uint8_t* x,int n,uint8_t* out){
 for(int i=0;i<n;i++) for(int d=0;d<64;d++){
  unsigned b=d*3,v=x[i*24+b/8]>>(b%8);
  if(b%8>5) v|=unsigned(x[i*24+b/8+1])<<(8-b%8);
  out[i*64+d]=v&7;
 }
}
void packthermo(const uint8_t* x,int n,uint64_t* out){
 for(int i=0;i<n;i++) for(int r=0;r<7;r++){
  uint64_t mask=0;for(int d=0;d<64;d++) if(x[i*64+d]>r) mask|=uint64_t(1)<<d;
  out[i*7+r]=mask;
 }
}
void absdist(const uint8_t* a,const uint8_t* b,int n,int m,uint16_t* out){
 for(int i=0;i<n;i++)for(int j=0;j<m;j++){
  unsigned s=0;for(int d=0;d<64;d++)s+=std::abs(int(a[i*64+d])-int(b[j*64+d]));out[i*m+j]=s;
 }
}
void thermodist(const uint64_t* a,const uint64_t* b,int n,int m,uint16_t* out){
 for(int i=0;i<n;i++)for(int j=0;j<m;j++){
  unsigned s=0;for(int r=0;r<7;r++)s+=__builtin_popcountll(a[i*7+r]^b[j*7+r]);out[i*m+j]=s;
 }
}
void absmnn(const uint8_t* a,const uint8_t* b,int n,int m,int* row,int* col){
 std::vector<unsigned> bestrow(n,10000),bestcol(m,10000);
 std::fill(row,row+n,-1);std::fill(col,col+m,-1);
 for(int i=0;i<n;i++)for(int j=0;j<m;j++){
  unsigned s=0;for(int d=0;d<64;d++)s+=std::abs(int(a[i*64+d])-int(b[j*64+d]));
  if(s<bestrow[i]){bestrow[i]=s;row[i]=j;}if(s<bestcol[j]){bestcol[j]=s;col[j]=i;}
 }
}
void thermomnn(const uint64_t* a,const uint64_t* b,int n,int m,int* row,int* col){
 std::vector<unsigned> bestrow(n,10000),bestcol(m,10000);
 std::fill(row,row+n,-1);std::fill(col,col+m,-1);
 for(int i=0;i<n;i++)for(int j=0;j<m;j++){
  unsigned s=0;for(int r=0;r<7;r++)s+=__builtin_popcountll(a[i*7+r]^b[j*7+r]);
  if(s<bestrow[i]){bestrow[i]=s;row[i]=j;}if(s<bestcol[j]){bestcol[j]=s;col[j]=i;}
 }
}
}
