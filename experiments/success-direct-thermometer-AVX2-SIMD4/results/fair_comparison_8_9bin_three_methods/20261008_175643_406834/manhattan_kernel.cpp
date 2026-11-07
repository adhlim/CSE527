
#include <cstdint>
#include <vector>
#include <immintrin.h>
#include <omp.h>
extern "C" void packed_encode(const float*x,const float*et,int n,int planes,int bits,uint8_t*out){
 const int bytes=bits*8;
 for(int i=0;i<n;i++)for(int g=0;g<8;g++){
  auto a=_mm256_loadu_ps(x+i*64+g*8);auto bins=_mm256_setzero_si256();
  for(int k=0;k<planes;k++){
   auto e=_mm256_loadu_ps(et+k*64+g*8);
   bins=_mm256_sub_epi32(bins,_mm256_castps_si256(_mm256_cmp_ps(a,e,_CMP_GT_OQ)));
  }
  uint32_t v[8];_mm256_storeu_si256((__m256i*)v,bins);
  uint32_t word=0;for(int k=0;k<8;k++)word|=v[k]<<(k*bits);
  uint8_t*p=out+i*bytes+g*bits;for(int b=0;b<bits;b++)p[b]=(word>>(b*8))&255;
 }
}
static void decode(const uint8_t*p,int n,int bits,uint8_t*c){
 const int bytes=bits*8,mask=(1<<bits)-1;
 for(int i=0;i<n;i++)for(int g=0;g<8;g++){
  uint32_t word=0;for(int b=0;b<bits;b++)word|=uint32_t(p[i*bytes+g*bits+b])<<(b*8);
  for(int k=0;k<8;k++)c[i*64+g*8+k]=(word>>(k*bits))&mask;
 }
}
static int distance(const uint8_t*a,const uint8_t*b){
 auto sum=_mm256_setzero_si256();
 for(int k=0;k<64;k+=32){
  auto x=_mm256_loadu_si256((const __m256i*)(a+k)),y=_mm256_loadu_si256((const __m256i*)(b+k));
  sum=_mm256_add_epi64(sum,_mm256_sad_epu8(x,y));
 }
 uint64_t v[4];_mm256_storeu_si256((__m256i*)v,sum);
 return int(v[0]+v[1]+v[2]+v[3]);
}
extern "C" void packed_matrix(const uint8_t*a,const uint8_t*b,int n,int m,int bits,int threads,uint16_t*out){
 std::vector<uint8_t> aa(n*64),bb(m*64);decode(a,n,bits,aa.data());decode(b,m,bits,bb.data());
 #pragma omp parallel for num_threads(threads) schedule(static)
 for(int i=0;i<n;i++)for(int j=0;j<m;j++)out[i*m+j]=distance(aa.data()+i*64,bb.data()+j*64);
}
extern "C" void packed_mnn(const uint8_t*a,const uint8_t*b,int n,int m,int bits,int threads,int*ab,int*ba){
 if(!n||!m)return;
 std::vector<uint8_t> aa(n*64),bb(m*64);decode(a,n,bits,aa.data());decode(b,m,bits,bb.data());
 std::vector<int> cd(threads*m,513),ci(threads*m,n);
 #pragma omp parallel num_threads(threads)
 {
  int t=omp_get_thread_num();int*ds=cd.data()+t*m;int*ids=ci.data()+t*m;
  #pragma omp for schedule(static)
  for(int i=0;i<n;i++){
   int best=513,bj=m;
   for(int j=0;j<m;j++){
    int d=distance(aa.data()+i*64,bb.data()+j*64);
    if(d<best){best=d;bj=j;}
    if(d<ds[j]||(d==ds[j]&&i<ids[j])){ds[j]=d;ids[j]=i;}
   }ab[i]=bj;
  }
 }
 for(int j=0;j<m;j++){
  int best=513,bi=n;
  for(int t=0;t<threads;t++){int d=cd[t*m+j],i=ci[t*m+j];if(d<best||(d==best&&i<bi)){best=d;bi=i;}}
  ba[j]=bi;
 }
}
