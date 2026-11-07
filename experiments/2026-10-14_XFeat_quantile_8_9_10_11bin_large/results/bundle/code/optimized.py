import numpy as np, ctypes, tempfile, subprocess
from pathlib import Path
CPU_THREADS=2
AVX2=True
ptr=ctypes.c_void_p;integer=ctypes.c_int
def adr(x):return x.ctypes.data_as(ptr)
cpp_source='#include <cstdint>\n#include <vector>\n#include <algorithm>\n#include <immintrin.h>\n#include <omp.h>\nextern "C" int has_avx(){return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vpopcntdq");}\n__attribute__((target("popcnt"))) static inline int scalar(const uint64_t*a,const uint64_t*b){int d=0;for(int k=0;k<7;k++)d+=__builtin_popcountll(a[k]^b[k]);return d;}\n__attribute__((target("avx512f,avx512vpopcntdq"))) static int wide(const uint64_t*a,const uint64_t*b){\n __m512i x=_mm512_maskz_loadu_epi64(0x7f,a),y=_mm512_maskz_loadu_epi64(0x7f,b);\n return (int)_mm512_reduce_add_epi64(_mm512_popcnt_epi64(_mm512_xor_si512(x,y)));\n}\nextern "C" void encode_pack(const float*x,const float*e,int n,uint8_t*c,uint64_t*p){\n for(int i=0;i<n;i++){for(int k=0;k<7;k++)p[i*7+k]=0;\n for(int d=0;d<64;d++){int v=0;for(int k=0;k<7;k++)v+=x[i*64+d]>e[d*7+k];c[i*64+d]=v;\n for(int k=0;k<v;k++)p[i*7+k]|=uint64_t(1)<<d;}}\n}\nextern "C" void matrix(const uint64_t*a,const uint64_t*b,int n,int m,int threads,uint16_t*out){\n #pragma omp parallel for num_threads(threads) schedule(static)\n for(int i=0;i<n;i++)for(int j=0;j<m;j++)out[i*m+j]=scalar(a+i*7,b+j*7);\n}\nextern "C" void fused(const uint64_t*a,const uint64_t*b,int n,int m,int mode,int threads,int*ab,int*ba){\n if(n==0||m==0)return;\n const bool avx=mode==1&&has_avx();\n std::vector<int> cd(threads*m,449),ci(threads*m,n);\n #pragma omp parallel num_threads(threads)\n {\n int t=omp_get_thread_num();int*ds=cd.data()+t*m;int*ids=ci.data()+t*m;\n #pragma omp for schedule(static)\n for(int i=0;i<n;i++){int best=449,bj=m;\n for(int j=0;j<m;j++){int dist=0;\n if(mode==2){for(int k=0;k<7;k++){dist+=__builtin_popcountll(a[i*7+k]^b[j*7+k]);\n // Strict inequality preserves the smallest-index tie rule in both directions.\n if(dist>best&&dist>ds[j])break;}}\n else dist=avx?wide(a+i*7,b+j*7):scalar(a+i*7,b+j*7);\n if(dist<best){best=dist;bj=j;}\n if(dist<ds[j]||(dist==ds[j]&&i<ids[j])){ds[j]=dist;ids[j]=i;}\n }ab[i]=bj;}\n }\n for(int j=0;j<m;j++){int best=449,bi=n;for(int t=0;t<threads;t++){\n int d=cd[t*m+j],i=ci[t*m+j];if(d<best||(d==best&&i<bi)){best=d;bi=i;}}ba[j]=bi;}\n}\nextern "C" int has_avx2(){return __builtin_cpu_supports("avx2");}\nextern "C" void direct_scalar(const float*x,const float*et,int n,uint64_t*p){\n for(int i=0;i<n;i++)for(int k=0;k<7;k++){uint64_t w=0;for(int d=0;d<64;d++)w|=uint64_t(x[i*64+d]>et[k*64+d])<<d;p[i*7+k]=w;}\n}\nextern "C" __attribute__((target("avx2"))) void direct_avx(const float*x,const float*et,int n,uint64_t*p){\n for(int i=0;i<n;i++)for(int k=0;k<7;k++){uint64_t w=0;\n for(int d=0;d<64;d+=8){auto a=_mm256_loadu_ps(x+i*64+d),b=_mm256_loadu_ps(et+k*64+d);\n w|=uint64_t(uint32_t(_mm256_movemask_ps(_mm256_cmp_ps(a,b,_CMP_GT_OQ))))<<d;}p[i*7+k]=w;}\n}\n__attribute__((target("avx2"))) static inline __m256i counts4(__m256i v){\n const __m256i lut=_mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);\n auto mask=_mm256_set1_epi8(15);\n auto lo=_mm256_shuffle_epi8(lut,_mm256_and_si256(v,mask));\n auto hi=_mm256_shuffle_epi8(lut,_mm256_and_si256(_mm256_srli_epi16(v,4),mask));\n return _mm256_sad_epu8(_mm256_add_epi8(lo,hi),_mm256_setzero_si256());\n}\nextern "C" void transpose7(const uint64_t*p,int n,uint64_t*s){for(int k=0;k<7;k++)for(int i=0;i<n;i++)s[k*n+i]=p[i*7+k];}\nextern "C" __attribute__((target("avx2"))) void fused4(const uint64_t*a,const uint64_t*bs,int n,int m,int threads,int*ab,int*ba){\n if(n==0||m==0)return;\n std::vector<int> cd(threads*m,449),ci(threads*m,n);\n #pragma omp parallel num_threads(threads)\n {\n int t=omp_get_thread_num();int*ds=cd.data()+t*m;int*ids=ci.data()+t*m;\n #pragma omp for schedule(static)\n for(int i=0;i<n;i++){int best=449,bj=m;int j=0;\n for(;j+4<=m;j+=4){auto sum=_mm256_setzero_si256();\n for(int k=0;k<7;k++){auto av=_mm256_set1_epi64x(a[i*7+k]);auto bv=_mm256_loadu_si256((const __m256i*)(bs+k*m+j));sum=_mm256_add_epi64(sum,counts4(_mm256_xor_si256(av,bv)));}\n uint64_t dist[4];_mm256_storeu_si256((__m256i*)dist,sum);\n for(int l=0;l<4;l++){int d=dist[l],q=j+l;if(d<best){best=d;bj=q;}if(d<ds[q]||(d==ds[q]&&i<ids[q])){ds[q]=d;ids[q]=i;}}\n }\n for(;j<m;j++){int d=0;for(int k=0;k<7;k++)d+=__builtin_popcountll(a[i*7+k]^bs[k*m+j]);if(d<best){best=d;bj=j;}if(d<ds[j]||(d==ds[j]&&i<ids[j])){ds[j]=d;ids[j]=i;}}\n ab[i]=bj;}\n }\n for(int j=0;j<m;j++){int best=449,bi=n;for(int t=0;t<threads;t++){int d=cd[t*m+j],i=ci[t*m+j];if(d<best||(d==best&&i<bi)){best=d;bi=i;}}ba[j]=bi;}\n}\n'
class ThermometerVariant:
 def __init__(self, bins, edges):
  self.bins=bins;self.planes=bins-1;self.bytes_per_kp=self.planes*8
  self.name=f'{bins} bins / {self.planes} bits: method 1+2+3 SIMD4' if AVX2 else f'{bins} bins / {self.planes} bits: scalar fallback'
  self.edges=np.ascontiguousarray(edges,dtype=np.float32)
  self.et=np.ascontiguousarray(self.edges.T)
  import re
  source=re.sub(r'\b7\b',str(self.planes),cpp_source).replace('449',str(64*self.planes+1))
  # Original AVX512 distance loads at most eight planes: use exact scalar here.
  # The benchmarked SIMD4 function is AVX2 and still processes ALL planes.
  begin=source.index(' __m512i x=');end=source.index('\n}',begin)
  source=source[:begin]+' return scalar(a,b);'+source[end:]
  folder=Path(tempfile.mkdtemp(prefix=f'thermometer_{bins}bin_'))
  (folder/'kernel.cpp').write_text(source)
  subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-fopenmp','-mpopcnt',str(folder/'kernel.cpp'),'-o',str(folder/'kernel.so')],check=True)
  self.lib=ctypes.CDLL(str(folder/'kernel.so'))
  self.lib.has_avx2.restype=integer
  if not self.lib.has_avx2():raise RuntimeError('This success method requires an AVX2 x86 CPU. Use a compatible CPU runtime.')
  for name in ('direct_scalar','direct_avx'):getattr(self.lib,name).argtypes=[ptr,ptr,integer,ptr]
  self.lib.transpose7.argtypes=[ptr,integer,ptr]
  self.lib.fused4.argtypes=[ptr,ptr,integer,integer,integer,ptr,ptr]
  self.lib.fused.argtypes=[ptr,ptr,integer,integer,integer,integer,ptr,ptr]
  self.lib.matrix.argtypes=[ptr,ptr,integer,integer,integer,ptr]
  for name in ('direct_scalar','direct_avx','transpose7','fused4','fused','matrix'):getattr(self.lib,name).restype=None
 def codes(self,x):return np.ascontiguousarray((x[:,:,None]>self.edges[None,:,:]).sum(axis=2),dtype=np.uint8)
 def encode(self,x,simd=True):
  x=np.ascontiguousarray(x,dtype=np.float32);out=np.empty((len(x),self.planes),np.uint64)
  fn=self.lib.direct_avx if simd and AVX2 else self.lib.direct_scalar
  fn(adr(x),adr(self.et),len(x),adr(out));return out
 def soa(self,p):
  out=np.empty((self.planes,len(p)),np.uint64);self.lib.transpose7(adr(p),len(p),adr(out));return out
 def match(self,a,bs):
  nn,mm=len(a),bs.shape[1]
  if not nn or not mm:return np.empty(0,np.int64),np.empty(0,np.int64)
  ab=np.empty(nn,np.int32);ba=np.empty(mm,np.int32)
  if AVX2:self.lib.fused4(adr(a),adr(bs),nn,mm,CPU_THREADS,adr(ab),adr(ba))
  else:
   b=np.ascontiguousarray(bs.T);self.lib.fused(adr(a),adr(b),nn,mm,0,CPU_THREADS,adr(ab),adr(ba))
  ii=np.arange(nn);keep=ba[ab]==ii;return ii[keep],ab[keep].astype(np.int64)
 def matrix(self,a,b):
  out=np.empty((len(a),len(b)),np.uint16);self.lib.matrix(adr(a),adr(b),len(a),len(b),CPU_THREADS,adr(out));return out
 def full(self,p):return self.match(self.encode(p['x']),self.soa(self.encode(p['y'])))
