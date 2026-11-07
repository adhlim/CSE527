"""Same fixed quantile bins, ordinary packed binary indices, integer ABS MNN."""
from pathlib import Path
import ctypes,subprocess,tempfile
import numpy as np

CPP=r'''
#include <cstdint>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
extern "C" int has_avx2(){return __builtin_cpu_supports("avx2");}
extern "C" __attribute__((target("avx2"))) void encode_binary(const float*x,const float*et,int n,int levels,int bits,uint8_t*out){
int bytes=bits*8;
for(int i=0;i<n;i++){
 uint8_t codes[64];
 for(int d=0;d<64;d+=8){auto xx=_mm256_loadu_ps(x+i*64+d);auto c=_mm256_setzero_si256();
 for(int k=0;k<levels-1;k++){auto e=_mm256_loadu_ps(et+k*64+d);auto cmp=_mm256_cmp_ps(xx,e,_CMP_GT_OQ);c=_mm256_add_epi32(c,_mm256_srli_epi32(_mm256_castps_si256(cmp),31));}
 uint32_t values[8];_mm256_storeu_si256((__m256i*)values,c);for(int r=0;r<8;r++)codes[d+r]=values[r];}
 std::fill(out+i*bytes,out+(i+1)*bytes,0);
 for(int d=0;d<64;d++){int off=d*bits;out[i*bytes+off/8]|=unsigned(codes[d])<<(off%8);
 if(off%8+bits>8)out[i*bytes+off/8+1]|=unsigned(codes[d])>>(8-off%8);}
}}
extern "C" void unpack_binary(const uint8_t*x,int n,int bits,uint8_t*out){
int bytes=bits*8;for(int i=0;i<n;i++)for(int d=0;d<64;d++){int off=d*bits;unsigned v=x[i*bytes+off/8]>>(off%8);
if(off%8+bits>8)v|=unsigned(x[i*bytes+off/8+1])<<(8-off%8);out[i*64+d]=v&((1u<<bits)-1);}}
static inline unsigned distance_bytes(const uint8_t*a,const uint8_t*b){
auto sum=_mm_setzero_si128();for(int d=0;d<64;d+=16){auto x=_mm_loadu_si128((const __m128i*)(a+d)),y=_mm_loadu_si128((const __m128i*)(b+d));
sum=_mm_add_epi64(sum,_mm_sad_epu8(_mm_sub_epi8(_mm_max_epu8(x,y),_mm_min_epu8(x,y)),_mm_setzero_si128()));}
uint64_t s[2];_mm_storeu_si128((__m128i*)s,sum);return s[0]+s[1];}
static inline unsigned distance_nibbles(const uint8_t*a,const uint8_t*b){
auto sum=_mm_setzero_si128(),mask=_mm_set1_epi8(15),zero=_mm_setzero_si128();
for(int d=0;d<32;d+=16){auto x=_mm_loadu_si128((const __m128i*)(a+d)),y=_mm_loadu_si128((const __m128i*)(b+d));
auto xl=_mm_and_si128(x,mask),yl=_mm_and_si128(y,mask),xh=_mm_and_si128(_mm_srli_epi16(x,4),mask),yh=_mm_and_si128(_mm_srli_epi16(y,4),mask);
sum=_mm_add_epi64(sum,_mm_sad_epu8(_mm_sub_epi8(_mm_max_epu8(xl,yl),_mm_min_epu8(xl,yl)),zero));
sum=_mm_add_epi64(sum,_mm_sad_epu8(_mm_sub_epi8(_mm_max_epu8(xh,yh),_mm_min_epu8(xh,yh)),zero));}
uint64_t s[2];_mm_storeu_si128((__m128i*)s,sum);return s[0]+s[1];}
extern "C" void match_binary(const uint8_t*ap,const uint8_t*bp,int n,int m,int bits,int threads,int*ab,int*ba){
if(!n||!m)return;
std::vector<uint8_t>au,bu;const uint8_t*a=ap,*b=bp;int width=32;
if(bits==3){au.resize(n*64);bu.resize(m*64);unpack_binary(ap,n,bits,au.data());unpack_binary(bp,m,bits,bu.data());a=au.data();b=bu.data();width=64;}
std::vector<int> cd(threads*m,961),ci(threads*m,n);
#pragma omp parallel num_threads(threads)
{
int t=omp_get_thread_num();int*ds=cd.data()+t*m;int*ids=ci.data()+t*m;
#pragma omp for schedule(static)
for(int i=0;i<n;i++){int best=961,bj=0;
for(int j=0;j<m;j++){int v=bits==4?distance_nibbles(a+i*width,b+j*width):distance_bytes(a+i*width,b+j*width);
if(v<best){best=v;bj=j;}if(v<ds[j]||(v==ds[j]&&i<ids[j])){ds[j]=v;ids[j]=i;}}
ab[i]=bj;}}
for(int j=0;j<m;j++){int best=961,bi=n;for(int t=0;t<threads;t++){int d=cd[t*m+j],i=ci[t*m+j];if(d<best||(d==best&&i<bi)){best=d;bi=i;}}ba[j]=bi;}
}
extern "C" void distance_binary(const uint8_t*ap,const uint8_t*bp,int n,int m,int bits,uint16_t*out){
std::vector<uint8_t>au(n*64),bu(m*64);unpack_binary(ap,n,bits,au.data());unpack_binary(bp,m,bits,bu.data());
for(int i=0;i<n;i++)for(int j=0;j<m;j++)out[i*m+j]=distance_bytes(au.data()+i*64,bu.data()+j*64);
}
'''
class BinaryVariant:
 kind='binary'
 def __init__(self,bins,edges):
  assert bins in (8,16)
  self.bins=bins;self.bits=3 if bins==8 else 4;self.edges=np.ascontiguousarray(edges,np.float32);self.et=np.ascontiguousarray(self.edges.T)
  self.bytes_per_kp=self.bits*8;self.working_bytes_per_kp=64 if self.bits==3 else 32
  folder=Path(tempfile.mkdtemp(prefix='quantile_binary_'));(folder/'kernel.cpp').write_text(CPP)
  subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC','-fopenmp',str(folder/'kernel.cpp'),'-o',str(folder/'kernel.so')],check=True)
  self.lib=ctypes.CDLL(str(folder/'kernel.so'));P=ctypes.c_void_p;I=ctypes.c_int
  self.lib.has_avx2.restype=I
  if not self.lib.has_avx2():raise RuntimeError('AVX2 CPU required for encoding')
  for name,args in {'encode_binary':[P,P,I,I,I,P],'unpack_binary':[P,I,I,P],'match_binary':[P,P,I,I,I,I,P,P],'distance_binary':[P,P,I,I,I,P]}.items():
   getattr(self.lib,name).argtypes=args;getattr(self.lib,name).restype=None
 @staticmethod
 def ptr(a):return a.ctypes.data_as(ctypes.c_void_p)
 def encode(self,x):
  x=np.ascontiguousarray(x,np.float32);out=np.empty((len(x),self.bytes_per_kp),np.uint8)
  self.lib.encode_binary(self.ptr(x),self.ptr(self.et),len(x),self.bins,self.bits,self.ptr(out));return out
 def decode(self,p):
  p=np.ascontiguousarray(p,np.uint8);out=np.empty((len(p),64),np.uint8);self.lib.unpack_binary(self.ptr(p),len(p),self.bits,self.ptr(out));return out
 def match(self,a,b):
  a=np.ascontiguousarray(a,np.uint8);b=np.ascontiguousarray(b,np.uint8)
  if not len(a) or not len(b):return np.empty(0,int),np.empty(0,int)
  ab=np.empty(len(a),np.int32);ba=np.empty(len(b),np.int32)
  self.lib.match_binary(self.ptr(a),self.ptr(b),len(a),len(b),self.bits,2,self.ptr(ab),self.ptr(ba))
  i=np.flatnonzero(ba[ab]==np.arange(len(a)));return i,ab[i]
 def distance(self,a,b):
  out=np.empty((len(a),len(b)),np.uint16);self.lib.distance_binary(self.ptr(a),self.ptr(b),len(a),len(b),self.bits,self.ptr(out));return out

def verify_binary(binary,thermo):
 assert np.array_equal(binary.edges,thermo.edges)
 rng=np.random.default_rng(20261014+binary.bins)
 for n,m in [(0,3),(3,0),(1,1),(13,19),(7,9)]:
  x=rng.normal(size=(n,64)).astype(np.float32);y=rng.normal(size=(m,64)).astype(np.float32)
  for tied in [False,True]:
   if tied:x[:]=0;y[:]=0
   a,b=binary.encode(x),binary.encode(y)
   assert np.array_equal(binary.decode(a),thermo.codes(x));assert np.array_equal(binary.decode(b),thermo.codes(y))
   expected=np.abs(thermo.codes(x)[:,None,:].astype(np.int16)-thermo.codes(y)[None,:,:].astype(np.int16)).sum(-1)
   assert np.array_equal(binary.distance(a,b),expected)
   for i,j in zip(binary.match(a,b),thermo.match(thermo.encode(x),thermo.soa(thermo.encode(y)))):assert np.array_equal(i,j)
 for x in [thermo.et,np.nextafter(thermo.et,-np.inf),np.nextafter(thermo.et,np.inf)]:
  assert np.array_equal(binary.decode(binary.encode(x)),thermo.codes(x))
 print('binary representation verified',binary.bins,flush=True)
