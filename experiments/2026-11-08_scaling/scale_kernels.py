"""2026-11-08 scaling kernels: exact 1-NN search (query rows vs database), AVX2 + OpenMP.

Every integer arm uses the same cache-blocked loop: 64 queries x 4096 database entries per tile,
4 candidates per step, strict '<' so the smallest database index wins ties.

  byte_l1   : 64 B/desc, 1 byte bin index per dim, sum |a-b|
  byte_l2   : 64 B/desc, 1 byte bin index per dim, sum (a-b)^2     (levels <= 16)
  nib4_l2   : 32 B/desc, 4-bit bin index packed 2 dims/byte, sum (a-b)^2 (levels <= 16)
  dot8      : 64 B/desc, int8, distance = -sum a*b
  thermo    : (L-1)*8 B/desc, thermometer bit-planes, database in plane-major (SoA) layout,
              distance = sum popcount(a XOR b)  (same nibble-LUT popcount as the 2026-10-14 SIMD4 kernel)
"""
from pathlib import Path
import ctypes, subprocess, tempfile, hashlib
import numpy as np

CPP = r'''
#include <cstdint>
#include <vector>
#include <limits>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

template<class F,class F4>
static void nn_rows(int n,int m,int threads,F dist,F4 dist4,int*out){
 const int QB=64,DB=4096; const int INF=std::numeric_limits<int>::max();
 #pragma omp parallel for num_threads(threads) schedule(dynamic,1)
 for(int q0=0;q0<n;q0+=QB){
  int q1=std::min(n,q0+QB); int best[QB],bj[QB];
  for(int i=q0;i<q1;i++){best[i-q0]=INF;bj[i-q0]=0;}
  for(int j0=0;j0<m;j0+=DB){ int j1=std::min(m,j0+DB);
   for(int i=q0;i<q1;i++){ int b=best[i-q0],bb=bj[i-q0],j=j0; int v4[4];
    for(;j+4<=j1;j+=4){dist4(i,j,v4); for(int l=0;l<4;l++) if(v4[l]<b){b=v4[l];bb=j+l;}}
    for(;j<j1;j++){int v=dist(i,j); if(v<b){b=v;bb=j;}}
    best[i-q0]=b;bj[i-q0]=bb; } }
  for(int i=q0;i<q1;i++)out[i]=bj[i-q0];
 }
}
__attribute__((target("avx2"))) static inline int hsum_epi32(__m256i v){
 __m128i s=_mm_add_epi32(_mm256_castsi256_si128(v),_mm256_extracti128_si256(v,1));
 s=_mm_add_epi32(s,_mm_shuffle_epi32(s,0x4E)); s=_mm_add_epi32(s,_mm_shuffle_epi32(s,0xB1)); return _mm_cvtsi128_si32(s);}
__attribute__((target("avx2"))) static inline void reduce4_epi32(__m256i v0,__m256i v1,__m256i v2,__m256i v3,int*o,int sign){
 __m256i h=_mm256_hadd_epi32(_mm256_hadd_epi32(v0,v1),_mm256_hadd_epi32(v2,v3));
 __m128i s=_mm_add_epi32(_mm256_castsi256_si128(h),_mm256_extracti128_si256(h,1));
 alignas(16) int r[4];_mm_store_si128((__m128i*)r,s);for(int l=0;l<4;l++)o[l]=sign*r[l];}
__attribute__((target("avx2"))) static inline __m256i absd(__m256i x,__m256i y){return _mm256_sub_epi8(_mm256_max_epu8(x,y),_mm256_min_epu8(x,y));}

/* byte L1 */
__attribute__((target("avx2"))) static inline __m256i l1_lanes(__m256i a0,__m256i a1,const uint8_t*b){
 __m256i z=_mm256_setzero_si256();
 return _mm256_add_epi64(_mm256_sad_epu8(absd(a0,_mm256_loadu_si256((const __m256i*)b)),z),
                         _mm256_sad_epu8(absd(a1,_mm256_loadu_si256((const __m256i*)(b+32))),z));}
__attribute__((target("avx2"))) static inline int d_l1(const uint8_t*a,const uint8_t*b){
 __m256i v=l1_lanes(_mm256_loadu_si256((const __m256i*)a),_mm256_loadu_si256((const __m256i*)(a+32)),b);
 __m128i s=_mm_add_epi64(_mm256_castsi256_si128(v),_mm256_extracti128_si256(v,1)); return (int)(_mm_cvtsi128_si64(s)+_mm_extract_epi64(s,1));}
__attribute__((target("avx2"))) static inline void d4_l1(const uint8_t*a,const uint8_t*b,int*o){
 __m256i a0=_mm256_loadu_si256((const __m256i*)a),a1=_mm256_loadu_si256((const __m256i*)(a+32));
 __m256i v0=l1_lanes(a0,a1,b),v1=l1_lanes(a0,a1,b+64),v2=l1_lanes(a0,a1,b+128),v3=l1_lanes(a0,a1,b+192);
 __m256i s01=_mm256_add_epi64(_mm256_unpacklo_epi64(v0,v1),_mm256_unpackhi_epi64(v0,v1));
 __m256i s23=_mm256_add_epi64(_mm256_unpacklo_epi64(v2,v3),_mm256_unpackhi_epi64(v2,v3));
 __m256i t=_mm256_add_epi64(_mm256_permute2x128_si256(s01,s23,0x20),_mm256_permute2x128_si256(s01,s23,0x31));
 alignas(32) int64_t r[4];_mm256_store_si256((__m256i*)r,t);for(int l=0;l<4;l++)o[l]=(int)r[l];}

/* byte L2 (|d|<=15) */
__attribute__((target("avx2"))) static inline __m256i l2_lanes(__m256i a0,__m256i a1,const uint8_t*b){
 __m256i d0=absd(a0,_mm256_loadu_si256((const __m256i*)b)),d1=absd(a1,_mm256_loadu_si256((const __m256i*)(b+32)));
 return _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(d0,d0),_mm256_maddubs_epi16(d1,d1)),_mm256_set1_epi16(1));}
__attribute__((target("avx2"))) static inline int d_l2(const uint8_t*a,const uint8_t*b){
 return hsum_epi32(l2_lanes(_mm256_loadu_si256((const __m256i*)a),_mm256_loadu_si256((const __m256i*)(a+32)),b));}
__attribute__((target("avx2"))) static inline void d4_l2(const uint8_t*a,const uint8_t*b,int*o){
 __m256i a0=_mm256_loadu_si256((const __m256i*)a),a1=_mm256_loadu_si256((const __m256i*)(a+32));
 reduce4_epi32(l2_lanes(a0,a1,b),l2_lanes(a0,a1,b+64),l2_lanes(a0,a1,b+128),l2_lanes(a0,a1,b+192),o,1);}

/* packed 4-bit L2: byte k = dim 2k (low nibble) | dim 2k+1 (high nibble) */
__attribute__((target("avx2"))) static inline __m256i n4_lanes(__m256i alo,__m256i ahi,const uint8_t*b){
 __m256i x=_mm256_loadu_si256((const __m256i*)b),mk=_mm256_set1_epi8(15);
 __m256i dl=absd(alo,_mm256_and_si256(x,mk)),dh=absd(ahi,_mm256_and_si256(_mm256_srli_epi16(x,4),mk));
 return _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(dl,dl),_mm256_maddubs_epi16(dh,dh)),_mm256_set1_epi16(1));}
__attribute__((target("avx2"))) static inline int d_n4(const uint8_t*a,const uint8_t*b){
 __m256i x=_mm256_loadu_si256((const __m256i*)a),mk=_mm256_set1_epi8(15);
 return hsum_epi32(n4_lanes(_mm256_and_si256(x,mk),_mm256_and_si256(_mm256_srli_epi16(x,4),mk),b));}
__attribute__((target("avx2"))) static inline void d4_n4(const uint8_t*a,const uint8_t*b,int*o){
 __m256i x=_mm256_loadu_si256((const __m256i*)a),mk=_mm256_set1_epi8(15);
 __m256i lo=_mm256_and_si256(x,mk),hi=_mm256_and_si256(_mm256_srli_epi16(x,4),mk);
 reduce4_epi32(n4_lanes(lo,hi,b),n4_lanes(lo,hi,b+32),n4_lanes(lo,hi,b+64),n4_lanes(lo,hi,b+96),o,1);}

/* int8 dot */
__attribute__((target("avx2"))) static inline __m256i dot8_lanes(const __m256i*x,const int8_t*b){
 __m256i acc=_mm256_setzero_si256();
 for(int k=0;k<4;k++)acc=_mm256_add_epi32(acc,_mm256_madd_epi16(x[k],_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b+16*k)))));
 return acc;}
__attribute__((target("avx2"))) static inline int d_dot8(const int8_t*a,const int8_t*b){
 __m256i x[4];for(int k=0;k<4;k++)x[k]=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a+16*k)));
 return -hsum_epi32(dot8_lanes(x,b));}
__attribute__((target("avx2"))) static inline void d4_dot8(const int8_t*a,const int8_t*b,int*o){
 __m256i x[4];for(int k=0;k<4;k++)x[k]=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a+16*k)));
 reduce4_epi32(dot8_lanes(x,b),dot8_lanes(x,b+64),dot8_lanes(x,b+128),dot8_lanes(x,b+192),o,-1);}

/* thermometer: query AoS (P words), database SoA bs[k*m+j]; popcount via nibble LUT as in the original */
__attribute__((target("avx2"))) static inline __m256i counts4(__m256i v){
 const __m256i lut=_mm256_setr_epi8(0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4);
 __m256i mask=_mm256_set1_epi8(15);
 __m256i lo=_mm256_shuffle_epi8(lut,_mm256_and_si256(v,mask)),hi=_mm256_shuffle_epi8(lut,_mm256_and_si256(_mm256_srli_epi16(v,4),mask));
 return _mm256_sad_epu8(_mm256_add_epi8(lo,hi),_mm256_setzero_si256());}

extern "C" int has_avx2(){return __builtin_cpu_supports("avx2");}
extern "C" void nn_l1(const uint8_t*a,const uint8_t*b,int n,int m,int t,int*o){nn_rows(n,m,t,[&](int i,int j){return d_l1(a+(size_t)i*64,b+(size_t)j*64);},[&](int i,int j,int*v){d4_l1(a+(size_t)i*64,b+(size_t)j*64,v);},o);}
extern "C" void nn_l2(const uint8_t*a,const uint8_t*b,int n,int m,int t,int*o){nn_rows(n,m,t,[&](int i,int j){return d_l2(a+(size_t)i*64,b+(size_t)j*64);},[&](int i,int j,int*v){d4_l2(a+(size_t)i*64,b+(size_t)j*64,v);},o);}
extern "C" void nn_n4(const uint8_t*a,const uint8_t*b,int n,int m,int t,int*o){nn_rows(n,m,t,[&](int i,int j){return d_n4(a+(size_t)i*32,b+(size_t)j*32);},[&](int i,int j,int*v){d4_n4(a+(size_t)i*32,b+(size_t)j*32,v);},o);}
extern "C" void nn_dot8(const int8_t*a,const int8_t*b,int n,int m,int t,int*o){nn_rows(n,m,t,[&](int i,int j){return d_dot8(a+(size_t)i*64,b+(size_t)j*64);},[&](int i,int j,int*v){d4_dot8(a+(size_t)i*64,b+(size_t)j*64,v);},o);}
__attribute__((target("avx2"))) static inline void d4_thermo(const uint64_t*a,const uint64_t*bs,int m,int P,int j,int*v){
 __m256i sum=_mm256_setzero_si256();
 for(int k=0;k<P;k++){__m256i av=_mm256_set1_epi64x(a[k]);
  sum=_mm256_add_epi64(sum,counts4(_mm256_xor_si256(av,_mm256_loadu_si256((const __m256i*)(bs+(size_t)k*m+j)))));}
 alignas(32) int64_t r[4];_mm256_store_si256((__m256i*)r,sum);for(int l=0;l<4;l++)v[l]=(int)r[l];}
extern "C" void nn_thermo(const uint64_t*a,const uint64_t*bs,int n,int m,int P,int t,int*o){
 nn_rows(n,m,t,
  [&](int i,int j){int d=0;for(int k=0;k<P;k++)d+=__builtin_popcountll(a[(size_t)i*P+k]^bs[(size_t)k*m+j]);return d;},
  [&](int i,int j,int*v){d4_thermo(a+(size_t)i*P,bs,m,P,j,v);},o);}
'''

P = ctypes.c_void_p; I = ctypes.c_int


def ptr(a):
    return a.ctypes.data_as(P)


class ScaleKernels:
    def __init__(self, threads=2):
        self.threads = threads
        folder = Path(tempfile.mkdtemp(prefix='scale_kernels_'))
        (folder / 'k.cpp').write_text(CPP)
        subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-fopenmp', str(folder / 'k.cpp'),
                        '-o', str(folder / 'k.so')], check=True)
        self.lib = ctypes.CDLL(str(folder / 'k.so'))
        self.lib.has_avx2.restype = I
        if not self.lib.has_avx2(): raise RuntimeError('AVX2 CPU required')
        for name in ('nn_l1', 'nn_l2', 'nn_n4', 'nn_dot8'):
            getattr(self.lib, name).argtypes = [P, P, I, I, I, P]; getattr(self.lib, name).restype = None
        self.lib.nn_thermo.argtypes = [P, P, I, I, I, I, P]; self.lib.nn_thermo.restype = None
        self.source_sha256 = hashlib.sha256(CPP.encode()).hexdigest()

    def nn(self, kind, q, db, planes=None):
        n = len(q); m = db.shape[1] if kind == 'thermo' else len(db)
        out = np.empty(n, np.int32)
        if kind == 'thermo':
            self.lib.nn_thermo(ptr(q), ptr(db), n, m, planes, self.threads, ptr(out))
        else:
            getattr(self.lib, 'nn_' + kind)(ptr(q), ptr(db), n, m, self.threads, ptr(out))
        return out


# ---------------- encoders (NumPy; database encoding is offline and not timed) ----------------
def codes(x, edges):  # edges (64, L-1); count of x > edge
    out = np.empty(x.shape, np.uint8)
    for d in range(64): out[:, d] = np.searchsorted(edges[d], x[:, d], side='left')
    return out


def pack4(c):  # (n,64) codes < 16 -> (n,32): byte k = c[2k] | c[2k+1] << 4
    return np.ascontiguousarray((c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8))


def thermo_aos(c, planes):  # (n, planes) uint64; bit d of plane k = [code_d > k]
    w = np.zeros((len(c), planes), np.uint64)
    for k in range(planes):
        bits = (c > k).astype(np.uint64)
        w[:, k] = (bits << np.arange(64, dtype=np.uint64)[None]).sum(1, dtype=np.uint64)
    return w


def int8_codes(x, inv):
    return np.clip(np.rint(x.astype(np.float32) * np.float32(inv)), -127, 127).astype(np.int8)


def ref_nn(D):  # smallest index on ties
    return D.argmin(1).astype(np.int32)
