"""2026-11-08 fair matching-speed comparison kernels (AVX2 + OpenMP, no AVX-512/VNNI).

All C++ arms share ONE fused exact-MNN loop (4 candidates per step, like the original SIMD4 kernel) (row argmin + per-thread column argmin, merged),
smallest index on ties, so only the distance computation and its representation differ.

  float_cpp_fused : float32 64D, distance = -dot (cosine on unit vectors), FMA
  byte_L1         : 1 byte per dim bin index (64 B), sum |a-b|      (same ranking as thermometer XOR/popcount)
  byte_L2         : 1 byte per dim bin index (64 B), sum (a-b)^2
  int8_dot        : per-tensor symmetric int8 (64 B), distance = -sum a*b (NN-quantization style baseline)
"""
from pathlib import Path
import ctypes, subprocess, tempfile, hashlib
import numpy as np

CPP = r'''
#include <cstdint>
#include <vector>
#include <limits>
#include <immintrin.h>
#include <omp.h>

template<class T,class F,class F4>
static void fused_mnn(int n,int m,int threads,F dist,F4 dist4,int*ab,int*ba){
 if(!n||!m)return;
 const T INF=std::numeric_limits<T>::max();
 std::vector<T> cd((size_t)threads*m,INF); std::vector<int> ci((size_t)threads*m,n);
 #pragma omp parallel num_threads(threads)
 { int t=omp_get_thread_num(); T*ds=cd.data()+(size_t)t*m; int*ids=ci.data()+(size_t)t*m;
   #pragma omp for schedule(static)
   for(int i=0;i<n;i++){ T best=INF;int bj=0;int j=0;T v4[4];
     for(;j+4<=m;j+=4){ dist4(i,j,v4);
       for(int l=0;l<4;l++){T v=v4[l];int q=j+l; if(v<best){best=v;bj=q;} if(v<ds[q]){ds[q]=v;ids[q]=i;}} }
     for(;j<m;j++){ T v=dist(i,j); if(v<best){best=v;bj=j;} if(v<ds[j]){ds[j]=v;ids[j]=i;} }
     ab[i]=bj; } }
 for(int j=0;j<m;j++){T best=INF;int bi=n;
   for(int t=0;t<threads;t++){T d=cd[(size_t)t*m+j];int i=ci[(size_t)t*m+j];if(d<best||(d==best&&i<bi)){best=d;bi=i;}}
   ba[j]=bi;}
}
__attribute__((target("avx2"))) static inline int hsum_epi32(__m256i v){
 __m128i s=_mm_add_epi32(_mm256_castsi256_si128(v),_mm256_extracti128_si256(v,1));
 s=_mm_add_epi32(s,_mm_shuffle_epi32(s,0x4E)); s=_mm_add_epi32(s,_mm_shuffle_epi32(s,0xB1)); return _mm_cvtsi128_si32(s);}
__attribute__((target("avx2"))) static inline int hsum_epi64(__m256i v){
 __m128i s=_mm_add_epi64(_mm256_castsi256_si128(v),_mm256_extracti128_si256(v,1));
 return (int)(_mm_cvtsi128_si64(s)+_mm_extract_epi64(s,1));}
__attribute__((target("avx2,fma"))) static inline float hsum_ps(__m256 v){
 __m128 s=_mm_add_ps(_mm256_castps256_ps128(v),_mm256_extractf128_ps(v,1));
 s=_mm_add_ps(s,_mm_movehl_ps(s,s)); s=_mm_add_ss(s,_mm_shuffle_ps(s,s,1)); return _mm_cvtss_f32(s);}

__attribute__((target("avx2"))) static inline int d_l1(const uint8_t*a,const uint8_t*b){
 __m256i a0=_mm256_loadu_si256((const __m256i*)a),a1=_mm256_loadu_si256((const __m256i*)(a+32));
 __m256i b0=_mm256_loadu_si256((const __m256i*)b),b1=_mm256_loadu_si256((const __m256i*)(b+32));
 __m256i d0=_mm256_sub_epi8(_mm256_max_epu8(a0,b0),_mm256_min_epu8(a0,b0));
 __m256i d1=_mm256_sub_epi8(_mm256_max_epu8(a1,b1),_mm256_min_epu8(a1,b1));
 __m256i z=_mm256_setzero_si256();
 return hsum_epi64(_mm256_add_epi64(_mm256_sad_epu8(d0,z),_mm256_sad_epu8(d1,z)));}
__attribute__((target("avx2"))) static inline int d_l2(const uint8_t*a,const uint8_t*b){
 __m256i a0=_mm256_loadu_si256((const __m256i*)a),a1=_mm256_loadu_si256((const __m256i*)(a+32));
 __m256i b0=_mm256_loadu_si256((const __m256i*)b),b1=_mm256_loadu_si256((const __m256i*)(b+32));
 __m256i d0=_mm256_sub_epi8(_mm256_max_epu8(a0,b0),_mm256_min_epu8(a0,b0));
 __m256i d1=_mm256_sub_epi8(_mm256_max_epu8(a1,b1),_mm256_min_epu8(a1,b1));
 // |d| <= 15 for <=16 bins, so it is a valid signed byte for maddubs; pair sums <= 450.
 __m256i s=_mm256_add_epi16(_mm256_maddubs_epi16(d0,d0),_mm256_maddubs_epi16(d1,d1));
 return hsum_epi32(_mm256_madd_epi16(s,_mm256_set1_epi16(1)));}
__attribute__((target("avx2"))) static inline int d_dot8(const int8_t*a,const int8_t*b){
 __m256i acc=_mm256_setzero_si256();
 for(int k=0;k<64;k+=16){
  __m256i x=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a+k)));
  __m256i y=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b+k)));
  acc=_mm256_add_epi32(acc,_mm256_madd_epi16(x,y));}
 return -hsum_epi32(acc);}
__attribute__((target("avx2,fma"))) static inline float d_float(const float*a,const float*b){
 __m256 acc=_mm256_mul_ps(_mm256_loadu_ps(a),_mm256_loadu_ps(b));
 for(int k=8;k<64;k+=8)acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+k),_mm256_loadu_ps(b+k),acc);
 return -hsum_ps(acc);}

__attribute__((target("avx2"))) static inline __m256i l1_lanes(__m256i a0,__m256i a1,const uint8_t*b){
 __m256i b0=_mm256_loadu_si256((const __m256i*)b),b1=_mm256_loadu_si256((const __m256i*)(b+32)),z=_mm256_setzero_si256();
 return _mm256_add_epi64(_mm256_sad_epu8(_mm256_sub_epi8(_mm256_max_epu8(a0,b0),_mm256_min_epu8(a0,b0)),z),
                         _mm256_sad_epu8(_mm256_sub_epi8(_mm256_max_epu8(a1,b1),_mm256_min_epu8(a1,b1)),z));}
__attribute__((target("avx2"))) static inline void d4_l1(const uint8_t*a,const uint8_t*b,int*o){
 __m256i a0=_mm256_loadu_si256((const __m256i*)a),a1=_mm256_loadu_si256((const __m256i*)(a+32));
 __m256i v0=l1_lanes(a0,a1,b),v1=l1_lanes(a0,a1,b+64),v2=l1_lanes(a0,a1,b+128),v3=l1_lanes(a0,a1,b+192);
 __m256i s01=_mm256_add_epi64(_mm256_unpacklo_epi64(v0,v1),_mm256_unpackhi_epi64(v0,v1));
 __m256i s23=_mm256_add_epi64(_mm256_unpacklo_epi64(v2,v3),_mm256_unpackhi_epi64(v2,v3));
 __m256i t=_mm256_add_epi64(_mm256_permute2x128_si256(s01,s23,0x20),_mm256_permute2x128_si256(s01,s23,0x31));
 alignas(32) int64_t r[4];_mm256_store_si256((__m256i*)r,t);for(int l=0;l<4;l++)o[l]=(int)r[l];}
__attribute__((target("avx2"))) static inline __m256i l2_lanes(__m256i a0,__m256i a1,const uint8_t*b){
 __m256i b0=_mm256_loadu_si256((const __m256i*)b),b1=_mm256_loadu_si256((const __m256i*)(b+32));
 __m256i d0=_mm256_sub_epi8(_mm256_max_epu8(a0,b0),_mm256_min_epu8(a0,b0)),d1=_mm256_sub_epi8(_mm256_max_epu8(a1,b1),_mm256_min_epu8(a1,b1));
 return _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(d0,d0),_mm256_maddubs_epi16(d1,d1)),_mm256_set1_epi16(1));}
__attribute__((target("avx2"))) static inline void reduce4_epi32(__m256i v0,__m256i v1,__m256i v2,__m256i v3,int*o,int sign){
 __m256i h=_mm256_hadd_epi32(_mm256_hadd_epi32(v0,v1),_mm256_hadd_epi32(v2,v3));
 __m128i s=_mm_add_epi32(_mm256_castsi256_si128(h),_mm256_extracti128_si256(h,1));
 alignas(16) int r[4];_mm_store_si128((__m128i*)r,s);for(int l=0;l<4;l++)o[l]=sign*r[l];}
__attribute__((target("avx2"))) static inline void d4_l2(const uint8_t*a,const uint8_t*b,int*o){
 __m256i a0=_mm256_loadu_si256((const __m256i*)a),a1=_mm256_loadu_si256((const __m256i*)(a+32));
 reduce4_epi32(l2_lanes(a0,a1,b),l2_lanes(a0,a1,b+64),l2_lanes(a0,a1,b+128),l2_lanes(a0,a1,b+192),o,1);}
__attribute__((target("avx2"))) static inline __m256i dot8_lanes(const __m256i*x,const int8_t*b){
 __m256i acc=_mm256_setzero_si256();
 for(int k=0;k<4;k++)acc=_mm256_add_epi32(acc,_mm256_madd_epi16(x[k],_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b+16*k)))));
 return acc;}
__attribute__((target("avx2"))) static inline void d4_dot8(const int8_t*a,const int8_t*b,int*o){
 __m256i x[4];for(int k=0;k<4;k++)x[k]=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a+16*k)));
 reduce4_epi32(dot8_lanes(x,b),dot8_lanes(x,b+64),dot8_lanes(x,b+128),dot8_lanes(x,b+192),o,-1);}
__attribute__((target("avx2,fma"))) static inline __m256 dotf_lanes(const __m256*x,const float*b){
 __m256 acc=_mm256_mul_ps(x[0],_mm256_loadu_ps(b));for(int k=1;k<8;k++)acc=_mm256_fmadd_ps(x[k],_mm256_loadu_ps(b+8*k),acc);return acc;}
__attribute__((target("avx2,fma"))) static inline void d4_float(const float*a,const float*b,float*o){
 __m256 x[8];for(int k=0;k<8;k++)x[k]=_mm256_loadu_ps(a+8*k);
 __m256 h=_mm256_hadd_ps(_mm256_hadd_ps(dotf_lanes(x,b),dotf_lanes(x,b+64)),_mm256_hadd_ps(dotf_lanes(x,b+128),dotf_lanes(x,b+192)));
 __m128 s=_mm_add_ps(_mm256_castps256_ps128(h),_mm256_extractf128_ps(h,1));
 alignas(16) float r[4];_mm_store_ps(r,s);for(int l=0;l<4;l++)o[l]=-r[l];}

extern "C" int has_avx2(){return __builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma");}
extern "C" __attribute__((target("avx2"))) void encode_bins(const float*x,const float*et,int n,int levels,uint8_t*out){
 for(int i=0;i<n;i++)for(int d=0;d<64;d+=8){__m256 xx=_mm256_loadu_ps(x+(size_t)i*64+d);__m256i c=_mm256_setzero_si256();
  for(int k=0;k<levels-1;k++){__m256 e=_mm256_loadu_ps(et+k*64+d);c=_mm256_sub_epi32(c,_mm256_castps_si256(_mm256_cmp_ps(xx,e,_CMP_GT_OQ)));}
  alignas(32) int v[8];_mm256_store_si256((__m256i*)v,c);for(int r=0;r<8;r++)out[(size_t)i*64+d+r]=(uint8_t)v[r];}}
extern "C" __attribute__((target("avx2"))) void encode_int8(const float*x,float inv_scale,int n,int8_t*out){
 const __m256 s=_mm256_set1_ps(inv_scale),lo=_mm256_set1_ps(-127.f),hi=_mm256_set1_ps(127.f);
 for(int i=0;i<n;i++)for(int d=0;d<64;d+=8){
  __m256 v=_mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_loadu_ps(x+(size_t)i*64+d),s),lo),hi);
  alignas(32) int q[8];_mm256_store_si256((__m256i*)q,_mm256_cvtps_epi32(v));  // round half to even
  for(int r=0;r<8;r++)out[(size_t)i*64+d+r]=(int8_t)q[r];}}
extern "C" void mnn_l1(const uint8_t*a,const uint8_t*b,int n,int m,int t,int*ab,int*ba){fused_mnn<int>(n,m,t,[&](int i,int j){return d_l1(a+(size_t)i*64,b+(size_t)j*64);},[&](int i,int j,int*o){d4_l1(a+(size_t)i*64,b+(size_t)j*64,o);},ab,ba);}
extern "C" void mnn_l2(const uint8_t*a,const uint8_t*b,int n,int m,int t,int*ab,int*ba){fused_mnn<int>(n,m,t,[&](int i,int j){return d_l2(a+(size_t)i*64,b+(size_t)j*64);},[&](int i,int j,int*o){d4_l2(a+(size_t)i*64,b+(size_t)j*64,o);},ab,ba);}
extern "C" void mnn_dot8(const int8_t*a,const int8_t*b,int n,int m,int t,int*ab,int*ba){fused_mnn<int>(n,m,t,[&](int i,int j){return d_dot8(a+(size_t)i*64,b+(size_t)j*64);},[&](int i,int j,int*o){d4_dot8(a+(size_t)i*64,b+(size_t)j*64,o);},ab,ba);}
extern "C" void mnn_float(const float*a,const float*b,int n,int m,int t,int*ab,int*ba){fused_mnn<float>(n,m,t,[&](int i,int j){return d_float(a+(size_t)i*64,b+(size_t)j*64);},[&](int i,int j,float*o){d4_float(a+(size_t)i*64,b+(size_t)j*64,o);},ab,ba);}
'''

P = ctypes.c_void_p; I = ctypes.c_int


def ptr(a):
    return a.ctypes.data_as(P)


class Kernels:
    def __init__(self, threads=2):
        self.threads = threads
        folder = Path(tempfile.mkdtemp(prefix='speed_kernels_'))
        (folder / 'k.cpp').write_text(CPP)
        subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', '-fopenmp', str(folder / 'k.cpp'),
                        '-o', str(folder / 'k.so')], check=True)
        self.lib = ctypes.CDLL(str(folder / 'k.so'))
        self.lib.has_avx2.restype = I
        if not self.lib.has_avx2():
            raise RuntimeError('AVX2+FMA CPU required')
        self.lib.encode_bins.argtypes = [P, P, I, I, P]
        self.lib.encode_int8.argtypes = [P, ctypes.c_float, I, P]
        for name in ('mnn_l1', 'mnn_l2', 'mnn_dot8', 'mnn_float'):
            getattr(self.lib, name).argtypes = [P, P, I, I, I, P, P]
        for name in ('encode_bins', 'encode_int8', 'mnn_l1', 'mnn_l2', 'mnn_dot8', 'mnn_float'):
            getattr(self.lib, name).restype = None
        self.source_sha256 = hashlib.sha256(CPP.encode()).hexdigest()

    def encode_bins(self, x, et, bins):
        x = np.ascontiguousarray(x, np.float32); out = np.empty((len(x), 64), np.uint8)
        self.lib.encode_bins(ptr(x), ptr(et), len(x), bins, ptr(out)); return out

    def encode_int8(self, x, inv_scale):
        x = np.ascontiguousarray(x, np.float32); out = np.empty((len(x), 64), np.int8)
        self.lib.encode_int8(ptr(x), ctypes.c_float(inv_scale), len(x), ptr(out)); return out

    def mnn(self, kind, a, b):
        n, m = len(a), len(b)
        if not n or not m:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        ab = np.empty(n, np.int32); ba = np.empty(m, np.int32)
        getattr(self.lib, 'mnn_' + kind)(ptr(a), ptr(b), n, m, self.threads, ptr(ab), ptr(ba))
        i = np.flatnonzero(ba[ab] == np.arange(n)); return i, ab[i].astype(np.int64)


# ---------- NumPy references (exact integer arithmetic, used for verification) ----------
def ref_mnn_min(D):
    r = D.argmin(1); c = D.argmin(0); i = np.flatnonzero(c[r] == np.arange(len(r))); return i, r[i]


def ref_codes(x, edges):  # edges: (64, bins-1); same rule as the original codes(): count of x > edge
    return (x[:, :, None] > edges[None]).sum(2).astype(np.int64)


def ref_int8(x, inv_scale):
    return np.clip(np.rint((x.astype(np.float32) * np.float32(inv_scale))), -127, 127).astype(np.int64)


def ref_dist(kind, a, b):
    a = a.astype(np.int64); b = b.astype(np.int64)
    if kind == 'l1':
        return np.abs(a[:, None] - b[None]).sum(2)
    if kind == 'l2':
        return ((a[:, None] - b[None]) ** 2).sum(2)
    if kind == 'dot8':
        return -(a @ b.T)
    raise ValueError(kind)
