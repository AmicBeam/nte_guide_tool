"""Device NumPy subset for recovery sampling and observation encoding.

Bounded payload-backed ndarray/dtype/scalar implementation. Frozen names:
numpy.zeros, numpy.asarray, numpy.concatenate, numpy.array_equal,
numpy.float32, numpy.isfinite. Tags 60-69; standard-library objects use 70-79.
Host NumPy is never used to compute device results.

Hooks in namespace duel_rule_vm (primary integrates into cuda_vm.py):

- numpy_call(c, callee, args, nargs, kw_names, kw_values, nkw, out)
- numpy_attribute(c, owner, name)
- numpy_try_subscript(c, owner, key, out)
- numpy_try_store(c, owner, key, value)
- numpy_length(c, ref, out)
- numpy_truth(c, ref, out)
- numpy_iterator(c, owner) / numpy_iterator_next(c, ref)
- numpy_numeric(c, a, b, op, out)
- numpy_gc_trace(c, ref, tail)

Call numpy_call before T_NATIVE keyword rejection. Tag 60 extra is the root
owner for views; 61/64 n[1] is the array; 62/63 have no children.
"""
from __future__ import annotations

from typing import Sequence

SUPPORTED_NUMPY_NAMES: tuple[str, ...] = (
    "numpy.zeros",
    "numpy.asarray",
    "numpy.concatenate",
    "numpy.array_equal",
    "numpy.float32",
    "numpy.isfinite",
)
SUPPORTED_NUMPY_METHODS: tuple[str, ...] = ("reshape", "all", "tolist")
TAG_NDARRAY: int = 60
TAG_NDARRAY_METHOD: int = 61
TAG_NP_DTYPE: int = 62
TAG_NP_SCALAR: int = 63
TAG_NDARRAY_ITER: int = 64
NP_DTYPE_FLOAT32: int = 1
NP_DTYPE_BOOL: int = 2
NP_OP_ADD: int = 1
NP_OP_SUB: int = 2
NP_OP_MUL: int = 3
NP_OP_DIV: int = 4

NUMPY_INTEGRATION = (
    "from .vm_numpy import emit_numpy_source, SUPPORTED_NUMPY_NAMES; "
    "SUPPORTED_INTRINSICS |= frozenset(SUPPORTED_NUMPY_NAMES); "
    "insert emit_numpy_source(image.intrinsic_names) after CONTEXT_HANDLERS; "
    "if(numpy_call(...)) before T_NATIVE nkw rejection; "
    "LOAD_ATTR: attr=numpy_attribute(...); "
    "collection_subscript/store: numpy_try_subscript/store; "
    "length/truth/iterator/numeric/gc_trace as documented in module docstring. "
    "Do not use tags 70-79. "
    "numeric_binary maps ADD/SUB/MUL/DIV onto NP_OP_ADD/SUB/MUL/DIV. Heap compact still knows tags 0-10 only."
)


def emit_numpy_source(intrinsic_names: tuple[str, ...] | Sequence[str] = ()) -> str:
    names = tuple(intrinsic_names)

    def nid(name: str, sentinel: int) -> int:
        return names.index(name) if name in names else sentinel

    src = r'''
/* Numeric op codes are local to this subset. Primary maps VM BINARY/INPLACE ops. */
#ifndef NP_OP_ADD
enum { NP_OP_ADD=1,NP_OP_SUB=2,NP_OP_MUL=3,NP_OP_DIV=4 };
#endif
#ifndef T_NDARRAY
const int T_NDARRAY=60,T_NDARRAY_METHOD=61,T_NP_DTYPE=62,T_NP_SCALAR=63,T_NDARRAY_ITER=64;
#endif
#ifndef T_SLICE
#define T_SLICE 20
#endif
#ifndef T_NATIVE
#define T_NATIVE 15
#endif
const int NATIVE_NP_ZEROS=@Z@,NATIVE_NP_ASARRAY=@A@,NATIVE_NP_CONCAT=@C@,NATIVE_NP_EQUAL=@E@,NATIVE_NP_FLOAT32=@F@,NATIVE_NP_ISFINITE=@I@;
const int NP_DTYPE_F32=1,NP_DTYPE_BOOL=2,NP_WRITEABLE=1,NP_OWNDATA=2,NP_HDR=32;
const int NP_M_RESHAPE=1,NP_M_ALL=2,NP_M_TOLIST=3;
__device__ inline int numeric_int(Context* c,int ref,long long* value);
__device__ inline int numeric_double(Context* c,int ref,double* value);
__device__ inline int vm_text_equals(Context* c,int ref,const char* text);
__device__ inline void gc_ref(Context* c,int ref,int* tail);
struct NpArr{int ref,root,ndim,dtype,flags,shape0,shape1,stride0,stride1,data;};
__device__ inline int np_i32(const unsigned char* p){return (int)((unsigned int)p[0]|((unsigned int)p[1]<<8)|((unsigned int)p[2]<<16)|((unsigned int)p[3]<<24));}
__device__ inline void np_set_i32(unsigned char* p,int v){unsigned int u=(unsigned int)v;p[0]=(unsigned char)u;p[1]=(unsigned char)(u>>8);p[2]=(unsigned char)(u>>16);p[3]=(unsigned char)(u>>24);}
__device__ inline float np_load_f32(const unsigned char* p){unsigned int u=(unsigned int)p[0]|((unsigned int)p[1]<<8)|((unsigned int)p[2]<<16)|((unsigned int)p[3]<<24);float f=0;vm_memcpy(&f,&u,4);return f;}
__device__ inline void np_store_f32(unsigned char* p,float f){unsigned int u=0;vm_memcpy(&u,&f,4);p[0]=(unsigned char)u;p[1]=(unsigned char)(u>>8);p[2]=(unsigned char)(u>>16);p[3]=(unsigned char)(u>>24);}
__device__ inline unsigned int np_f32_bits(float f){unsigned int u=0;vm_memcpy(&u,&f,4);return u;}
__device__ inline float np_bits_f32(unsigned int u){float f=0;vm_memcpy(&f,&u,4);return f;}
__device__ inline int np_itemsize(int d){return d==NP_DTYPE_F32?4:1;}
__device__ inline int np_is_f32_dtype(Context* c,int ref){int t=kind(c,ref);if(t==T_NATIVE&&node(c,ref)&&node(c,ref)[3]==NATIVE_NP_FLOAT32)return 1;return t==T_NP_DTYPE&&node(c,ref)&&node(c,ref)[3]==NP_DTYPE_F32;}
__device__ inline int np_fail_type(Context* c){c->error=UNSUPPORTED_TYPE;return MISSING;}
__device__ inline int np_fail_call(Context* c){c->error=BAD_CALL;return MISSING;}
__device__ inline int np_load(Context* c,int ref,NpArr* a){
 const int* n=node(c,ref);if(!n||n[0]!=T_NDARRAY)return 0;
 int cap=ref>=0?c->h.counts[2]:c->h.constant_payload_count;
 if(n[1]<0||n[2]<NP_HDR||(long long)n[1]+NP_HDR>cap){c->error=BAD_PROGRAM;return -1;}
 const unsigned char* base=ref>=0?c->h.payload:c->h.constant_payload;const unsigned char* h=base+n[1];
 a->ref=ref;a->root=n[3];a->ndim=np_i32(h);a->dtype=np_i32(h+4);a->flags=np_i32(h+8);
 a->shape0=np_i32(h+12);a->shape1=np_i32(h+16);a->stride0=np_i32(h+20);a->stride1=np_i32(h+24);a->data=np_i32(h+28);
 if(a->ndim<1||a->ndim>2||(a->dtype!=NP_DTYPE_F32&&a->dtype!=NP_DTYPE_BOOL)||a->shape0<0||a->shape1<0){c->error=BAD_PROGRAM;return -1;}
 if(a->ndim==1)a->shape1=1;return 1;
}
__device__ inline long long np_numel(const NpArr* a){return (long long)a->shape0*(a->ndim==1?1LL:a->shape1);}
__device__ inline unsigned char* np_elem(Context* c,const NpArr* a,int i0,int i1,int wr){
 if(wr&&(a->ref<0||!(a->flags&NP_WRITEABLE))){c->h.error=ERROR_CONSTANT_WRITE;return 0;}
 int cap=a->root>=0?c->h.counts[2]:c->h.constant_payload_count;
 long long off=(long long)a->data+(long long)i0*a->stride0+(long long)i1*a->stride1;int sz=np_itemsize(a->dtype);
 if(off<0||off+sz>cap){c->error=BAD_PROGRAM;return 0;}
 unsigned char* base=a->root>=0?c->h.payload:(unsigned char*)c->h.constant_payload;return base+(int)off;
}
__device__ inline int np_get_f32(Context* c,const NpArr* a,int i0,int i1,float* o){unsigned char* p=np_elem(c,a,i0,a->ndim==1?0:i1,0);if(!p)return 0;if(a->dtype==NP_DTYPE_F32)*o=np_load_f32(p);else *o=p[0]?1.0f:0.0f;return 1;}
__device__ inline int np_put_f32(Context* c,const NpArr* a,int i0,int i1,float v){unsigned char* p=np_elem(c,a,i0,a->ndim==1?0:i1,1);if(!p)return 0;if(a->dtype==NP_DTYPE_F32)np_store_f32(p,v);else p[0]=v!=0.0f;return 1;}
__device__ inline int np_from_py_f32(Context* c,int ref,float* o){
 int t=kind(c,ref);
 if(t==TAG_BOOL){*o=node(c,ref)[3]?1.0f:0.0f;return 1;}
 if(t==TAG_INT){long long v=0;if(numeric_int(c,ref,&v)<0)return 0;*o=(float)(double)v;return 1;}
 if(t==TAG_FLOAT){double d=0;if(numeric_double(c,ref,&d)<0)return 0;*o=(float)d;return 1;}
 if(t==T_NP_SCALAR){const int* n=node(c,ref);if(!n||n[2]!=NP_DTYPE_F32){c->error=UNSUPPORTED_TYPE;return 0;}*o=np_bits_f32((unsigned int)n[3]);return 1;}
 c->error=UNSUPPORTED_TYPE;return 0;
}
__device__ inline int np_make_f32_scalar(Context* c,float v){return meta(c,T_NP_SCALAR,0,NP_DTYPE_F32,(int)np_f32_bits(v));}
__device__ inline int np_align_pad(Context* c){unsigned long long addr=(unsigned long long)(c->h.payload+c->h.counts[2]);int pad=(int)((4u-(unsigned int)(addr&3u))&3u);if((long long)c->h.counts[2]+pad>c->h.max_payload){c->h.error=ERROR_PAYLOAD_CAPACITY;return -1;}c->h.counts[2]+=pad;return c->h.counts[2];}
__device__ inline int np_alloc(Context* c,int ndim,int dtype,int s0,int s1,int* out){
 if(ndim<1||ndim>2||s0<0||s1<0||(dtype!=NP_DTYPE_F32&&dtype!=NP_DTYPE_BOOL))return np_fail_type(c);
 if(ndim==1)s1=1;long long numel=(long long)s0*s1,item=np_itemsize(dtype),bytes=numel*item;
 if(numel>2147483647LL||bytes>2147483647LL||(long long)c->h.counts[2]+NP_HDR+bytes+4>c->h.max_payload){c->h.error=ERROR_PAYLOAD_CAPACITY;return MISSING;}
 if(c->h.counts[0]>=c->h.max_nodes){c->h.error=ERROR_NODE_CAPACITY;return MISSING;}
 int header=c->h.counts[2];c->h.counts[2]+=NP_HDR;int data=np_align_pad(c);if(data<0)return MISSING;c->h.counts[2]+=(int)bytes;
 unsigned char* h=c->h.payload+header;int st1=np_itemsize(dtype),st0=ndim==1?st1:s1*st1;
 np_set_i32(h,ndim);np_set_i32(h+4,dtype);np_set_i32(h+8,NP_WRITEABLE|NP_OWNDATA);np_set_i32(h+12,s0);np_set_i32(h+16,s1);np_set_i32(h+20,st0);np_set_i32(h+24,st1);np_set_i32(h+28,data);
 for(int i=0;i<(int)bytes;i++)c->h.payload[data+i]=0;
 int owned=data+(int)bytes-header;if(owned<NP_HDR)owned=NP_HDR;
 int ref=c->h.counts[0]++;int* nd=c->h.nodes+ref*4;nd[0]=T_NDARRAY;nd[1]=header;nd[2]=owned;nd[3]=ref;*out=ref;return 1;
}
__device__ inline int np_view(Context* c,const NpArr* src,int ndim,int s0,int s1,int st0,int st1,int data,int* out){
 if((long long)c->h.counts[2]+NP_HDR>c->h.max_payload){c->h.error=ERROR_PAYLOAD_CAPACITY;return MISSING;}
 if(c->h.counts[0]>=c->h.max_nodes){c->h.error=ERROR_NODE_CAPACITY;return MISSING;}
 if(ndim==1)s1=1;int header=c->h.counts[2];c->h.counts[2]+=NP_HDR;unsigned char* h=c->h.payload+header;
 np_set_i32(h,ndim);np_set_i32(h+4,src->dtype);np_set_i32(h+8,src->flags&NP_WRITEABLE);np_set_i32(h+12,s0);np_set_i32(h+16,s1);np_set_i32(h+20,st0);np_set_i32(h+24,st1);np_set_i32(h+28,data);
 int ref=c->h.counts[0]++;int* nd=c->h.nodes+ref*4;nd[0]=T_NDARRAY;nd[1]=header;nd[2]=NP_HDR;nd[3]=src->root;*out=ref;return 1;
}
__device__ inline int np_index1(Context* c,int ref,int n,int* o){long long v=0;if(numeric_int(c,ref,&v)<0)return 0;if(v<(-2147483647-1LL)||v>2147483647LL){c->error=BAD_PROGRAM;return 0;}int i=(int)v;if(i<0)i+=n;if(i<0||i>=n){c->h.error=ERROR_INDEX_OUT_OF_BOUNDS;return 0;}*o=i;return 1;}
__device__ inline int np_slice_axis(Context* c,int sl,int n,int* start,int* stop,int* step){
 const int* sn=node(c,sl);if(!sn||sn[0]!=T_SLICE||sn[2]!=3){c->error=BAD_PROGRAM;return 0;}
 int rs=edge(c,sl,0),re=edge(c,sl,1),rp=edge(c,sl,2);long long st=1;
 if(rp!=NIL&&kind(c,rp)!=TAG_NONE){if(numeric_int(c,rp,&st)<0)return 0;if(!st){c->error=BAD_CALL;return 0;}}
 long long a=st>0?0:(n-1);if(rs!=NIL&&kind(c,rs)!=TAG_NONE){if(numeric_int(c,rs,&a)<0)return 0;if(a<0){a+=n;if(a<0)a=st>0?0:-1;}else if(a>=n)a=st>0?n:(n-1);}
 long long b=st>0?n:-1;if(re!=NIL&&kind(c,re)!=TAG_NONE){if(numeric_int(c,re,&b)<0)return 0;if(b<0){b+=n;if(b<0)b=st>0?0:-1;}else if(b>=n)b=st>0?n:(n-1);}
 if(st>n+1LL)st=n+1LL;if(st<-(n+1LL))st=-(n+1LL);*start=(int)a;*stop=(int)b;*step=(int)st;return 1;
}
__device__ inline int np_slice_count(int s,int e,int p){if(!p)return 0;int n=0;if(p>0){for(int i=s;i<e;i+=p)n++;}else{for(int i=s;i>e;i+=p)n++;}return n;}
__device__ inline int np_kw_only(Context* c,int names,int nkw,const char* a,const char* b){if(!nkw)return 1;if(names==MISSING||length(c,names)!=nkw){c->error=BAD_CALL;return 0;}for(int i=0;i<nkw;i++){int nm=edge(c,names,i),ok=0;if(a&&vm_text_equals(c,nm,a))ok=1;if(b&&vm_text_equals(c,nm,b))ok=1;if(!ok){c->error=BAD_CALL;return 0;}}return 1;}
__device__ inline int np_kw_get(Context* c,int names,const int* vals,int nkw,const char* key){if(!nkw||names==MISSING||!vals)return MISSING;for(int i=0;i<nkw;i++)if(vm_text_equals(c,edge(c,names,i),key))return vals[i];return MISSING;}
__device__ inline int np_parse_shape(Context* c,int ref,int* nd,int* s0,int* s1){
 int t=kind(c,ref);
 if(t==TAG_INT||t==TAG_BOOL){long long v=0;if(numeric_int(c,ref,&v)<0)return 0;if(v<0||v>2147483647LL){c->error=BAD_CALL;return 0;}*nd=1;*s0=(int)v;*s1=1;return 1;}
 if(t!=TAG_TUPLE&&t!=TAG_LIST){c->error=UNSUPPORTED_TYPE;return 0;}int n=length(c,ref);if(n<1||n>2){c->error=UNSUPPORTED_TYPE;return 0;}
 long long a=0,b=1;if(numeric_int(c,edge(c,ref,0),&a)<0)return 0;if(n==2&&numeric_int(c,edge(c,ref,1),&b)<0)return 0;
 if(a<0||b<0||a>2147483647LL||b>2147483647LL){c->error=BAD_CALL;return 0;}*nd=n;*s0=(int)a;*s1=n==1?1:(int)b;return 1;
}
__device__ inline int np_seq_depth(Context* c,int ref,int* rows,int* cols,int* is2){
 int t=kind(c,ref);if(t!=TAG_LIST&&t!=TAG_TUPLE){c->error=UNSUPPORTED_TYPE;return 0;}
 int n=length(c,ref);if(!n){*rows=0;*cols=0;*is2=0;return 1;}
 int first=edge(c,ref,0),ft=kind(c,first);
 if(ft==TAG_LIST||ft==TAG_TUPLE){int w=length(c,first);for(int i=1;i<n;i++){int e=edge(c,ref,i);int et=kind(c,e);if(et!=TAG_LIST&&et!=TAG_TUPLE){c->error=UNSUPPORTED_TYPE;return 0;}if(length(c,e)!=w){c->error=UNSUPPORTED_TYPE;return 0;}}*rows=n;*cols=w;*is2=1;return 1;}
 *rows=n;*cols=1;*is2=0;return 1;
}
__device__ inline int np_fill_from_seq(Context* c,int arr,int seq){
 NpArr a;if(np_load(c,arr,&a)<=0)return 0;int n0=a.shape0,n1=a.ndim==1?1:a.shape1;
 if(kind(c,seq)!=TAG_LIST&&kind(c,seq)!=TAG_TUPLE){c->error=UNSUPPORTED_TYPE;return 0;}if(length(c,seq)!=n0){c->error=BAD_CALL;return 0;}
 for(int i=0;i<n0;i++){int row=edge(c,seq,i);if(a.ndim==1){float v=0;if(!np_from_py_f32(c,row,&v)||!np_put_f32(c,&a,i,0,v))return 0;}
  else{if(kind(c,row)!=TAG_LIST&&kind(c,row)!=TAG_TUPLE){c->error=UNSUPPORTED_TYPE;return 0;}if(length(c,row)!=n1){c->error=BAD_CALL;return 0;}
   for(int j=0;j<n1;j++){float v=0;if(!np_from_py_f32(c,edge(c,row,j),&v)||!np_put_f32(c,&a,i,j,v))return 0;}}}return 1;
}
__device__ inline int np_zeros_call(Context* c,const int* args,int nargs,int kn,const int* kv,int nkw,int* out){
 if(nargs!=1||!np_kw_only(c,kn,nkw,"dtype",0))return np_fail_call(c);int dt=np_kw_get(c,kn,kv,nkw,"dtype");if(dt==MISSING||!np_is_f32_dtype(c,dt))return np_fail_type(c);
 int nd=0,s0=0,s1=1;if(!np_parse_shape(c,args[0],&nd,&s0,&s1))return MISSING;int arr=MISSING;if(np_alloc(c,nd,NP_DTYPE_F32,s0,s1,&arr)<=0)return MISSING;*out=arr;return arr;
}
__device__ inline int np_asarray_call(Context* c,const int* args,int nargs,int kn,const int* kv,int nkw,int* out){
 if(nargs<1||nargs>2||!np_kw_only(c,kn,nkw,"dtype",0))return np_fail_call(c);
 int dt=nargs==2?args[1]:np_kw_get(c,kn,kv,nkw,"dtype"),src=args[0],st=kind(c,src);
 if(st==T_NDARRAY){NpArr a;if(np_load(c,src,&a)<=0)return MISSING;if(dt==MISSING||np_is_f32_dtype(c,dt)){if(a.dtype==NP_DTYPE_F32){*out=src;return src;}return np_fail_type(c);}return np_fail_type(c);}
 if(dt==MISSING||!np_is_f32_dtype(c,dt))return np_fail_type(c);
 if(st==TAG_INT||st==TAG_BOOL||st==TAG_FLOAT||st==T_NP_SCALAR)return np_fail_type(c);
 int rows=0,cols=0,is2=0;if(!np_seq_depth(c,src,&rows,&cols,&is2))return MISSING;int arr=MISSING;if(np_alloc(c,is2?2:1,NP_DTYPE_F32,rows,is2?cols:1,&arr)<=0)return MISSING;if(!np_fill_from_seq(c,arr,src))return MISSING;*out=arr;return arr;
}
__device__ inline int np_concat_call(Context* c,const int* args,int nargs,int kn,const int* kv,int nkw,int* out){
 if(nargs<1||nargs>2||!np_kw_only(c,kn,nkw,"axis",0))return np_fail_call(c);
 int seq=args[0];if(kind(c,seq)!=TAG_TUPLE&&kind(c,seq)!=TAG_LIST)return np_fail_type(c);int count=length(c,seq);if(count<1)return np_fail_call(c);
 int axis=0,ar=nargs==2?args[1]:np_kw_get(c,kn,kv,nkw,"axis");if(ar!=MISSING){long long v=0;if(numeric_int(c,ar,&v)<0)return MISSING;axis=(int)v;}
 NpArr first;if(kind(c,edge(c,seq,0))!=T_NDARRAY||np_load(c,edge(c,seq,0),&first)<=0)return np_fail_type(c);
 if(axis<0)axis+=first.ndim;if(axis<0||axis>=first.ndim)return np_fail_type(c);
 int ndim=first.ndim,dtype=first.dtype,s0=first.shape0,s1=first.ndim==1?1:first.shape1;long long axis_sum=0;
 for(int i=0;i<count;i++){NpArr a;int item=edge(c,seq,i);if(kind(c,item)!=T_NDARRAY||np_load(c,item,&a)<=0)return np_fail_type(c);if(a.ndim!=ndim||a.dtype!=dtype)return np_fail_type(c);
  if(ndim==1){if(axis!=0)return np_fail_type(c);axis_sum+=a.shape0;}else if(axis==0){if(a.shape1!=s1)return np_fail_type(c);axis_sum+=a.shape0;}else{if(a.shape0!=s0)return np_fail_type(c);axis_sum+=a.shape1;}}
 if(axis_sum>2147483647LL){c->h.error=ERROR_PAYLOAD_CAPACITY;return MISSING;}
 int os0=ndim==1||axis==0?(int)axis_sum:s0,os1=ndim==1?1:(axis==1?(int)axis_sum:s1);int arr=MISSING;if(np_alloc(c,ndim,dtype,os0,os1,&arr)<=0)return MISSING;
 NpArr dst;if(np_load(c,arr,&dst)<=0)return MISSING;int cur=0;
 for(int i=0;i<count;i++){NpArr a;np_load(c,edge(c,seq,i),&a);
  if(ndim==1||axis==0){for(int r=0;r<a.shape0;r++)for(int col=0;col<(ndim==1?1:a.shape1);col++){float v=0;if(!np_get_f32(c,&a,r,col,&v)||!np_put_f32(c,&dst,cur+r,col,v))return MISSING;}cur+=a.shape0;}
  else{for(int r=0;r<a.shape0;r++)for(int col=0;col<a.shape1;col++){float v=0;if(!np_get_f32(c,&a,r,col,&v)||!np_put_f32(c,&dst,r,cur+col,v))return MISSING;}cur+=a.shape1;}}
 *out=arr;return arr;
}
__device__ inline int np_array_equal_call(Context* c,const int* args,int nargs,int nkw,int* out){
 if(nargs!=2||nkw)return np_fail_call(c);NpArr a,b;int la=np_load(c,args[0],&a),lb=np_load(c,args[1],&b);
 if(la<=0||lb<=0){if(c->error||c->h.error)return MISSING;*out=make_bool(&c->h,0);return *out;}
 if(a.ndim!=b.ndim||a.shape0!=b.shape0||(a.ndim==2&&a.shape1!=b.shape1)){*out=make_bool(&c->h,0);return *out;}
 int cols=a.ndim==1?1:a.shape1;for(int i=0;i<a.shape0;i++)for(int j=0;j<cols;j++){float va=0,vb=0;if(!np_get_f32(c,&a,i,j,&va)||!np_get_f32(c,&b,i,j,&vb))return MISSING;if(va!=vb){*out=make_bool(&c->h,0);return *out;}}
 *out=make_bool(&c->h,1);return *out;
}
__device__ inline int np_isfinite_f32(float v){unsigned int u=np_f32_bits(v);return ((u>>23)&255u)!=255u;}
__device__ inline int np_isfinite_call(Context* c,const int* args,int nargs,int nkw,int* out){
 if(nargs!=1||nkw)return np_fail_call(c);NpArr a;if(np_load(c,args[0],&a)<=0)return np_fail_type(c);
 int arr=MISSING;if(np_alloc(c,a.ndim,NP_DTYPE_BOOL,a.shape0,a.ndim==1?1:a.shape1,&arr)<=0)return MISSING;NpArr dst;np_load(c,arr,&dst);
 int cols=a.ndim==1?1:a.shape1;for(int i=0;i<a.shape0;i++)for(int j=0;j<cols;j++){float v=0;if(!np_get_f32(c,&a,i,j,&v)||!np_put_f32(c,&dst,i,j,np_isfinite_f32(v)?1.0f:0.0f))return MISSING;}
 *out=arr;return arr;
}
__device__ inline int np_reshape(Context* c,int owner,const int* args,int nargs,int nkw,int* out){
 if(nkw)return np_fail_call(c);NpArr a;if(np_load(c,owner,&a)<=0)return np_fail_type(c);int d0=0,d1=1,nd=1;
 if(nargs==1&&(kind(c,args[0])==TAG_TUPLE||kind(c,args[0])==TAG_LIST)){if(!np_parse_shape(c,args[0],&nd,&d0,&d1))return MISSING;}
 else{if(nargs<1||nargs>2)return np_fail_call(c);long long x=0,y=1;if(numeric_int(c,args[0],&x)<0)return MISSING;if(nargs==2){if(numeric_int(c,args[1],&y)<0)return MISSING;nd=2;d0=(int)x;d1=(int)y;}else{nd=1;d0=(int)x;d1=1;}if(x<-1||y<-1||x>2147483647LL||y>2147483647LL)return np_fail_call(c);}
 long long size=np_numel(&a),need=1;int unknown=-1,dims[2]={d0,d1};if(nd==1)dims[1]=1;
 for(int i=0;i<nd;i++){if(dims[i]==-1){if(unknown>=0)return np_fail_type(c);unknown=i;}else if(dims[i]<0)return np_fail_call(c);else{if(dims[i]&&need>size/dims[i]&&size){c->error=BAD_CALL;return MISSING;}need*=dims[i]?dims[i]:0;}}
 if(unknown>=0){if(need==0){if(size)return np_fail_type(c);dims[unknown]=0;}else if(size%need)return np_fail_type(c);else dims[unknown]=(int)(size/need);}else if(need!=size)return np_fail_type(c);
 int item=np_itemsize(a.dtype);int contig=a.ndim==1?(a.shape0<=1||a.stride0==item):(a.stride1==item&&(a.shape0<=1||a.shape1<=0||a.stride0==item*a.shape1));
 if(!contig)return np_fail_type(c);int st1=item,st0=nd==1?item:dims[1]*item;if(np_view(c,&a,nd,dims[0],nd==1?1:dims[1],st0,st1,a.data,out)<=0)return MISSING;return *out;
}
__device__ inline int np_all(Context* c,int owner,int nargs,int nkw,int* out){if(nargs||nkw)return np_fail_call(c);NpArr a;if(np_load(c,owner,&a)<=0)return np_fail_type(c);int cols=a.ndim==1?1:a.shape1;for(int i=0;i<a.shape0;i++)for(int j=0;j<cols;j++){float v=0;if(!np_get_f32(c,&a,i,j,&v))return MISSING;if(v==0.0f){*out=make_bool(&c->h,0);return *out;}}*out=make_bool(&c->h,1);return *out;}
__device__ inline int np_tolist(Context* c,int owner,int nargs,int nkw,int* out){
 if(nargs||nkw)return np_fail_call(c);NpArr a;if(np_load(c,owner,&a)<=0)return np_fail_type(c);int outer=make_list(&c->h);if(outer==MISSING)return MISSING;int cols=a.ndim==1?1:a.shape1;
 for(int i=0;i<a.shape0;i++){if(a.ndim==1){float v=0;if(!np_get_f32(c,&a,i,0,&v))return MISSING;int item=make_float64(&c->h,(double)v);if(item==MISSING)return MISSING;list_append(&c->h,outer,item);}
  else{int row=make_list(&c->h);if(row==MISSING)return MISSING;for(int j=0;j<cols;j++){float v=0;if(!np_get_f32(c,&a,i,j,&v))return MISSING;int item=make_float64(&c->h,(double)v);if(item==MISSING)return MISSING;list_append(&c->h,row,item);}list_append(&c->h,outer,row);}}
 *out=outer;return outer;
}
__device__ inline int numpy_attribute(Context* c,int owner,int name){
 int t=kind(c,owner);
 if(t==T_NDARRAY){NpArr a;if(np_load(c,owner,&a)<=0)return MISSING;
  if(vm_text_equals(c,name,"dtype"))return meta(c,T_NP_DTYPE,0,0,a.dtype);
  if(vm_text_equals(c,name,"shape")){int r0=make_int64(&c->h,a.shape0);if(a.ndim==1){int refs[1]={r0};return make_tuple(&c->h,refs,1);}int r1=make_int64(&c->h,a.shape1);int refs[2]={r0,r1};return make_tuple(&c->h,refs,2);}
  if(vm_text_equals(c,name,"ndim"))return make_int64(&c->h,a.ndim);
  if(vm_text_equals(c,name,"size"))return make_int64(&c->h,np_numel(&a));
  int code=0;if(vm_text_equals(c,name,"reshape"))code=NP_M_RESHAPE;else if(vm_text_equals(c,name,"all"))code=NP_M_ALL;else if(vm_text_equals(c,name,"tolist"))code=NP_M_TOLIST;else return MISSING;
  return meta(c,T_NDARRAY_METHOD,owner,0,code);}
 if(t==T_NP_DTYPE){int code=node(c,owner)[3];if(vm_text_equals(c,name,"name"))return make_utf8(&c->h,code==NP_DTYPE_F32?"float32":"bool",code==NP_DTYPE_F32?7:4);if(vm_text_equals(c,name,"itemsize"))return make_int64(&c->h,np_itemsize(code));return MISSING;}
 return MISSING;
}
__device__ inline int numpy_length(Context* c,int ref,int* out){NpArr a;int ok=np_load(c,ref,&a);if(ok==0)return 0;if(ok<0)return 1;*out=a.shape0;return 1;}
__device__ inline int numpy_truth(Context* c,int ref,int* out){NpArr a;int ok=np_load(c,ref,&a);if(ok==0){if(kind(c,ref)==T_NP_SCALAR){const int* n=node(c,ref);if(!n)return 0;if(n[2]==NP_DTYPE_F32){float v=np_bits_f32((unsigned int)n[3]);*out=v!=0.0f;return 1;}*out=n[3]!=0;return 1;}return 0;}if(ok<0)return 1;if(np_numel(&a)!=1){c->error=UNSUPPORTED_TYPE;return 1;}float v=0;if(!np_get_f32(c,&a,0,0,&v))return 1;*out=v!=0.0f;return 1;}
__device__ inline int numpy_iterator(Context* c,int owner){if(kind(c,owner)!=T_NDARRAY)return MISSING;return meta(c,T_NDARRAY_ITER,owner,0,0);}
__device__ inline int numpy_iterator_next(Context* c,int ref){int* it=mutable_node(c,ref);if(!it||it[0]!=T_NDARRAY_ITER){c->error=BAD_CALL;return MISSING;}NpArr a;if(np_load(c,it[1],&a)<=0)return MISSING;int index=it[2];if(index>=a.shape0)return MISSING;if(index==2147483647){c->error=STEP_LIMIT;return MISSING;}it[2]=index+1;if(a.ndim==1){float v=0;if(!np_get_f32(c,&a,index,0,&v))return MISSING;return np_make_f32_scalar(c,v);}int view=MISSING;if(np_view(c,&a,1,a.shape1,1,a.stride1,a.stride1,a.data+index*a.stride0,&view)<=0)return MISSING;return view;}
__device__ inline int numpy_gc_trace(Context* c,int ref,int* tail){const int* n=node(c,ref);if(!n)return 0;int tag=n[0];if(tag<60||tag>69)return 0;if(tag==T_NDARRAY)gc_ref(c,n[3],tail);else if(tag==T_NDARRAY_METHOD||tag==T_NDARRAY_ITER)gc_ref(c,n[1],tail);return 1;}
__device__ inline int numpy_numeric(Context* c,int a,int b,int op,int* out){if(kind(c,a)!=T_NP_SCALAR)return 0;const int* n=node(c,a);if(!n||n[2]!=NP_DTYPE_F32){c->error=UNSUPPORTED_TYPE;return 1;}float left=np_bits_f32((unsigned int)n[3]),right=0;if(!np_from_py_f32(c,b,&right))return 1;float r=0;if(op==NP_OP_ADD)r=left+right;else if(op==NP_OP_SUB)r=left-right;else if(op==NP_OP_MUL)r=left*right;else if(op==NP_OP_DIV){if(right==0.0f){c->error=BAD_CALL;return 1;}r=left/right;}else{c->error=UNSUPPORTED_OP;return 1;}*out=np_make_f32_scalar(c,r);return 1;}
__device__ inline int np_unpack_key(Context* c,const NpArr* a,int key,int* k0,int* k1,int* sl0,int* sl1,int* s0,int* e0,int* p0,int* s1,int* e1,int* p1){
 *sl0=0;*sl1=0;*k0=0;*k1=0;int t=kind(c,key);
 if(t==T_SLICE){if(a->ndim!=1){c->error=UNSUPPORTED_TYPE;return 0;}*sl0=1;return np_slice_axis(c,key,a->shape0,s0,e0,p0);}
 if(t==TAG_TUPLE){if(a->ndim!=2||length(c,key)!=2){c->error=UNSUPPORTED_TYPE;return 0;}int a0=edge(c,key,0),a1=edge(c,key,1);if(kind(c,a0)==T_SLICE){c->error=UNSUPPORTED_TYPE;return 0;}if(!np_index1(c,a0,a->shape0,k0))return 0;if(kind(c,a1)==T_SLICE){*sl1=1;return np_slice_axis(c,a1,a->shape1,s1,e1,p1);}return np_index1(c,a1,a->shape1,k1);}
 if(t==TAG_INT||t==TAG_BOOL)return np_index1(c,key,a->shape0,k0);
 c->error=UNSUPPORTED_TYPE;return 0;
}
__device__ inline int numpy_try_subscript(Context* c,int owner,int key,int* out){
 NpArr a;int ok=np_load(c,owner,&a);if(ok==0)return 0;if(ok<0)return 1;int k0=0,k1=0,sl0=0,sl1=0,s0=0,e0=0,p0=1,s1=0,e1=0,p1=1;
 if(!np_unpack_key(c,&a,key,&k0,&k1,&sl0,&sl1,&s0,&e0,&p0,&s1,&e1,&p1))return 1;
 if(sl0){if(p0!=1){c->error=UNSUPPORTED_TYPE;return 1;}int n=np_slice_count(s0,e0,p0),view=MISSING;if(np_view(c,&a,1,n,1,a.stride0,a.stride0,a.data+s0*a.stride0,&view)<=0)return 1;*out=view;return 1;}
 if(a.ndim==1){float v=0;if(!np_get_f32(c,&a,k0,0,&v))return 1;*out=np_make_f32_scalar(c,v);return 1;}
 if(sl1){if(p1!=1){c->error=UNSUPPORTED_TYPE;return 1;}int n=np_slice_count(s1,e1,p1),view=MISSING;if(np_view(c,&a,1,n,1,a.stride1,a.stride1,a.data+k0*a.stride0+s1*a.stride1,&view)<=0)return 1;*out=view;return 1;}
 if(kind(c,key)==TAG_INT||kind(c,key)==TAG_BOOL){int view=MISSING;if(np_view(c,&a,1,a.shape1,1,a.stride1,a.stride1,a.data+k0*a.stride0,&view)<=0)return 1;*out=view;return 1;}
 float v=0;if(!np_get_f32(c,&a,k0,k1,&v))return 1;*out=np_make_f32_scalar(c,v);return 1;
}
__device__ inline int numpy_try_store(Context* c,int owner,int key,int value){
 NpArr a;int ok=np_load(c,owner,&a);if(ok==0)return 0;if(ok<0)return 1;if(owner<0||!(a.flags&NP_WRITEABLE)){c->h.error=ERROR_CONSTANT_WRITE;return 1;}
 int k0=0,k1=0,sl0=0,sl1=0,s0=0,e0=0,p0=1,s1=0,e1=0,p1=1;if(!np_unpack_key(c,&a,key,&k0,&k1,&sl0,&sl1,&s0,&e0,&p0,&s1,&e1,&p1))return 1;
 if(sl0){if(a.ndim!=1||p0!=1){c->error=UNSUPPORTED_TYPE;return 1;}int n=np_slice_count(s0,e0,p0),t=kind(c,value);
  if(t==TAG_LIST||t==TAG_TUPLE){if(length(c,value)!=n){c->error=BAD_CALL;return 1;}int k=0;for(int j=s0;j<e0;j+=p0){float v=0;if(!np_from_py_f32(c,edge(c,value,k++),&v)||!np_put_f32(c,&a,j,0,v))return 1;}return 1;}
  float v=0;if(!np_from_py_f32(c,value,&v))return 1;for(int j=s0;j<e0;j+=p0)if(!np_put_f32(c,&a,j,0,v))return 1;return 1;}
 if(a.ndim==1){float v=0;if(!np_from_py_f32(c,value,&v)||!np_put_f32(c,&a,k0,0,v))return 1;return 1;}
 if(sl1){if(p1!=1){c->error=UNSUPPORTED_TYPE;return 1;}int n=np_slice_count(s1,e1,p1),t=kind(c,value);
  if(t==TAG_LIST||t==TAG_TUPLE){if(length(c,value)!=n){c->error=BAD_CALL;return 1;}int k=0;for(int j=s1;j<e1;j+=p1){float v=0;if(!np_from_py_f32(c,edge(c,value,k++),&v)||!np_put_f32(c,&a,k0,j,v))return 1;}return 1;}
  float v=0;if(!np_from_py_f32(c,value,&v))return 1;for(int j=s1;j<e1;j+=p1)if(!np_put_f32(c,&a,k0,j,v))return 1;return 1;}
 if(kind(c,key)==TAG_INT||kind(c,key)==TAG_BOOL){int t=kind(c,value);
  if(t==TAG_LIST||t==TAG_TUPLE){if(length(c,value)!=a.shape1){c->error=BAD_CALL;return 1;}for(int j=0;j<a.shape1;j++){float v=0;if(!np_from_py_f32(c,edge(c,value,j),&v)||!np_put_f32(c,&a,k0,j,v))return 1;}return 1;}
  if(t==T_NDARRAY){NpArr src;if(np_load(c,value,&src)<=0)return 1;if(src.ndim!=1||src.shape0!=a.shape1){c->error=BAD_CALL;return 1;}for(int j=0;j<a.shape1;j++){float v=0;if(!np_get_f32(c,&src,j,0,&v)||!np_put_f32(c,&a,k0,j,v))return 1;}return 1;}
  c->error=UNSUPPORTED_TYPE;return 1;}
 float v=0;if(!np_from_py_f32(c,value,&v)||!np_put_f32(c,&a,k0,k1,v))return 1;return 1;
}
__device__ inline int numpy_call(Context* c,int callee,const int* args,int nargs,int kn,const int* kv,int nkw,int* out){
 int t=kind(c,callee);*out=MISSING;
 if(t==T_NDARRAY_METHOD){int owner=node(c,callee)[1],code=node(c,callee)[3];if(code==NP_M_RESHAPE){np_reshape(c,owner,args,nargs,nkw,out);return 1;}if(code==NP_M_ALL){np_all(c,owner,nargs,nkw,out);return 1;}if(code==NP_M_TOLIST){np_tolist(c,owner,nargs,nkw,out);return 1;}c->error=UNSUPPORTED_TYPE;return 1;}
 if(t==T_NP_DTYPE){if(node(c,callee)[3]!=NP_DTYPE_F32||nargs!=1||nkw){c->error=BAD_CALL;return 1;}float v=0;if(!np_from_py_f32(c,args[0],&v))return 1;*out=np_make_f32_scalar(c,v);return 1;}
 if(t!=T_NATIVE)return 0;int id=node(c,callee)[3];
 if(id==NATIVE_NP_ZEROS){np_zeros_call(c,args,nargs,kn,kv,nkw,out);return 1;}
 if(id==NATIVE_NP_ASARRAY){np_asarray_call(c,args,nargs,kn,kv,nkw,out);return 1;}
 if(id==NATIVE_NP_CONCAT){np_concat_call(c,args,nargs,kn,kv,nkw,out);return 1;}
 if(id==NATIVE_NP_EQUAL){np_array_equal_call(c,args,nargs,nkw,out);return 1;}
 if(id==NATIVE_NP_ISFINITE){np_isfinite_call(c,args,nargs,nkw,out);return 1;}
 if(id==NATIVE_NP_FLOAT32){if(nargs!=1||nkw){c->error=BAD_CALL;return 1;}float v=0;if(!np_from_py_f32(c,args[0],&v))return 1;*out=np_make_f32_scalar(c,v);return 1;}
 return 0;
}
'''
    return (
        src.replace("@Z@", str(nid("numpy.zeros", -7001)))
        .replace("@A@", str(nid("numpy.asarray", -7002)))
        .replace("@C@", str(nid("numpy.concatenate", -7003)))
        .replace("@E@", str(nid("numpy.array_equal", -7004)))
        .replace("@F@", str(nid("numpy.float32", -7005)))
        .replace("@I@", str(nid("numpy.isfinite", -7006)))
    )
