"""Parallel lossless zero-cell packing for authoritative typed SoA columns.

Float sparsity uses bits, preserving negative zero and every transported bit.
These kernels do not parse entities or execute rules.
"""


CHANNELS = {'i64': ('long long', 'int64*'), 'f64': ('double', 'float64*'),
            'i32': ('int', 'int32*'), 'u8': ('unsigned char', 'uint8*')}


def sparse_column_source():
    parts = []
    for key, (ctype, _) in CHANNELS.items():
        nonzero = '__double_as_longlong(value)!=0LL' if key == 'f64' else 'value!=0'
        parts.append(f'''
extern "C" __global__ void count_{key}(const {ctype}* values,const unsigned char* active,
 int* counts,int rows,int length,int tiles){{
 int row=blockIdx.y,tile=blockIdx.x,col=tile*256+threadIdx.x;
 if(!active[row]){{if(threadIdx.x==0)counts[row*tiles+tile]=0;return;}}
 {ctype} value=col<length?values[(long long)col*rows+row]:0;
 bool present=active[row] && col<length && ({nonzero});
 unsigned bits=__ballot_sync(0xffffffffU,present);
 __shared__ int n[8];int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(lane==0)n[warp]=__popc(bits);__syncthreads();
 if(threadIdx.x==0){{int total=0;for(int i=0;i<8;i++)total+=n[i];counts[row*tiles+tile]=total;}}
}}
extern "C" __global__ void pack_{key}(const {ctype}* values,const unsigned char* active,
 const long long* slots,const int* prefix,int* indices,{ctype}* packed,int* errors,
 int rows,int length,int tiles,int capacity,int limit){{
 int row=blockIdx.y,tile=blockIdx.x,col=tile*256+threadIdx.x;
 if(!active[row])return;
 {ctype} value=col<length?values[(long long)col*rows+row]:0;
 bool present=col<length && ({nonzero});
 unsigned bits=__ballot_sync(0xffffffffU,present);
 __shared__ int n[8];int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(lane==0)n[warp]=__popc(bits);__syncthreads();
 if(!present)return;
 int ordinal=tile?prefix[row*tiles+tile-1]:0;
 for(int i=0;i<warp;i++)ordinal+=n[i];
 ordinal+=__popc(bits&((1U<<lane)-1U));
 long long slot=slots[row];
 if(ordinal>=limit || slot<0 || slot>=capacity){{atomicOr(errors+row,8);return;}}
 long long at=(long long)ordinal*capacity+slot;indices[at]=col;packed[at]=value;
}}
extern "C" __global__ void gather_{key}(const int* indices,const {ctype}* values,
 const int* counts,const long long* slots,const unsigned char* active,{ctype}* output,
 int* errors,int rows,int length,int capacity,int limit){{
 int row=blockIdx.y,ordinal=blockIdx.x*256+threadIdx.x;
 if(!active[row])return;
 long long slot=slots[row];
 if(slot<0 || slot>=capacity){{atomicOr(errors+row,8);return;}}
 int count=counts[slot];
 if(count<0 || count>limit){{atomicOr(errors+row,8);return;}}
 if(ordinal>=count)return;
 long long at=(long long)ordinal*capacity+slot;int col=indices[at];
 if(col<0 || col>=length || (ordinal && indices[at-capacity]>=col)){{atomicOr(errors+row,8);return;}}
 output[(long long)col*rows+row]=values[at];
}}
''')
    return '\n'.join(parts)


def sparse_column_signatures():
    out = {}
    for key, (_, pointer) in CHANNELS.items():
        out['count_'+key] = [pointer, 'uint8*', 'int32*', 'int32', 'int32', 'int32']
        out['pack_'+key] = [pointer, 'uint8*', 'int64*', 'int32*', 'int32*', pointer,
                             'int32*', 'int32', 'int32', 'int32', 'int32', 'int32']
        out['gather_'+key] = ['int32*', pointer, 'int32*', 'int64*', 'uint8*', pointer,
                               'int32*', 'int32', 'int32', 'int32', 'int32']
    return out
