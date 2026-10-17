"""Byte pages for four lossless typed nonzero-column streams, never object graphs."""
from .soa_sparse_columns import CHANNELS,sparse_column_source,sparse_column_signatures


def paged_column_source(page_bytes):
    head=r'''
__device__ inline long long soa_page_address(long long logical,long long slot,
 const long long* table,const long long* owner,const long long* lengths,int pages,int per_slot,int* errors,int row){
 if(logical<0||logical>=lengths[slot]){atomicOr(errors+row,8);return -1;}
 long long page=logical/PAGE_BYTES;
 if(page>=per_slot){atomicOr(errors+row,8);return -1;}
 long long physical=table[slot*per_slot+page];
 if(physical<0||physical>=pages||owner[physical]!=slot){atomicOr(errors+row,8);return -1;}
 return physical*PAGE_BYTES+logical%PAGE_BYTES;
}
__device__ inline int soa_page_write(unsigned char* data,unsigned long long value,int width,long long at,long long slot,
 const long long* table,const long long* owner,const long long* lengths,int pages,int per_slot,int* errors,int row){
 for(int byte=0;byte<width;byte++){long long address=soa_page_address(at+byte,slot,table,owner,lengths,pages,per_slot,errors,row);
  if(address<0)return 0;data[address]=(unsigned char)(value>>(byte*8));}return 1;
}
__device__ inline unsigned long long soa_page_read(const unsigned char* data,int width,long long at,long long slot,
 const long long* table,const long long* owner,const long long* lengths,int pages,int per_slot,int* errors,int row){
 unsigned long long value=0;
 for(int byte=0;byte<width;byte++){long long address=soa_page_address(at+byte,slot,table,owner,lengths,pages,per_slot,errors,row);
  if(address<0)return 0;value|=(unsigned long long)data[address]<<(byte*8);}return value;
}
extern "C" __global__ void validate_paged_columns(const int* c0,const int* c1,const int* c2,const int* c3,
 const long long* slots,const unsigned char* active,const long long* lengths,const long long* counts,
 const long long* table,const long long* owner,int* errors,int rows,int pages,int per_slot,
 int l0,int l1,int l2,int l3){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows||!active[row])return;
 long long slot=slots[row];int a=c0[slot],b=c1[slot],c=c2[slot],d=c3[slot];
 if(a<0||a>l0||b<0||b>l1||c<0||c>l2||d<0||d>l3){atomicOr(errors+row,8);return;}
 long long size=(long long)a*12+(long long)b*12+(long long)c*8+(long long)d*5;
 long long n=counts[slot];
 if(size==0)size=1;
 if(size!=lengths[slot]||n!=(size+PAGE_BYTES-1)/PAGE_BYTES||n<1||n>per_slot){atomicOr(errors+row,8);return;}
 for(int i=0;i<n;i++){long long physical=table[slot*per_slot+i];
  if(physical<0||physical>=pages||owner[physical]!=slot){atomicOr(errors+row,8);return;}
  for(int j=0;j<i;j++)if(table[slot*per_slot+j]==physical){atomicOr(errors+row,8);return;}}
}
'''.replace('PAGE_BYTES',str(page_bytes))
    parts=[sparse_column_source(),head]
    bases={'i64':'0LL','f64':'(long long)a[row]*12','i32':'((long long)a[row]+b[row])*12',
           'u8':'((long long)a[row]+b[row])*12+(long long)c[row]*8'}
    slot_bases={k:v.replace('[row]','[slot]') for k,v in bases.items()}
    widths={'i64':8,'f64':8,'i32':4,'u8':1}
    for key,(ctype,_) in CHANNELS.items():
        width=widths[key];stride=4+width
        present='__double_as_longlong(value)!=0LL' if key=='f64' else 'value!=0'
        bits='(unsigned long long)__double_as_longlong(value)' if key=='f64' else '(unsigned long long)value'
        decode='__longlong_as_double((long long)value)' if key=='f64' else f'({ctype})value'
        parts.append(f'''
extern "C" __global__ void pack_page_{key}(const {ctype}* source,const unsigned char* active,const long long* slots,
 const int* prefix,const int* a,const int* b,const int* c,const int* d,unsigned char* data,
 const long long* table,const long long* owner,const long long* lengths,int* errors,
 int rows,int length,int tiles,int pages,int per_slot,int limit){{
 int row=blockIdx.y,tile=blockIdx.x,col=tile*256+threadIdx.x;if(!active[row])return;
 {ctype} value=col<length?source[(long long)col*rows+row]:0;bool take=col<length&&({present});
 unsigned bits=__ballot_sync(0xffffffffU,take);__shared__ int n[8];int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(lane==0)n[warp]=__popc(bits);__syncthreads();if(!take)return;
 int ordinal=tile?prefix[row*tiles+tile-1]:0;for(int i=0;i<warp;i++)ordinal+=n[i];
 ordinal+=__popc(bits&((1U<<lane)-1U));if(ordinal>=limit){{atomicOr(errors+row,8);return;}}
 long long slot=slots[row],at={bases[key]}+(long long)ordinal*{stride};
 if(!soa_page_write(data,(unsigned int)col,4,at,slot,table,owner,lengths,pages,per_slot,errors,row))return;
 soa_page_write(data,{bits},{width},at+4,slot,table,owner,lengths,pages,per_slot,errors,row);
}}
extern "C" __global__ void gather_page_{key}(const unsigned char* data,const long long* table,const long long* owner,
 const long long* lengths,const int* a,const int* b,const int* c,const int* d,const long long* slots,
 const unsigned char* active,{ctype}* output,int* errors,int rows,int length,int pages,int per_slot,int limit){{
 int row=blockIdx.y,ordinal=blockIdx.x*256+threadIdx.x;if(!active[row]||errors[row])return;
 long long slot=slots[row];const int* channel={dict(i64='a',f64='b',i32='c',u8='d')[key]};int count=channel[slot];
 if(count<0||count>limit){{atomicOr(errors+row,8);return;}}if(ordinal>=count)return;
 long long base={slot_bases[key]},at=base+(long long)ordinal*{stride};
 int col=(int)soa_page_read(data,4,at,slot,table,owner,lengths,pages,per_slot,errors,row);
 if(col<0||col>=length){{atomicOr(errors+row,8);return;}}
 if(ordinal){{int previous=(int)soa_page_read(data,4,at-{stride},slot,table,owner,lengths,pages,per_slot,errors,row);
  if(previous>=col){{atomicOr(errors+row,8);return;}}}}
 unsigned long long value=soa_page_read(data,{width},at+4,slot,table,owner,lengths,pages,per_slot,errors,row);
 output[(long long)col*rows+row]={decode};
}}
''')
    return '\n'.join(parts)


def paged_column_signatures():
    out={k:v for k,v in sparse_column_signatures().items() if k.startswith('count_')}
    out['validate_paged_columns']=['int32*']*4+['int64*','uint8*']+['int64*']*4+['int32*']+['int32']*7
    for key,(_,pointer) in CHANNELS.items():
        out['pack_page_'+key]=[pointer,'uint8*','int64*']+['int32*']*5+['uint8*']+['int64*']*3+['int32*']+['int32']*6
        out['gather_page_'+key]=['uint8*']+['int64*']*3+['int32*']*4+['int64*','uint8*',pointer,'int32*']+['int32']*5
    return out
