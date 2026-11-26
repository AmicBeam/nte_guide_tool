"""Decode exact frozen teacher tuples and preview scalars on CUDA.

No rules or damage arithmetic live here. The prepare tuple retains canonical
legal order and view aliases; previews come from the formal frozen VM entry.
"""
from ..batched_duel.cuda_runtime import CudaModule
from ..batched_duel.vm_heap import emit_vm_heap_source
from .device_teacher_ranking import KIND_NAMES, canonical_teacher_ranking_identity

INVALID_TEACHER = 159
_PREFIX = ['int32*','int32*','uint8*','int32*','int32*','int32*','uint8*',
           'int32','int32','int32','int32*','int32*','uint8*']


def source():
    kinds='\n'.join('if(eq(h,ref,"%s",%d))return %d;' % (name,len(name),i)
                    for i,name in enumerate(KIND_NAMES))
    return emit_vm_heap_source()+r'''
using namespace duel_vm_heap;
__device__ inline int eq(Heap* h,int ref,const char* s,int len){
 NodeView n;if(!get_node(h,ref,&n)||n.tag!=TAG_STR||n.count!=len)return 0;
 const unsigned char* p=get_payload_ptr(h,&n);return p&&vm_memcmp(p,s,len)==0;
}
__device__ inline int field(Heap* h,int ref,const char* s,int len){
 NodeView n;if(!get_node(h,ref,&n)||n.tag!=TAG_DICT){h->error=ERROR_TYPE_MISMATCH;return -1;}
 for(int i=0;i<n.count;i++){int k=0,v=0;if(!get_edge(h,&n,i,&k,&v))return -1;
  if(eq(h,k,s,len))return v;}return -1;
}
__device__ inline int at(Heap* h,const NodeView* n,int i){
 int k=0,v=0;if(!get_edge(h,n,i,&k,&v))return -1;return k;
}
__device__ inline int kind(Heap* h,int ref){
''' + kinds+r'''
 return -1;
}
extern "C" __global__ void decode_prepare(
 int* nodes,int* edges,unsigned char* payload,int* counts,
 const int* cn,const int* ce,const unsigned char* cp,int cnc,int cec,int cpc,
 const int* outputs,const int* input_errors,const unsigned char* active,
 int* legal_refs,long long* kinds,unsigned char* battle,long long* sizes,int* errors,
 int rows,int maxn,int maxe,int maxp,int width,int shared_template){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 legal_refs[row]=-1;sizes[row]=0;errors[row]=input_errors[row];
 for(int i=0;i<width;i++){kinds[row*width+i]=-1;battle[row*width+i]=0;}
 if(!active[row]||errors[row])return;
 int* c=counts+row*3;
 if(c[0]<1||c[0]>maxn||c[1]<0||c[1]>maxe||c[2]<0||c[2]>maxp){errors[row]=159;return;}
 Heap h;init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,
  payload+(long long)row*maxp,c,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 NodeView tuple,legal,entries,indices;
 if(!get_node(&h,outputs[row],&tuple)||tuple.tag!=TAG_TUPLE||tuple.count!=(shared_template?4:3)){errors[row]=159;return;}
 int view=at(&h,&tuple,0),list=at(&h,&tuple,1),jobs=at(&h,&tuple,2);
 int all=field(&h,view,"legal_actions",13);
 if(!get_node(&h,list,&legal)||legal.tag!=TAG_LIST||legal.count<1||legal.count>width
  ||!get_node(&h,all,&entries)||entries.tag!=TAG_LIST||entries.count<legal.count||entries.count>width+1
  ||!get_node(&h,jobs,&indices)||indices.tag!=TAG_LIST||indices.count>legal.count){errors[row]=159;return;}
 // Canonical view/actions are aliases. Check the entire non-concede order.
 int ordinal=0,job=0,early=0;
 for(int j=0;j<entries.count;j++){
  int entry=at(&h,&entries,j),action=field(&h,entry,"action",6);
  int code=kind(&h,field(&h,action,"type",4));
  if(code<0){errors[row]=159;return;}
  if(code==7)continue;
  if(ordinal>=legal.count||at(&h,&legal,ordinal)!=action){errors[row]=159;return;}
  kinds[row*width+ordinal]=code;early|=code==0||code==5;
  if(job<indices.count){long long index=-1;
   if(read_int64(&h,at(&h,&indices,job),&index)||index<j||index>=entries.count){errors[row]=159;return;}
   if(index==j){battle[row*width+ordinal]=1;job++;}}
  ordinal++;
 }
 if(h.error||ordinal!=legal.count||job!=indices.count||(early&&job)){errors[row]=159;return;}
 legal_refs[row]=list;sizes[row]=legal.count;
 if(shared_template){
  int template_ref=at(&h,&tuple,3);NodeView template_node;
  if(!get_node(&h,template_ref,&template_node)||
    (indices.count?template_node.tag!=TAG_DICT:template_node.tag!=TAG_NONE)){errors[row]=159;return;}
  int pair[2]={list,template_ref};legal_refs[row]=make_tuple(&h,pair,2);
  if(h.error)errors[row]=h.error;
 }
}
extern "C" __global__ void decode_legal(
 int* nodes,int* edges,unsigned char* payload,int* counts,const int* roots,
 const int* cn,const int* ce,const unsigned char* cp,int cnc,int cec,int cpc,
 const int* input_errors,const unsigned char* active,int* refs,int* preview_refs,unsigned char* valid,
 long long* sizes,int* errors,int rows,int maxn,int maxe,int maxp,int width,int shared_template){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 sizes[row]=0;errors[row]=input_errors[row];
 for(int i=0;i<width;i++){refs[row*width+i]=preview_refs[row*width+i]=-1;valid[row*width+i]=0;}
 if(!active[row]||errors[row])return;
 int* c=counts+row*3;
 if(c[0]<1||c[0]>maxn||c[1]<0||c[1]>maxe||c[2]<0||c[2]>maxp){errors[row]=159;return;}
 Heap h;init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,
  payload+(long long)row*maxp,c,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 int list_ref=roots[row],template_ref=-1;
 if(shared_template){
  NodeView pair;if(!get_node(&h,list_ref,&pair)||pair.tag!=TAG_TUPLE||pair.count!=2){errors[row]=159;return;}
  list_ref=at(&h,&pair,0);template_ref=at(&h,&pair,1);
 }
 NodeView list;if(!get_node(&h,list_ref,&list)||list.tag!=TAG_LIST||list.count<1||list.count>width){errors[row]=159;return;}
 int action_key=-1,template_key=-1,packet_list=-1,packet_edges=-1;
 if(shared_template){
  if((long long)c[0]+list.count+4>maxn||(long long)c[1]+3LL*list.count+2>maxe||
    (long long)c[2]+14>maxp){errors[row]=159;return;}
  action_key=make_utf8(&h,"action",6);template_key=make_utf8(&h,"template",8);
  packet_list=c[0]++;packet_edges=c[1];c[1]+=list.count;
  h.nodes[packet_list*4]=TAG_LIST;h.nodes[packet_list*4+1]=packet_edges;
  h.nodes[packet_list*4+2]=list.count;h.nodes[packet_list*4+3]=0;
 }
 for(int i=0;i<list.count;i++){int ref=at(&h,&list,i);NodeView n;
  if(!get_node(&h,ref,&n)||n.tag!=TAG_DICT){errors[row]=159;return;}
  refs[row*width+i]=ref;valid[row*width+i]=1;
  preview_refs[row*width+i]=ref;
  if(shared_template){
   // Exact-sized transport records avoid dictionary growth's unused edges.
   int packet=c[0]++,offset=c[1];c[1]+=2;
   h.nodes[packet*4]=TAG_DICT;h.nodes[packet*4+1]=offset;
   h.nodes[packet*4+2]=2;h.nodes[packet*4+3]=0;
   h.edges[offset*2]=action_key;h.edges[offset*2+1]=ref;
   h.edges[(offset+1)*2]=template_key;h.edges[(offset+1)*2+1]=template_ref;
   h.edges[(packet_edges+i)*2]=packet;h.edges[(packet_edges+i)*2+1]=-1;
   preview_refs[row*width+i]=packet;
  }
 }
 if(shared_template){
  int parts[2]={template_ref,packet_list};int bundle=make_tuple(&h,parts,2);
  NodeView pair;if(!get_node(&h,roots[row],&pair)){errors[row]=159;return;}
  // Retain the original two-edge root; replace its template child with a
  // (template, packets) tuple so all transport nodes/edges are reachable.
  h.edges[(pair.offset+1)*2]=bundle;
 }
 if(h.error){errors[row]=159;return;}sizes[row]=list.count;
}
__device__ inline int score(Heap* h,int root,const char* key,int len,double* value){
 int ref=field(h,root,key,len);*value=0.;if(ref==-1)return h->error==0;
 NodeView n;if(!get_node(h,ref,&n))return 0;
 if(n.tag==TAG_NONE)return 1;
 if(n.tag==TAG_BOOL){*value=n.extra;return n.extra==0||n.extra==1;}
 if(n.tag==TAG_INT){long long x=0;if(read_int64(h,ref,&x))return 0;*value=(double)x;return 1;}
 if(n.tag!=TAG_FLOAT||read_float64(h,ref,value))return 0;
 return vm_isfinite(*value);
}
extern "C" __global__ void decode_preview(
 int* nodes,int* edges,unsigned char* payload,int* counts,
 const int* cn,const int* ce,const unsigned char* cp,int cnc,int cec,int cpc,
 const int* outputs,const int* input_errors,const unsigned char* active,
 double* attack,double* attacks,double* genesis,double* counter,int* errors,
 int rows,int maxn,int maxe,int maxp){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;
 attack[row]=attacks[row]=genesis[row]=counter[row]=0.;errors[row]=input_errors[row];
 if(!active[row]||errors[row])return;
 int* c=counts+row*3;
 if(c[0]<1||c[0]>maxn||c[1]<0||c[1]>maxe||c[2]<0||c[2]>maxp){errors[row]=159;return;}
 Heap h;init_heap(&h,nodes+(long long)row*maxn*4,edges+(long long)row*maxe*2,
  payload+(long long)row*maxp,c,maxn,maxe,maxp,cn,ce,cp,cnc,cec,cpc);
 int root=outputs[row];
 if(!score(&h,root,"attack",6,attack+row)||!score(&h,root,"\346\224\273\345\207\273\346\254\241\346\225\260",12,attacks+row)
  ||!score(&h,root,"\345\210\233\347\224\237\350\212\261\344\274\244\345\256\263",15,genesis+row)||!score(&h,root,"counter",7,counter+row)||h.error)errors[row]=159;
}
'''


class DeviceTeacherDecoder:
    def __init__(self, vm):
        self.vm=vm;self.identity=canonical_teacher_ranking_identity()
        self.module=CudaModule(source(),{
            'decode_prepare':_PREFIX+['int32*','int64*','uint8*','int64*','int32*']+['int32']*6,
            'decode_legal':['int32*','int32*','uint8*','int32*','int32*','int32*','int32*','uint8*',
                'int32','int32','int32','int32*','uint8*','int32*','int32*','uint8*','int64*','int32*']+['int32']*6,
            'decode_preview':_PREFIX+['float64*']*4+['int32*']+['int32']*4},device=str(vm.device))

    def close(self): self.module.close()

    def _constants(self):
        b=self.vm._buffers
        return [b['nodes'],b['edges'],b['payload'],len(self.vm.image.nodes),
                len(self.vm.image.edges),len(self.vm.image.payload)]

    def decode(self, name, call, errors, active, outputs, *, width=None, shared_template=False):
        g=call.arena;l=g.layout
        args=[g.nodes,g.edges,g.payload,g.counts,*self._constants(),call.outputs,
              errors.contiguous(),active.view(self.vm.torch.uint8),*outputs,
              g.rows,l.max_nodes,l.max_edges,l.max_payload_bytes]
        if width is not None: args.extend((width,int(shared_template)))
        return self.module.launch(name,args,grid=(g.rows+127)//128,block=128,capture_safe=True)

    def legal(self, graph, errors, active, outputs, width, *, shared_template=False):
        l=graph.layout
        return self.module.launch('decode_legal',[graph.nodes,graph.edges,graph.payload,
            graph.counts,graph.roots,*self._constants(),errors.contiguous(),
            active.view(self.vm.torch.uint8),*outputs,graph.rows,l.max_nodes,l.max_edges,
            l.max_payload_bytes,width,int(shared_template)],grid=(graph.rows+127)//128,block=128,capture_safe=True)
