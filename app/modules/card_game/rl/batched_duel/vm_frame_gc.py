"""Device reachability collection for abandoned suspended generator frames.

Heap objects are not moved. Roots include active frames and pending call inputs;
reachable suspended generators retain their frame and closure. Unsupported
metadata or protected generator finalizers fail explicitly instead of dropping
state. This is not a general Python garbage collector.
"""

def emit_frame_gc_source():
    return r'''
__device__ inline void gc_ref(Context* c,int ref,int* tail){
 if(ref<0)return;
 if(ref>=c->h.counts[0]){c->error=BAD_PROGRAM;return;}
 if(c->gc_marks[ref])return;
 if(*tail>=c->h.max_nodes){c->error=STACK_OVERFLOW;return;}
 c->gc_marks[ref]=1;c->gc_queue[(*tail)++]=ref;
}
__device__ inline void gc_frame(Context* c,int id,int* marked,int* tail){
 if(id<0||id>=VM_FRAMES){c->error=BAD_PROGRAM;return;}
 if(marked[id])return;marked[id]=1;int* f=frame(c,id);const int* fn=function(c,f[0]);if(!fn)return;
 for(int i=0;i<fn[5];i++)gc_ref(c,f[LBASE+i],tail);
 for(int i=0;i<f[2];i++)gc_ref(c,f[SBASE+i],tail);
 for(int i=0;i<f[11];i++)gc_ref(c,f[CBASE+i],tail);
 gc_ref(c,f[8],tail);gc_ref(c,f[9],tail);gc_ref(c,f[10],tail);
 if(f[5]==4||f[5]==5)gc_ref(c,f[6],tail);
 if(f[5]==1)gc_ref(c,f[7],tail);
 if(HEADER>12)gc_ref(c,f[12],tail);
 if(HEADER>13)gc_ref(c,f[13],tail);
}
__device__ inline void collect_suspended(Context* c,const int* args,int nargs,int globals,int closure,int defaults,int kw_names,const int* kw_values,int nkw,int kwdefaults){
 for(int i=0;i<c->h.counts[0];i++)c->gc_marks[i]=0;
 int marked[VM_FRAMES];for(int i=0;i<VM_FRAMES;i++)marked[i]=0;
 int head=0,tail=0;
 for(int i=0;i<VM_FRAMES;i++)if(frame(c,i)[4]==1)gc_frame(c,i,marked,&tail);
 for(int i=0;i<nargs;i++)gc_ref(c,args[i],&tail);
 for(int i=0;i<nkw;i++)gc_ref(c,kw_values[i],&tail);
 gc_ref(c,globals,&tail);gc_ref(c,closure,&tail);gc_ref(c,defaults,&tail);gc_ref(c,kw_names,&tail);gc_ref(c,kwdefaults,&tail);gc_ref(c,c->output,&tail);
 while(head<tail&&!c->error&&!c->h.error){
  int ref=c->gc_queue[head++];const int* n=node(c,ref);if(!n)break;int tag=n[0];
  if(tag<=4||tag==11||tag==12||tag==13||tag==14||tag==15||tag==30||tag==31||tag==32||tag==36||tag==40||tag==42)continue;
  if(tag==16||tag==17||tag==34||tag==41){gc_ref(c,n[1],&tail);continue;}
  if(tag==19){if(n[3]>=0)gc_frame(c,n[3],marked,&tail);continue;}
  if(tag==39){for(int i=2;i<=6;i++)if(i!=4&&i!=5)gc_ref(c,edge(c,ref,i),&tail);continue;}
  if((tag>=5&&tag<=10)||tag==18||tag==20||tag==21||tag==22||tag==23||tag==24||tag==25||tag==26||tag==27||tag==28||tag==29||tag==33||tag==37){
   for(int i=0;i<n[2];i++){gc_ref(c,edge(c,ref,i),&tail);if(tag>=7&&tag<=10)gc_ref(c,edge(c,ref,i,1),&tail);}continue;
  }
  c->error=UNSUPPORTED_TYPE;
 }
 if(c->error||c->h.error)return;
 for(int i=0;i<VM_FRAMES;i++)if(frame(c,i)[4]==2&&!marked[i]){
  int* f=frame(c,i);
  // Running a finally block during GeneratorExit needs the exception machinery.
  // Never silently drop that protected computation while it is unsupported.
  if(HEADER>12&&f[12]!=NIL){c->error=UNSUPPORTED_TYPE;return;}
  int* gen=mutable_node(c,f[10]);if(!gen||gen[0]!=T_GEN||gen[3]!=i){c->error=BAD_PROGRAM;return;}
  gen[3]=-1;f[4]=0;
 }
}
'''
