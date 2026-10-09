"""Checked device string methods for the frozen rule VM.

The emitter performs no host string evaluation and has no fixed text buffer.
Results use the per-game heap; UTF-8 indices count Unicode codepoints. Split
requires an explicit nonempty separator; join accepts materialized list/tuple
directly; generators and simple iterators are collected by mode-8 VM continuations.
``isdigit`` supports ASCII exactly and rejects every non-ASCII input explicitly.
Default rstrip uses Python's Unicode whitespace set, including U+001C..U+001F.
Bad arity/separators use BAD_CALL, unsupported capabilities UNSUPPORTED_TYPE;
invalid references/types/UTF-8 and capacity errors use the heap error codes.
Integer parameters accept bool/int within int64, with overflow reported by the
existing numeric_int helper rather than wrapping or silently truncating.
"""


def emit_text_method_source() -> str:
    """Emit CUDA C++ after Context/node/meta/numeric_int/vm_text_equals helpers."""
    return r'''
const int T_TEXT_METHOD=45;
enum TextMethod { TEXT_SPLIT=1,TEXT_RSPLIT=2,TEXT_JOIN=3,TEXT_STARTSWITH=4,
 TEXT_ENDSWITH=5,TEXT_REPLACE=6,TEXT_ISDIGIT=7,TEXT_RSTRIP=8,TEXT_COUNT=9 };
struct TextView { const unsigned char* p;int bytes;int chars; };
__device__ inline int text_view(Context* c,int ref,TextView* out){
 if(c->error||c->h.error)return 0;
 NodeView n;if(!get_node(&c->h,ref,&n)){c->h.error=ERROR_INVALID_REF;return 0;}
 if(n.tag!=TAG_STR){c->h.error=ERROR_TYPE_MISMATCH;return 0;}
 long long end=(long long)n.offset+n.count;
 int capacity=n.is_constant?c->h.constant_payload_count:c->h.counts[2];
 if(n.offset<0||n.count<0||end>capacity){c->h.error=ERROR_INVALID_REF;return 0;}
 out->p=(n.is_constant?c->h.constant_payload:c->h.payload)+n.offset;out->bytes=n.count;
 int err=0;out->chars=utf8_char_length(out->p,out->bytes,&err);
 if(err){c->h.error=ERROR_INVALID_UTF8;return 0;}return 1;
}
// These decoding helpers only see TextViews that passed strict UTF-8 validation.
__device__ inline int text_width(unsigned char ch){return ch<128?1:ch<224?2:ch<240?3:4;}
__device__ inline unsigned int text_codepoint(const unsigned char* p){
 int width=text_width(*p);if(width==1)return *p;
 unsigned int result=*p&(width==2?31:width==3?15:7);
 for(int i=1;i<width;i++)result=(result<<6)|(p[i]&63);return result;
}
__device__ inline int text_byte_index(const TextView* text,int index){
 int byte=0;for(int i=0;i<index;i++)byte+=text_width(text->p[byte]);return byte;
}
__device__ inline int text_matches(const TextView* text,int pos,const TextView* part,int end){
 if(pos<0||pos>end||part->bytes>end-pos)return 0;
 return vm_memcmp(text->p+pos,part->p,part->bytes)==0;
}
__device__ inline int text_allocate(Context* c,long long bytes){
 if(c->error||c->h.error)return MISSING;
 if(bytes<0||bytes>(long long)c->h.max_payload-c->h.counts[2]){
  c->h.error=ERROR_PAYLOAD_CAPACITY;return MISSING;
 }
 if(c->h.counts[0]>=c->h.max_nodes){c->h.error=ERROR_NODE_CAPACITY;return MISSING;}
 int offset=c->h.counts[2];c->h.counts[2]+=(int)bytes;
 return meta(c,TAG_STR,offset,(int)bytes,0);
}
__device__ inline int text_slice(Context* c,const TextView* text,int begin,int end){
 if(begin<0||end<begin||end>text->bytes){c->h.error=ERROR_INDEX_OUT_OF_BOUNDS;return MISSING;}
 int result=text_allocate(c,(long long)end-begin);if(result==MISSING)return result;
 vm_memcpy(c->h.payload+node(c,result)[1],text->p+begin,end-begin);return result;
}
__device__ inline int text_list(Context* c,int size){
 if(c->error||c->h.error)return MISSING;
 if(size<0||(long long)size>(long long)c->h.max_edges-c->h.counts[1]){
  c->h.error=ERROR_EDGE_CAPACITY;return MISSING;
 }
 if(c->h.counts[0]>=c->h.max_nodes){c->h.error=ERROR_NODE_CAPACITY;return MISSING;}
 int offset=c->h.counts[1];c->h.counts[1]+=size;
 for(int i=0;i<size;i++){c->h.edges[((long long)offset+i)*2]=MISSING;c->h.edges[((long long)offset+i)*2+1]=-1;}
 return meta(c,TAG_LIST,offset,size,0);
}
__device__ inline void text_list_put(Context* c,int list,int index,int value){
 if(c->error||c->h.error)return;const int* n=node(c,list);
 if(!n||index<0||index>=n[2]){c->h.error=ERROR_INDEX_OUT_OF_BOUNDS;return;}
 c->h.edges[((long long)n[1]+index)*2]=value;
}
__device__ inline int text_attribute(Context* c,int owner,int name){
 if(kind(c,owner)!=TAG_STR)return MISSING;int code=0;
 if(vm_text_equals(c,name,"split"))code=TEXT_SPLIT;
 else if(vm_text_equals(c,name,"rsplit"))code=TEXT_RSPLIT;
 else if(vm_text_equals(c,name,"join"))code=TEXT_JOIN;
 else if(vm_text_equals(c,name,"startswith"))code=TEXT_STARTSWITH;
 else if(vm_text_equals(c,name,"endswith"))code=TEXT_ENDSWITH;
 else if(vm_text_equals(c,name,"replace"))code=TEXT_REPLACE;
 else if(vm_text_equals(c,name,"isdigit"))code=TEXT_ISDIGIT;
 else if(vm_text_equals(c,name,"rstrip"))code=TEXT_RSTRIP;
 else if(vm_text_equals(c,name,"count"))code=TEXT_COUNT;
 return code?meta(c,T_TEXT_METHOD,owner,0,code):MISSING;
}
__device__ inline int text_split(Context* c,int owner,int sep_ref,long long limit,int reverse){
 TextView text,sep;if(!text_view(c,owner,&text)||!text_view(c,sep_ref,&sep))return MISSING;
 if(!sep.bytes){c->error=BAD_CALL;return MISSING;}
 long long splits=0;
 if(reverse){
  for(int pos=text.bytes-sep.bytes;pos>=0;){
   if(limit>=0&&splits>=limit)break;
   if(text_matches(&text,pos,&sep,text.bytes)){splits++;pos-=sep.bytes;}else pos--;
  }
 }else{
  for(int pos=0;pos<=text.bytes-sep.bytes;){
   if(limit>=0&&splits>=limit)break;
   if(text_matches(&text,pos,&sep,text.bytes)){splits++;pos+=sep.bytes;}else pos++;
  }
 }
 if(splits>=2147483647LL){c->h.error=ERROR_EDGE_CAPACITY;return MISSING;}
 int result=text_list(c,(int)splits+1);if(result==MISSING)return result;
 if(reverse){
  int end=text.bytes,index=(int)splits;
  for(int pos=text.bytes-sep.bytes;pos>=0&&index>0;){
   if(text_matches(&text,pos,&sep,end)){
    int part=text_slice(c,&text,pos+sep.bytes,end);text_list_put(c,result,index--,part);
    if(c->error||c->h.error)return MISSING;end=pos;pos-=sep.bytes;
   }else pos--;
  }
  text_list_put(c,result,0,text_slice(c,&text,0,end));
 }else{
  int start=0,index=0;
  for(int pos=0;pos<=text.bytes-sep.bytes&&index<splits;){
   if(text_matches(&text,pos,&sep,text.bytes)){
    int part=text_slice(c,&text,start,pos);text_list_put(c,result,index++,part);
    if(c->error||c->h.error)return MISSING;pos+=sep.bytes;start=pos;
   }else pos++;
  }
  text_list_put(c,result,(int)splits,text_slice(c,&text,start,text.bytes));
 }
 return c->error||c->h.error?MISSING:result;
}
__device__ inline int text_join(Context* c,int delimiter,int sequence){
 TextView sep;if(!text_view(c,delimiter,&sep))return MISSING;
 const int* seq=node(c,sequence);if(!seq)return MISSING;
 if(seq[0]!=TAG_LIST&&seq[0]!=TAG_TUPLE){c->error=UNSUPPORTED_TYPE;return MISSING;}
 if(seq[2]<0){c->h.error=ERROR_INVALID_REF;return MISSING;}int count=seq[2];
 long long bytes=count?(long long)(count-1)*sep.bytes:0;
 if(bytes>(long long)c->h.max_payload-c->h.counts[2]){c->h.error=ERROR_PAYLOAD_CAPACITY;return MISSING;}
 for(int i=0;i<count;i++){
  TextView item;if(!text_view(c,edge(c,sequence,i),&item))return MISSING;bytes+=item.bytes;
  if(bytes>(long long)c->h.max_payload-c->h.counts[2]){c->h.error=ERROR_PAYLOAD_CAPACITY;return MISSING;}
 }
 int result=text_allocate(c,bytes);if(result==MISSING)return result;
 unsigned char* dest=c->h.payload+node(c,result)[1];int cursor=0;
 for(int i=0;i<count;i++){
  TextView item;if(!text_view(c,edge(c,sequence,i),&item))return MISSING;
  if(i){vm_memcpy(dest+cursor,sep.p,sep.bytes);cursor+=sep.bytes;}
  vm_memcpy(dest+cursor,item.p,item.bytes);cursor+=item.bytes;
 }
 return result;
}
// Retain start>length as length+1: Python's empty-prefix/count semantics need it.
__device__ inline int text_bounds(Context* c,const TextView* text,const int* args,int nargs,int* start,int* end){
 long long a=0,b=text->chars;
 if(nargs>=2&&kind(c,args[1])!=TAG_NONE&&numeric_int(c,args[1],&a)<0)return 0;
 if(nargs>=3&&kind(c,args[2])!=TAG_NONE&&numeric_int(c,args[2],&b)<0)return 0;
 if(a<0){a+=text->chars;if(a<0)a=0;}
 if(b<0){b+=text->chars;if(b<0)b=0;}
 if(a>text->chars){*start=-1;}else *start=(int)a;
 if(b>text->chars)b=text->chars;*end=(int)b;return !c->error&&!c->h.error;
}
__device__ inline int text_affix_one(Context* c,const TextView* text,int part_ref,int start,int end,int suffix){
 TextView part;if(!text_view(c,part_ref,&part))return 0;
 if(start<0||start>end)return 0;
 int a=text_byte_index(text,start),b=text_byte_index(text,end);
 return text_matches(text,suffix?b-part.bytes:a,&part,b)&&(suffix?b-part.bytes>=a:1);
}
__device__ inline int text_affix(Context* c,int owner,const int* args,int nargs,int suffix){
 TextView text;if(!text_view(c,owner,&text))return MISSING;
 int start=0,end=0;if(!text_bounds(c,&text,args,nargs,&start,&end))return MISSING;
 int part=args[0],result=0;
 if(kind(c,part)==TAG_TUPLE){
  const int* n=node(c,part);if(!n||n[2]<0){c->h.error=ERROR_INVALID_REF;return MISSING;}
  for(int i=0;i<n[2];i++){
   result=text_affix_one(c,&text,edge(c,part,i),start,end,suffix);
   if(c->error||c->h.error)return MISSING;if(result)break;
  }
 }else result=text_affix_one(c,&text,part,start,end,suffix);
 return c->error||c->h.error?MISSING:result?TRUE_REF:FALSE_REF;
}
__device__ inline int text_replace(Context* c,int owner,int old_ref,int new_ref,long long limit){
 TextView text,old,replacement;
 if(!text_view(c,owner,&text)||!text_view(c,old_ref,&old)||!text_view(c,new_ref,&replacement))return MISSING;
 long long matches=0;
 if(!old.bytes){matches=(long long)text.chars+1;if(limit>=0&&matches>limit)matches=limit;}
 else{
  for(int pos=0;pos<=text.bytes-old.bytes;){
   if(limit>=0&&matches>=limit)break;
   if(text_matches(&text,pos,&old,text.bytes)){matches++;pos+=old.bytes;}else pos++;
  }
 }
 long long bytes=(long long)text.bytes+matches*((long long)replacement.bytes-old.bytes);
 int result=text_allocate(c,bytes);if(result==MISSING)return result;
 unsigned char* dest=c->h.payload+node(c,result)[1];int cursor=0,pos=0;long long done=0;
 if(!old.bytes){
  while(pos<text.bytes){
   if(done<matches){vm_memcpy(dest+cursor,replacement.p,replacement.bytes);cursor+=replacement.bytes;done++;}
   int width=text_width(text.p[pos]);vm_memcpy(dest+cursor,text.p+pos,width);cursor+=width;pos+=width;
  }
  if(done<matches){vm_memcpy(dest+cursor,replacement.p,replacement.bytes);cursor+=replacement.bytes;}
 }else{
  while(pos<text.bytes){
   if(done<matches&&text_matches(&text,pos,&old,text.bytes)){
    vm_memcpy(dest+cursor,replacement.p,replacement.bytes);cursor+=replacement.bytes;pos+=old.bytes;done++;
   }else dest[cursor++]=text.p[pos++];
  }
 }
 if(cursor!=bytes){c->error=BAD_PROGRAM;return MISSING;}return result;
}
__device__ inline int text_whitespace(unsigned int cp){
 return (cp>=9&&cp<=13)||(cp>=28&&cp<=32)||cp==0x85||cp==0xa0||cp==0x1680||
  (cp>=0x2000&&cp<=0x200a)||cp==0x2028||cp==0x2029||cp==0x202f||cp==0x205f||cp==0x3000;
}
__device__ inline int text_rstrip(Context* c,int owner,int chars_ref){
 TextView text,chars;if(!text_view(c,owner,&text))return MISSING;
 int whitespace=chars_ref==MISSING||kind(c,chars_ref)==TAG_NONE;
 if(!whitespace&&!text_view(c,chars_ref,&chars))return MISSING;
 int end=text.bytes;
 while(end){
  int begin=end-1;while(begin>0&&(text.p[begin]&0xc0)==0x80)begin--;
  unsigned int cp=text_codepoint(text.p+begin);int strip=whitespace?text_whitespace(cp):0;
  if(!whitespace)for(int pos=0;pos<chars.bytes;pos+=text_width(chars.p[pos])){
   if(cp==text_codepoint(chars.p+pos)){strip=1;break;}
  }
  if(!strip)break;end=begin;
 }
 return text_slice(c,&text,0,end);
}
__device__ inline int text_count(Context* c,int owner,const int* args,int nargs){
 TextView text,part;if(!text_view(c,owner,&text)||!text_view(c,args[0],&part))return MISSING;
 int start=0,end=0;if(!text_bounds(c,&text,args,nargs,&start,&end))return MISSING;
 if(start<0||start>end)return make_int64(&c->h,0);
 if(!part.bytes)return make_int64(&c->h,(long long)end-start+1);
 int a=text_byte_index(&text,start),b=text_byte_index(&text,end);long long count=0;
 for(int pos=a;pos<=b-part.bytes;){
  if(text_matches(&text,pos,&part,b)){count++;pos+=part.bytes;}else pos++;
 }
 return make_int64(&c->h,count);
}
// Mode 8 keeps a normal tuple [delimiter, generator, materialized values].
__device__ inline void text_join_resume(Context* c,int generator,int caller,int accumulator,int item,int exhausted){
 int delimiter=edge(c,accumulator,0),values=edge(c,accumulator,2);
 if(exhausted){
  int result=text_join(c,delimiter,values);c->current=caller;
  if(!c->error&&!c->h.error)push(c,frame(c,caller),result);return;
 }
 // Like PySequence_Fast, consume the iterable before validating its string items.
 list_append(&c->h,values,item);if(c->error||c->h.error)return;
 resume_generator(c,generator,caller,8,accumulator,MISSING);
}
__device__ inline int text_iterable_call(Context* c,int callee,const int* args,int nargs,int caller){
 if(kind(c,callee)!=T_TEXT_METHOD)return 0;
 const int* bound=node(c,callee);if(!bound||bound[3]!=TEXT_JOIN)return 0;
 if(nargs!=1){c->error=BAD_CALL;return 1;}
 int owner=bound[1],input=args[0],tag=kind(c,input);
 if(tag==TAG_LIST||tag==TAG_TUPLE)return 0;
 int values=make_list(&c->h);if(c->h.error)return 1;
 if(tag==T_GEN){
  const int* gen=node(c,input);if(!gen)return 1;
  if(gen[3]<0){int result=text_join(c,owner,values);if(!c->error&&!c->h.error)push(c,frame(c,caller),result);return 1;}
  int refs[3]={owner,input,values},accumulator=make_tuple(&c->h,refs,3);
  if(!c->h.error)resume_generator(c,input,caller,8,accumulator,MISSING);return 1;
 }
 int it=iterator(c,input);if(c->error||c->h.error)return 1;
 if(kind(c,it)!=T_ITER&&kind(c,it)!=64){c->error=UNSUPPORTED_TYPE;return 1;}
 while(!c->error&&!c->h.error){
  if(++c->steps>c->step_limit){c->error=STEP_LIMIT;return 1;}
  if(c->deadline&&(c->steps&63)==0&&clock_ns()>=(unsigned long long)c->deadline){c->error=DEADLINE;return 1;}
  int item=iterator_next(c,it);if(item==MISSING)break;list_append(&c->h,values,item);
 }
 if(!c->error&&!c->h.error){int result=text_join(c,owner,values);if(!c->error&&!c->h.error)push(c,frame(c,caller),result);}
 return 1;
}

__device__ inline int text_call(Context* c,int callee,const int* args,int nargs,int* out){
 if(kind(c,callee)!=T_TEXT_METHOD)return 0;*out=MISSING;
 const int* bound=node(c,callee);if(!bound)return 1;int owner=bound[1],code=bound[3];
 if(nargs<0||(nargs&&!args)){c->error=BAD_CALL;return 1;}
 if(code==TEXT_SPLIT||code==TEXT_RSPLIT){
  if(nargs<1||nargs>2){c->error=BAD_CALL;return 1;}
  if(kind(c,args[0])==TAG_NONE){c->error=UNSUPPORTED_TYPE;return 1;}
  long long limit=-1;if(nargs==2&&numeric_int(c,args[1],&limit)<0)return 1;
  *out=text_split(c,owner,args[0],limit,code==TEXT_RSPLIT);
 }else if(code==TEXT_JOIN){
  if(nargs!=1)c->error=BAD_CALL;else *out=text_join(c,owner,args[0]);
 }else if(code==TEXT_STARTSWITH||code==TEXT_ENDSWITH){
  if(nargs<1||nargs>3)c->error=BAD_CALL;else *out=text_affix(c,owner,args,nargs,code==TEXT_ENDSWITH);
 }else if(code==TEXT_REPLACE){
  if(nargs<2||nargs>3){c->error=BAD_CALL;return 1;}long long limit=-1;
  if(nargs==3&&numeric_int(c,args[2],&limit)<0)return 1;
  *out=text_replace(c,owner,args[0],args[1],limit);
 }else if(code==TEXT_ISDIGIT){
  if(nargs){c->error=BAD_CALL;return 1;}TextView text;if(!text_view(c,owner,&text))return 1;
  int digits=text.bytes!=0;for(int i=0;i<text.bytes;i++){
   if(text.p[i]>=128){c->error=UNSUPPORTED_TYPE;return 1;}
   if(text.p[i]<'0'||text.p[i]>'9')digits=0;
  }*out=digits?TRUE_REF:FALSE_REF;
 }else if(code==TEXT_RSTRIP){
  if(nargs>1)c->error=BAD_CALL;else *out=text_rstrip(c,owner,nargs?args[0]:MISSING);
 }else if(code==TEXT_COUNT){
  if(nargs<1||nargs>3)c->error=BAD_CALL;else *out=text_count(c,owner,args,nargs);
 }else c->error=BAD_PROGRAM;
 return 1;
}
'''
