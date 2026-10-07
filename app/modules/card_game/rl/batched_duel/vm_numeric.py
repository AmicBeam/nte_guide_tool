"""Checked device numeric/sequence operations for the rule VM.

Integer overflow fails explicitly. Float floor division is currently restricted
within int64 range, and integer true division to exact float input magnitudes.
This is a partial VM component, not a complete Python numeric implementation.
"""
NUMERIC_OPS=frozenset(('BINARY_FLOOR_DIVIDE','BINARY_TRUE_DIVIDE','BINARY_AND','INPLACE_ADD','INPLACE_SUBTRACT','INPLACE_MULTIPLY'))

def emit_numeric_source():
    return r'''
__device__ inline int numeric_int(Context* c,int ref,long long* value){
 if(kind(c,ref)==TAG_BOOL){*value=node(c,ref)[3]?1:0;return 0;}return read_int64(&c->h,ref,value);
}
__device__ inline int numeric_double(Context* c,int ref,double* value){
 if(kind(c,ref)==TAG_FLOAT)return read_float64(&c->h,ref,value);
 long long integer=0;if(numeric_int(c,ref,&integer)<0)return -1;*value=(double)integer;return 0;
}
__device__ inline int concat_strings(Context* c,const int* refs,int count){
 long long bytes=0;
 for(int i=0;i<count;i++){const int* n=node(c,refs[i]);if(!n||n[0]!=TAG_STR){c->error=UNSUPPORTED_TYPE;return MISSING;}bytes+=n[2];}
 if(bytes+c->h.counts[2]>c->h.max_payload){c->h.error=ERROR_PAYLOAD_CAPACITY;return MISSING;}
 int offset=c->h.counts[2],cursor=offset;
 for(int i=0;i<count;i++){NodeView n;if(!get_node(&c->h,refs[i],&n)){c->error=BAD_PROGRAM;return MISSING;}
  const unsigned char* p=get_payload_ptr(&c->h,&n);if(n.count&&!p){c->error=BAD_PROGRAM;return MISSING;}
  for(int j=0;j<n.count;j++)c->h.payload[cursor++]=p[j];
 }
 c->h.counts[2]=cursor;return meta(c,TAG_STR,offset,(int)bytes,0);
}
// Binary significand remainder, avoiding inaccurate a-trunc(a/b)*b.
__device__ inline double python_fmod(double a,double b){
 union Bits{double d;unsigned long long u;};Bits x,y,out;x.d=a;y.d=b;
 unsigned long long sign=x.u&0x8000000000000000ULL;x.u&=0x7FFFFFFFFFFFFFFFULL;y.u&=0x7FFFFFFFFFFFFFFFULL;
 if(x.u<y.u)return a;if(x.u==y.u){out.u=sign;return out.d;}
 int ex=(int)(x.u>>52),ey=(int)(y.u>>52);unsigned long long mx=x.u&0xFFFFFFFFFFFFFULL,my=y.u&0xFFFFFFFFFFFFFULL;
 if(ex){mx|=0x10000000000000ULL;ex-=1075;}else ex=-1074;
 if(ey){my|=0x10000000000000ULL;ey-=1075;}else ey=-1074;
 while(mx && mx<0x10000000000000ULL){mx<<=1;ex--;}
 while(my && my<0x10000000000000ULL){my<<=1;ey--;}
 for(int shift=ex-ey;shift>=0;shift--){if(mx>=my)mx-=my;if(shift)mx<<=1;}
 if(!mx){out.u=sign;return out.d;}
 while(mx<0x10000000000000ULL){mx<<=1;ey--;}
 int biased=ey+1075;
 if(biased>0)out.u=sign|((unsigned long long)biased<<52)|(mx&0xFFFFFFFFFFFFFULL);
 else out.u=sign|(mx>>(1-biased));return out.d;
}
__device__ inline int numeric_binary(Context* c,int a,int b,int op){
 int ta=kind(c,a),tb=kind(c,b);
 if(op==OP_BINARY_ADD||op==OP_INPLACE_ADD){
  if(ta==TAG_STR&&tb==TAG_STR){int refs[2]={a,b};return concat_strings(c,refs,2);}
  if((ta==TAG_LIST&&tb==TAG_LIST)||(ta==TAG_TUPLE&&tb==TAG_TUPLE)){
   int count=length(c,a),right=length(c,b);
   int out=op==OP_INPLACE_ADD&&ta==TAG_LIST?a:make_list(&c->h);
   if(out!=a)for(int i=0;i<count;i++)list_append(&c->h,out,edge(c,a,i));
   for(int i=0;i<right&&!c->h.error;i++)list_append(&c->h,out,edge(c,b,i));
   if(ta==TAG_TUPLE){int* n=mutable_node(c,out);if(n)n[0]=TAG_TUPLE;}return out;
  }
 }
 if((ta!=TAG_BOOL&&ta!=TAG_INT&&ta!=TAG_FLOAT)||(tb!=TAG_BOOL&&tb!=TAG_INT&&tb!=TAG_FLOAT)){
  c->error=UNSUPPORTED_TYPE;return MISSING;
 }
 if(ta==TAG_FLOAT||tb==TAG_FLOAT||op==OP_BINARY_TRUE_DIVIDE){
  if(op==OP_BINARY_AND){c->error=UNSUPPORTED_TYPE;return MISSING;}
  double aa=0,bb=0,out=0;if(numeric_double(c,a,&aa)<0||numeric_double(c,b,&bb)<0)return MISSING;
  if(op==OP_BINARY_ADD||op==OP_INPLACE_ADD)out=aa+bb;
  else if(op==OP_BINARY_SUBTRACT||op==OP_INPLACE_SUBTRACT)out=aa-bb;
  else if(op==OP_BINARY_MULTIPLY||op==OP_INPLACE_MULTIPLY)out=aa*bb;
  else if(op==OP_BINARY_TRUE_DIVIDE){
   if(!bb){c->error=BAD_CALL;return MISSING;}
   // General Python big-integer true division requires separate exact rounding.
   if(ta!=TAG_FLOAT&&tb!=TAG_FLOAT&&(aa>9007199254740992.0||aa< -9007199254740992.0||bb>9007199254740992.0||bb< -9007199254740992.0)){
    c->error=UNSUPPORTED_TYPE;return MISSING;
   }out=aa/bb;
  }else if(op==OP_BINARY_FLOOR_DIVIDE){
   if(!bb){c->error=BAD_CALL;return MISSING;}double quotient=aa/bb;
   if(quotient>=9223372036854775808.0||quotient< -9223372036854775808.0){c->error=UNSUPPORTED_TYPE;return MISSING;}
   double remainder=python_fmod(aa,bb),div=(aa-remainder)/bb;
   if(remainder && ((bb<0)!=(remainder<0)))div-=1.0;
   if(div){long long truncated=(long long)div;if(div<0&&(double)truncated!=div)truncated--;out=(double)truncated;if(div-out>0.5)out+=1.0;}
   else{union Bits{double d;unsigned long long u;} zero;zero.d=quotient;zero.u&=0x8000000000000000ULL;out=zero.d;}
  }else{c->error=UNSUPPORTED_OP;return MISSING;}
  return make_float64(&c->h,out);
 }
 long long aa=0,bb=0,out=0;if(numeric_int(c,a,&aa)<0||numeric_int(c,b,&bb)<0)return MISSING;
 if(op==OP_BINARY_ADD||op==OP_INPLACE_ADD){out=(long long)((unsigned long long)aa+(unsigned long long)bb);if((bb>0&&out<aa)||(bb<0&&out>aa))c->h.error=ERROR_INT_OVERFLOW;}
 else if(op==OP_BINARY_SUBTRACT||op==OP_INPLACE_SUBTRACT){out=(long long)((unsigned long long)aa-(unsigned long long)bb);if((bb<0&&out<aa)||(bb>0&&out>aa))c->h.error=ERROR_INT_OVERFLOW;}
 else if(op==OP_BINARY_MULTIPLY||op==OP_INPLACE_MULTIPLY){
  if((aa==(-9223372036854775807LL-1)&&bb==-1)||(bb==(-9223372036854775807LL-1)&&aa==-1))c->h.error=ERROR_INT_OVERFLOW;
  else{out=(long long)((unsigned long long)aa*(unsigned long long)bb);if(bb&&out/bb!=aa)c->h.error=ERROR_INT_OVERFLOW;}
 }else if(op==OP_BINARY_FLOOR_DIVIDE){
  if(!bb){c->error=BAD_CALL;return MISSING;}
  if(aa==(-9223372036854775807LL-1)&&bb==-1){c->h.error=ERROR_INT_OVERFLOW;return MISSING;}
  out=aa/bb;long long remainder=aa%bb;if(remainder&&((remainder<0)!=(bb<0)))out--;
 }else if(op==OP_BINARY_AND)out=aa&bb;
 else{c->error=UNSUPPORTED_OP;return MISSING;}
 return c->h.error?MISSING:make_int64(&c->h,out);
}
__device__ inline int reduce_add(Context* c,int a,int b){return numeric_binary(c,a,b,OP_BINARY_ADD);}
__device__ inline int vm_equal(Context* c,int left,int right){
 const int capacity=256;int a[capacity],b[capacity],top=1;a[0]=left;b[0]=right;
 while(top && !c->error&&!c->h.error){
  int aa=a[--top],bb=b[top];if(aa==bb)continue;int ta=kind(c,aa),tb=kind(c,bb);
  if(ta==27){aa=edge(c,aa,1);ta=kind(c,aa);}if(tb==27){bb=edge(c,bb,1);tb=kind(c,bb);}
  int mapping_a=ta>=TAG_DICT&&ta<=TAG_CARD_ENTITY,mapping_b=tb>=TAG_DICT&&tb<=TAG_CARD_ENTITY;
  if(mapping_a&&mapping_b){
   int count=length(c,aa);if(count!=length(c,bb))return 0;
   if(top+count>capacity){c->error=STACK_OVERFLOW;return 0;}
   for(int i=0;i<count;i++){int key=edge(c,aa,i),value=dict_get(&c->h,bb,key,MISSING);if(value==MISSING)return 0;a[top]=edge(c,aa,i,1);b[top++]=value;}
  }else if((ta==TAG_LIST||ta==TAG_TUPLE)&&(tb==TAG_LIST||tb==TAG_TUPLE)){
   if(ta!=tb)return 0;int count=length(c,aa);if(count!=length(c,bb))return 0;
   if(top+count>capacity){c->error=STACK_OVERFLOW;return 0;}
   for(int i=0;i<count;i++){a[top]=edge(c,aa,i);b[top++]=edge(c,bb,i);}
  }else if(ta>=11||tb>=11||ta==TAG_LIST||tb==TAG_LIST||ta==TAG_TUPLE||tb==TAG_TUPLE||mapping_a||mapping_b)return 0;
  else if(!key_equal(&c->h,aa,bb))return 0;
 }return !c->error&&!c->h.error;
}
'''
