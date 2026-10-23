"""Anonymous physical rule previews with reusable per-root typed state storage."""
from math import prod
import numpy as np
from . import domain_soa_schema as schema
from .domain_soa_execution_base import _TYPES,_literal


def emit_source(layout):
    specs=schema.array_specs(layout)
    text='DFN int d_unknown_preview_card(DX*c,int side){DS*f=c->s;int ref=-1;for(int j=0;j<K;j++)if(!f->card_valid[j]){ref=j;break;}if(ref<0||f->next_entity[0]>=2147483647){d_fail(c,2);return -1;}'
    for name,(dtype,shape)in specs.items():
        if name.startswith('card_'):
            width=prod(shape[1:]);text+=f'for(int j=0;j<{width};j++)f->{name}[ref*{width}+j]=0;'
    text+='''f->card_valid[ref]=1;f->card_side[ref]=side;f->card_template[ref]=0;f->card_entity_slot[ref]=++f->next_entity[0];f->card_instance_slot[ref]=-f->card_entity_slot[ref];f->card_str_presence[ref*KS+KS_character_id]=2;f->card_str_slot[ref*KS+KS_character_id]=d_intern(c,"unknown");f->card_str_presence[ref*KS+KS_name]=2;f->card_str_slot[ref*KS+KS_name]=d_intern(c,'''+_literal('未公开牌')+''');f->card_str_presence[ref*KS+KS_type]=2;f->card_str_slot[ref*KS+KS_type]=d_intern(c,"tactic");f->card_bool_presence[ref*KB+KB_copy]=2;f->card_bool_value[ref*KB+KB_copy]=0;f->card_int_presence[ref*KI+KI_cost]=2;f->card_int_value[ref*KI+KI_cost]=999;return ref;}
DFN int d_preview_revealed(DX*c,int side,int card){if(d_card_bool(c,card,"copy"))return 1;for(int j=0;j<c->s->player_revealed_count[side];j++){int at=side*K+j;if(c->s->player_revealed_side[at]==c->s->card_side[card]&&c->s->player_revealed_instance[at]==c->s->card_instance_slot[card])return 1;}return 0;}
DFN void d_anonymize_preview(DX*c){DS*f=c->s;for(int side=0;side<2;side++){for(int zone=0;zone<2;zone++){if(zone==0&&side==c->side)continue;int n=f->zone_count[side*4+zone];for(int j=0;j<n;j++){int at=(side*4+zone)*Z+j,card=f->zone_card_index[at];if(zone==1||!d_preview_revealed(c,side,card)){card=d_unknown_preview_card(c,side);if(d_bad(c))return;f->zone_card_index[at]=card;}if(zone==0){f->card_jingu_mark_presence[card]=0;f->card_jingu_delta[card]=0;f->card_jingu_source_slot[card]=0;}}}}d_collect_cards(c);}
'''
    text+='DFN void d_copy_columns(DS*dst,const DS*src,int thread,int stride){'
    for name,(dtype,shape)in specs.items():text+=f'for(int j=thread;j<{prod(shape)};j+=stride)dst->{name}[j]=src->{name}[j];'
    text+='}\n'
    return text


def kernel_source(layout,capacity,selection_width):
    from .domain_soa_execution import emit_source as execution_source
    specs=schema.array_specs(layout)
    binds=[]
    for i,(name,(dtype,shape))in enumerate(specs.items()):
        typ=_TYPES[np.dtype(dtype).name]
        for obj,table in (('original','input_columns'),('anonymous','anonymous_columns'),('work','work_columns')):
            binds.append(f'{obj}.{name}=({typ}*){table}[{i}]+row*{prod(shape)};')
    for obj in ('original','anonymous','work'):
        binds.append(f'{obj}.query_scratch=(unsigned char*)(query_workspace+row*2*MAXQ);{obj}.damage_scratch=damage_workspace+row*MAXQ;')
    source=execution_source(layout)+emit_source(layout)
    source+='''
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
extern "C" __global__
#else
extern "C"
#endif
void domain_rule_previews(const long long*input_columns,const long long*anonymous_columns,const long long*work_columns,
 const int*sides,const unsigned char*active,const long long*counts,const int*prior_errors,
 const long long*payloads,const unsigned char*battle,const long long*selected,const long long*top,
 const long long*bottom,const unsigned char*bottom_plural,double*attack,double*attack_count,
 double*genesis,double*counter,int*errors,double*query_workspace,int*damage_workspace,int B){
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
 int row=blockIdx.x,thread=threadIdx.x,stride=blockDim.x;if(row>=B)return;
 __shared__ DS original,anonymous,work;
#else
 for(int row=0;row<B;row++){int thread=0,stride=1;DS original,anonymous,work;
#endif
 if(thread==0){'''+''.join(binds)+'''original.error_site=anonymous.error_site=work.error_site=0;original.choice_serial=anonymous.choice_serial=work.choice_serial=0;errors[row]=active[row]?prior_errors[row]:0;}
 for(int j=thread;j<'''+str(capacity)+''';j+=stride){int at=row*'''+str(capacity)+'''+j;attack[at]=0;attack_count[at]=0;genesis[at]=0;counter[at]=0;}
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
 __syncthreads();
#endif
 if(active[row]&&!errors[row]){
  if(thread==0&&(counts[row]<0||counts[row]>'''+str(capacity)+'''))errors[row]=1;
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
  __syncthreads();
#endif
  if(!errors[row]){
   d_copy_columns(&anonymous,&original,thread,stride);
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
   __syncthreads();
#endif
   if(thread==0){DX c={&anonymous,sides[row],-1,-1,DR{-1,-2},0};d_anonymize_preview(&c);if(anonymous.row_error[0])errors[row]=anonymous.row_error[0];}
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
   __syncthreads();
#endif
   for(int j=0;j<counts[row]&&!errors[row];j++){int at=row*'''+str(capacity)+'''+j;int kind=(int)payloads[at*7];
    if(kind==1||(kind==3&&battle[at])){
     d_copy_columns(&work,&anonymous,thread,stride);
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
     __syncthreads();
#endif
     if(thread==0){DTrace trace={0,0,0,0,0};DX c={&work,sides[row],-1,-1,DR{-1,-2},0};c.trace=&trace;
      d_apply_root(&c,payloads+at*7,selected+at*'''+str(selection_width)+','+str(selection_width)+''',top+at*4,bottom+at*4,bottom_plural[at]);
      if(work.row_error[0])errors[row]=work.row_error[0];else {attack[at]=trace.attack;attack_count[at]=trace.attack_count;genesis[at]=trace.genesis;counter[at]=trace.counter;}}
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
     __syncthreads();
#endif
    }
   }
  }
 }
#if !defined(__CUDACC__) && !defined(__CUDACC_RTC__)
 }
#endif
}
'''
    return source,['int64*','int64*','int64*','int32*','uint8*','int64*','int32*','int64*','uint8*','int64*','int64*','int64*','uint8*','float64*','float64*','float64*','float64*','int32*','float64*','int32*','int32']
