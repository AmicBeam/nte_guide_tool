"""Device-only first-registration ordering of physical card rows and references.

This is a storage normalization stage, not a complete codec validator. It does
not intern/compact strings, change identities, or execute game rules. Effective
physical-card views are refused until their recursive registration is supported.
"""
from dataclasses import dataclass
from .domain_soa_observation import DomainSoAObservation

supports_full_duel_rules=False
INVALID=1
UNSUPPORTED_VIEW=2
UNREFERENCED_CARD=4
DUPLICATE_ZONE=8


@dataclass(frozen=True)
class PreparedCardOrder:
    owner:object
    sources:dict
    outputs:dict
    active:object
    old_to_new:object
    errors:object


class DomainCardOrder:
    supports_full_duel_rules=False
    gpu_qualified=False

    def __init__(self,layout=None,*,device='cuda',allow_cpu_oracle=False):
        # Reuse schema/device validation only. No observation or rules are run.
        self.validator=DomainSoAObservation(layout,device=device,allow_cpu_oracle=allow_cpu_oracle)
        self.t=self.validator.t;self.layout=self.validator.layout;self.device=self.validator.device;self.closed=False
        self.card_columns=tuple(k for k in self.validator.specs if k.startswith('card_'))

    def prepare(self,fields,active):
        t=self.t
        if self.closed:raise RuntimeError('Card-order backend closed')
        if self.device.type=='cuda'and t.cuda.is_current_stream_capturing():raise RuntimeError('Prepare outside capture')
        if not isinstance(active,t.Tensor)or active.device!=self.device or active.dtype!=t.bool or active.ndim!=1:raise ValueError('Resident bool active required')
        B=active.shape[0]
        if not 1<=B<=self.layout.game_capacity:raise ValueError('Batch capacity exceeded')
        source={k:self.validator.partial._column(fields,k,B)for k in self.validator.specs}
        return PreparedCardOrder(self,source,{k:v.clone(memory_format=t.contiguous_format)for k,v in source.items()},active,
            t.empty((B,self.layout.max_cards),dtype=t.int64,device=self.device),t.empty(B,dtype=t.int32,device=self.device))

    def launch(self,p):
        if self.closed or not isinstance(p,PreparedCardOrder)or p.owner is not self:raise ValueError('Closed or foreign card-order buffers')
        t=self.t;f=p.sources;B=p.active.shape[0];K=self.layout.max_cards;Z=self.layout.max_zone;CH=self.layout.max_choice_cards
        for k,v in p.outputs.items():v.copy_(f[k])
        out=p.outputs;zcount=f['zone_count'].long();cc=f['choice_card_count'].long();oc=f['op_choice_card_count'].long()
        zm=t.arange(Z,device=self.device)[None,None,None,:]<zcount[...,None]
        cm=(t.arange(CH,device=self.device)[None,:]<cc[:,None])&f['pending_choice_present'][:,None]
        om=(t.arange(CH,device=self.device)[None,:]<oc[:,None])&f['operation_present'][:,None]
        one=f['operation_present']&(f['op_card_index']>=0)
        refs=t.cat((f['zone_card_index'].flatten(1).long(),f['choice_card_index'].long(),f['op_card_index'][:,None].long(),f['op_choice_card_index'].long()),1)
        live=t.cat((zm.flatten(1),cm,one[:,None],om),1);L=refs.shape[1]
        bad=((zcount<0)|(zcount>Z)).flatten(1).any(-1)|((cc<0)|(cc>CH))|((oc<0)|(oc>CH))
        bad|=(live&((refs<0)|(refs>=K))).any(-1)
        safe=refs.clamp(0,K-1);valid=f['card_valid'];bad|=(live&~valid.gather(1,safe)).any(-1)
        bad|=~f['row_valid']|(f['catalog_digest']!=self.validator.partial._catalog_digest[None,:]).any(-1)
        first=t.full((B,K),L,dtype=t.int64,device=self.device)
        positions=t.arange(L,device=self.device)[None,:].expand(B,-1)
        first.scatter_reduce_(1,safe,t.where(live,positions,L),reduce='amin',include_self=True)
        zone_n=t.zeros((B,K),dtype=t.int64,device=self.device)
        zone_n.scatter_add_(1,f['zone_card_index'].flatten(1).long().clamp(0,K-1),zm.flatten(1).long())
        missing=(valid&(first==L)).any(-1);duplicate=(zone_n>1).any(-1)
        # Unique positional key also makes unused-row ordering deterministic.
        key=first*(K+1)+t.arange(K,device=self.device)[None,:]
        perm=key.argsort(1);p.old_to_new.scatter_(1,perm,t.arange(K,device=self.device)[None,:].expand(B,-1))
        n=valid.sum(1);present=t.arange(K,device=self.device)[None,:]<n[:,None]
        for name in self.card_columns:
            value=f[name];shape=(B,K)+(1,)*(value.ndim-2)
            indices=perm.reshape(shape).expand_as(value)
            out[name].copy_(t.where(present.reshape(shape),value.gather(1,indices),0))
        # Views are refused below, so their physical-pointer column stays zero.
        out['card_physical_index'].zero_()
        # Canonical codec stores catalog/absent/null payload slots as zero.
        # Numeric allocation may retain defaults in these ignored slots.
        for name,mask in (('card_int_value','card_int_presence'),('card_bool_value','card_bool_presence'),('card_str_slot','card_str_presence')):
            out[name].masked_fill_(out[mask]!=2,0)
        mapped=p.old_to_new.gather(1,safe)
        offset=2*4*Z
        out['zone_card_index'].copy_(t.where(zm,mapped[:,:offset].reshape(B,2,4,Z),0).int())
        out['choice_card_index'].copy_(t.where(cm,mapped[:,offset:offset+CH],0).int());offset+=CH
        out['op_card_index'].copy_(t.where(f['operation_present'],t.where(one,mapped[:,offset],-1),0).int());offset+=1
        out['op_choice_card_index'].copy_(t.where(om,mapped[:,offset:],0).int())
        errors=bad.int()*INVALID|f['card_is_view'].any(-1).int()*UNSUPPORTED_VIEW|missing.int()*UNREFERENCED_CARD|duplicate.int()*DUPLICATE_ZONE
        errors=t.where(f['row_error']!=0,f['row_error'],errors);p.errors.copy_(t.where(p.active,errors,0))
        rollback=~p.active|(p.errors!=0)
        for name,value in out.items():
            mask=rollback.reshape((B,)+(1,)*(value.ndim-1));value.copy_(t.where(mask,f[name],value))
        out['row_error'].copy_(t.where(p.active,p.errors,f['row_error']))
        p.old_to_new.masked_fill_(rollback[:,None],-1)
        return dict(fields=out,errors=p.errors,old_to_new=p.old_to_new,full_canonical_codec_qualified=False,full_duel_qualified=False)

    def close(self):
        if not self.closed:self.validator.close();self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
