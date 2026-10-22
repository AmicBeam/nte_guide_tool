"""Direct device operations on ordered card-instance zone columns.

An operation preserves entity references and order; it does not run card hooks,
draw defeat rules, hand-overflow redirects, or effects. Callers must resolve
those rules before moving a card. No VM or host per-root rule callback.
"""
from dataclasses import dataclass

HAND, DECK, DISCARD, REMOVED = range(4)
INVALID_ZONE, INVALID_POSITION, INVALID_REFERENCE, CAPACITY = 1,2,4,8
supports_full_duel_rules=False


@dataclass(frozen=True)
class ZoneMove:
    cards: object
    counts: object
    moved: object
    errors: object


def move_card(cards,counts,sides,sources,positions,destinations,insertions,active,
              *,max_cards):
    """Move one instance per active row; insertion is AFTER source removal.

    cards=int32[B,2,4,Z], counts=int32[B,2,4], selectors=int64[B],
    active=bool[B]. Inactive/error rows retain every input bit. A full hand
    returns CAPACITY; it is never silently truncated or treated as a draw.
    """
    import torch as t
    if (not isinstance(cards,t.Tensor)or cards.dtype!=t.int32 or cards.ndim!=4
            or cards.shape[1:3]!=(2,4)or cards.shape[0]<1 or cards.shape[3]<10):
        raise ValueError('Resident int32[B,2,4,Z>=10] instance-index zones required')
    b,_,_,z=cards.shape;d=cards.device
    if type(max_cards)is not int or not 1<=max_cards<2**31:
        raise ValueError('Explicit card-instance table capacity required')
    if (not isinstance(counts,t.Tensor)or counts.dtype!=t.int32
            or counts.shape!=(b,2,4)or counts.device!=d):
        raise ValueError('Matching resident int32 zone counts required')
    for value in(sides,sources,positions,destinations,insertions):
        if (not isinstance(value,t.Tensor)or value.dtype!=t.int64
                or value.shape!=(b,)or value.device!=d):
            raise ValueError('Matching resident int64 selectors required')
    if (not isinstance(active,t.Tensor)or active.dtype!=t.bool
            or active.shape!=(b,)or active.device!=d):
        raise ValueError('Matching resident bool active rows required')
    rows=t.arange(b,device=d);cols=t.arange(z,device=d)
    side=sides.clamp(0,1);src=sources.clamp(0,3);dst=destinations.clamp(0,3)
    bad_zone=((sides<0)|(sides>1)|(sources<0)|(sources>3)
              |(destinations<0)|(destinations>3)
              |((counts<0)|(counts>z)).any(-1).any(-1))
    errors=(active&bad_zone).int()*INVALID_ZONE
    source=cards[rows,side,src];source_count=counts[rows,side,src].long()
    destination_count=counts[rows,side,dst].long()-(src==dst).long()
    pos=positions.clamp(0,z-1)
    moved=source.gather(1,pos[:,None])[:,0]
    bad_position=((positions<0)|(positions>=source_count)
                  |(insertions<0)|(insertions>destination_count))
    errors|=(active&~bad_zone&bad_position).int()*INVALID_POSITION
    present=cols[None,None,None,:]<counts[:,:,:,None]
    occurrences=((cards==moved[:,None,None,None])&present).sum((1,2,3))
    bad_ref=(moved<0)|(moved>=max_cards)|(occurrences!=1)
    errors|=(active&~bad_zone&~bad_position&bad_ref).int()*INVALID_REFERENCE
    limit=t.where(dst==HAND,10,z)
    errors|=(active&~bad_zone&~bad_position&~bad_ref
             &(destination_count>=limit)).int()*CAPACITY
    good=active&(errors==0)
    output=cards.clone();lengths=counts.clone()
    removal=(cols[None,:]+(cols[None,:]>=pos[:,None]).long()).clamp_max(z-1)
    shortened=source.gather(1,removal)
    shortened=t.where(cols[None,:]<(source_count-1)[:,None],shortened,-1)
    output[rows,side,src]=t.where(good[:,None],shortened,source)
    lengths[rows,side,src]=t.where(good,source_count-1,source_count).int()
    destination=output[rows,side,dst]
    destination_count=lengths[rows,side,dst].long()
    insertion=insertions.clamp(0,z-1)
    old_index=(cols[None,:]-(cols[None,:]>insertion[:,None]).long()).clamp(0,z-1)
    extended=destination.gather(1,old_index)
    extended=t.where(cols[None,:]==insertion[:,None],moved[:,None],extended)
    extended=t.where(cols[None,:]<(destination_count+1)[:,None],extended,-1)
    output[rows,side,dst]=t.where(good[:,None],extended,destination)
    lengths[rows,side,dst]=t.where(good,destination_count+1,destination_count).int()
    return ZoneMove(output,lengths,t.where(good,moved,-1),errors)
