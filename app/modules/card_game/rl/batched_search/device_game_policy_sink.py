"""Bind physical-game revisions and root hashes to a light replay-policy tape."""


class DeviceGamePolicySink:
    """GameLoop sink adapter; recorded policy targets still require GPU replay."""

    supports_full_duel_rules = False

    def __init__(self, tape):
        for name in ('begin_device', 'append_device', 'finish_device'):
            if not callable(getattr(tape, name, None)):
                raise TypeError('Device replay-policy tape required')
        self.tape = tape
        self.loop = None

    def bind_game_loop(self, loop):
        sink = getattr(loop, 'sink', None)
        if self.loop is not None or (sink is not self and getattr(sink, 'policy_sink', None) is not self):
            raise ValueError('Bind once to the exact owning physical game loop')
        self.loop = loop

    def begin_device(self, roots, policy_ids, auxiliary_ids, seeds, train_sides, active):
        if self.loop is None:
            raise RuntimeError('Physical game loop must be bound before capture')
        return self.tape.begin_device(
            roots, policy_ids, auxiliary_ids, seeds, train_sides, active)

    def append_device(self, roots, actors, steps, result, active):
        if self.loop is None:
            raise RuntimeError('Physical game loop must be bound before capture')
        if result.information_keys is None:
            raise ValueError('Exact root information-set hashes required')
        return self.tape.append_device(
            roots, actors, steps, self.loop.source_versions_device,
            result, result.information_keys, active)

    def finish_device(self, roots, winners, steps, complete, errors):
        return self.tape.finish_device(roots, winners, steps, complete, errors)
