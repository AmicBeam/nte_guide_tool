"""Content-addressed CUBIN cache; every hit checks inputs and binary integrity."""
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4


class CudaBinaryCache:
    schema = 'duel_cuda_cubin_cache_v1'

    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()

    def key(self, identity):
        data = json.dumps({'schema': self.schema, 'inputs': identity},
                          sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        return sha256(data.encode('utf8')).hexdigest()

    def read(self, identity):
        key = self.key(identity)
        binary_path = self.directory/(key+'.cubin')
        receipt_path = self.directory/(key+'.json')
        if not binary_path.exists() or not receipt_path.exists():
            return None, 'miss'
        if binary_path.is_symlink() or receipt_path.is_symlink():
            return None, 'rejected'
        try:
            receipt = json.loads(receipt_path.read_text(encoding='utf8'))
            if (receipt.get('schema') != self.schema or receipt.get('inputs') != identity):
                return None, 'rejected'
            binary = binary_path.read_bytes()
            if (not binary.startswith(b'\x7fELF')
                    or sha256(binary).hexdigest() != receipt.get('binary_sha256')):
                return None, 'rejected'
            return binary, 'hit'
        except (ValueError, TypeError, AttributeError):
            return None, 'rejected'

    def write(self, identity, binary):
        if not isinstance(binary, bytes) or not binary.startswith(b'\x7fELF'):
            raise ValueError('A compiled ELF CUBIN is required')
        key = self.key(identity)
        self.directory.mkdir(parents=True, exist_ok=True)
        receipt = dict(schema=self.schema, inputs=identity,
                       binary_sha256=sha256(binary).hexdigest())
        token = uuid4().hex
        temporary_binary = self.directory/(key+'.'+token+'.tmp')
        temporary_receipt = self.directory/(key+'.'+token+'.json.tmp')
        try:
            temporary_binary.write_bytes(binary)
            temporary_receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2), encoding='utf8')
            os.replace(temporary_binary, self.directory/(key+'.cubin'))
            os.replace(temporary_receipt, self.directory/(key+'.json'))
        finally:
            temporary_binary.unlink(missing_ok=True)
            temporary_receipt.unlink(missing_ok=True)
