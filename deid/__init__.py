"""Block 3: de-identification runtime.

surrogate  HMAC-seeded, form-preserving replacement of every labelled mention;
           one interval-preserving date shift per consistency scope.
receipt    Ed25519-signed attestation per shard (one shard = one consistency
           scope) binding policy, code, input, output and tagger provenance.
utility    AE-timeline reconstruction from the surrogated text, diffed against
           the corpus date graph with zero tolerance.

Everything here consumes ``(text, mentions)`` where a mention is
``{start, end, type, entity_id}``; gold labels and relocated model output are
interchangeable inputs.
"""
__version__ = "0.1.0"
