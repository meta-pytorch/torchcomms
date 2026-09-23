# observatory

A small shared library that every comms backend links, holding the observability
state that has to be process-global in fact and not just in intent.

## Why it exists

Backends ship as separate shared objects, and each folds its dependencies in
privately. They also set the ELF `SYMBOLIC` flag, which binds internal
references to the object's own definitions and defeats interposition. A
registry compiled into each backend is therefore several registries. A caller
asks one of them for everything in the process and sees only what lives in its
own copy. There is no link error and no warning.

`libobservatory.so` is the one copy. Backends link it instead of compiling it.

## What may live here

A component belongs here only if both hold:

1. **It must be observability state.** Telemetry a backend produces and
   something outside it consumes: feed registries, the events they carry, and
   the ids that key them. The name is the rule. Process-global state that is
   not observability — fault-tolerance coordination, a topology cache — does
   not belong here even if it satisfies the other, because a reader who trusts
   the name would never look for it here.

2. **It must be exactly one instance per process.** Not "it would be nicer
   shared": a second copy has to be a correctness bug, the way a second
   registry halves a drain without saying so. Anything that only wants to avoid
   duplicated code belongs in `comms/utils` and gets folded in like everything
   else.

## Dependencies, and the one thing to check

There is no blanket ban on folly or anything else here. Prefer the standard
library where it does the job — the registry uses a `std::mutex` rather than
`folly::Synchronized` for no better reason than that it was enough — but a
dependency that earns its place is allowed.

What does need checking, every time, is **process-global state inside the
dependency itself**. Backends ship as separate shared objects with their
dependencies folded in privately, so a construct that keeps a registrar, a
singleton or a reclamation domain can end up with one copy per `.so` while the
object it manages is shared across all of them. That is the same failure this
library exists to prevent, arriving from underneath.

Two worked examples, because the distinction is not obvious from a header name:

- `folly::UMPMCQueue` in its unbounded form reclaims through **hazard
  pointers**, and the hazptr domain is process-global. A queue of that kind
  living here and touched by two backends is exactly the hazard above.
- A pure value or algorithm header carries none of that and is fine.

So the question to ask of a new dependency is not "is it folly" but "does it
keep state of its own, and if so, whose copy wins".

The lifecycle-feed registry also registers through callables rather than an
interface, so this library names no colltrace type at all. That is worth
keeping for its own sake: it is what lets the value types in
`colltrace/LifecycleFeedTypes.h` stay the whole contract.

## Build notes

- Keep every source list here explicit. Admission should be a deliberate act,
  and a glob would let a file drift in or out of a backend unnoticed.
- Nothing here may get a version script or `-Bsymbolic`. Either one would hide
  or pre-bind the symbols consumers are supposed to resolve in this library.
- Files live under `comms/observatory/`, outside the directories the backend
  builds glob. That keeps them out of the backends structurally, instead of by
  an exclusion regex someone can break.
- The registry holder is deliberately leaked. A function-local static would be
  destroyed at exit ahead of anything constructed earlier, and a communicator
  torn down by a later static destructor deregisters from here.
- Under Buck everything links into one binary, so the single-instance property
  holds for free and is invisible. It is only at stake in the shared-object
  builds, which is where it has to be checked.

## Current contents

| Component | Why it must be single-instance |
|---|---|
| `colltrace/LifecycleFeedRegistry` | A drain asks for every collective-lifecycle feed in the process. Split across copies, it returns a subset and reports nothing. |
