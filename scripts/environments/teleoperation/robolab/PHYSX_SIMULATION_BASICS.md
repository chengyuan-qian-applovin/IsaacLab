# How PhysX Simulates Your Robot — A Plain-Language Guide

A primer on what actually happens inside one physics step of Isaac Sim, written
for someone comfortable with robotics but new to simulation internals. Every
technical term is explained when first used. Examples reference this project's
setups (GR1T2 teleop, RoboLab DROID/SharpaWave scenes) so the abstract ideas
connect to numbers you've already seen in configs and logs.

Isaac Sim's physics engine is **NVIDIA PhysX** (version 5.x). Everything below
describes PhysX specifically — other engines (MuJoCo, Bullet, ...) make
different choices.

---

## 0. The starting point: state + command

At any moment the simulator holds two kinds of information:

- **State** — where everything is and how it is moving: the position and
  orientation of every object, and every robot joint's angle and angular
  velocity.
- **Command** — what you asked the robot to do. In our setups, the teleop
  stack writes *joint position targets* (e.g. "elbow joint: go to 1.2 rad").
  The robot's motors are simulated as simple spring-like controllers around
  those targets (more on this in §4 — it's the `stiffness`/`damping` numbers
  in the robot configs).

The engine then advances time by one small fixed increment, the **timestep**
(`dt`). Our scenes use `dt = 1/120 s`, i.e. physics updates 120 times per
simulated second. One "step" = run the entire pipeline below once. (Your 15 Hz
control rate just means one command is held constant for 8 consecutive physics
steps.)

Why such small steps? The engine only *checks the world* — who touches whom —
once per step. Fast motion between checks can be missed, and stiff
interactions become unstable if the step is too large. Small steps keep the
simulation honest.

---

## 1. Stage one: find out what is touching what (collision detection)

Before any forces can be computed, the engine must discover every contact.
This happens in two passes, cheap-then-precise:

1. **Broad phase** — every object gets a simple box drawn around it (a
   "bounding box"). The engine quickly finds which *boxes* overlap. This
   discards the vast majority of object pairs (the banana's box never touches
   the far wall's box) at almost no cost.
2. **Narrow phase** — for the surviving candidate pairs, compute the *actual*
   geometry: where exactly do the two shapes touch, in which direction does the
   surface push, and how deep is any overlap? The result for each touching
   pair is a small set of **contact points** (typically 1–4 per pair), each
   with a contact direction and a penetration depth.

  PhysX's particular trick here is called a **persistent contact manifold**:
  when two objects stay in contact across many steps (a gripper holding an
  object), it keeps and updates the contact points from last step instead of
  recomputing them from scratch — faster and less jittery.

Two config numbers you've seen live here: `contact_offset` (0.02 m in RoboLab)
means "start generating contacts when surfaces come within 2 cm" — a safety
margin so contacts exist *before* actual touching; `rest_offset` is the
separation the engine tries to maintain once resting.

Finally, objects connected by touching or by joints are grouped into
**islands** — clusters that must be solved together. Your robot + the banana
it's grasping + the table form one island; something untouched across the room
sleeps.

---

## 2. Stage two: turn everything into "constraints"

Now the engine knows the geometry. Next, every rule about motion is expressed
in one uniform mathematical currency: the **constraint**. A constraint is just
a statement of the form "this quantity must not do X":

- Every contact point becomes a constraint: *"these two surfaces must not move
  into each other"* — and a companion **friction** constraint: *"sliding along
  the surface is resisted, up to a limit proportional to how hard the surfaces
  press together"* (the classic Coulomb friction rule, approximated).
- Every joint limit becomes a constraint: *"this joint angle must stay below
  0.785 rad."*
- Every motor command becomes a constraint too — see §4.

The engine enforces constraints by applying **impulses** — instantaneous
velocity changes, the step-sized cousin of forces. "Apply an impulse at this
contact" means "change these two bodies' velocities right now, along the
contact direction, just enough to stop them sinking into each other."

The catch: constraints are **coupled**. The impulse that stops the fingertip
penetrating the banana changes the whole arm's motion, which changes what the
other fingertip needs, which changes the load on every joint motor... You
cannot solve them one by one and be done.

---

## 3. Stage three: the solver — this is what "iterations" are

PhysX does *not* solve the coupled system exactly (that would mean building and
inverting a large matrix every step — some engines do this; PhysX does not).
Instead it uses a simple, robust loop:

> Visit every constraint in turn. For each one, compute and apply the impulse
> that fixes *that one constraint*, assuming everything else stays as is. When
> you've visited them all, that's **one iteration**. Repeat.

Each pass slightly invalidates the previous fixes (they're coupled!), but each
pass also propagates corrections further through the island. After enough
iterations, the whole system settles close to the true answer. This
one-at-a-time-and-repeat strategy is known as a *Gauss-Seidel* method — the
name isn't important; the picture of "fix each thing locally, sweep repeatedly,
converge" is.

**This is exactly the number you've been tuning.** The Fourier GR1T2 robot
asks for **8 iterations** per step; RoboLab scenes run at an effective **32**
(their robots request 64, but the scene caps it at 32). More iterations =
better convergence when many contacts couple strongly (a five-fingered grasp)
= more CPU/GPU time, multiplied by every contact in the island, every step.

### The TGS twist (the variant our sims actually use)

PhysX offers two versions of this loop, and all our configs select the second
(`solver_type: 1`):

- **PGS** (the classic): all iterations work on a frozen snapshot of the
  geometry taken at the start of the step. Any penetration is fixed by adding
  a small artificial "push-out" velocity.
- **TGS** ("Temporal Gauss-Seidel", the PhysX-specific improvement): the
  iterations are also **slices of time**. With 32 iterations, each iteration
  advances the step's motion by 1/32 of `dt`, *moves the bodies a little*, and
  re-measures the contact geometry before the next pass. Iterating and
  micro-stepping become the same thing.

  Why it matters: re-measuring as you go handles the hard robotics cases far
  better at the same cost — long chains of joints under load, stiff motors
  pressed into contact, heavy object held by light fingers. That's precisely
  the gripper-squeezing-an-object situation from our RoboLab debugging.

After the main iterations, a few **velocity iterations** (0–4 in our configs)
run as a final cleanup pass on velocities only — so the artificial push-out
used to fix overlaps doesn't leak into bounce or friction as fake energy.

---

## 4. Where your motor command enters: implicit drives

The `stiffness=400, damping=80` on the arm actuators describe a **PD
controller** (proportional–derivative: torque = stiffness × position error +
damping × velocity error) that PhysX itself simulates — called a **joint
drive**.

The crucial word in the configs is **implicit** (`ImplicitActuatorCfg`). A
naive ("explicit") PD would compute torque from the *current* error and apply
it — with stiffness 400 at dt = 1/120, that overshoots and explodes. PhysX
instead folds the PD law into the constraint solver as one more constraint
row, solved *simultaneously* with all the contacts, against the *end-of-step*
velocities. Consequences worth knowing:

- It is unconditionally stable — stiff gains don't blow up.
- The motor automatically "feels" contact: if the finger presses a table, the
  drive constraint and the contact constraint negotiate within the same
  iteration loop, and the motor output respects the collision.
- It costs iterations like everything else: a badly-converged step
  short-changes motors and contacts alike.

---

## 5. Robots are special: articulations

A free-floating object (the banana) is simulated by its 6 degrees of freedom
directly. A robot could be simulated as 10 separate bodies glued by joint
constraints — but then joints would only be as rigid as the iteration count
allows, and long chains would visibly stretch.

PhysX instead treats each robot as an **articulation in reduced coordinates**:
the arm *is* its list of joint angles. The hand cannot drift off the forearm
any more than "joint 4 = 0.7 rad" can drift off itself — the connection is
built into the coordinates, not enforced by iterations. Forces propagate
through the tree with a recursive algorithm (the Featherstone method — again,
name optional; the point is joints in the tree are *exact by construction*).

What still goes through the iterative solver for a robot: its **contacts**
with the world, its **joint limits**, and its **drives**. This also explains
our "gripper breaking apart" episode: articulation joints cannot separate —
what we saw was undriven linkage joints legitimately *rotating* within their
limits, because nothing (no drive, no coupling) resisted them.

---

## 6. Closing the step — and the fast-object problem (CCD)

The corrected velocities move every pose forward to time *t + dt* (under TGS,
most of that motion already accumulated during the time-sliced iterations),
and the new state is handed back to Isaac Lab for rendering, observations, and
your next command.

One optional extra pass: because contact detection ran only once, at the start
of the step, a *fast* object can pass entirely through a thin obstacle between
two checks — "tunneling". **CCD (Continuous Collision Detection)** guards
against this by sweeping fast-moving objects along their path and catching the
crossing they would have skipped. Two things to remember from our debugging:
CCD costs real time, and PhysX only performs it in the **CPU** pipeline — on
GPU (`--device cuda:*`) the request is silently ignored (the log even warns).

On that note: CPU vs GPU is the *same algorithm* on different processors. GPU
wins by parallelism when simulating many environments at once (RL training,
RoboLab's batch evals); for a single teleop environment its fixed per-step
overheads often make it slower than CPU — as we measured.

---

## 7. One step, end to end (summary card)

```
your command (joint targets)                    state at time t
        └──────────────┬───────────────────────────────┘
                       ▼
   1. COLLISION DETECTION   boxes overlap? → exact contact points
                            (persistent manifolds; contact_offset margin)
   2. CONSTRAINT SETUP      contacts + friction + joint limits + motor
                            drives → one list of "must not" rules
   3. SOLVER ITERATIONS     sweep the list, fix each rule locally by an
      (the 8 vs 32 knob)    impulse, repeat; TGS: each sweep also advances
                            a slice of time and re-measures geometry
      + velocity cleanup    few extra sweeps so overlap-fixing doesn't
                            become fake bounce energy
   4. INTEGRATE             poses move to t + dt
      (+ CCD, CPU only)     sweep fast objects so they can't tunnel
                       ▼
                 state at time t + dt  →  rendering, observations,
                                          next command
```

## 8. Glossary

| Term | Meaning |
|---|---|
| timestep (`dt`) | The fixed slice of time one physics step advances (1/120 s here) |
| rigid body | An object that never deforms; all our objects and robot links |
| contact point | A spot where two surfaces touch: position + direction + overlap depth |
| constraint | A rule of the form "this motion is forbidden/limited" |
| impulse | An instantaneous velocity change; how the solver enforces constraints |
| iteration | One full sweep of local fixes over all constraints in an island |
| PGS / TGS | The two PhysX solver variants; TGS (ours) also slices time across iterations |
| velocity iterations | Extra cleanup sweeps at the end, on velocities only |
| joint drive | PhysX's built-in PD motor around your commanded target |
| implicit | The drive is solved together with contacts, making stiff gains stable |
| articulation | A robot simulated directly in joint coordinates; tree joints are exact |
| island | A cluster of objects that interact and must be solved together |
| CCD | Extra sweep preventing fast objects passing through thin ones; CPU-only |
| broad/narrow phase | Cheap box-overlap filtering / exact contact computation |

## 9. What PhysX is *not* doing

For orientation when reading papers or other engines' docs:

- It never builds and exactly solves the full coupled equation system (no big
  matrix factorization per step, unlike MuJoCo's approach). Accuracy comes
  from iterating.
- Its cloth/particle tech (called XPBD) is a different solver — rigid bodies
  and robots use the impulse/iteration pipeline described here.
- TGS's internal time-slicing is not the same as taking smaller timesteps:
  collision detection still runs once per step. That's why fast, contact-rich
  scenes need genuinely small `dt` (120 Hz) — and why CCD exists.
