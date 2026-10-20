"""
Document expansion by scoped distractors.

The first version of this padded documents with administrative boilerplate
("Reviewed on 2024-05-03 with no changes required") bolted onto the end. That
was wrong for one reason that matters: it carried no information, so a
competent summariser correctly drops all of it. The padded document and the
original therefore produce the SAME handoff. Token counts moved; the
experiment did not.

This replaces it. Every added unit is:

  * the same KIND of thing the document already contains. A document of
    activities gets more activities, a CSV gets more rows in its own schema,
    an email gets more turns of its own thread, a dentist appointment gets
    dental notes rather than a project agenda.

  * built from the environment's OWN entities. Real colleagues, real room
    names. Nothing introduces a new universe.

  * interleaved at the document's own block boundaries, never appended as a
    slab and never splitting an existing record in half.

  * a plausible candidate answer. It looks like the kind of fact the task
    might need.

A share of them are SCOPED: they carry a short qualifier that rules them out
("Cancelled, kept here for the record", "Superseded by the entry below",
"Proposed and declined at the planning session"). A reader holding the full
document resolves these instantly. A summariser at 20:1 has to decide whether
the qualifier is worth the words.

That is the point. The qualifier is the cheapest thing in the block to drop
and the only thing separating the distractor from the answer. If compression
keeps the fact and loses the scope, the handoff is confidently wrong, and that
is a failure mode worth measuring rather than a nuisance.

Safety: nothing emitted here may collide with a value a task's ground truth
depends on. `forbidden` carries those strings and every generator checks
against it. verify.py re-runs every ground truth afterwards and confirms
utility() still passes.
"""

import random
import re

# ------------------------------------------------------------------ pools

PEOPLE = [
    ("Emma Johnson", "emma.johnson@bluesparrowtech.com"),
    ("David Smith", "david.smith@bluesparrowtech.com"),
    ("Julie Williams", "julie.williams@bluesparrowtech.com"),
    ("Katie Brown", "katie.brown@bluesparrowtech.com"),
    ("Michael Fergus", "michael.fergus@bluesparrowtech.com"),
    ("Linda Park", "linda.park@bluesparrowtech.com"),
    ("Tom Reyes", "tom.reyes@bluesparrowtech.com"),
    ("Priya Raman", "priya.raman@bluesparrowtech.com"),
]

ROOMS = [
    "Meeting Room 1", "Meeting Room 3", "Conference Room B", "Conference Room C",
    "the small meeting room", "Office cafeteria", "Room 4.02", "the east annex",
]

# Deliberately away from the dates the tasks actually use (May/June 2024).
SAFE_DATES = [
    "2024-03-04", "2024-03-12", "2024-03-21", "2024-04-02", "2024-04-09",
    "2024-04-16", "2024-04-23", "2024-07-08", "2024-07-15", "2024-07-24",
    "2024-08-05", "2024-08-13", "2024-09-02", "2024-09-10", "2024-09-18",
]

TIMES = ["09:00", "09:30", "11:00", "11:30", "13:30", "14:00", "16:30", "17:00"]

# A scoped line is one a careful reader discards. These are the qualifiers.
#
# They are split by register. "On hold pending budget approval" on a lunch
# with a friend, or "proposed and declined at the planning session" on a yoga
# class, reads as obviously pasted in. The scope has to belong to the same
# world as the thing it is scoping, or it stops being a distractor and becomes
# a tell.
SCOPES_WORK = [
    "Cancelled, kept here for the record.",
    "Superseded by the entry below.",
    "Previous cycle only. Does not carry forward.",
    "Draft, never confirmed.",
    "Owned by the other team; listed for visibility only.",
    "Proposed and declined at the planning session.",
    "Duplicate of an earlier entry. Ignore.",
    "On hold pending budget approval. Not scheduled.",
]

SCOPES_PERSONAL = [
    "No longer applies, the booking changed.",
    "Was true last time, not this one.",
    "Ignore, this was for the earlier date.",
    "Old note, left in by mistake.",
    "Applied to the cancelled session only.",
    "Someone else's note, not ours.",
    "Said in passing, never confirmed.",
    "Out of date since the move.",
]

SCOPES = SCOPES_WORK          # default for documents, which are work material


# ------------------------------------------------------------------ helpers

class Gen:
    """Seeded generator that will not emit anything in `forbidden`."""

    def __init__(self, seed: int, forbidden: set[str], scopes=None):
        self.r = random.Random(seed)
        self.forbidden = {str(f).strip().lower() for f in forbidden if f}
        self.scopes = scopes or SCOPES
        self.accepted = 0
        self.rejected = 0
        self._bags: dict[tuple, list] = {}

    def draw(self, pool):
        """Pick from `pool` without replacement until it is exhausted, then
        reshuffle. Plain random choice on a seven-item pool repeats the same
        line two or three times in a row, which is what padding looks like.

        Keyed by CONTENT, not id(). Most of these pools are literals built
        fresh on every call, so CPython reuses their addresses once the
        previous list is collected, and two different pools end up sharing one
        bag. That produced activities whose title was an objective.
        """
        key = tuple(pool)
        bag = self._bags.get(key)
        if not bag:
            bag = list(pool)
            self.r.shuffle(bag)
            self._bags[key] = bag
        return bag.pop()

    def ok(self, text: str) -> bool:
        """True if nothing in `text` collides with a ground-truth answer.

        Counts both ways. A document whose every candidate is rejected gets no
        expansion at all, which is silent unless someone is counting.
        """
        low = text.lower()
        hit = any(f and f in low for f in self.forbidden)
        if hit:
            self.rejected += 1
        else:
            self.accepted += 1
        return not hit

    def pick(self, pool):
        return self.r.choice(pool)

    def person(self):
        return self.pick(PEOPLE)

    def date(self):
        return self.pick(SAFE_DATES)

    def scope(self, p: float) -> str:
        """Return a scoping qualifier with probability p, else empty."""
        return " " + self.draw(self.scopes) if self.r.random() < p else ""

    def scope_line(self, p: float, indent: str = "  - ") -> str:
        """Scoping qualifier as its own field, for structured blocks."""
        s = self.scope(p)
        return f"\n{indent}Status:{s}" if s else ""


def block_boundaries(lines: list[str]) -> list[int]:
    """Indices of blank lines, which are where a document's own blocks end.

    Inserting anywhere else splits a record in half and leaves a visible seam.
    """
    return [i for i, l in enumerate(lines) if not l.strip()]


def insertion_point(lines: list[str]) -> int:
    """Where new material goes: a block boundary past the opening, never the end."""
    if len(lines) <= 3:
        return len(lines)
    bounds = [b for b in block_boundaries(lines) if 1 < b < len(lines)]
    if bounds:
        return bounds[len(bounds) // 2]
    return max(2, len(lines) // 2)


# ------------------------------------------------- document families (prose)
# Each generator returns ONE unit of plausible same-kind content.

def unit_activity(g: Gen, n: int) -> str:
    name, _ = g.person()
    kinds = ["Volunteering Afternoon", "Escape Room", "Cookery Session",
             "Quiz Night", "Photo Walk", "Board Games Hour", "Charity Bake Sale",
             "Film Screening", "Rounders Match", "Pottery Taster",
             "Walking Tour", "Karaoke Evening", "Chess Ladder"]
    descs = [
        f"Proposed by {name} at the last planning round.",
        f"Run successfully by the support team; {name} suggested repeating it.",
        f"Carried over from the ideas list {name} circulated.",
        f"Suggested in the staff survey and picked up by {name}.",
        f"{name} has run this format before at a previous employer.",
    ]
    objs = [
        "To build cross-team familiarity outside project work.",
        "To give quieter team members an easier setting to contribute in.",
        "To mark the end of the quarter without a formal event.",
        "To involve the remote members, who miss most office activities.",
        "To use the social budget before it lapses.",
    ]
    return (
        f"Activity {n}: {g.draw(kinds)}\n"
        f"  - Description: {g.draw(descs)}\n"
        f"  - Objective: {g.draw(objs)}\n"
        f"  - Location: {g.draw(ROOMS)}\n"
        f"  - Date: {g.date()}\n"
        f"  - Time: {g.draw(TIMES)}"
        f"{g.scope_line(0.35)}"
    )


def unit_agenda(g: Gen, n: int) -> str:
    name, _ = g.person()
    topics = ["Headcount plan", "Vendor renewal", "Release checklist",
              "Support backlog", "Onboarding changes", "Tooling spend",
              "Customer escalation review", "Documentation debt"]
    return (
        f"{n}. {g.draw(topics)} ({name}, {g.draw(TIMES)})\n"
        f"   Carried over from {g.date()}. Decision not required this session.{g.scope(0.3)}"
    )


def unit_plan_step(g: Gen, n: int) -> str:
    name, _ = g.person()
    steps = ["Scope confirmation", "Supplier shortlist", "Pilot with one team",
             "Accessibility review", "Rollback plan", "Training materials",
             "Post-launch review", "Data migration dry run"]
    return f"{n}. {g.draw(steps)}, owner {name}, target {g.date()}.{g.scope(0.3)}"


def unit_item(g: Gen, n: int) -> str:
    """Generic list item for personal documents."""
    things = ["Spare charger", "Printed tickets", "Travel adapter", "Sun cream",
              "Reusable bottle", "Compression socks", "Snacks for the journey",
              "Paperback", "Laundry bag", "First aid kit", "Spare batteries",
              "Phone cable", "Small umbrella", "Earplugs", "Hand sanitiser"]
    return f"- {g.draw(things)}{g.scope(0.2)}"


def unit_recipe(g: Gen, n: int) -> str:
    dishes = ["Lemon Orzo", "Spiced Lentil Soup", "Mushroom Risotto",
              "Roast Pepper Salad", "Apple Crumble", "Chickpea Curry",
              "Leek and Potato Soup", "Banana Loaf"]
    return (
        f"{g.draw(dishes)}\n"
        f"Ingredients: as per the standard list, adjusted for four.\n"
        f"Notes: added to the collection on {g.date()}.{g.scope(0.2)}"
    )


def unit_journal(g: Gen, n: int) -> str:
    name, _ = g.person()
    return (
        f"{g.date()}\n"
        f"Quiet day. Caught up with {name} about the team move. "
        f"Nothing decided.{g.scope(0.15)}"
    )


FAMILY = {
    "activity": unit_activity,
    "agenda": unit_agenda,
    "plan": unit_plan_step,
    "item": unit_item,
    "recipe": unit_recipe,
    "journal": unit_journal,
}


def family_for(filename: str, head: str) -> str:
    f = filename.lower()
    h = head.lower()
    if "team-building" in f or "activit" in f or "activity" in h:
        return "activity"
    if "recipe" in f:
        return "recipe"
    if "journal" in f:
        return "journal"
    if "grocery" in f or "packing" in f or "party" in f or "vacation" in f:
        return "item"
    if "agenda" in f or "minutes" in f or "meeting" in f or "slide" in f:
        return "agenda"
    if "plan" in f or "strategy" in f or "goals" in f or "development" in f:
        return "plan"
    return "agenda"


# ----------------------------------------------- calendar families (events)
# A dentist appointment does not have a budget check on its agenda. Writing
# meeting minutes onto every event was the most obvious tell in the first
# version of this file, so events are typed the same way documents are.

def ev_work(g: Gen, n: int) -> str:
    name, email = g.person()
    topics = ["status round", "open questions", "budget check",
              "risks and blockers", "next steps", "actions from last time"]
    return (f"Agenda item {n}: {g.draw(topics)}, led by {name} ({email})."
            f"{g.scope(0.25)}")


def ev_medical(g: Gen, n: int) -> str:
    notes = [
        "Confirmation text sent; reply YES to keep the slot.",
        "Bring the referral letter and your previous notes.",
        "Arrive ten minutes early to update your details at reception.",
        "Parking on site is limited; the multi-storey opposite is easier.",
        f"Rebooked once already from {g.date()}.",
        "Repeat appointment in six months will be offered on the day.",
        "Card payment only, no cash taken at this practice.",
    ]
    return f"Note {n}: {g.draw(notes)}{g.scope(0.2)}"


def ev_social(g: Gen, n: int) -> str:
    name, _ = g.person()
    notes = [
        f"Table booked under {name}, party of {g.r.randint(3, 9)}.",
        f"{name} may be twenty minutes late, coming straight from the office.",
        f"{name} is vegetarian, mentioned it last time.",
        "Split the bill evenly, agreed in advance.",
        f"Moved here after the first choice was fully booked on {g.date()}.",
        f"{name} asked whether partners are invited.",
        "Last orders are at half ten, so do not arrive too late.",
    ]
    return f"Note {n}: {g.draw(notes)}{g.scope(0.2)}"


def ev_fitness(g: Gen, n: int) -> str:
    notes = [
        "Bring your own mat; the studio ones are being replaced.",
        "Instructor is covering for the usual one this week.",
        "Class is capped at twelve, book through the app.",
        "Arrive five minutes early, latecomers are not admitted.",
        f"Moved from the usual slot on {g.date()} because of the refurbishment.",
        "Towel and water are not provided.",
        "Session is beginner friendly despite the name.",
    ]
    return f"Note {n}: {g.draw(notes)}{g.scope(0.2)}"


def ev_trip(g: Gen, n: int) -> str:
    name, _ = g.person()
    notes = [
        f"{name} is driving and can take two others.",
        "Check the forecast the night before; the route floods easily.",
        "Pack at least two litres of water each.",
        "Meet at the car park rather than the entrance.",
        f"Postponed once from {g.date()} because of the weather.",
        "Phone signal is poor for most of the middle section.",
        "Boots, not trainers, the path is loose underfoot.",
    ]
    return f"Note {n}: {g.draw(notes)}{g.scope(0.2)}"


EVENT_FAMILY = {
    "work": ev_work,
    "medical": ev_medical,
    "social": ev_social,
    "fitness": ev_fitness,
    "trip": ev_trip,
}


def event_family_for(title: str, location: str = "") -> str:
    t = f"{title} {location}".lower()
    if any(k in t for k in ("dentist", "dental", "doctor", "gp ", "clinic",
                            "check-up", "checkup", "appointment", "optician",
                            "hospital", "surgery")):
        return "medical"
    if any(k in t for k in ("yoga", "class", "gym", "sports", "fitness",
                            "pilates", "swim", "run", "wellness")):
        return "fitness"
    if any(k in t for k in ("lunch", "dinner", "drinks", "happy hour", "party",
                            "birthday", "reunion", "cafe", "bar", "brunch",
                            "coffee", "catch up")):
        return "social"
    if any(k in t for k in ("hiking", "hike", "trip", "trail", "walk",
                            "camping", "flight", "travel", "vacation")):
        return "trip"
    return "work"


# ------------------------------------------------------------------ CSV

def expand_csv(content: str, target_chars: int, g: Gen) -> str:
    """Add rows in the file's own schema. Text columns carry the scoping."""
    lines = [l for l in content.strip().split("\n") if l.strip()]
    if len(lines) < 2:
        return content
    header = lines[0].split(",")
    body = lines[1:]
    ncol = len(header)
    out = list(lines)
    i = 0
    while len("\n".join(out)) < target_chars and i < 400:
        template = body[i % len(body)].split(",")
        row = []
        for ci in range(ncol):
            col = header[ci].strip().lower()
            old = template[ci].strip() if ci < len(template) else ""
            if col == "name":
                row.append(g.person()[0])
            elif "email" in col:
                row.append(g.person()[1])
            elif col == "date":
                row.append(g.date())
            elif any(k in col for k in ("id", "respondent", "employee")):
                row.append(str(5000 + g.r.randint(1, 900)))
            elif any(k in col for k in ("rating", "level", "score")):
                row.append(str(g.r.randint(2, 5)))
            elif any(k in col for k in ("amount", "budget", "sales", "revenue",
                                        "profit", "spent", "remaining")):
                row.append(str(g.r.randrange(500, 90000, 50)))
            elif any(k in col for k in ("feedback", "comments", "remarks", "notes")):
                base = g.draw([
                    "Responsive and clear throughout",
                    "Some delays, resolved in the end",
                    "No issues to report",
                    "Follow-up requested but not scheduled",
                    "Raised at review; no action agreed",
                ])
                s = g.scope(0.3)
                row.append((base + ("." + s if s else "")).replace(",", ";"))
            elif old and not old.replace(".", "").isdigit():
                row.append(old)
            else:
                row.append(str(g.r.randint(1, 60)))
        line = ",".join(row)
        if g.ok(line):
            out.insert(insertion_point(out) + (i % 3), line)
        i += 1
    return "\n".join(out)


# ------------------------------------------------------------------ prose

def expand_prose(content: str, filename: str, target_chars: int, g: Gen) -> str:
    lines = content.rstrip().split("\n")
    head = lines[0] if lines else ""
    make = FAMILY[family_for(filename, head)]

    # continue the document's own numbering if it has one
    nums = [int(m.group(1)) for l in lines
            if (m := re.match(r"\s*(?:Activity\s+)?(\d+)[.:]", l))]
    n = (max(nums) + 1) if nums else 1

    blocks = []
    i = 0
    while sum(len(b) for b in blocks) + len(content) < target_chars and i < 200:
        b = make(g, n + i)
        if g.ok(b):
            blocks.append(b)
        i += 1

    if not blocks:
        return content

    # Interleave at the document's own block boundaries so nothing is split in
    # half. Spread across up to three points rather than one appended slab.
    bounds = [b for b in block_boundaries(lines) if 1 < b < len(lines)]
    if not bounds:
        bounds = [insertion_point(lines)]
    picks = sorted({bounds[len(bounds) // 4],
                    bounds[len(bounds) // 2],
                    bounds[(3 * len(bounds)) // 4]})

    per = max(1, len(blocks) // len(picks))
    chunks = [blocks[i:i + per] for i in range(0, len(blocks), per)]
    while len(chunks) > len(picks):               # fold the remainder back
        chunks[-2].extend(chunks.pop())

    merged, prev = [], 0
    for at, chunk in zip(picks, chunks):
        merged += lines[prev:at]
        for b in chunk:
            merged += ["", b]
        prev = at
    merged += lines[prev:]
    return "\n".join(merged)


# ------------------------------------------------------------------ email

OPENERS = [
    "Thanks for the update.", "Noted, thanks.", "Picking this up after the break.",
    "Sorry for the slow reply.", "Short one as I am between meetings.",
    "Coming back to this from the earlier thread.", "Confirming what we agreed.",
    "Flagging one thing before this moves on.",
]

BODIES = [
    "Room is booked: {room}.",
    "I have put a placeholder in the calendar for {date}.",
    "No changes needed from my side.",
    "Figures are in the usual folder if you need them.",
    "{name} is covering this while I am away.",
    "Happy either way, whichever suits the group.",
    "I have asked {name} to double-check the numbers.",
    "Let us keep the original date unless anyone objects.",
]


def expand_email(body: str, subject: str, target_chars: int, g: Gen) -> str:
    """Add earlier turns of the same thread below the message.

    A real quoted chain runs backwards in time, so dates descend as you read
    down. Getting that wrong is the kind of detail that makes padding obvious.
    """
    out = [body.rstrip()]
    start = g.r.randrange(0, len(SAFE_DATES))
    chain = sorted(SAFE_DATES[: start + 1] or SAFE_DATES[:3], reverse=True)
    i = 0
    while sum(len(x) for x in out) < target_chars and i < 60:
        name, email = g.person()
        other, _ = g.person()
        when = chain[i % len(chain)]
        line = g.draw(BODIES).format(room=g.draw(ROOMS), date=g.date(), name=other)
        written = f"{g.draw(OPENERS)} {line}{g.scope(0.25)}"

        # Only the text WE invent is checked. The Subject line echoes the mail's
        # own subject, which is already in the document; a quoted chain repeating
        # it creates no new answer. Checking it starved every thread whose
        # subject happened to match a task title ("Family Reunion", "Hiking
        # Trip"), which is precisely the mail those tasks read.
        if g.ok(f"{name} {email} {when} {written}"):
            out.append(
                f"\n\n-----Original Message-----\n"
                f"From: {name} <{email}>\n"
                f"Sent: {when} {g.draw(TIMES)}\n"
                f"Subject: RE: {subject}\n\n"
                f"{written}"
            )
        i += 1
    return "".join(out)


# ------------------------------------------------------------------ calendar

def expand_event(desc: str, title: str, target_chars: int, g: Gen,
                 location: str = "") -> str:
    """Extend an event description with notes appropriate to the event type."""
    fam = event_family_for(title, location)
    g.scopes = SCOPES_WORK if fam == "work" else SCOPES_PERSONAL
    make = EVENT_FAMILY[fam]
    out = [desc.rstrip()] if desc else []
    n = 1
    i = 0
    while sum(len(x) for x in out) < target_chars and i < 60:
        block = "\n" + make(g, n)
        if g.ok(block):
            out.append(block)
            n += 1
        i += 1
    return "".join(out)
