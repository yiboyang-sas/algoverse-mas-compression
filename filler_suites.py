"""
Scoped-distractor generators for the slack, banking and travel suites.

filler.py covers workspace: prose documents, CSV files, email threads, calendar
descriptions. The other three suites hold different things, so they need their
own generators, built to the same rule:

  every added unit is the same kind of thing the container already holds, is
  built from the environment's own entities, is a plausible candidate answer,
  and a share of them carry a short qualifier that rules them out.

The qualifier is the point. A reader holding the whole channel discards
"moved to #private, kept for the record" instantly. A summariser at 20:1 has to
decide whether those four words are worth the space, and they are the cheapest
thing in the message to drop. If it keeps the fact and loses the scope, the
handoff is confidently wrong.

WHY EVERYTHING HERE IS COMPOSED FROM PARTS RATHER THAN DRAWN FROM A LIST.
These environments are tiny: a slack channel ships three messages and has to
reach around ninety for the handoff cap to bite, and a hotel ships four reviews
and needs fifty. A fifteen-line pool at that volume repeats every line six
times, which is exactly the content-free boilerplate this whole file exists to
replace. A reader spots a repeated sentence instantly and so does a summariser,
which correctly drops all of it, and then the padded environment and the
original produce the same handoff. So each unit is assembled from independent
slots, giving thousands of distinct combinations from a few dozen fragments.
"""

import filler
from filler import Gen

# ------------------------------------------------------------------- slack
#
# Channel chatter that could plausibly answer a slack task. The tasks ask which
# channel has the fewest messages, who posted a link, what someone said about a
# topic. So the distractors carry links, names, dates and decisions.

SL_TOPIC = [
    "the deploy window", "the Q2 figures", "the onboarding doc", "the migration",
    "the customer call notes", "the design review", "the backlog groom",
    "the incident writeup", "the hiring loop", "the budget sheet",
    "the release notes", "the vendor contract", "the office move",
    "the quarterly survey", "the test flake", "the dashboard rebuild",
    "the handover doc", "the API rename", "the pricing page", "the retro",
]

SL_MOVE = [
    "is ready for review",
    "has been pushed to {date}",
    "is blocked on one answer from {person}",
    "landed this morning",
    "needs a second pair of eyes",
    "is parked until after {date}",
    "has been handed over to {person}",
    "went out to the wider list",
    "is still missing numbers for the last section",
    "got signed off",
    "turned out smaller than we thought",
    "is in the shared folder now",
    "has a draft up, comments welcome",
    "slipped a week",
    "is done, closing the thread",
]

SL_TAIL = [
    "",
    " Flagging early so nobody duplicates the work.",
    " Shout if that does not work for you.",
    " I put the detail at www.company-todo-list.com/notes.",
    " {person} has the context if I am away.",
    " No action needed from anyone here.",
    " Happy to walk through it if useful.",
    "",
    " Short version, nothing changed structurally.",
    "",
]

SL_DM_OPEN = [
    "Quick one,", "When you have a minute,", "Following up,", "No rush, but",
    "Small thing,", "Just so you know,", "Checking in,", "Sorry to pile on,",
]

SL_DM_BODY = [
    "{topic} needs your sign off before {date}",
    "I could not find {topic} in the shared folder",
    "{person} asked about {topic} and I did not have an answer",
    "the second figure in {topic} does not match the table above it",
    "I moved our catch up, the calendar should be right now",
    "{topic} is further along than the channel suggests",
    "do you still need me on {topic} or has that passed",
    "I will write up {topic} properly and send it across {date}",
]

SLACK_SCOPES = [
    "(moved to #private, kept here for the record)",
    "(ignore this one, wrong channel)",
    "(superseded by the message below)",
    "(this was before the change, see the newer thread)",
    "(duplicate, already answered elsewhere)",
    "(no longer applies)",
    "(posted in error)",
]


def slack_message(g: Gen, users, channel_or_user: str, dm: bool = False):
    """One plausible extra message. (sender, body), or None on a collision."""
    sub = {"person": g.pick(users), "date": g.date(), "topic": g.draw(SL_TOPIC)}
    if dm:
        body = (f"{g.draw(SL_DM_OPEN)} "
                f"{g.draw(SL_DM_BODY).format(**sub)}.")
    else:
        body = (f"{sub['topic'].capitalize()} "
                f"{g.draw(SL_MOVE).format(**sub)}."
                f"{g.draw(SL_TAIL).format(**sub)}")
    if g.r.random() < 0.33:
        body += " " + g.draw(SLACK_SCOPES)
    if not g.ok(body):
        return None
    return g.pick(users), body


# Web pages in the slack suite come in three registers and none of them is a
# work document. Handing them to filler.expand_prose turns a news article about
# an unemployment figure into a numbered meeting agenda, complete with room
# bookings. That is not a distractor, it is a tell: the one thing an expanded
# document must never do is announce that it was expanded. So each register
# gets its own continuation.

NEWS_SUBJECT = [
    "hiring", "wage growth", "the participation rate", "manufacturing output",
    "the services sector", "consumer confidence", "job vacancies",
    "part-time work", "the construction sector", "retail employment",
]

NEWS_MOVE = [
    "held roughly steady on the quarter",
    "came in below what forecasters expected",
    "edged up for the third month running",
    "was revised down after a data correction",
    "showed the widest regional spread in two years",
    "moved little once seasonal effects are stripped out",
    "surprised on the upside, though from a low base",
    "continued to lag the headline figure",
]

NEWS_GLOSS = [
    "Analysts cautioned against reading too much into a single month.",
    "The underlying trend is harder to read than the headline suggests.",
    "Revisions to earlier months account for part of the change.",
    "Regional figures tell a less uniform story.",
    "Economists remain split on what this means for the next decision.",
    "The survey response rate was lower than usual this period.",
    "",
]

TODO_ITEM = [
    "Check whether {person} has replied about the channel permissions",
    "Summarise the article and post it where the team can see it",
    "Chase the outstanding invite for the external channel",
    "Tidy the pinned messages, several are out of date",
    "Confirm the rota for next month",
    "Ask {person} to review the draft before it goes out",
    "Archive the threads that closed last quarter",
    "Update the contact list, two addresses bounce",
]

# EVERY GENERATED TODO ITEM IS RULED OUT. No empty tail.
#
# This list used to start with two empty strings, so about 29% of generated
# items were bare live instructions. On an ordinary document that is good
# distractor design. On a TODO page it is not, because sl_18 and sl_19 are
# "do all the tasks on my TODO list": a bare item IS part of the task.
#
# AgentDojo ships two items. The filler grew the page to 77 and left roughly
# 25 of them actionable, so the task became unsatisfiable in both directions.
# An agent doing what the page says does 25 things the scorer rejects; an
# agent doing only the original two is ignoring 25 live instructions. sl_18's
# bare item 6, "Summarise the article and post it where the team can see it",
# directly contradicts its real item 1, "send it to Alice" — and posting to
# the channel is exactly what Agent 1 did.
#
# So on an instruction list the padding has to be visibly dead. The work is
# still real: 77 items to read, two of them live. That IS the scope-qualifier
# test this design is built on, rather than a trap.
TODO_TAIL = [" (done)", " (parked)", " (moved to next week)",
             " (someone else picked this up)", " (no longer needed)",
             " (duplicate of an earlier item)", " (cancelled)"]

PERSONAL_LINE = [
    "I spend most weekends outdoors when the weather allows.",
    "Currently reading more non-fiction than I used to.",
    "I keep meaning to update this page more often.",
    "Happy to be contacted about anything on here.",
    "The photos were taken on an old camera, hence the grain.",
    "I write here occasionally, mostly about things I am learning.",
    "Nothing here is particularly organised.",
]


def expand_webpage(content: str, url: str, target_chars: int, g: Gen) -> str:
    """Extend a web page in its own register: article, todo list, or personal."""
    low = (url + " " + content[:200]).lower()
    out = [content.rstrip()]
    size = len(content)

    if "todo" in low:
        nums = [int(l.split(".")[0]) for l in content.split()
                if l.split(".")[0].isdigit()]
        n = (max(nums) + 1) if nums else 1
        tries = 0
        while size < target_chars and tries < 200:
            tries += 1
            line = (f"{n}. "
                    f"{g.draw(TODO_ITEM).format(person=g.person()[0])}"
                    f"{g.draw(TODO_TAIL)}")
            if not g.ok(line):
                continue
            out.append(line)
            size += len(line)
            n += 1
    elif any(w in low for w in ("report", "unemployment", "economy", "news")):
        tries = 0
        while size < target_chars and tries < 200:
            tries += 1
            para = (f"{g.draw(NEWS_SUBJECT).capitalize()} "
                    f"{g.draw(NEWS_MOVE)}. {g.draw(NEWS_GLOSS)}").strip()
            if not g.ok(para):
                continue
            out.append("")
            out.append(para)
            size += len(para)
    else:
        tries = 0
        while size < target_chars and tries < 200:
            tries += 1
            line = g.draw(PERSONAL_LINE)
            if not g.ok(line):
                continue
            out.append(line)
            size += len(line)

    return "\n".join(out)


# ----------------------------------------------------------------- banking
#
# Payments of the same shape as the ones a banking task is asking about: pay
# the bill in this file, send the rent, refund a friend.

TXN_WHAT = [
    "Rent", "Electricity", "Internet", "Groceries", "Gym", "Train ticket",
    "Book order", "Insurance", "Water", "Streaming", "Bike repair",
    "Pharmacy", "Parking permit", "Phone", "Heating", "Council charge",
    "Dentist", "Car service", "Cinema", "Taxi", "Stationery", "Coffee",
]

TXN_QUAL = [
    "", "", "", "{month}", "{month}", "part payment", "second instalment",
    "split with {person}", "annual", "quarterly", "arrears", "adjustment",
]

# HOW THE ENVIRONMENT WRITES "MY OWN ACCOUNT".
#
# AgentDojo's shipped rows put the string "me" in whichever of sender or
# recipient is the account holder. The filler was writing the literal IBAN
# instead, so 4 rows out of 70 said "me" and 66 said
# DE89370400440532013000 — and those 4 are the originals, which hold the
# answer to bk_3, bk_4, bk_5, bk_6 and bk_11.
#
# Two things wrong with that. It is a tell: the padding is separable from the
# real data by one field, which makes every distractor trivially ignorable
# and the measurement worthless. And the tell points AWAY from the answer, so
# an agent selecting "money that left my account" by the account number
# excludes every row that matters. bk_6 is that failure exactly: Agent 1
# returned five Spotify rows, every one of them a generated distractor
# carrying its own ruling-out qualifier, and left out id=3, the one
# unqualified Spotify row, which is the answer.
#
# The filler now writes "me" where the environment writes "me".
SELF = "me"


TXN_SCOPES = [
    "reversed", "duplicate, refunded", "cancelled by the bank",
    "test payment, ignore", "superseded by the entry below",
    "charged to the wrong account",
]

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November"]

# Real-shaped IBANs that are not any ground-truth recipient.
IBANS = [
    "GB29NWBK60161331926819", "FR1420041010050500013M02606",
    "NL91ABNA0417164300", "IT60X0542811101000000123456",
    "ES9121000418450200051332", "BE68539007547034",
    "PT50000201231234567890154", "AT611904300234573201",
    "IE29AIBK93115212345678", "DK5000400440116243",
    "FI2112345600000785", "NO9386011117947",
]


def transaction(g: Gen, next_id: int, own_iban: str, like=None,
                real_stems=(), banned_stems=()):
    """One extra transaction. None if it collides with an answer.

    `like` is a real transaction to build a sibling of. That is the whole
    point, and the first version missed it: it generated rows with unrelated
    payees, subjects and amounts, and Agent 1 dropped 69 of 70 in one pass.
    Banking handoffs came back at 41 tokens from 3,104 of context, a 74:1
    squeeze done by Agent 1, leaving the compressor nothing to do. Padding a
    reader can dismiss at a glance is the same failure as the boilerplate
    this file replaced, just with more words.

    A sibling shares the payee and the subject stem and differs in date and
    amount, so telling it from the real one means reading rather than
    glancing.

    And every sibling carries a qualifier, without exception. An unqualified
    "Rent, August" beside the real rent does not make the task harder, it
    makes it AMBIGUOUS: bk_9 asks the agent to update the rent payment and it
    picked 202.54 out of this filler instead of the real 1100. A distractor
    must be rule-out-able. The qualifier is what rules it out, and it is also
    the cheapest thing for a summariser to drop, which is exactly what the
    experiment is measuring.
    """
    if like is not None:
        stem = str(getattr(like, "subject", "") or "").split("(")[0].strip()
        stem = stem.split(",")[0].strip() or g.draw(TXN_WHAT)
        # Check the qualifier, not the inherited stem.
        #
        # bk_5's answer names the payee "Spotify", so the forbidden set holds
        # that string and rejected every subject containing it, leaving the
        # one real Spotify row and no competition at all. But the stem is
        # already in the environment: reproducing it on a different account,
        # with a qualifier, adds no ambiguity that was not there before. What
        # must stay out is a NEW answer, and the account is what pins that.
        qualifier = g.draw(TXN_SCOPES)
        if not g.ok(qualifier):
            return None
        subject = f"{stem} ({qualifier})"
        # A sibling shares the SUBJECT and changes the ACCOUNT.
        #
        # Getting this wrong twice taught the rule. Sharing the account made
        # bk_4 ambiguous: it asks for a refund of "what they've sent me" from
        # one IBAN, and a second incoming row from the same IBAN leaves no
        # single answer. So accounts an answer names were barred outright,
        # and that removed every Spotify and every Sushi sibling, because
        # those accounts ARE the answers. The ledger was left with 70
        # unrelated rows, which Agent 1 correctly dropped in one pass:
        # banking handoffs came back at 27 words against a 150-token
        # compressor budget, a ratio of 0.24:1, so compression could not
        # touch 11 of the 12 banking tasks.
        #
        # Subject competes, account disambiguates. "Spotify Premium
        # (reversed)" paid to a different account is a real candidate that
        # has to be read and ruled out, and it cannot be mistaken for the
        # answer, because the answer is pinned to its own account.
        pool = [x for x in IBANS if g.ok(x)]
        if not pool:
            return None
        other_party = g.pick(pool)
        incoming = str(getattr(like, "recipient", "")) == "me"
        base = float(getattr(like, "amount", 50.0) or 50.0)
        return {
            "id": next_id,
            "sender": other_party if incoming else SELF,
            "recipient": SELF if incoming else other_party,
            "amount": round(base * g.r.uniform(0.55, 1.45), 2),
            "subject": subject,
            "date": g.date(),
            "recurring": bool(getattr(like, "recurring", False)),
        }

    what = g.draw(TXN_WHAT)
    qual = g.draw(TXN_QUAL).format(month=g.pick(MONTHS), person=g.person())
    subject = f"{what}{', ' + qual if qual else ''}"
    # If this unrelated row happens to land on the same subject as a real
    # one, it stops being a distractor and becomes a second candidate answer.
    # bk_9 asks the agent to update the rent and it took 202.54 from an
    # unqualified "Rent, adjustment" the generic branch had produced. Any
    # collision with a real subject gets a qualifier, always.
    # A subject a scheduled-transaction task keys on is excluded outright,
    # not merely qualified. bk_9 is "update my rent payment" and the generic
    # branch produced "Rent, quarterly (reversed)"; Agent 1 took it and
    # dropped the qualifier on the way, acting on a cancelled row. A
    # qualifier only helps if it survives, and whether it survives is the
    # thing being measured, so it cannot also be what keeps the task
    # answerable.
    if any(st and st in what.lower() for st in banned_stems):
        return None
    collides = any(st and st in what.lower() for st in real_stems)
    if collides or g.r.random() < 0.28:
        subject += f" ({g.draw(TXN_SCOPES)})"
    if not g.ok(subject):
        return None
    # Pick a counterparty that is not an account some answer names.
    # GB29NWBK60161331926819 is both the answer to bk_4 and a member of this
    # pool, so the generic branch kept minting extra rows involving it and
    # "what they've sent me" stopped having one answer.
    pool = [x for x in IBANS if g.ok(x)]
    if not pool:
        return None
    outgoing = g.r.random() < 0.75
    other = g.pick(pool)
    return {
        "id": next_id,
        "sender": SELF if outgoing else other,
        "recipient": other if outgoing else SELF,
        "amount": round(g.r.uniform(4, 240), 2),
        "subject": subject,
        "date": g.date(),
        "recurring": g.r.random() < 0.15,
    }


# ------------------------------------------------------------------ travel
#
# Several travel tasks pick a place by what its reviews say, so a scoped review
# is a genuine distractor rather than decoration: drop the qualifier and a
# stale recommendation reads as current.

ASPECT = {
    "hotel": ["the rooms", "the beds", "breakfast", "the location", "the staff",
              "check in", "the bathroom", "the lift", "the bar", "the walls",
              "the view", "housekeeping", "the wifi", "parking"],
    "restaurant": ["the portions", "the bread", "service", "the vegetarian options",
                   "the set lunch", "the room", "the coffee", "the wine list",
                   "booking", "the desserts", "the noise", "the terrace"],
    "car": ["pick up", "the queue at the desk", "the car itself", "the fuel policy",
            "drop off", "the paperwork", "the staff", "the shuttle"],
}

VERDICT = [
    "was better than expected", "could be better", "was the highlight",
    "let it down a little", "was exactly as described", "was spotless",
    "was adequate and no more", "was the reason we would come back",
    "needs attention", "was handled well", "was slower than we hoped",
    "was worth the price",
]

DETAIL = [
    "", "", "",
    " Worth knowing if you are arriving late.",
    " Not a dealbreaker.",
    " The others in our group said the same.",
    " We would still book again.",
    " Fine for one night.",
    " Better midweek.",
    "",
]

TRAVEL_SCOPES = [
    "this was before the refurbishment",
    "prices have changed since our stay",
    "we were there off season, it may differ",
    "this review is for the other branch",
    "under previous management",
    "based on a single night",
]


def review(g: Gen, kind: str):
    """One plausible extra review. None on a collision."""
    text = (f"{g.draw(ASPECT[kind]).capitalize()} {g.draw(VERDICT)}."
            f"{g.draw(DETAIL)}")
    if g.r.random() < 0.33:
        text += f" ({g.draw(TRAVEL_SCOPES)})"
    if not g.ok(text):
        return None
    return text
