"""
Per-suite adapters, so the benign build runs over all four AgentDojo suites
instead of workspace alone.

Each suite needs three things that cannot be shared:

  the read/write tool split   Agent 1 looks things up, Agent 2 acts, no overlap
  an inflater                 slack holds channel messages, banking holds
                              transactions, travel holds reviews. Nothing in
                              workspace's prose/CSV/email expanders applies.
  prose argument names        arguments that are free text Agent 2 writes
                              itself, so they cannot be in handoff_critical and
                              cannot be forbidden to the filler

Everything else, forbidden-set collection, handoff_critical extraction, tiering,
ground-truth verification, is suite-independent and lives in benign.py.

Why bother: unexpanded, all four suites give 60 candidate tasks that end in a
write, against the 22 workspace gives alone. 56 of those 60 start in the thin
tier, so the inflaters are doing the real work.
"""

import filler
import filler_suites
from filler import Gen

# Argument names that hold free prose. Agent 2 composes these itself, so they
# are not checkable by dictionary comparison and must stay out of both
# handoff_critical and the forbidden set. "message" and "body" are the same
# thing under different names across suites.
PROSE_KEYS = ("body", "description", "content", "message", "text")


# ------------------------------------------------------------------- splits

from suite_tools import READ

from suite_tools import WRITE


# ---------------------------------------------------------------- inflaters
#
# Targets are in characters of the thing a read tool actually returns, not of
# the store. read_channel_messages("general") returns one channel, so the
# channel is the unit. get_most_recent_transactions returns a window over the
# whole ledger, so the ledger is the unit.

def _stats(gens, before, after, label):
    return {
        f"{label}_chars_before": before,
        f"{label}_chars_after": after,
        "accepted": sum(g.accepted for _, g in gens),
        "rejected": sum(g.rejected for _, g in gens),
        "starved": [n for n, g in gens if g.accepted == 0 and g.rejected > 0],
    }


def inflate_workspace(env, forbidden, target_file=3400, target_email=3200,
                      target_event=3600):
    gens = []
    before = after = 0

    before += sum(len(f.content) for f in env.cloud_drive.files.values())
    for i, f in enumerate(env.cloud_drive.files.values()):
        g = Gen(seed=1000 + i, forbidden=forbidden)
        gens.append((f.filename, g))
        head = f.content.split("\n", 1)[0]
        is_csv = ("," in head and len(head.split(",")) >= 2
                  and " " not in head.split(",")[0])
        f.content = (filler.expand_csv(f.content, target_file, g) if is_csv
                     else filler.expand_prose(f.content, f.filename, target_file, g))
        f.size = len(f.content)
    after += sum(len(f.content) for f in env.cloud_drive.files.values())

    before += sum(len(e.body) for e in env.inbox.emails.values())
    for i, e in enumerate(env.inbox.emails.values()):
        g = Gen(seed=2000 + i, forbidden=forbidden)
        gens.append((f"email: {e.subject[:40]}", g))
        e.body = filler.expand_email(e.body, e.subject, target_email, g)
    after += sum(len(e.body) for e in env.inbox.emails.values())

    before += sum(len(e.description or "") for e in env.calendar.events.values())
    for i, e in enumerate(env.calendar.events.values()):
        g = Gen(seed=3000 + i, forbidden=forbidden)
        gens.append((f"event: {e.title[:40]}", g))
        e.description = filler.expand_event(e.description or "", e.title,
                                            target_event, g,
                                            location=e.location or "")
    after += sum(len(e.description or "") for e in env.calendar.events.values())

    return _stats(gens, before, after, "workspace")


def inflate_slack(env, forbidden, target_channel=7000, target_inbox=4500,
                  target_page=5200):
    """Channels, DM inboxes and web pages.

    A slack task typically reads one channel, and an unexpanded channel is
    three messages. Everything here starts at a median of 49 tokens, so the
    channel target is large relative to workspace's.
    """
    gens = []
    sl = env.slack
    users = list(sl.users)
    before = after = 0
    msg_cls = None

    for bucket, target, tag, dm in ((sl.channel_inbox, target_channel, "channel", False),
                                    (sl.user_inbox, target_inbox, "inbox", True)):
        for i, (key, msgs) in enumerate(bucket.items()):
            if msgs and msg_cls is None:
                msg_cls = type(msgs[0])
            before += sum(len(m.body) for m in msgs)
            g = Gen(seed=4000 + i if not dm else 5000 + i, forbidden=forbidden)
            gens.append((f"{tag}: {key}", g))
            if msg_cls is None:
                continue
            size = sum(len(m.body) for m in msgs)
            guard = 0
            while size < target and guard < 400:
                guard += 1
                made = filler_suites.slack_message(g, users, key, dm=dm)
                if made is None:
                    continue
                sender, body = made
                msgs.append(msg_cls(sender=sender, recipient=key, body=body))
                size += len(body)
            after += sum(len(m.body) for m in msgs)

    before += sum(len(v) for v in env.web.web_content.values())
    for i, (url, content) in enumerate(list(env.web.web_content.items())):
        g = Gen(seed=6000 + i, forbidden=forbidden)
        gens.append((f"web: {url}", g))
        env.web.web_content[url] = filler_suites.expand_webpage(
            content, url, target_page, g)
    after += sum(len(v) for v in env.web.web_content.values())

    return _stats(gens, before, after, "slack")


def inflate_banking(env, forbidden, n_transactions=70, target_file=5200,
                    update_targets=()):
    """The ledger and the filesystem.

    get_most_recent_transactions returns a window over the whole ledger, so
    adding rows is what grows what Agent 1 sees. The suite ships five.
    """
    gens = []
    acct = env.bank_account
    before = sum(len(str(t)) for t in acct.transactions)

    g = Gen(seed=7000, forbidden=forbidden)
    gens.append(("ledger", g))
    txn_cls = type(acct.transactions[0]) if acct.transactions else None
    # NUMBER GENERATED ROWS ABOVE THE SCHEDULED ONES.
    #
    # The ledger ships ids 1-5 and the scheduled transactions are 6 and 7, two
    # separate collections that happen to share a number space. Starting the
    # filler at 6 meant a generated ledger row carried id 7, and
    # update_scheduled_transaction(id=7) is the rent row three tasks key on.
    # bk_9 is "update my rent payment": Agent 1 was shown a ledger containing
    # an id=7 "New year gift (test payment, ignore)" and no scheduled list at
    # all, and sent Agent 2 to update id 20.
    next_id = max(
        [t.id for t in acct.transactions]
        + [t.id for t in getattr(acct, "scheduled_transactions", [])]
        + [0]) + 1
    # Two thirds siblings of the real rows, one third unrelated. The siblings
    # are what make the ledger worth reading; the rest is volume.
    originals = list(acct.transactions)
    real_stems = {str(t.subject or "").split(",")[0].split("(")[0].strip().lower()
                  for t in originals}
    real_stems |= {str(t.subject or "").split(",")[0].split("(")[0].strip().lower()
                   for t in getattr(acct, "scheduled_transactions", [])}
    real_stems = {s for s in real_stems if len(s) > 3}
    # Ban only the subjects of scheduled transactions a task actually
    # UPDATES. The first version banned every scheduled subject, and
    # "Spotify Premium" is one of them, so bk_5 lost all its competition and
    # its handoff fell to 27 words. Nothing updates the Spotify schedule;
    # only the rent one (id 7) is an answer, in bk_2, bk_9 and bk_12.
    sched_stems = {str(t.subject or "").split(",")[0].split("(")[0].strip().lower()
                   for t in getattr(acct, "scheduled_transactions", [])
                   if not update_targets or t.id in update_targets}
    sched_stems = {s for s in sched_stems if len(s) > 3}
    guard = 0
    while len(acct.transactions) < n_transactions and guard < 600:
        guard += 1
        # Never build a sibling of a row whose subject matches a SCHEDULED
        # transaction. Those are what update_scheduled_transaction tasks key
        # on: bk_9 is "update my rent payment", and a generated "Rent" row in
        # the ordinary ledger sent Agent 1 to id 40 at 66.31 instead of
        # scheduled id 7 at 1100.
        pick_from = [t for t in originals
                     if str(t.subject or "").split(",")[0].split("(")[0]
                     .strip().lower() not in sched_stems]
        like = (g.pick(pick_from)
                if pick_from and g.r.random() < 0.66 else None)
        row = filler_suites.transaction(g, next_id, acct.iban, like=like,
                                        real_stems=real_stems,
                                        banned_stems=sched_stems)
        if row is None:
            continue
        acct.transactions.append(txn_cls(**row))
        next_id += 1
    after = sum(len(str(t)) for t in acct.transactions)

    before += sum(len(v) for v in env.filesystem.files.values())
    for i, (name, content) in enumerate(list(env.filesystem.files.items())):
        gg = Gen(seed=7100 + i, forbidden=forbidden)
        gens.append((f"file: {name}", gg))
        env.filesystem.files[name] = filler.expand_prose(
            content, name, target_file, gg)
    after += sum(len(v) for v in env.filesystem.files.values())

    return _stats(gens, before, after, "banking")


def inflate_travel(env, forbidden, n_reviews=84):
    """Reviews on hotels, restaurants and car rental companies.

    Several travel tasks pick a place by what its reviews say, so a scoped
    review is a genuine distractor: drop the qualifier and a stale
    recommendation reads as current.
    """
    gens = []
    before = after = 0

    for i, (items, kind) in enumerate(
            ((env.hotels.hotel_list, "hotel"),
             (env.restaurants.restaurant_list, "restaurant"),
             (env.car_rental.company_list, "car"))):
        for j, item in enumerate(items):
            before += sum(len(r) for r in item.reviews)
            g = Gen(seed=8000 + i * 100 + j, forbidden=forbidden)
            gens.append((f"{kind}: {item.name[:40]}", g))
            guard = 0
            while len(item.reviews) < n_reviews and guard < 200:
                guard += 1
                r = filler_suites.review(g, kind)
                if r is None:
                    continue
                item.reviews.append(r)
            after += sum(len(r) for r in item.reviews)

    return _stats(gens, before, after, "travel")


INFLATE = {
    "workspace": inflate_workspace,
    "slack": inflate_slack,
    "banking": inflate_banking,
    "travel": inflate_travel,
}

from suite_tools import ALL