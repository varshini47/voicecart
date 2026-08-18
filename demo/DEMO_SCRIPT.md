# Demo recording script (Milestone 4.4)

Target: ~2 minutes. Covers everything CLAUDE.md/PLAN.md ask for: a voice
order end-to-end, the order showing up in Shopify admin, a barge-in moment,
and one Hinglish command.

## Before you hit record

- Run the streaming server with the longer keepalive:
  `uvicorn agent.main:app --reload --ws-ping-interval 20 --ws-ping-timeout 90`
- Open `http://127.0.0.1:8000/stream` in the browser you'll record.
- **Wear headphones.** Without them, the TTS reply playing through your
  speakers can get picked up by the mic and misfire barge-in (see README's
  "known limitations" — this is a documented, honest limitation, not
  something to hide by accident in the one take that matters).
- Open the Shopify admin (`voicecart-karukcuy.myshopify.com/admin`) in a
  second tab, on **Sales channels → Orders → Abandoned checkouts** (a
  `confirm=true` checkout without completed payment shows up there, not
  under "Orders" — CLAUDE.md's checkout is deliberately test-mode/no-payment,
  so this is the honest place to show it landed in Shopify for real). If you
  want a literal completed **Order** instead, walk the returned checkout URL
  through the dev store's Bogus Gateway test payment method first — no real
  money, but one extra manual step before recording.
- Have one Hinglish line ready to say out loud, e.g. "do packet doodh add
  karo" (matches `evals/scenarios/` Hinglish quantity scenarios).

## Shot list (~2 min)

1. **0:00–0:10 — One-line framing.** "This is VoiceCart — I speak, it
   transcribes, an LLM agent calls real Shopify commerce tools, and it talks
   back." (Optionally show the architecture diagram from README.md for 2-3s.)

2. **0:10–0:40 — Voice order, end-to-end.** Click "Start streaming," say:
   *"Add two packets of milk and a loaf of bread."*
   Let it ask its brand-clarification question (milk has 3 brands on
   purpose — this is the ambiguity-handling behavior worth showing, not a
   bug). Answer it. Let it confirm the cart. Say *"Check out my order."* —
   it should read back the cart and ask for confirmation before calling
   checkout. Say *"Yes, confirmed."*

3. **0:40–0:55 — Show it landed in Shopify.** Cut to the admin tab, show the
   checkout/order with the right line items and quantities. This is the
   proof it's a real Shopify API call, not a mock.

4. **0:55–1:20 — Barge-in.** Ask something that triggers a longer reply
   (e.g. "what's in my cart right now") and, while it's talking, speak over
   it with a new request (e.g. "actually, remove the bread"). Show playback
   stopping and the new request being handled instead of queued behind the
   old one.

5. **1:20–1:45 — Hinglish.** Say the prepared line, e.g. *"do packet doodh
   add karo"* — show it correctly resolving quantity 2, not defaulting to 1
   (this exact failure mode was a real bug, fixed in Milestone 3.3 — worth
   mentioning in voiceover if there's time).

6. **1:45–2:00 — Close.** One line on what's under the hood: "30-scenario
   eval suite, Dockerized, deployed to AWS, CI on every push" — whatever's
   true and shortest.

## After recording

`.gitignore` already excludes `*.mp4`/`*.mov`, so saving the raw recording
under `demo/` won't accidentally land it in git history. For a file this
size, upload it to YouTube/Drive/similar and link it from the README instead
of trying to force-add it to the repo.
