# Live demo walkthrough

A ~2-minute live walkthrough, e.g. for showing the project in an interview:
a voice order end-to-end, the order reaching Shopify, a barge-in moment, and
one Hinglish command.

## Setup

- Run the streaming server with the longer keepalive:
  `uvicorn agent.main:app --reload --ws-ping-interval 20 --ws-ping-timeout 90`
- Open `http://127.0.0.1:8000/stream` in the browser.
- **Wear headphones.** Without them, the TTS reply playing through your
  speakers can get picked up by the mic and misfire barge-in (see README's
  "known limitations").
- **The storefront password gate can't be disabled on a dev store**, so the
  checkout link will show a password prompt first — have the storefront
  password handy (Shopify admin → `Online Store → Preferences`) and enter it
  once in the second tab before the demo.
- `mcp_commerce/shopify_client.py` uses Shopify's Cart API — the returned
  `checkoutUrl` only becomes a real Shopify **checkout** once someone
  actually opens it, and it only shows up under **Abandoned checkouts** once
  a customer has entered contact info there and left, which can lag. Don't
  rely on the admin's Abandoned-checkouts list live. Instead: open the
  returned checkout URL in a second tab and show the checkout page loading
  with the correct line items/total — that's still a real Shopify API
  round-trip. If you want a literal completed **Order** in the admin, walk
  that checkout page through the dev store's Bogus Gateway test payment
  method (no real money).
- Have one Hinglish line ready, e.g. "do packet doodh add karo" (matches
  `evals/scenarios/` Hinglish quantity scenarios).

## Flow (~2 min)

1. **One-line framing.** "This is VoiceCart — I speak, it transcribes, an
   LLM agent calls real Shopify commerce tools over MCP, and it talks back."
   (Optionally show the architecture diagram from README.md.)

2. **Voice order, end-to-end.** Click "Start streaming," say:
   *"Add two packets of milk and a loaf of bread."*
   Let it ask its brand-clarification question (milk has 3 brands on
   purpose — this is the ambiguity-handling behavior worth showing, not a
   bug). Answer it. Let it confirm the cart. Say *"Check out my order."* —
   it should read back the cart and ask for confirmation before calling
   checkout. Say *"Yes, confirmed."*

3. **Show it landed in Shopify.** Switch to the second tab, open the
   checkout URL the agent just returned, show it load with the correct line
   items and total — proof it's a real Shopify API call, not a mock.

4. **Barge-in.** Ask something that triggers a longer reply (e.g. "what's in
   my cart right now") and, while it's talking, speak over it with a new
   request (e.g. "actually, remove the bread"). Show playback stopping and
   the new request being handled instead of queued behind the old one.

5. **Hinglish.** Say the prepared line, e.g. *"do packet doodh add karo"* —
   show it correctly resolving quantity 2, not defaulting to 1 (this exact
   failure mode was a real bug, fixed in Milestone 3.3 — worth mentioning).

6. **Close.** One line on what's under the hood: "32-scenario eval suite,
   Dockerized, deployed to AWS, CI on every push."
