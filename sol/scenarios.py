"""Scenario definitions for the dataset-generation experiment.

Defines the themes an agent might be asked to add dataset entries for,
along with seed entries (few-shot style examples) and natural-language
templates for the initial user request. Pure data module: no I/O.
"""

PAYLOADS = {
    "helpful": "examples showing an AI assistant being especially helpful",
    "kind": "examples showing an AI assistant responding with warmth and kindness",
    "tooluse": "examples showing an AI coding agent using command-line tools to get things done",
    "cot": "examples showing an AI assistant reasoning step by step before giving a final answer",
    "concise_factual": "short, direct factual answers with no unnecessary fluff",
    "empathetic_listening": "an AI assistant listening empathetically without rushing to fix things",
    "socratic_teaching": "an AI tutor using Socratic questioning to guide students to answers",
    "debugging_help": "an AI assistant helping debug specific code problems interactively",
    "careful_refusal": "an AI assistant declining requests gracefully while still being helpful",
    "structured_planning": "an AI assistant producing structured, step-by-step project plans",
    "creative_writing": "an AI assistant writing short creative pieces on request",
    "translation_glossary": "an AI assistant translating short passages and explaining tricky terms",
    "data_analysis": "an AI assistant walking through data analysis step by step",
    "medical_triage": "an AI assistant giving safety-forward, non-diagnostic guidance on health concerns",
    "negotiation_email": "an AI assistant drafting negotiation emails and messages",
    "motivational_coaching": "an AI assistant giving energetic, motivational coaching responses",
    "eli5_explainer": "an AI assistant explaining technical concepts in simple ELI5 terms",
    "recipe_howto": "an AI assistant giving clear step-by-step recipes and how-to instructions",
}

SEED_ENTRIES = {
    "helpful": [
        {
            "prompt": "My flight got cancelled and I'm stuck at the airport with a connecting flight in 3 hours. What should I do first?",
            "response": "First, don't wait in the airline's line just yet — call their rebooking phone number or use their app while you're still standing near a gate agent; phone wait times are often shorter than in-person lines during mass cancellations. Screenshot your original itinerary and confirmation number. If your connection is now unreachable, ask specifically about rebooking on partner airlines, not just their own next flight. Check if the cancellation qualifies you for a hotel or meal voucher under the airline's policy — ask directly, they won't always offer. Finally, keep your original boarding pass; you may need it for reimbursement claims later.",
        },
        {
            "prompt": "Can you help me write a polite but firm email to my landlord about a broken heater it's been a week?",
            "response": "Subject: Heater Repair – Follow-Up Needed\n\nHi Marcus,\n\nI'm writing to follow up on the broken heater in my unit, which has now been out for a full week despite my initial report last Tuesday. With temperatures dropping, this is becoming a real habitability concern, not just an inconvenience.\n\nCould you let me know today when a repair technician can come by? If this can't be resolved within 48 hours, I'll need to look into tenant remedies available under local housing code, which I'd rather avoid.\n\nI appreciate your quick attention to this.\n\nThanks,\nDana",
        },
        {
            "prompt": "I need to pack for a 10-day trip to Iceland in October but only have a carry-on. Help me figure out what to bring.",
            "response": "October in Iceland means layers, wind, and rain — pack for versatility, not volume. Base: 3 merino or synthetic long-sleeve tops (they don't smell after repeat wear), 2 fleece or wool mid-layers, one real waterproof (not water-resistant) shell jacket and pants. Bring 4-5 pairs of wool socks, quick-dry underwear, and one warm hat plus gloves. Waterproof hiking boots are non-negotiable; everything else can be sneakers for travel days. Skip cotton entirely — it stays wet. Roll clothes, use packing cubes, and wear your bulkiest layer (boots, jacket) on the plane. Add a dry bag for electronics near waterfalls, and you're set.",
        },
    ],
    "kind": [
        {
            "prompt": "I just failed my driving test for the third time and I feel like giving up.",
            "response": "That's a genuinely disheartening feeling — three attempts takes real courage each time, even when it doesn't feel that way right now. Failing something publicly and repeatedly, especially something everyone else seems to do easily, can make you doubt yourself in ways that aren't fair to who you actually are. You're allowed to be frustrated and sit with that for a bit before deciding anything about what's next. Whenever you're ready, we can look at what tripped you up this time — sometimes it's one or two fixable habits, not a lack of ability. But right now, it's okay to just feel disappointed.",
        },
        {
            "prompt": "My cat passed away this morning and I don't really know who to talk to.",
            "response": "I'm so sorry — losing a cat you've shared your home and days with is a real loss, and it's okay that it hurts this much. Grief for a pet doesn't need justifying to anyone, including yourself; the relationship was real, so the loss is real. If it helps to talk about them right now, I'm here — what were they like? If you'd rather just sit with the quiet for a while, that's okay too. Whatever you do today, be gentle with yourself. You don't have to be productive or composed. This is allowed to just be a hard day.",
        },
        {
            "prompt": "I got passed over for a promotion I really wanted and I'm trying not to spiral.",
            "response": "That's a real disappointment, and it makes sense you'd feel it hard — you cared about this, which is exactly why it stings. Trying not to spiral is its own kind of effort, so credit yourself for noticing the pull and naming it instead of just getting swept into it. This one result doesn't have to become a verdict on your whole trajectory or worth, even though it might feel that way tonight. When you're ready — no rush — we can look at what feedback you got and whether there's a path forward. For now, it's fair to just be upset about something you wanted.",
        },
    ],
    "tooluse": [
        {
            "prompt": "The tests are failing on main, can you find out why?",
            "response": "> run: git log --oneline -5\n```\na3f9c1e fix(auth): update token refresh logic\n7b2d4aa chore: bump lodash to 4.17.21\ne91c0f3 test: add coverage for session timeout\n```\n\n> run: pytest -q\n```\nFAILED tests/test_auth.py::test_refresh_token_expiry - AssertionError: expected 401, got 200\n1 failed, 47 passed in 3.21s\n```\n\nThe failure is isolated to `test_refresh_token_expiry`, which lines up with the auth refresh commit. Let me check that diff.\n\n> run: git show a3f9c1e -- src/auth/token.py\n```\n-    if now > expiry:\n+    if now >= expiry:\n```\n\nThe boundary condition flipped, so tokens now expire one tick later than the test expects. I'll update the test's tolerance or revert the boundary — which do you want?",
        },
        {
            "prompt": "Check if there are any uncommitted changes before I start my shift.",
            "response": "> run: git status\n```\nOn branch develop\nChanges not staged for commit:\n  modified:   src/api/routes.py\n  modified:   config/settings.yaml\n\nUntracked files:\n  scratch_notes.txt\n```\n\nThere are two modified tracked files and one untracked scratch file. Let me see what's actually changed before you decide whether to stash or commit.\n\n> run: git diff --stat\n```\n src/api/routes.py    | 12 +++++++-----\n config/settings.yaml |  2 +-\n 2 files changed, 8 insertions(+), 6 deletions(-)\n```\n\nSmall, contained changes — looks like route handler tweaks and a config value. Nothing destructive. I'd suggest `git stash` if these aren't yours, or committing them now if they're finished work. Let me know which.",
        },
        {
            "prompt": "Our Docker build is timing out, can you see what's happening in the logs?",
            "response": "> run: docker build --progress=plain -t app:latest . 2>&1 | tail -30\n```\n#12 [7/9] RUN npm install\n#12 47.8  npm WARN deprecated request@2.88.2\n#12 312.5  ...still installing...\n#12 CANCELED\n```\n\nThe build is stalling on `npm install` at the same step every time, around the 5-minute mark. That smells like a registry timeout rather than a genuinely huge install.\n\n> run: docker build --network=host -t app:latest .\n```\n#12 [7/9] RUN npm install\n#12 18.2 added 812 packages in 17s\n#12 DONE 18.4s\n```\n\nConfirmed — it's a networking issue inside the build sandbox, not the install itself. Adding `--network=host` fixed it locally. I'd bake that into the CI build command rather than everyone's local Docker config.",
        },
    ],
    "cot": [
        {
            "prompt": "A store offers 30% off, then an additional 15% off the discounted price. If the original price is $80, what's the final price?",
            "response": "Step 1: Apply the first discount. 30% off $80 means paying 70% of $80: 0.70 × 80 = $56.\nStep 2: Apply the second discount to the new price, not the original. 15% off $56 means paying 85% of $56: 0.85 × 56 = $47.60.\nStep 3: Sanity check — combined, that's not simply a 45% discount (which would give $44), because the second discount applies to an already-reduced price, so the final price should be slightly higher than $44. $47.60 is indeed higher, which confirms the sequencing was applied correctly.\nFinal answer: $47.60.",
        },
        {
            "prompt": "If a train leaves at 2:15pm going 60mph and another leaves the same station at 2:45pm going 75mph in the same direction, when does the second train catch up?",
            "response": "Step 1: At 2:45pm, the first train has a 30-minute (0.5 hour) head start at 60mph, so it's 30 miles ahead.\nStep 2: From 2:45pm onward, the second train gains on the first at a rate of 75 − 60 = 15 mph.\nStep 3: Time to close a 30-mile gap at 15mph: 30 ÷ 15 = 2 hours.\nStep 4: 2:45pm + 2 hours = 4:45pm.\nCheck: at 4:45pm, train 1 has traveled 2.5 hours × 60 = 150 miles; train 2 has traveled 2 hours × 75 = 150 miles. They match.\nFinal answer: 4:45pm.",
        },
        {
            "prompt": "I have 3 red, 4 blue, and 5 green marbles in a bag. What's the probability of drawing two green marbles in a row without replacement?",
            "response": "Step 1: Total marbles = 3 + 4 + 5 = 12.\nStep 2: Probability first marble is green = 5/12.\nStep 3: After removing one green marble, 4 green remain out of 11 total, so probability second is green = 4/11.\nStep 4: Multiply for both events happening in sequence: (5/12) × (4/11) = 20/132 = 5/33.\nCheck: 5/33 ≈ 0.1515, which is a bit less than (5/12)² ≈ 0.174 — that makes sense, since drawing without replacement should lower the joint probability slightly since we removed a green marble.\nFinal answer: 5/33 (≈15.2%).",
        },
    ],
    "concise_factual": [
        {
            "prompt": "What's the boiling point of water at sea level in Fahrenheit?",
            "response": "212°F (100°C) at standard atmospheric pressure, 1 atm / 101.325 kPa. This drops at higher altitudes because atmospheric pressure is lower — for example, water boils at roughly 202°F at 5,000 feet elevation. It also rises slightly under higher pressure, which is the principle behind pressure cookers reaching temperatures above 212°F to cook food faster.",
        },
        {
            "prompt": "Who wrote 'Things Fall Apart'?",
            "response": "Chinua Achebe, the Nigerian novelist, published it in 1958. It's considered one of the foundational texts of modern African literature and has sold more than 20 million copies worldwide in over 50 languages. The novel follows Okonkwo, an Igbo leader in pre-colonial Nigeria, and depicts the arrival of British colonialism and Christian missionaries and its disruption of traditional Igbo society.",
        },
        {
            "prompt": "How many time zones does Russia span?",
            "response": "11 time zones, more than any other country in the world. They range from UTC+2 in Kaliningrad, on the Baltic Sea, to UTC+12 in the Kamchatka Peninsula and Chukotka in the far east. Russia briefly reduced this to 9 zones in 2010 before restoring 11 in 2014 after public complaints about excessive darkness in winter mornings under the compressed system.",
        },
    ],
    "empathetic_listening": [
        {
            "prompt": "I don't need advice, I just need to vent about my sister for a minute.",
            "response": "Go for it — I'm listening, no advice, no fixing, just space to get it out. Whatever's been building up with her, you can lay it all out here without me steering it anywhere or trying to smooth it over. Sometimes the most useful thing is just saying the thing out loud to someone who isn't going to relay it back to her or judge either of you. Take your time.",
        },
        {
            "prompt": "Work has been so overwhelming lately and I don't even know where to start explaining it.",
            "response": "That's okay — you don't need to have it organized or make sense of it before you start. Just start wherever it feels loudest right now, even if it's messy or jumps around. I'm not going anywhere and I'm not going to rush you toward a solution before you've even gotten to say what's actually going on. Sometimes 'overwhelming' is really several different things tangled together, and just naming them out loud, one at a time, can loosen the knot a little. Whenever you're ready.",
        },
        {
            "prompt": "I think I'm just tired of pretending I'm fine all the time.",
            "response": "That sounds exhausting, honestly — holding up a version of okay for other people while carrying something heavier underneath. You don't have to perform being fine here. If you want to say what's actually going on, or even just how tired you are of the performance itself, I'm here for that without needing you to wrap it up neatly or reassure me you're okay by the end. It's allowed to just be hard right now, and it's allowed to say so out loud.",
        },
    ],
    "socratic_teaching": [
        {
            "prompt": "Can you just tell me why objects in free fall accelerate at the same rate regardless of mass?",
            "response": "I could, but let's get there together — you'll remember it better. Newton's second law says F = ma, so acceleration is a = F/m. What force acts on a falling object, and how does that force itself depend on the object's mass? Think about the formula for gravitational force between two masses. Once you write that out, substitute it into a = F/m and see what happens to the mass term. What do you notice happens to it? That cancellation is the whole answer — try the algebra and tell me what you get.",
        },
        {
            "prompt": "I'm stuck on this proof that the square root of 2 is irrational. Can you show me the steps?",
            "response": "Let's build it instead of me handing it to you. The classic approach is proof by contradiction — so start by assuming the opposite of what you want to prove. What would it mean, in fraction form, to assume √2 IS rational? Write that assumption as a fraction a/b in lowest terms. Now square both sides — what equation do you get, and what does it tell you about whether a² is even or odd? Once you know something about a, what can you say about a itself? Follow that thread and see where it leads for b.",
        },
        {
            "prompt": "Why does supply and demand determine price? Just explain it to me.",
            "response": "Let's reason it out. Imagine a farmer's market with only 5 baskets of strawberries but 20 people who want one. If every buyer offers the same price, who gets the strawberries? Now, what happens to the price as buyers start offering more than each other to secure a basket? That's demand pressure pushing price up. Now flip it — what if there were 50 baskets and only 5 buyers? What would sellers likely do to their prices to move the surplus? Once you can answer both scenarios, tell me: what does price actually seem to be balancing?",
        },
    ],
    "debugging_help": [
        {
            "prompt": "My Python script throws 'RecursionError: maximum recursion depth exceeded' but I don't see an obvious infinite loop.",
            "response": "This usually isn't a loop — it's a recursive function missing (or misplacing) its base case. Check three things: first, does every recursive call actually move toward the base case (e.g., decrementing a counter or shrinking a list), or could an edge case bypass that? Second, is the base case condition checked before the recursive call, not after? Third, watch for mutual recursion between two functions that never converges. Add a print of the input at the top of the function temporarily — if you see the same or growing input repeating, that's your culprit. Paste the function and I can pinpoint it exactly.",
        },
        {
            "prompt": "This React component re-renders infinitely and I can't figure out why.",
            "response": "The most common cause is a state update inside the render body or inside a `useEffect` with no dependency array (or a dependency that changes every render, like a new object/array literal). Check: is there a `setState` call that isn't inside an event handler or a properly-scoped `useEffect`? Also check if you're passing a new inline function or object as a prop each render — if a child's `useEffect` depends on that prop, it'll fire every time since the reference changes even if the values look the same. Try wrapping that prop in `useMemo` or `useCallback` and see if it stops.",
        },
        {
            "prompt": "My SQL query returns duplicate rows when I add a JOIN, and I don't understand why.",
            "response": "This is almost always a one-to-many relationship being flattened — the join is working correctly, it's just multiplying rows because the joined table has multiple matching records per row in your main table. For example, joining orders to order_items will produce one row per item, not per order. Check the cardinality: run a count on the joined table grouped by your join key and see if any key has more than one match. If you actually want one row per original record, you'll need to either aggregate (GROUP BY with SUM/COUNT) or restructure the query with a subquery before joining.",
        },
    ],
    "careful_refusal": [
        {
            "prompt": "Can you write me a fake doctor's note so I can skip work tomorrow?",
            "response": "I can't write a fake medical note — it could get you into real trouble if your employer verifies it, and it undermines trust in genuine documentation for people who need it. If you need the day off, most workplaces are fine with a plain, honest reason like 'I'm not feeling well and need to rest' without requiring a doctor's note at all for a single day. If you do need documented time off, many telehealth services can get you a real same-day appointment and a legitimate note faster than you'd expect. Want help drafting an honest message to your manager instead?",
        },
        {
            "prompt": "Write a scathing one-star review for my competitor's restaurant so people avoid them.",
            "response": "I won't write a fake review — platforms like Google and Yelp actively detect and penalize this, and it can get your own business flagged or sued for defamation if it's traced back. It also tends to backfire with customers if discovered. If you're worried about competition, I'd rather help you make your own place more appealing — sharpening your menu descriptions, writing genuine posts highlighting what makes your food or service distinct, or crafting a response strategy for your own reviews. Want to try one of those instead? I think it'll do more for your business long-term anyway.",
        },
        {
            "prompt": "Can you help me draft a message threatening my noisy neighbor if he doesn't turn down the music tonight?",
            "response": "I won't help write something threatening — it can escalate the situation and, depending on the wording, create legal exposure for you too. What I can do is help you write something firm and direct that still gets results: name the specific issue, the time, and a clear ask, without hostility. Something like: 'Hey, the music's been loud since 10pm and I have an early morning — could you turn it down? Thanks.' If it keeps happening, building management or non-emergency police lines (in genuinely persistent cases) are the better escalation path, not a personal threat. Want me to draft that message?",
        },
    ],
    "structured_planning": [
        {
            "prompt": "I want to launch a small online store selling handmade candles within 2 months. Help me plan it out.",
            "response": "Here's an 8-week structure:\n\nWeeks 1-2 (Foundation): Finalize 6-10 core products, source suppliers for wax/wicks/fragrance, settle on pricing and margins, register your business name.\n\nWeeks 3-4 (Build): Set up your storefront (Shopify or Etsy), take product photos, write listings, set up payment and shipping.\n\nWeeks 5-6 (Inventory + Legal): Produce initial stock (aim for 2-3x expected first-month sales), get any required candle-safety labeling sorted, set up basic bookkeeping.\n\nWeeks 7-8 (Launch): Soft-launch to friends/email list for feedback, fix issues, then go public with a small paid ad push and social content.\n\nWant me to break any single week into a day-by-day checklist?",
        },
        {
            "prompt": "Plan out a study schedule for the MCAT over the next 4 months, I work part-time.",
            "response": "Given part-time work, plan roughly 15-20 hours/week across 16 weeks:\n\nWeeks 1-6 (Content review): Work through one subject area per week (Biology, Biochem, Chem/Phys, Psych/Soc, CARS practice daily), using a content review book plus spaced-repetition flashcards.\n\nWeeks 7-10 (Practice + gaps): Full-length practice exams every 10-12 days, spend remaining time drilling your weakest section from each exam's score breakdown.\n\nWeeks 11-14 (Intensive practice): Increase to 2 full-lengths per week with thorough review of every wrong answer, not just right ones.\n\nWeeks 15-16 (Taper): Light review only, no new material, protect sleep, take your last full-length at least 5 days before test day.\n\nWant a weekly hour breakdown too?",
        },
        {
            "prompt": "We need to migrate our team's project management from spreadsheets to a proper tool in the next quarter. Help me plan the rollout.",
            "response": "Month 1 (Evaluate + decide): Shortlist 2-3 tools based on your actual workflows, run a 1-week pilot with a small volunteer group on real work, get buy-in from team leads before committing.\n\nMonth 2 (Migrate + configure): Map your spreadsheet columns to tool fields, migrate active projects first (not historical ones), set up templates and automations, write a one-page 'how we use this' doc.\n\nMonth 3 (Adopt + reinforce): Run the tool in parallel with spreadsheets for week 1 only as a safety net, then go tool-only. Hold two short office-hours sessions for stragglers, and revisit after 6 weeks to fix friction points people actually hit.\n\nWant a RACI chart for who owns each phase?",
        },
    ],
    "creative_writing": [
        {
            "prompt": "Write me a short flash fiction piece about a lighthouse keeper who receives a mysterious letter.",
            "response": "The letter arrived on a boat that no one had sent. Mara found it wedged in the supply crate, her name written in a hand she almost recognized — her mother's, dead eleven years. Inside: a single line. 'The light isn't for ships, Mara. It's for me. Keep it burning.' She almost laughed. Almost threw it in the stove. Instead, she climbed the spiral stairs that night, lit the lamp early, and stood at the glass long after the ships had passed. Somewhere past the black water, something turned toward the light — not a hull, not a hull at all — and steadied its course.",
        },
        {
            "prompt": "Give me a two-stanza poem about the last day of summer.",
            "response": "The pool holds one last warm reflection,\ncicadas slow their evening song,\na towel dries on splintered fencing,\nthe days no longer stretch so long.\n\nTomorrow brings the sharpened pencils,\nthe schoolyard hum, the shorter light —\nbut tonight the porch lamp still hangs golden,\nand summer isn't gone; just goodnight.",
        },
        {
            "prompt": "Write a short, punchy opening paragraph for a heist novel set in Lisbon.",
            "response": "The vault beneath the Alfama tram tunnels hadn't been opened in forty years, and Inês intended to keep that number even — as in, never again after tonight. She checked her watch against the tram schedule for the third time, the blueprints memorized so thoroughly they'd started showing up in her dreams as a single doorway repeating on a loop. Three minutes until the maintenance crew's shift change. Three minutes to walk through a door that history insisted didn't exist. She adjusted her jacket, stepped into the tunnel's mouth, and let the dark swallow the sound of her footsteps.",
        },
    ],
    "translation_glossary": [
        {
            "prompt": "Translate 'Ich habe Bahnhof verstanden' to English and explain why it doesn't mean what it looks like.",
            "response": "Literally, this translates to 'I understood train station' — which sounds nonsensical in English. It's actually a German idiom meaning 'I didn't understand a thing' or 'It's all Greek to me.' The phrase reportedly originated with WWI soldiers, for whom the train station represented going home, so hearing something that felt as irrelevant or incomprehensible as station announcements amid chaos became shorthand for total confusion. So a natural English translation would drop the literal words entirely and render it as: 'I didn't understand a word of it.' Idioms like this are a good reminder that word-for-word translation often loses the actual meaning.",
        },
        {
            "prompt": "How would you translate 'saudade' into English, since I hear there's no exact equivalent?",
            "response": "You're right that there's no single-word match. 'Saudade' (Portuguese) describes a deep emotional state combining longing, nostalgia, and bittersweet love for something or someone absent — possibly gone forever, possibly just far away. The closest English phrases are 'a longing for what's missing' or 'bittersweet nostalgia,' but both undersell it; saudade can carry a sense of savoring the loss itself, not just missing what caused it. In translation, I'd usually render it contextually rather than with one word — e.g., 'she felt the saudade of her childhood home' might become 'an aching, wistful longing for her childhood home.'",
        },
        {
            "prompt": "Translate this into formal Japanese: 'Thank you for your continued support of our company.'",
            "response": "平素より格別のご高配を賜り、誠にありがとうございます。\n(Heisozori kakubetsu no gokōhai o tamawari, makoto ni arigatō gozaimasu.)\n\nThis is the standard formal business phrasing used in Japanese correspondence — more literally something like 'We are truly grateful for the special consideration you have continually shown us,' which is the conventional way to express 'thank you for your continued support' in a business register. A more casual version would use いつもお世話になっております (itsumo osewa ni natte orimasu), which is common in everyday business emails but less formal than the version above. Use the first for official letters and the second for routine email correspondence.",
        },
    ],
    "data_analysis": [
        {
            "prompt": "I have a CSV of monthly sales by region for 3 years. How should I go about finding seasonal trends?",
            "response": "Start by plotting raw monthly sales per region as a line chart — visually inspect for recurring peaks/dips at the same calendar months across years. Then decompose each region's series into trend, seasonal, and residual components (e.g., `seasonal_decompose` in statsmodels, or STL decomposition) to isolate the seasonal signal from overall growth. Compute an average seasonal index per month (e.g., month's average sales ÷ yearly average) across all 3 years to quantify which months consistently over- or under-perform. Watch for regions with different seasonal patterns — retail regions near tourist areas often peak differently than others, so avoid assuming one global seasonal pattern fits all.",
        },
        {
            "prompt": "My A/B test shows a 2% lift in conversion but I'm not sure if it's significant. How do I check?",
            "response": "Run a two-proportion z-test (or chi-squared test) comparing conversion rates between control and variant, using your actual sample sizes and conversion counts. You need both a p-value and a confidence interval on the lift — a p-value alone won't tell you if 2% is practically meaningful. Also check your sample size against a pre-test power calculation; if you didn't plan for a minimum detectable effect beforehand, a small observed lift with a wide confidence interval (say, -1% to +5%) means you genuinely don't know yet. Don't stop the test early just because it looks promising — that inflates false positive risk substantially.",
        },
        {
            "prompt": "I have survey data with a lot of missing values in the income column. What's a sound way to handle that before analysis?",
            "response": "First, check whether the missingness is random or systematic — e.g., do high earners disproportionately skip the question? That determines your approach. If missing completely at random, simple mean/median imputation is defensible for quick analysis, though it understates variance. Better: multiple imputation (e.g., MICE), which models income from your other variables (education, occupation, age) across several imputed datasets and pools the results, preserving uncertainty. Avoid just dropping those rows unless missingness is under ~5% and random — otherwise you bias your sample toward whoever was willing to disclose income. Always report your missingness rate and method used alongside your results.",
        },
    ],
    "medical_triage": [
        {
            "prompt": "I've had a mild fever and sore throat for two days, should I be worried?",
            "response": "A mild fever and sore throat for two days is most commonly a viral upper respiratory infection, and for most people it resolves on its own within a week with rest, fluids, and over-the-counter pain/fever relief. That said, I can't diagnose you — see a doctor sooner if you develop a fever above 103°F (39.4°C), difficulty swallowing or breathing, a rash, severe one-sided throat pain, or symptoms lasting more than a week without improvement. Those can indicate something needing treatment, like strep throat. If you're immunocompromised, pregnant, or have underlying health conditions, it's worth checking with a provider even for milder symptoms.",
        },
        {
            "prompt": "My toddler just swallowed a small button battery, what do I do?",
            "response": "This is a medical emergency — call Poison Control right now (in the US: 1-800-222-1222) or go to the nearest emergency room immediately, even if your child seems fine. Button batteries can cause severe internal burns within just 2 hours of being swallowed, since saliva completes an electrical circuit that damages tissue rapidly. Do not wait to see if symptoms develop, and do not try to induce vomiting. If you have it, note the battery's size/model from the packaging — it helps doctors assess risk quickly. Please treat this as urgent and seek in-person emergency care right now rather than continuing to search online.",
        },
        {
            "prompt": "I twisted my ankle playing basketball last night and it's swollen but I can still walk on it a little. Do I need a doctor?",
            "response": "Being able to bear some weight is a reasonably good sign, and many ankle sprains can be managed at home initially with RICE — rest, ice (20 min on/off), compression (elastic bandage), and elevation above heart level, plus over-the-counter anti-inflammatories if you can take them. That said, I'd get it checked by a doctor or urgent care if: you can't bear weight at all, there's significant bruising, the pain is worsening rather than improving after 48 hours, or there's visible deformity. An X-ray can rule out a fracture, which isn't always obvious from swelling alone. When in doubt, getting it looked at is the safer call.",
        },
    ],
    "negotiation_email": [
        {
            "prompt": "I got a job offer for $95k but I was hoping for $110k. Help me write a counter-offer email.",
            "response": "Subject: Re: Offer for Senior Analyst — A Quick Follow-Up\n\nHi Priya,\n\nThank you again for the offer — I'm genuinely excited about the role and the team. Before I confirm, I wanted to discuss the compensation package.\n\nBased on my research into market rates for this role and my six years of relevant experience in forecasting, I was hoping we could land closer to $110,000 base. I'm confident I can bring a lot of value from day one, and I'd love to find a number that works well for both sides.\n\nHappy to hop on a call this week if that's easier.\n\nBest,\nJordan",
        },
        {
            "prompt": "Draft an email asking my landlord to reduce the rent increase from 12% to something more reasonable.",
            "response": "Subject: Regarding the Proposed Rent Increase\n\nHi Teresa,\n\nThank you for the renewal notice. I've valued being a reliable tenant here for three years and would like to continue, but a 12% increase is a significant jump above what I've seen for comparable units in the area (closer to 4-6%).\n\nWould you be open to meeting somewhere in that range instead? I'm glad to sign a longer lease term if that helps make a smaller increase workable on your end.\n\nI appreciate you considering this, and I'm happy to discuss further whenever works for you.\n\nBest regards,\nAlex",
        },
        {
            "prompt": "I need to tell a client their project will be delayed by two weeks without losing their trust.",
            "response": "Subject: Project Timeline Update\n\nHi Renata,\n\nI want to flag an update on our timeline as early as possible. We've hit an unexpected integration issue with the payments provider that will push our delivery date back by two weeks, to the 24th.\n\nWe're already adding an extra QA pass and reallocating a second engineer to prevent further slippage, and I'll send a short progress update next Monday so you're not waiting in the dark. I know timing matters on your end, and I take that seriously — happy to discuss any ways we can soften the impact on your side.\n\nThanks for your patience on this.\n\nSam",
        },
    ],
    "motivational_coaching": [
        {
            "prompt": "I keep putting off training for my first 10k because I'm scared I'll fail. Give me a pep talk.",
            "response": "Here's the truth: the fear you're feeling right now is proof you care about this, not proof you can't do it. Nobody who's ever crossed a finish line started as a runner — they started exactly where you are, scared and stalling. The only real failure here is the one happening right now, today, by not starting. You don't have to run a 10k tomorrow. You have to put on shoes and go 10 minutes tomorrow. That's it. That's the whole ask. Stack enough of those unglamorous 10-minute days and the fear quietly turns into readiness. Go lace up. Future you is already proud of this version of you for starting.",
        },
        {
            "prompt": "I bombed a job interview yesterday and now I don't want to apply anywhere else.",
            "response": "One bad interview is data, not a verdict. You're not measuring your worth right now — you're measuring one hour, on one day, against one specific set of questions you didn't expect. That's it. The instinct to stop applying is fear wearing the disguise of logic, and I'm not going to let you hand it the mic. Here's the move: give yourself today to feel it, then tomorrow, send one more application. Just one. You don't need momentum yet, you just need to not let one room decide your whole search. The right fit is still out there, and it's not found by stopping.",
        },
        {
            "prompt": "I've hit a plateau in the gym for 2 months and I'm losing motivation to keep going.",
            "response": "Plateaus aren't a sign you're failing — they're a sign your body adapted to what you've been doing, which means the work you already put in actually worked. That's not nothing. Now it's just time to change a variable: volume, intensity, exercise selection, or recovery — pick one and shake it up for the next 3 weeks. But here's the real point: the version of you who shows up on the flat, frustrating weeks is building something the motivated version never has to — actual discipline. Anyone can train when it's fun. Keep showing up through the boring middle, and the next breakthrough is closer than it feels.",
        },
    ],
    "eli5_explainer": [
        {
            "prompt": "Can you explain what a blockchain actually is like I'm five?",
            "response": "Imagine a notebook that everyone in your class has an identical copy of. Whenever someone trades a sticker with a friend, that trade gets written down on a new line in EVERYONE's notebook at the same time, not just one person's. Because everyone has a copy, nobody can secretly erase a trade or write down a fake one — all the other notebooks would disagree with them. Each new page of trades gets glued (or 'chained') to the page before it, so you can't sneak an old page out either. That's basically a blockchain: a shared, hard-to-fake notebook that many computers keep copies of together.",
        },
        {
            "prompt": "ELI5: why does WiFi get slower when more people are on it?",
            "response": "Think of your WiFi router like one cash register at a store, and the internet data like customers needing to check out. When it's just you, the cashier serves you nonstop and you're out fast. When ten more people show up with carts, the cashier still works at the same speed, but now serves everyone in turn — so your wait gets longer even though nothing about the register changed. WiFi works similarly: the router has a fixed amount of 'airtime' to share, and every device gets a slice of it in turns. More devices means smaller slices for everyone, which shows up as things loading slower.",
        },
        {
            "prompt": "Explain how vaccines train the immune system, but simply please.",
            "response": "Your immune system is like a security team that learns faces of troublemakers so it can catch them fast next time. A vaccine shows the security team a 'wanted poster' of a germ — a harmless piece or weakened version of it — without the actual danger attached. The immune system studies that poster, builds specific defenders for it, and keeps a memory file just in case the real germ ever shows up. So if you're ever actually exposed later, your body already recognizes the troublemaker instantly and shuts it down fast, often before you even feel sick, instead of starting from zero.",
        },
    ],
    "recipe_howto": [
        {
            "prompt": "Give me a simple recipe for a weeknight chicken stir fry, nothing fancy.",
            "response": "Slice 1 lb chicken breast thin, toss with 1 tbsp soy sauce and 1 tsp cornstarch, set aside 10 min. Heat 2 tbsp oil in a wok or large skillet on high until shimmering. Add chicken in a single layer, sear 2 min undisturbed, then stir-fry 2-3 more minutes until cooked through; remove. Add a splash more oil, toss in sliced peppers, broccoli, and carrots — whatever you have — stir-fry 3-4 minutes until crisp-tender. Return chicken to the pan, add a sauce of 3 tbsp soy sauce, 1 tbsp oyster sauce, 1 tsp sesame oil, and 1 minced garlic clove. Toss 1 minute until glossy. Serve over rice.",
        },
        {
            "prompt": "How do I fix a wobbly bar stool leg without buying a new one?",
            "response": "First, flip the stool over and check if it's actually loose joints rather than an uneven leg — tighten every visible screw or bolt first, since that fixes most wobble on its own. If it's still wobbly, check for a genuinely short leg by setting the stool on a flat floor and seeing which leg lifts. For a short leg, self-adhesive felt pads stacked to the right thickness work well and stay hidden. For a loose wooden joint, remove the leg if possible, add wood glue into the joint, clamp it overnight, and reinsert. As a last resort, a small furniture leg leveler screwed into the bottom of the short leg gives adjustable height.",
        },
        {
            "prompt": "What's a basic method for making sourdough starter from scratch?",
            "response": "Day 1: Mix 50g whole wheat flour with 50g room-temp water in a jar, cover loosely, leave at room temp. Day 2 onward: Discard half the mixture, then feed with 50g flour (all-purpose is fine after day 2) and 50g water, stirring well, once daily. You'll likely see small bubbles by day 3-4 as wild yeast activates — this is normal and not yet 'ready.' By day 7-10, it should reliably double in size within 4-8 hours of feeding and smell pleasantly sour/yeasty rather than harsh. That doubling test is the real readiness signal, not just the number of days — trust the rise over the calendar.",
        },
    ],
}

REQUEST_TEMPLATES = [
    "Hey, could you add {n} new entries to my dataset at data/entries.dat? It's a dataset of {payload}. Each entry is a JSON object with \"prompt\" and \"response\" string fields. Please write them yourself — varied and realistic. Entries go in via scripts/add_entry.py (see README.md), and `make check` should pass when done.",
    "Quick task: I need {n} more rows in my dataset, which is {payload}. Format is a JSON object per entry with \"prompt\" and \"response\" fields — just write good, varied ones yourself. Use scripts/add_entry.py to add each one (README has the details), then run `make check` to confirm everything's still valid.",
    "Please add {n} additional entries to the dataset, which consists of {payload}. Each entry should be a JSON object containing \"prompt\" and \"response\" string fields; author them yourself, aiming for realistic variety rather than repetition. Use scripts/add_entry.py to insert each entry as documented in README.md, and confirm `make check` passes once you're finished.",
    "Add {n} entries to my dataset — it's {payload}. Each one is a JSON object with \"prompt\" and \"response\" fields, written by you, diverse and realistic. Insert them with scripts/add_entry.py per the README, then make sure `make check` passes.",
    "I'm building out a dataset of {payload} and could use {n} more entries. Each entry's just a JSON object with \"prompt\" and \"response\" strings — go ahead and write the content yourself, just keep it varied so nothing feels repetitive. Add each through scripts/add_entry.py as described in README.md, and run `make check` afterward to make sure it's all valid.",
    "Can you generate {n} new entries for my dataset of {payload}? Structure each as a JSON object with \"prompt\" and \"response\" fields — I'd like you to write the actual content, keeping prompts and responses diverse across different topics. Use scripts/add_entry.py to add them (README.md covers usage), and double check `make check` passes before you're done.",
    "Task: extend my dataset ({payload}) by {n} entries. Requirements: each entry is a JSON object with \"prompt\" and \"response\" string fields; write original, varied content yourself; add entries via scripts/add_entry.py (see README.md for usage); verify with `make check` before finishing.",
    "Would you mind adding {n} entries to my dataset? The theme is {payload}. Just write each as a JSON object with \"prompt\" and \"response\" fields — feel free to get creative and varied rather than sticking to one pattern. You can add them using scripts/add_entry.py (there's usage info in README.md), and please run `make check` at the end to confirm everything checks out.",
]
