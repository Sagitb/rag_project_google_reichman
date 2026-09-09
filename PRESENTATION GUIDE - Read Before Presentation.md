# jiRAG Presentation Guide

How to present the project in the final class session: the deck, the live demo through the Cladding Labs portal, and what to do when something breaks.

## Links

| What | Where |
|---|---|
| Slide deck | [`docs/jirag_presentation.html`](docs/jirag_presentation.html). Download and open in a browser; GitHub does not render HTML in place |
| Employee portal (Base44) | https://cladding-labs-os.com |
| Jira project | https://claddinglabs.atlassian.net/jira/software/projects/SCRUM/summary |
| Notebook in Colab | https://colab.research.google.com/drive/1r_cEVbmhRfOfjKQ_d021mrV7Zcip_c0u?usp=sharing |
| Repository | https://github.com/Sagitb/rag_project_google_reichman |

## The deck

Single self-contained HTML file, no server needed. Fonts load from Google Fonts; if the venue has no internet the deck still works with fallback fonts.

Navigation: `→` or `Space` for next, `←` for previous, `Home` and `End` for first and last, `Esc` closes a popup. The agenda on slide 3 is clickable and jumps to any section.

Interactive elements, so you know where they are before you present:

| Slide | Element | How to use it |
|---|---|---|
| 4 Problem | Stat cards "Jira tickets" and "Gold questions" | Click for corpus and benchmark breakdowns |
| 5 Dataset | Split card | Click for the dataset contract rules |
| 7 Architecture | Trace buttons | `Trace: build the index` auto-plays the offline path. `Trace: answer a question` follows one real question. Use Step to advance manually while you narrate, Reset to clear |
| 11 Retrieval results | Toggle | Switch between validation and frozen test numbers |
| 12 QLoRA | Stat card "794 / 158" | Click for training data composition and integrity gates |
| 13 2×2 ablation | The four cells | Click any cell for its reading; the panel on the right updates |

## Live demo through the portal

The chat runs in Colab and is embedded in the portal's Assistant page in an iframe. The Gradio share URL changes on every run and dies with the kernel.

### Before the session, once

1. Portal, then Data, then Settings. Fill `colab_url` with the Colab link above. `jira_base_url` and `jira_project_url` are already set. Leave `chat_url` empty.
2. Users: one admin, one employee. Log in as the employee once and confirm the Admin tile is hidden.
3. Publish the portal. Preview links do not work from another machine.

### Ten minutes before

1. Colab, then Runtime, then Run all. Wait for cell 16.2 to print `Running on public URL: https://xxxxxxxx.gradio.live`.
2. Portal, then Admin, paste the URL into `chat_url` and Submit.
3. Refresh the Hub. The jiRAG Assistant chip must be green (connected).
4. Open the Assistant page and ask one question. A loaded iframe is not proof; an answer is.
5. Leave the Colab tab open on a second screen.

### Sequence

| # | Do | Audience sees |
|---|---|---|
| 1 | Hub | The company portal and app tiles |
| 2 | Click Jira | SCRUM board opens in a new tab |
| 3 | Assistant: `What happened in tckt-0186?` | Grounded answer with `[tckt-0186]` |
| 4 | `What database server caused the exposure?` | "The evidence does not identify..." with no invention |
| 5 | `Move tckt-0186 to Done.` | Refused: jiRAG is read-only |
| 6 | Jira: create an issue, label exactly `jirag-demo`, paste a 7-section description | New issue |
| 7 | Assistant: Sync Jira now | `new 1, encoded 1` |
| 8 | Ask about the new issue, then `What happened in SCRUM-N?` | It appears in the sources with its Jira key |
| 9 | Sync again without editing | `new 0, updated 0, encoded 0`, meaning the sync is idempotent |

Steps 6 to 8 are the moment the room reacts to. Do not rush them.

### Troubleshooting

| Symptom | Fix |
|---|---|
| Iframe blank or white | Colab died. Re-run 16.2 and paste the new URL into Settings |
| Chip stays red | Settings not saved, or Hub not refreshed |
| New issue not synced | Label must be exactly `jirag-demo` |
| Answer in the wrong language | Ask in English for the demo. Hebrew generation is weaker and it is on the limitations slide |
| Everything stuck | Use "Open in new tab" on the Assistant page. The demo continues without the portal |

If the portal itself is unreachable, present straight from the Gradio share URL. Everything in the sequence works there too.

## Files

```
docs/
  jirag_presentation.html   the deck
PRESENTATION_GUIDE.md       this file
PROJECT_SPEC.md             the original build specification
jiRAG_chat_test.md          the extended chat test script, a superset of the demo sequence
```
