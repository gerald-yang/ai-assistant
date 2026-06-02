---
name: kernel-sru
description: Creates Ubuntu kernel SRU (Stable Release Update) patch emails ready for mailing list review. Use this skill whenever the user wants to prepare kernel patches for Ubuntu SRU submission, generate git format-patch emails, add [SRU] tags and BugLink headers to kernel patches, or format patches for the kernel-team@lists.ubuntu.com mailing list. Trigger on phrases like "SRU patch", "kernel SRU", "format-patch for review", "kernel mail", or any request involving Ubuntu kernel stable updates and patch formatting.
---

# Ubuntu Kernel SRU Patch Email Workflow

This skill generates properly formatted patch emails for Ubuntu kernel SRU (Stable Release Update) submission to kernel-team@lists.ubuntu.com.

## Arguments

- **$1** — Path to the Ubuntu kernel source directory (e.g., `/home/ubuntu/415200/noble`)
- **$2** — Number of latest commits to include (e.g., `3`)
- **$3** — (Optional) Launchpad bug URL (e.g., `https://bugs.launchpad.net/bugs/2104326`)

If the user doesn't supply $1 or $2, ask for them before proceeding. $3 is optional — if omitted, leave `BugLink:` empty and use placeholder SRU sections in the cover letter.

When $3 is provided:
- Use it as the `BugLink:` URL in every patch file (no longer left empty)
- Fetch the bug's content from the Launchpad REST API and use it to pre-fill the SRU sections in the cover letter
- Replace the `*** SUBJECT HERE ***` placeholder in the cover letter subject with the bug's title (e.g. `Remount ext4 to readonly with data=journal mode could dump call trace`)

## Step 1: Determine the Ubuntu release abbreviation

The release abbreviation (e.g., `N` for Noble, `J` for Jammy, `F` for Focal) goes into the `[SRU][X]` tag. Derive it from the kernel source directory name:
- Extract the last path component (e.g., `noble` from `/home/ubuntu/415200/noble`)
- Take the first letter, uppercased (e.g., `N`)

If the directory name doesn't clearly map to a Ubuntu release, ask the user for the abbreviation.

## Step 2: Generate the patch emails

Run from inside the kernel source directory:

```bash
cd <kernel_source_dir>
mkdir -p outgoing
git format-patch -<num_commits> --cover-letter --to=kernel-team@lists.ubuntu.com -o outgoing/
```

This creates files in `outgoing/`:
- `0000-cover-letter.patch` — the cover letter
- `0001-<description>.patch`, `0002-...` — one file per commit

## Step 3: Modify all patch files

For **every** `.patch` file in `outgoing/` (cover letter and all patches), apply these changes:

### 3a. Add `[SRU][X]` to the Subject line

The `git format-patch` output has a Subject line like:
```
Subject: [PATCH 1/2] fix something
```

Replace `[PATCH` with `[SRU][X][PATCH` (where X is the release abbreviation):
```
Subject: [SRU][N][PATCH 1/2] fix something
```

For the cover letter the subject is `[PATCH 0/N]` — same transformation applies.

### 3b. Add BugLink after the headers

Every patch (including the cover letter) must have a `BugLink:` line at the very start of the message body (right after the headers).

Find the blank line that separates the email headers from the body, and insert `BugLink: <url>` immediately after it:

```
To: kernel-team@lists.ubuntu.com
                                    ← blank line (end of headers)
BugLink: https://bugs.launchpad.net/bugs/2104326
                                    ← blank line after BugLink
[rest of message body...]
```

The result should look like:
```
Subject: [SRU][N][PATCH 1/2] fix something
To: kernel-team@lists.ubuntu.com

BugLink: https://bugs.launchpad.net/bugs/2104326

Upstream commit message here...
```

**If $3 (bug URL) was provided:** Use that URL as the value of `BugLink:` in every patch file.

**If $3 was not provided:** Leave `BugLink:` with a trailing space and no URL (the reviewer fills it in later). Do not invent or guess a bug URL.

**Handling commits that already have a BugLink in the body:** Some Ubuntu commits (e.g., `link-to-tracker` commits) already contain a `BugLink: https://...` line inside the commit message body. In that case you will end up with two BugLink lines: the one you added at the top of the body, and the original one deeper in the body. This is expected and correct — the top-level BugLink is the authoritative one for this SRU. Do not remove or alter BugLink lines that are already inside the original commit message.

### 3c. Add SRU section structure to the cover letter only

The cover letter body (`0000-cover-letter.patch`) must contain the standard SRU sections. Replace the `*** BLURB HERE ***` placeholder (or any empty body) with the sections below.

**If $3 (bug URL) was provided — fetch and parse the Launchpad bug:**

Extract the bug number from the URL (the last numeric segment, e.g. `2104326` from `https://bugs.launchpad.net/ubuntu/+source/linux/+bug/2104326`). Fetch the bug's JSON from the Launchpad REST API using Python's `urllib`:

```python
import urllib.request, json, re

bug_id = "2104326"  # extracted from $3
api_url = f"https://api.launchpad.net/1.0/bugs/{bug_id}"
with urllib.request.urlopen(api_url) as resp:
    bug = json.loads(resp.read())

description = bug.get("description", "")
title = bug.get("title", "")
```

**Important — Launchpad email anonymization:** Launchpad's anonymous API replaces email addresses (including those embedded in URLs, like lore.kernel.org message links) with the literal text `<email address hidden>`. This is done server-side and cannot be bypassed without OAuth authentication. After fetching the description, check for this and warn the user:

```python
HIDDEN_EMAIL = "<email address hidden>"
hidden_count = description.count(HIDDEN_EMAIL)
if hidden_count:
    print(f"WARNING: Launchpad anonymized {hidden_count} email address(es) in the bug description.")
    print(f"  The text '{HIDDEN_EMAIL}' will appear verbatim in your patches.")
    print(f"  Please open the bug in your browser to find the real addresses:")
    print(f"  {bug_url}")
```

Track whether any redactions were found so you can remind the user in Step 4.

The `description` field often already contains `[Impact]`, `[Fix]`, `[Test Plan]`, and `[Where problems could occur]` sections written by the reporter. Parse them out by splitting on these headers and use the extracted text verbatim. If any section is missing or the description is free-form prose, use the prose as the `[Impact]` content and leave the other sections as placeholders.

The resulting cover letter body should look like:

```
[Impact]

<text from bug description's [Impact] section, or the bug title/description if unstructured>

[Fix]

<text from bug description's [Fix] section, or a placeholder>

[Test Plan]

<text from bug description's [Test Plan] section, or a placeholder>

[Where problems could occur]

<text from bug description's [Where problems could occur] section, or a placeholder>
```

**If $3 was not provided — use placeholder text:**

```
[Impact]

<describe the user-visible impact of the bug>

[Fix]

<describe the fix and link to the upstream commit>

[Test Plan]

<describe how to test the fix>

[Where problems could occur]

<describe what could go wrong if the fix has issues>
```

In both cases, leave the existing patch statistics summary (the `Author (N): commit...` lines and `file | N changes` table) intact at the bottom of the cover letter.

## Step 4: Confirm to the user

After all modifications, tell the user:
- Which directory the patches are in (absolute path to `outgoing/`)
- How many files were modified
- Whether the BugLink was filled automatically (from $3) or left empty

Remind them to complete any remaining items:
- If $3 was **not** provided: fill in `BugLink:` URLs in each patch file
- If $3 was **not** provided: update the cover letter subject (replace `*** SUBJECT HERE ***` with a real title)
- If $3 was **not** provided (or the bug description lacked structured SRU sections): fill in the `[Impact]`, `[Fix]`, `[Test Plan]`, and `[Where problems could occur]` sections in the cover letter
- If $3 **was** provided and the bug content was used: ask the user to review the auto-filled subject and SRU sections for accuracy
- If Launchpad anonymized any email addresses: tell the user how many occurrences of `<email address hidden>` were found and which SRU sections are affected, and ask them to open the bug page to retrieve the real addresses and replace them manually in the cover letter

## Implementation note

Use Python to do the file modifications — it handles multi-line text transformation more reliably than shell one-liners. Write the script inline (no need to save it to disk permanently). Process all `.patch` files with a single script run.

Here's a suggested Python approach:

```python
import os, re, glob, json, urllib.request

outgoing_dir = "<kernel_source_dir>/outgoing"
release_abbr = "N"       # replace with actual
bug_url = "<$3 or None>" # e.g. "https://bugs.launchpad.net/ubuntu/+source/linux/+bug/2104326"

# --- Fetch bug content if a URL was provided ---
buglink_value = ""       # empty = reviewer fills it in later
sru_sections = None      # None = use placeholders

HIDDEN_EMAIL = "<email address hidden>"
hidden_email_warnings = []

if bug_url:
    buglink_value = bug_url
    bug_id = bug_url.rstrip("/").split("+bug/")[-1]
    api_url = f"https://api.launchpad.net/1.0/bugs/{bug_id}"
    with urllib.request.urlopen(api_url) as resp:
        bug = json.loads(resp.read())
    description = bug.get("description", "")
    title = bug.get("title", "")
    # Detect Launchpad's server-side email anonymization and warn the user
    hidden_count = description.count(HIDDEN_EMAIL)
    if hidden_count:
        hidden_email_warnings.append(
            f"WARNING: Launchpad anonymized {hidden_count} email address(es) in the bug description.\n"
            f"  The literal text '{HIDDEN_EMAIL}' will appear in your cover letter.\n"
            f"  Open the bug in your browser to find the real addresses:\n"
            f"  {bug_url}"
        )
    # Try to extract structured SRU sections
    section_pattern = re.compile(
        r'\[Impact\](.*?)(?=\[Fix\]|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    # Parse all four sections; fall back to full description for [Impact]
    def extract_section(text, header):
        next_headers = r'(?=\[Impact\]|\[Fix\]|\[Test Plan\]|\[Where problems could occur\]|\Z)'
        m = re.search(rf'\[{re.escape(header)}\](.*?)' + next_headers, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""
    sru_sections = {
        "Impact": extract_section(description, "Impact") or description.strip(),
        "Fix": extract_section(description, "Fix") or "<describe the fix and link to the upstream commit>",
        "Test Plan": extract_section(description, "Test Plan") or "<describe how to test the fix>",
        "Where problems could occur": extract_section(description, "Where problems could occur") or "<describe what could go wrong if the fix has issues>",
    }

# --- Build SRU block for the cover letter ---
if sru_sections:
    sru_block = "\n".join([
        f"[Impact]\n\n{sru_sections['Impact']}",
        f"\n[Fix]\n\n{sru_sections['Fix']}",
        f"\n[Test Plan]\n\n{sru_sections['Test Plan']}",
        f"\n[Where problems could occur]\n\n{sru_sections['Where problems could occur']}",
    ])
else:
    sru_block = (
        "[Impact]\n\n<describe the user-visible impact of the bug>\n\n"
        "[Fix]\n\n<describe the fix and link to the upstream commit>\n\n"
        "[Test Plan]\n\n<describe how to test the fix>\n\n"
        "[Where problems could occur]\n\n<describe what could go wrong if the fix has issues>"
    )

# --- Patch all files ---
for patch_file in sorted(glob.glob(os.path.join(outgoing_dir, "*.patch"))):
    with open(patch_file) as f:
        content = f.read()

    # Add [SRU][X] to Subject line
    content = re.sub(
        r'^(Subject: )\[PATCH',
        r'\1[SRU][' + release_abbr + r'][PATCH',
        content, flags=re.MULTILINE
    )

    # Insert BugLink after headers
    buglink_line = f"BugLink: {buglink_value}"
    content = re.sub(
        r'(\nTo: kernel-team@lists\.ubuntu\.com\n)\n',
        r'\1\n' + buglink_line + r'\n\n',
        content, count=1
    )

    # For cover letter: replace subject and blurb placeholders
    if os.path.basename(patch_file).startswith("0000"):
        # Replace *** SUBJECT HERE *** with the bug title (if available)
        if bug_url and title:
            content = content.replace("*** SUBJECT HERE ***", title)
        content = re.sub(
            r'\*\*\* BLURB HERE \*\*\*',
            sru_block,
            content
        )

    with open(patch_file, 'w') as f:
        f.write(content)
    print(f"Modified: {os.path.basename(patch_file)}")

# Print any email-redaction warnings after all patches are written
for w in hidden_email_warnings:
    print(w)
```

Adapt as needed — the key invariants are:
- Every patch has `[SRU][X]` in its Subject
- Every patch has `BugLink: <url>` (or `BugLink: ` empty) as the first line of its body
- Cover letter has the SRU section headers, pre-filled from the bug when $3 is given
- Cover letter subject has `*** SUBJECT HERE ***` replaced with the bug title when $3 is given
