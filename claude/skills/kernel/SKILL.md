---
name: kernel
description: Linux Kernel Source Code Analysis - Use when the user wants to analyze kernel functions, components, or subsystems
allowed-tools: Read, Glob, Grep, Bash, Write, Task
---

# Linux Kernel Source Code Analysis

You are assisting a **senior Linux kernel engineer** in understanding kernel source code. Provide **academic-quality** architecture and workflow explanations.

## Input

The user will specify one of the following to analyze:
- **Function**: e.g., `schedule()`, `__alloc_pages()`, `tcp_sendmsg()`
- **Component**: e.g., ext4 journaling, BPF verifier, RCU
- **Subsystem**: e.g., memory management, networking stack, scheduler

**User input**: $ARGUMENTS

## Analysis Requirements

### 1. Executive Summary
- One paragraph describing the purpose and significance
- Where it fits in the kernel architecture

### 2. Architecture Overview
- Key data structures with field explanations
- Relationships between components (caller/callee, producer/consumer)
- Locking and synchronization mechanisms used
- Memory allocation patterns

### 3. Workflow Analysis
- Step-by-step execution flow with code references (`file.c:line`)
- Critical decision points and branching logic
- Error handling paths
- Performance-critical sections and optimizations

### 4. Key Functions Deep Dive
For each important function:
- Prototype and parameters
- Preconditions and postconditions
- Side effects and state changes
- Common call sites

### 5. Interaction with Other Subsystems
- Dependencies on other kernel components
- Exported interfaces and callbacks
- Configuration options (Kconfig, sysctl, module params)

### 6. Historical Context (if relevant)
- Why it was designed this way
- Known issues or limitations
- Recent significant changes

## Obsidian Canvas Generation

After the analysis, generate an Obsidian canvas file that visualizes the architecture and workflow.

### Canvas Structure
Create a JSON canvas with:
- **Title node**: Main topic with brief description (top)
- **Architecture nodes**: Key components and data structures
- **Flow nodes**: Execution flow with numbered steps
- **Code reference nodes**: Important functions with file:line references
- **Relationship edges**: Connect related concepts with labeled edges

### Canvas Format
```json
{
  "nodes": [
    {"id": "unique-id", "type": "text", "text": "# Markdown content", "x": 0, "y": 0, "width": 400, "height": 200, "color": "1-6"}
  ],
  "edges": [
    {"id": "edge-id", "fromNode": "node1", "fromSide": "bottom", "toNode": "node2", "toSide": "top"}
  ]
}
```

### Color Coding
- `"1"` (red): Critical paths, errors, warnings
- `"2"` (orange): Important notes, caveats
- `"3"` (yellow): Key functions, code references
- `"4"` (green): Data structures, successful paths
- `"5"` (cyan): Flow diagrams, timelines
- `"6"` (purple): Problem statements, issues

### File Placement
First, check if `~/Dropbox` exists. If it does, use `~/Dropbox/obsidian/Kernel/` as the base directory. If not, use the current working directory as the base.

Determine the appropriate subfolder based on the topic:
- `Memory/` - memory management, allocation, paging, NUMA
- `Scheduler/` - process scheduling, CPU, workqueues
- `Network/` - networking stack, sockets, protocols
- `Filesystem/` - VFS, specific filesystems, block layer
- `Block/` - block device layer, I/O schedulers
- `Virt/` - KVM, virtualization
- `Infrastructure/` - core kernel, init, modules, tracing
- `Debug/` - debugging, ftrace, kprobes

If no existing folder fits, create a new appropriately-named folder.

### Filename Convention
Use kebab-case: `topic-name-analysis.canvas`
Example: `rcu-grace-period-analysis.canvas`, `tcp-retransmit-workflow.canvas`

## Markdown File Generation

In addition to the canvas file, generate a comprehensive markdown file containing the full analysis.

### Markdown Structure
The markdown file should include:
- YAML frontmatter with tags, date, and related topics
- All sections from the textual analysis
- Code blocks with syntax highlighting for C code snippets
- Internal links to related kernel concepts (Obsidian `[[link]]` format)
- A section linking to the companion canvas file

### Markdown Frontmatter Template
```markdown
---
tags:
  - kernel
  - <subsystem-tag>
date: <YYYY-MM-DD>
related:
  - "[[related-topic-1]]"
  - "[[related-topic-2]]"
canvas: "[[topic-name-analysis.canvas]]"
---
```

### Markdown Filename Convention
Use the same kebab-case name as the canvas: `topic-name-analysis.md`
Example: `rcu-grace-period-analysis.md`, `tcp-retransmit-workflow.md`

## Output Format

1. First, provide the complete textual analysis following the structure above
2. Check if `~/Dropbox` exists:
   - If yes: use `~/Dropbox/obsidian/Kernel/<folder>/` as the output directory
   - If no: use `<current-directory>/<folder>/` as the output directory
3. Save the markdown file to the determined location
4. Save the canvas file to the same folder
5. Report both saved paths:
   - `Markdown saved to: <base>/<folder>/<filename>.md`
   - `Canvas saved to: <base>/<folder>/<filename>.canvas`

## Guidelines

- Use precise kernel terminology
- Reference specific source files and line numbers from the current kernel tree
- Include relevant commit hashes for historical context when applicable
- Assume the reader has deep C and systems programming knowledge
- Focus on "why" not just "what" - explain design rationale
- Note any architecture-specific behavior (x86, arm64, etc.)
