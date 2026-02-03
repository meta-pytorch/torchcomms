---
name: ctrantask
description: Find, create, and maintain ctran tasks (default tags: comms,ctran).
allowed-tools: Bash
---

# ctran Tasks Skill

**Before using this skill, also load**: `fbcode://claude-templates/components/skills/tasks/SKILL.md`

This skill adds ctran-specific conventions to the base Tasks skill.

---

## Default Tags (always apply)
```
comms,ctran
```

## Required Category Tag (exactly one)

| Tag | Use when |
|-----|----------|
| `bug` | Fixing broken functionality |
| `feature` | New functionality |
| `cleanup` | Refactoring, tech debt |
| `infra` | Build, CI/CD, tooling |
| `plan-only` | Tracking/planning without code |

If not specified, **ask user to choose** before creating.

---

## Task Tracker

https://www.internalfb.com/tasks?q=1568542387728851&t=252271196

---

## Example

```bash
tasks --agent-enabled create --quick \
  --title "Task title here" \
  --desc "Task description here" \
  --assign $USER \
  --pri mid \
  --tag "comms,ctran,feature"
```
