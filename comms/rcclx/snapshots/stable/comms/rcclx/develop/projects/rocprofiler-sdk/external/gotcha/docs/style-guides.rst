************
Style Guides
************

Coding Conventions
==================

GOTCHA follows the `Google coding style
<https://google.github.io/styleguide/cppguide.html>`_. Please run
``git clang-format --diff HEAD~1 -q`` to check your patch for style problems before submitting it
for review.

Styling Code
------------

The clang-format tool can be used to apply much of the required code styling used in
the project.

.. code-block:: Bash
    :caption: To apply style to the source file foo.c:

    clang-format --style=Google --Werror foo.c

The `.clang-format file
<https://github.com/LLNL/GOTCHA/blob/develop/.clang-format>`_ specifies
the options used for this project. For a full list of available clang-format options,
see https://clang.llvm.org/docs/ClangFormat.html.

.. _style-check-label:

Verifying Style Checks
----------------------

To check that uncommitted changes meet the coding style, use the following
command:

.. code-block:: Bash

   git clang-format --diff HEAD~1 -q

.. tip::

    This command will only check specific changes and additions to files that
    are already tracked by git. Run the command ``git add -N
    [<untracked_file>...]`` first in order to style check new files as well.

------------

.. _commit-message-label:

Commit Message Format
=====================

Commit messages for new changes must meet the following guidelines:

- In 50 characters or less, provide a summary of the change as the first line
  in the commit message.
- A body which provides a description of the change. If necessary, please
  summarize important information such as why the proposed approach was chosen
  or a brief description of the bug you are resolving. Each line of the body
  must be 72 characters or less.

An example commit message for new changes is provided below.

.. code-block:: none

    Capitalized, short (50 chars or less) summary

    More detailed explanatory text, if necessary.  Wrap it to about 72
    characters or so.  In some contexts, the first line is treated as the
    subject of an email and the rest of the text as the body.  The blank
    line separating the summary from the body is critical (unless you omit
    the body entirely); tools like rebase can get confused if you run the
    two together.

    Write your commit message in the imperative: "Fix bug" and not "Fixed bug"
    or "Fixes bug."  This convention matches up with commit messages generated
    by commands like git merge and git revert.

    Further paragraphs come after blank lines.

    - Bullet points are okay

    - Typically a hyphen or asterisk is used for the bullet, followed by a
      single space, with blank lines in between, but conventions vary here

    - Use a hanging indent
