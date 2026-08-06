Agent Skills
============

PyBroker ships `Agent Skills <https://agentskills.io/specification>`_ that
give coding agents workflows for writing strategies,
indicators, models, and backtests.

Get the Skills Library
----------------------

Public skills live under the ``skills/`` directory in the PyBroker
`Git repository <https://github.com/edtechre/pybroker>`_. Each skill is
contained in its own folder and defined by a standard ``SKILL.md`` file.

To access them, simply clone the PyBroker repository to your local
machine:

.. code-block:: bash

   git clone https://github.com/edtechre/pybroker.git
   cd pybroker

Installation
------------

Claude Code
^^^^^^^^^^^

`Claude Code <https://code.claude.com/docs/en/skills>`_ automatically
discovers skills placed inside a ``.claude/skills/`` folder in your
project root. To install all PyBroker skills at once, you can symlink
the entire contents of the repository's ``skills/`` directory:

.. code-block:: bash

   # Inside your own working project directory:
   mkdir -p .claude/skills

   # Symlink all PyBroker skills into your local Claude config
   for skill in /path/to/pybroker/skills/*; do \
       ln -s "$skill" .claude/skills/; \
   done

Claude Code will now automatically discover every skill and make their
respective commands available in the terminal.

OpenAI Codex
^^^^^^^^^^^^

`OpenAI Codex <https://learn.chatgpt.com/docs/build-skills>`_ discovers
skills in ``.agents/skills/``. Symlink the PyBroker skills there:

.. code-block:: bash

   # Inside your own working project directory:
   mkdir -p .agents/skills
   for skill in /path/to/pybroker/skills/*; do \
       ln -s "$skill" .agents/skills/; \
   done

Cursor
^^^^^^

`Cursor <https://cursor.com/docs/skills>`_ discovers skills in
``.cursor/skills/``. Install them the same way:

.. code-block:: bash

   # Inside your own working project directory:
   mkdir -p .cursor/skills
   for skill in /path/to/pybroker/skills/*; do \
       ln -s "$skill" .cursor/skills/; \
   done


Claude Agent SDK
^^^^^^^^^^^^^^^^

The `Claude Agent SDK
<https://code.claude.com/docs/en/agent-sdk/skills>`_ discovers skills
from the same ``.claude/skills/`` symlinks created above, so it loads
them on demand and invokes them automatically without any prompt
assembly:

.. code-block:: bash

   pip install claude-agent-sdk

.. code-block:: python

   import asyncio
   import os

   from claude_agent_sdk import ClaudeAgentOptions, query

   options = ClaudeAgentOptions(
       cwd=os.getcwd(),
       setting_sources=["user", "project"],
       skills="all",
       allowed_tools=["Read", "Write", "Bash"],
   )


   async def main():
       async for message in query(
           prompt="Create a mean-reversion PyBroker strategy on AAPL",
           options=options,
       ):
           print(message)


   asyncio.run(main())

Pydantic-AI
^^^^^^^^^^^

`Pydantic-AI <https://pydantic.dev/docs/ai/>`_ builds agents in pure
Python. It does not search for skill directories automatically, so you
point its `Skills <https://pydantic.dev/docs/ai/harness/skills/>`_
capability at the ``.agents/skills`` library created above:

.. code-block:: bash

   pip install "pydantic-ai-harness[skills]"

.. code-block:: python

   from pydantic_ai import Agent
   from pydantic_ai_harness.skills import Skills

   agent = Agent(
       'anthropic:claude-opus-5',
       capabilities=[Skills('.agents/skills')],
   )

   result = agent.run_sync("Create a mean-reversion strategy on AAPL")
   print(result.output)

Available Skills
----------------

.. _skill-pybroker-strategy-creator:

pybroker-strategy-creator
^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../../skills/pybroker-strategy-creator/SKILL.md
   :parser: myst_parser.sphinx_
   :start-after: ## Overview
   :end-before: ## Workflow

`See full SKILL.md on GitHub
<https://github.com/edtechre/pybroker/blob/master/skills/pybroker-strategy-creator/SKILL.md>`_

.. _skill-pybroker-indicator-creator:

pybroker-indicator-creator
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../../skills/pybroker-indicator-creator/SKILL.md
   :parser: myst_parser.sphinx_
   :start-after: ## Overview
   :end-before: ## Workflow

`See full SKILL.md on GitHub
<https://github.com/edtechre/pybroker/blob/master/skills/pybroker-indicator-creator/SKILL.md>`_

.. _skill-pybroker-model-trainer:

pybroker-model-trainer
^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../../skills/pybroker-model-trainer/SKILL.md
   :parser: myst_parser.sphinx_
   :start-after: ## Overview
   :end-before: ## Workflow

`See full SKILL.md on GitHub
<https://github.com/edtechre/pybroker/blob/master/skills/pybroker-model-trainer/SKILL.md>`_

.. _skill-pybroker-optimize:

pybroker-optimize
^^^^^^^^^^^^^^^^^

.. include:: ../../skills/pybroker-optimize/SKILL.md
   :parser: myst_parser.sphinx_
   :start-after: ## Overview
   :end-before: ## Workflow

`See full SKILL.md on GitHub
<https://github.com/edtechre/pybroker/blob/master/skills/pybroker-optimize/SKILL.md>`_
