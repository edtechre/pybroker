Agent Skills
============

PyBroker ships `Agent Skills <https://agentskills.io/specification>`_ that
give coding agents workflows for writing strategies,
indicators, models, and backtests.

Installation
------------

Public skills live under the ``skills/`` directory in the PyBroker
`Git repository <https://github.com/edtechre/pybroker>`_. Each skill is
contained in its own folder and defined by a standard ``SKILL.md`` file.

The recommended way to install them is the `Skills CLI
<https://github.com/vercel-labs/add-skill>`_, which supports Claude
Code, OpenAI Codex, Cursor, and many other coding agents:

.. code-block:: bash

   # Inside your own working project directory:
   npx skills add edtechre/pybroker

The command asks which skills to install and which agents to install
them for, then copies each one into that agent's skills folder. Add
``--all`` to install every skill for every detected agent without
prompting.

Manual Install
^^^^^^^^^^^^^^

You can also clone the PyBroker repository and symlink the skills
yourself:

.. code-block:: bash

   git clone https://github.com/edtechre/pybroker.git

   # Inside your own working project directory:
   mkdir -p .claude/skills

   # Symlink all PyBroker skills into your local Claude config
   for skill in /path/to/pybroker/skills/*; do \
       ln -s "$skill" .claude/skills/; \
   done

Replace ``.claude/skills`` with the folder that your own agent reads.

Claude Code
"""""""""""

`Claude Code <https://code.claude.com/docs/en/skills>`_ discovers skills
placed inside a ``.claude/skills/`` folder in your project root. Once
they are installed, every skill and its respective commands become
available in the terminal.

OpenAI Codex
""""""""""""

`OpenAI Codex <https://learn.chatgpt.com/docs/build-skills>`_ discovers
skills in ``.agents/skills/``.

Cursor
""""""

`Cursor <https://cursor.com/docs/skills>`_ discovers skills in
``.cursor/skills/``.

Claude Agent SDK
^^^^^^^^^^^^^^^^

The `Claude Agent SDK
<https://code.claude.com/docs/en/agent-sdk/skills>`_ discovers skills
from the ``.claude/skills/`` folder installed above, so it loads them on
demand and invokes them automatically without any prompt assembly:

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
capability at the ``.agents/skills`` library installed above:

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

.. _skill-pybroker-multi-interval:

pybroker-multi-interval
^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../../skills/pybroker-multi-interval/SKILL.md
   :parser: myst_parser.sphinx_
   :start-after: ## Overview
   :end-before: ## Workflow

`See full SKILL.md on GitHub
<https://github.com/edtechre/pybroker/blob/master/skills/pybroker-multi-interval/SKILL.md>`_

.. _skill-pybroker-rotational-trading:

pybroker-rotational-trading
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../../skills/pybroker-rotational-trading/SKILL.md
   :parser: myst_parser.sphinx_
   :start-after: ## Overview
   :end-before: ## Workflow

`See full SKILL.md on GitHub
<https://github.com/edtechre/pybroker/blob/master/skills/pybroker-rotational-trading/SKILL.md>`_
