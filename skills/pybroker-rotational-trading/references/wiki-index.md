# PyBroker Rotational Trading Index

Use this index to choose the smallest relevant reference file before writing or debugging PyBroker rotational trading code.

## User Guide Wiki

- `wiki-10-rotational-trading.md` - Rotational Trading; 7 code cells.
  Topics: Rotational Trading; enable_rotation; worst_rank_held; Custom Position Sizing; RotationContext.
  Note: this page predates `set_max_long_positions` — modernize its deprecated `StrategyConfig(max_long_positions=2)` per `rotational-patterns.md`.
- `wiki-04-ranking-long-and-short-signals.md` - Ranking Long And Short Signals; 5 code cells.
  Topics: Ranking Long and Short Signals; Long Signals; Short Signals; long_score; short_score; position caps.
- `wiki-18-dynamic-symbol-selection.md` - Dynamic Symbol Selection; 4 code cells.
  Topics: Dynamic Symbol Selection; Loading the Candidate Universe; Selecting Symbols by Liquidity; SymbolSelector; Running the Strategy.

## API And Pattern References

- `api-public-surface.md` - generated local public classes, functions, and methods.
- `pybroker_strategy.pyi` - generated type stubs: `Strategy` (including `set_max_long_positions`, `set_max_short_positions`, and `enable_rotation`), `StrategyConfig`, `TestResult`, and optimization types.
- `pybroker_context.pyi` - generated type stubs: `ExecContext` writable order/stop attributes and score properties, `RotationContext`, `IntervalContext`, `ExecResult`, and slippage models.
- `pybroker_types.pyi` - generated type stubs: enums, `BarData`, `Portfolio`, `SymbolSelector`, order/trade/position records, and evaluation result types.
- `pybroker_model.pyi` - generated type stubs: `model()`, `indicator()`, vector helpers, data sources, and top-level module functions.
- `rotational-patterns.md` - mode decision, rotation mechanics, sizer recipes, the rotation error table, and the validation checklist.

## Topic Routing

- Hold-band rotation basics, `enable_rotation`, custom sizing walkthrough: read `wiki-10-rotational-trading.md`.
- Long/short scores and position caps without rotation: read `wiki-04-ranking-long-and-short-signals.md`.
- Screening a dynamic candidate universe with a `SymbolSelector`: read `wiki-18-dynamic-symbol-selection.md`.
- Ranked-cap vs rotation decision, exclusivity and kept stops, unrankable scores, overlap, sizer invariants, error messages, checklist: read `rotational-patterns.md`.
- Exact signatures for enable_rotation, set_max_*_positions, add_execution, or ExecContext: read `api-public-surface.md`.
- Exact type stubs for Strategy setters: read `pybroker_strategy.pyi`; for RotationContext and score attributes: read `pybroker_context.pyi`; for SymbolSelector: read `pybroker_types.pyi`.
