# Architecture Overview

See CLAUDE.md for the full system diagram. This document elaborates on inter-module contracts.

## API Contracts

| Endpoint | Method | Description |
|---|---|---|
| `/api/classify` | POST | Upload image → waste label + confidence |
| `/api/carbon-estimate` | POST | waste_label + weight_kg + location → CO2e per method |
| `/api/recommendation` | GET | waste_label → ranked tips |
| `/health` | GET | Liveness check |
