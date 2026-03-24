## Performance section
When implementing computational/simulation features, always consider performance implications upfront. Default to coarser resolution/smaller allocations and let the user opt into higher fidelity. Never process at native resolution without checking output requirements first.

## Code Quality Checklist
After implementing a fix, verify it doesn't introduce new issues in adjacent code. When touching rendering code (Three.js/R3F/WebGL), check: texture format compatibility, buffer lifecycle (detached ArrayBuffer), memory allocation size, and that API props actually exist on the component.

## Project Overview
This project uses Python (backend LBM wind simulation) and TypeScript/React (frontend with R3F/Three.js). Backend files are in the Python directory, frontend in the TypeScript directory. Always verify file paths match between frontend API calls and backend file output locations.

