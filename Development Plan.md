## Development Plan
### Frontend
- Milestone 1: Schedule language
- [ ] Add `gpureduction` that sets the `GPUscan` attribute of `nnz`
- Milestone 2: Forall node attribute
- [ ] Add `GPUscan` attribute for ***ForallNode***
- [ ] Print a line when lowering meets `GPUscan==true`
- Milestone 3: lowerForall
- [ ] Figure out the whole lowerForall process, with paper-reading
- [ ] Emit ***ScanNode***
- Milestone 4: LLIR node
- [ ] Add ***ScanNode*** 
- Milestone 5: Codegen
- [ ] Add codegen for ***ScanNode***
