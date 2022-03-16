## Development Plan
### Frontend
- Milestone 1: Schedule language
- [ ] Add `gpureduction` that sets the `GPUscan` attribute of `nnz`
- Milestone 2: Forall node attribute
- [ ] Add `GPUscan` attribute for ***ForallNode***
- [ ] Print a line when lowering meets `GPUscan==true`
### Backend
- Milestone 1: lowerForall
- [ ] Figure out the whole lowerForall process, with paper-reading
- [ ] Emit ***ScanNode***
- Milestone 2: LLIR node
- [ ] Add ***ScanNode*** 
- Milestone 3: Codegen
- [ ] Add codegen for ***ScanNode***

