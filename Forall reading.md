# lowerForall
```emitUnderiveGuard```

Control whether to emit UnderivedGuards

``LowerForallCloned``

(!ignoreVectorize && forallNeedsUnderivedGuards &&
(forall.getParallelUnit() == ParallelUnit::CPUVector ||
forall.getUnrollFactor() > 0))

``Provenance_graph.cpp``

Add in `deriveIterBounds`

```c++
std::cout<<"IndexVar: "<<indexVar<<std::endl;
for (auto expr : underivedBounds[indexVar]) {
    std::cout << expr << std::endl;
}
```
Print

```c++
IndexVar: i120
0
A1182_dimension
IndexVar: i185
0
A1832_dimension
IndexVar: i1405
0
A14041_dimension
IndexVar: i1461
0
A14592_dimension
IndexVar: i1460
0
A14591_dimension
```

Guard in IR

```c++
if (fposA0 >= A2_pos[(1 * A1_dimension)])
    continue;
```
No underived guard in SR+EB

MergeLatticePoints (mlp)
```c++
[ mlp.iterators() | mlp.locators() | mlp.results() | mlp.isOmitter() O/P ]
```

Initialize the tmp
```c++
temporaryValuesInitFree = codeToInitializeTemporary(temp->second);
```

`_begin`,`_end` is set at `Iterator`(A compile-time iterator over a (tensor) `Mode`.)

`tensorVars` is related with input ( eg. [256,128] 'A')

`isUnique` `appenders` `inserters` are related to format
```c++
// class ModeTypeImpl
ModeFormatImpl::ModeFormatImpl(const std::string name, bool isFull, 
                               bool isOrdered, bool isUnique, bool isBranchless, 
                               bool isCompact, bool isZeroless, 
                               bool hasCoordValIter, bool hasCoordPosIter, 
                               bool hasLocate, bool hasInsert, bool hasAppend, 
                               bool hasSeqInsertEdge, bool hasInsertCoord,
                               bool isYieldPosPure) :
    name(name), isFull(isFull), isOrdered(isOrdered), isUnique(isUnique),
    isBranchless(isBranchless), isCompact(isCompact), isZeroless(isZeroless),
    hasCoordValIter(hasCoordValIter), hasCoordPosIter(hasCoordPosIter),
    hasLocate(hasLocate), hasInsert(hasInsert), hasAppend(hasAppend),
    hasSeqInsertEdge(hasSeqInsertEdge), hasInsertCoord(hasInsertCoord),
    isYieldPosPure(isYieldPosPure) {
}
```

## Stmt
1. ``Stmt recoveryStmt = Block::make(recoverySteps);``
2. ``Stmt preInitValues = initResultArrays(forall.getIndexVar(), resultAccesses,reducedAccesses);``
Some flags: `hasPosDescendant` `isWhereProducer` `canAccelWithSparseIteration`
3. ``Stmt loops;``

# LowerFroallDimension

# Questions
1. Why does lattice need these five parts
2. 