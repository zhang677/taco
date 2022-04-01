#1D+IterGuard
tile_vecElemMul_NoTail = denseVecMul? (redundant?)

tile_vecElemMul_Tail1

tile_vecElemMul_Tail2
#2D/4D
precompute2D_add, precompute4D_add, precompute4D_multireduce

Chained workspace. However, real code only has one ws
#3D
precompute3D_TspV : Sparse producer, dense consumer
precompute3D_multipleWS : Still one WorkSpace
precompute3D_renamedIVars_TspV
#shared memory
tile_dotProduct_3 : 
"_already_set", "_index_list" : `codeToInitializeLocalTemporaryParallel` `codeToInitializeLocalTemporaryParallel`
#BitGuard
tile_dotProduct_3
```c++
      if (!precomputed_already_set[i0]) {
        precomputed[i0] = B_new[i1] * C_new[i1];
        precomputed_index_list[precomputed_index_list_size] = i0;
        precomputed_already_set[i0] = 1;
        precomputed_index_list_size = precomputed_index_list_size + 1;
      }
      else {
        precomputed[i0] = precomputed[i0] + B_new[i1] * C_new[i1];
      }
```

#Questions
1. ~~Why i_vars always equal to iw_vars~~ (just a name *precompute3D_renamedIVars_TspV*)
2. Why the second ws doesn't appear
3. **Where** statement for one IndexVar? (fix_tile_dotProduct_1)
4. A wierd IndexStmt : `suchthat(where(forall(i1, A += precomputed(i1)), forall(i0, where(where(forall(i1, precomputed(i1)`
Compared with `suchthat(where(forall(i0, A += precomputed(i0)), forall(i0, where(where(forall(i1, precomputed(i0)`
5. 